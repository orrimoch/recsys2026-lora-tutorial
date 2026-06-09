"""Build LightGBM LambdaMART training features from train conversations.

For each music turn in each (sampled) train session:
  1. Build retrieval_input = newline-joined chat history up to the user turn.
  2. Run wRRF retrieval -> top-N candidate tids (N=100 by default).
  3. For each candidate, compute the feature vector. The CANONICAL, up-to-date
     feature list is the row dict emitted by extract_features() below — this
     docstring only sketches the main numeric ones. (Historical note:
     dense_meta_cos / dense_lyrics_cos were listed here but NEVER emitted; the
     real query-doc cosine feature is bge_cos, gated behind --bge-model.)
       Numeric (selected):
         - wrrf_score         : fusion score from RRF
         - bm25_score         : standalone BM25 score (proxy for name-match)
         - cfbpr_score        : user.track dot product (0 for cold users)
         - pop_log            : log1p(track popularity)
         - recency_years      : years from 2026 to release_date (0 if unknown)
         - tag_count          : number of tags on track
         - artist_in_query    : 1 if any artist name token appears verbatim in query
       Categorical (LGBM native):
         - goal_category      : conversation_goal.category (11 classes)
         - goal_specificity   : conversation_goal.specificity (4 classes)
         - user_age_group     : '10s'/'20s'/... (or 'unknown')
         - user_country       : country_code (or 'OTHER' for long-tail)
         - user_gender        : 'male'/'female'/'unknown'
  4. Label: 1 if candidate == gold_tid for this turn, else 0.
  5. query_id = f"{session_id}#{turn_number}" for LGBM query grouping.

Writes a parquet file with one row per (query_id, candidate_tid). Downstream
notebook trains a LambdaMART ranker using LightGBM's LambdaRank objective.

Usage:
    python scripts/build_lgbm_features.py --n-sessions 2000 --topk 100
    python scripts/build_lgbm_features.py --n-sessions 15000 --topk 50 --out data/lgbm_train.parquet

Default output: data/lgbm_features.parquet. Intended as a one-time
feature dump; re-run with different --seed to create a separate holdout
for validation.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

# Disable JAX GPU preallocation BEFORE importing datasets/transformers (they pull
# JAX transitively, and JAX grabs ~75% of VRAM on first use). Without this, this
# script — run as a !python subprocess from the notebook — would have JAX steal
# the GPU from its own torch encoder. Must precede the imports below. Mirrors the
# nb 70/71 cell-1 env setup.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402
from datasets import load_dataset, concatenate_datasets  # noqa: E402

from mcrs.db_item import MusicCatalogDB  # noqa: E402
from mcrs.crs_baseline import build_retrieval_query  # noqa: E402
from mcrs.retrieval_modules.sasrec_model import build_user_dialog, prior_turns  # noqa: E402
from mcrs.retrieval_modules import load_retrieval_module  # noqa: E402
from mcrs.retrieval_modules.cf_bpr import CF_BPR  # noqa: E402
from mcrs.retrieval_modules.rrf import RRF_MODEL  # noqa: E402
# session_match_features lives in the shared session_history module so that
# training (this script) and inference (mcrs/rerankers/lgbm_rerank.py) use the
# IDENTICAL implementation (no train/serve feature skew). Re-exported here so
# existing callers/tests that import it from this script keep working.
from mcrs.retrieval_modules.session_history import (  # noqa: E402,F401
    session_match_features,
)
from mcrs.retrieval_modules.clap_similarity import (  # noqa: E402
    clap_has_vector, clap_mean_vector, clap_session_similarity, load_clap_lookup,
)


def session_fold(session_id, num_folds):
    """Deterministic OOF fold for a session, in [0, num_folds). MUST be byte-for-
    byte identical to train_sasrec.session_fold — the OOF leak guarantee depends
    on train_sasrec EXCLUDING fold k from training while this script SELECTS fold
    k for feature-building, so both must map a given session to the same k.
    Pinned by tests/test_oof_fold_assignment.py."""
    h = int(hashlib.sha1(str(session_id).encode()).hexdigest()[:8], 16)
    return h % num_folds


# ------------------------------------------------------------------ cached lookups
def load_track_meta_lookup(item_db: MusicCatalogDB) -> dict[str, dict]:
    """{track_id -> flat metadata dict for feature extraction}."""
    out: dict[str, dict] = {}
    for tid, meta in item_db.metadata_dict.items():
        # Some fields are lists (artist_name etc); normalize to first string.
        def _first(v):
            if isinstance(v, list):
                return v[0] if v else None
            return v
        out[tid] = {
            "artist_name": _first(meta.get("artist_name")),
            "album_name": _first(meta.get("album_name")),
            "tag_list": meta.get("tag_list") or [],
            "popularity": float(meta.get("popularity") or 0.0),
            "release_date": meta.get("release_date"),
        }
    return out


def load_track_cfbpr(cache_dir: str) -> tuple[dict[str, int], np.ndarray, dict]:
    """{track_id -> row_idx} + the L2-normalized (T, 128) track matrix +
    {user_id -> 128-vec} warm-user embeddings (used for the cfbpr_score feature)."""
    cf = CF_BPR("", ["all_tracks"], [], cache_dir=cache_dir)
    tid_to_idx = {t: i for i, t in enumerate(cf.track_ids)}
    return tid_to_idx, cf.track_mat, cf.user_embs


def load_user_meta() -> dict[str, dict]:
    um = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    concat = concatenate_datasets([um[s] for s in um])
    out: dict[str, dict] = {}
    for r in concat:
        out[r["user_id"]] = {
            "age_group": r.get("age_group") or "unknown",
            "country_code": r.get("country_code") or "unknown",
            "gender": r.get("gender") or "unknown",
        }
    return out


# ------------------------------------------------------------------ wRRF runner
def _union_extra_config(use_sasrec: bool = False, w_sasrec: float = 1.0,
                        sasrec_model_dir: str = "sasrec_v1",
                        use_segment_routing: bool = False,
                        use_two_tower: bool = False, w_two_tower: float = 0.7,
                        two_tower_model_dir: str = "two_tower_v1") -> dict:
    """Assemble the wrrf_union_v1 extra_config for feature-build time. Any channel
    enabled here changes the fused wrrf_rank, so it MUST match serve — the reranker
    has to train on the exact pool it will serve on. Pure (no I/O) for testing."""
    extra: dict = {"use_segment_routing": True} if use_segment_routing else {}
    if use_sasrec:
        extra.update({"use_sasrec": True, "w_sasrec": w_sasrec,
                      "sasrec_model_dir": sasrec_model_dir})
    if use_two_tower:
        extra.update({"use_two_tower": True, "w_two_tower": w_two_tower,
                      "two_tower_model_dir": two_tower_model_dir})
    return extra


class WRRFRunner:
    """Thin wrapper that runs the 021-champion wRRF stack and returns ranked
    candidates with fusion score + position. Inference-friendly: the LGBM
    reranker at inference only needs (tid, wrrf_rank) per candidate — no
    per-sub ranks — so feature computation is cheap and deterministic."""

    def __init__(self, cache_dir: str, corpus_types: list[str],
                 use_sasrec: bool = False,
                 w_sasrec: float = 1.0,
                 sasrec_model_dir: str = "sasrec_v1",
                 use_segment_routing: bool = False,
                 use_two_tower: bool = False,
                 w_two_tower: float = 0.7,
                 two_tower_model_dir: str = "two_tower_v1"):
        # wrrf_union_v1 is the 3-channel recall union (lexical + frozen-Qwen
        # semantic + same-artist session continuity); session_cf was dropped
        # after the G1 ablation. The same-artist channel needs
        # batch_context['history_tids'] to fire — see run().
        self.use_sasrec = use_sasrec
        extra = _union_extra_config(
            use_sasrec=use_sasrec, w_sasrec=w_sasrec, sasrec_model_dir=sasrec_model_dir,
            use_segment_routing=use_segment_routing,
            use_two_tower=use_two_tower, w_two_tower=w_two_tower,
            two_tower_model_dir=two_tower_model_dir)
        self.wrrf = load_retrieval_module(
            "wrrf_union_v1",
            "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            ["all_tracks"],
            corpus_types,
            cache_dir,
            extra_config=extra,
        )

    def run(self, queries: list[str], topk: int,
            batch_context=None, user_ids=None) -> list[list[dict]]:
        """Per-query list of {tid, wrrf_rank} for the top-K fused candidates.
        wrrf_rank = 1 for the top of wRRF output, K for the bottom.

        When use_sasrec=True, each candidate dict also carries:
          "sasrec_rank": 1-indexed position in the SASRec sub's ranking,
                         or 10000 (SENTINEL) for candidates not in that sub.

        batch_context (per-query {history_tids: [...]}) + user_ids feed the
        session-aware union channels. RRF_MODEL accepts both; keep a try/except
        fallback to the no-kwargs call for safety against older retrievers."""
        if self.use_sasrec:
            per_sub, labels = self.wrrf.batch_per_sub_rankings(
                queries, user_ids=user_ids, batch_context=batch_context,
            )
            weights = [s["weight"] for s in self.wrrf.subs]
            fused_per_q = RRF_MODEL.fuse_per_sub(per_sub, weights, self.wrrf.k, topk)
            sidx = labels.index("sasrec_seq")
            result = []
            for q, tids in enumerate(fused_per_q):
                # Build tid -> 1-indexed rank map for the sasrec sub.
                rankmap = {tid: (rank + 1) for rank, tid in enumerate(per_sub[sidx][q])}
                # Cross-channel agreement (Lever 2): how many of the union's
                # channels surfaced each tid. Computed from this query's per-sub
                # lists (already in hand). Golds tend to be multiply-surfaced.
                hit_count: dict[str, int] = {}
                for s in range(len(per_sub)):
                    for t in per_sub[s][q]:
                        hit_count[t] = hit_count.get(t, 0) + 1
                result.append([
                    {
                        "tid": tid,
                        "wrrf_rank": r + 1,
                        "sasrec_rank": rankmap.get(tid, 10000),
                        "n_channels_hit": hit_count.get(tid, 1),
                    }
                    for r, tid in enumerate(tids)
                ])
            return result
        else:
            try:
                fused_per_q = self.wrrf.batch_text_to_item_retrieval(
                    queries, topk=topk, user_ids=user_ids, batch_context=batch_context,
                )
            except TypeError:
                fused_per_q = self.wrrf.batch_text_to_item_retrieval(queries, topk=topk)
            return [
                [{"tid": tid, "wrrf_rank": r + 1} for r, tid in enumerate(tids)]
                for tids in fused_per_q
            ]


# ------------------------------------------------------------------ features
def _tokenize_simple(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def compute_release_year_cyclical(release_date):
    """Sin/cos of year-mod-century. (0.0, 0.0) on missing/unparseable."""
    if not release_date:
        return 0.0, 0.0
    try:
        year = int(str(release_date)[:4])
    except (ValueError, TypeError):
        return 0.0, 0.0
    phase = 2 * math.pi * (year % 100) / 100.0
    return math.sin(phase), math.cos(phase)


def compute_tag_overlap(query, tag_list):
    """Count of tags present in the query (case-insensitive substring)."""
    if not tag_list:
        return 0
    q_lower = (query or "").lower()
    return sum(1 for t in tag_list if t and t.lower() in q_lower)


def last_turn_moved_toward_goal(assessments):
    """1 / 0 / -1 from the last assessment value. -1 for empty/None."""
    if not assessments:
        return -1
    last = assessments[-1]
    if last == "MOVES_TOWARD_GOAL":
        return 1
    if last == "DOES_NOT_MOVE_TOWARD_GOAL":
        return 0
    return -1


def query_drift_score(current_query, prior_queries, embedder=None):
    """Cosine sim between current and turn-1 query embeddings. 1.0 when no prior or no embedder."""
    if not prior_queries or embedder is None:
        return 1.0
    embs = embedder.encode([current_query, prior_queries[0]], normalize_embeddings=True)
    return float(embs[0] @ embs[1])


def build_pop_rank_pct_map(track_meta):
    """Returns {track_id: rank_pct in [0, 1]} where 0 = most popular, 1 = least.

    Tracks with missing popularity get pct=0.5 (neutral).
    """
    items = [(tid, float(m.get("popularity") or 0.0)) for tid, m in track_meta.items()]
    items.sort(key=lambda x: -x[1])
    n = max(1, len(items))
    out = {}
    for rank, (tid, pop) in enumerate(items):
        if pop <= 0.0:
            out[tid] = 0.5
        else:
            out[tid] = rank / n
    return out


def extract_features(
    query: str,
    candidates: list[dict],
    gold_tid: str,
    session_info: dict,
    user_info: dict,
    track_meta: dict[str, dict],
    cfbpr_tid_to_idx: dict[str, int],
    cfbpr_track_mat: np.ndarray,
    cfbpr_user_embs: dict[str, np.ndarray],
    query_tokens: set[str],
    pop_rank_pct: dict[str, float] | None = None,
    played_meta: list[dict] | None = None,
    clap_lookup: dict | None = None,
    played_tids: list | None = None,
    clap_mean: "np.ndarray | None" = None,
    with_bge: bool = False,
    with_relevance: bool = False,
) -> list[dict]:
    """One dict per candidate — becomes one row in the parquet output."""
    if played_meta is None:
        played_meta = []
    user_id = session_info["user_id"]
    user_emb = cfbpr_user_embs.get(user_id)
    goal_cat = (session_info.get("conversation_goal") or {}).get("category") or "unknown"
    goal_spec = (session_info.get("conversation_goal") or {}).get("specificity") or "unknown"
    age_group = user_info.get("age_group", "unknown")
    country = user_info.get("country_code", "unknown")
    gender = user_info.get("gender", "unknown")
    # Session-level (candidate-independent) — compute once, not once per candidate.
    last_goal_move = last_turn_moved_toward_goal(
        session_info.get("goal_progress_assessments")
    )

    rows: list[dict] = []
    for c in candidates:
        tid = c["tid"]
        m = track_meta.get(tid, {})
        artist = m.get("artist_name") or ""
        artist_in_query = 1 if artist and artist.lower() in " ".join(query_tokens) else 0
        tag_count = len(m.get("tag_list") or [])
        pop = m.get("popularity") or 0.0
        rd = m.get("release_date")
        try:
            year = int(str(rd)[:4]) if rd else None
        except ValueError:
            year = None
        recency = (2026 - year) if year else 0.0

        # cf-bpr score
        if user_emb is not None and tid in cfbpr_tid_to_idx:
            tv = cfbpr_track_mat[cfbpr_tid_to_idx[tid]]
            cfbpr_score = float(np.dot(user_emb, tv))
        else:
            cfbpr_score = 0.0

        rs_sin, rs_cos = compute_release_year_cyclical(rd)
        tag_overlap = compute_tag_overlap(query, m.get("tag_list"))
        pop_pct = pop_rank_pct.get(tid, 0.5) if pop_rank_pct is not None else 0.5

        row = {
            # ids
            "query_id": f"{session_info['session_id']}#{session_info['turn_number']}",
            "session_id": session_info["session_id"],
            "user_id": user_id,
            "turn_number": session_info["turn_number"],
            "candidate_tid": tid,
            # existing numeric features
            "wrrf_rank": c["wrrf_rank"],
            "cfbpr_score": cfbpr_score,
            "pop_log": float(np.log1p(pop)),
            "recency_years": float(recency),
            "tag_count": tag_count,
            "artist_in_query": artist_in_query,
            # NEW track-temporal features (15, 16)
            "release_year_sin": rs_sin,
            "release_year_cos": rs_cos,
            # NEW query-track feature (17)
            "tag_overlap_count": tag_overlap,
            # NEW session-state feature (18)
            "last_turn_moved_toward_goal": last_goal_move,
            # NOTE: bm25_rank_inv / dense_meta_rank_inv / dense_lyrics_rank_inv
            # were REMOVED — WRRFRunner never populated their per-channel ranks,
            # so each collapsed to 1/wrrf_rank (three collinear duplicates of the
            # dominant feature, wasting tree capacity). Do NOT re-add them by
            # wiring real per-channel ranks: that is the documented-dead
            # RRF-reweight lever (+recall, -dev) and sasrec_rank carries an
            # in-sample leak. Requires a clean_full retrain to take effect; the
            # f_idx-gated serve mirror (lgbm_rerank.py) auto-disables them then.
            # NEW reranker-output features — caller supplies; default 0.
            "ce_score": float(c.get("ce_score", 0.0)),
            "ce_rank_inv": 1.0 / max(1, c.get("ce_rank", c["wrrf_rank"])),
            # NEW session-position features (24, 25)
            "turn_number_feat": int(session_info["turn_number"]),
            "prior_track_count": int(session_info.get("prior_track_count", 0)),
            # NEW session-state feature (26) — caller supplies precomputed drift score
            "query_drift_score": float(session_info.get("query_drift_score", 1.0)),
            # NEW track-popularity feature (27)
            "pop_rank_pct": float(pop_pct),
            # NEW user-state feature (28)
            "is_warm_user": int(user_emb is not None),
            # categorical features (strings; LGBM can consume as category)
            "goal_category": str(goal_cat),
            "goal_specificity": str(goal_spec),
            "user_age_group": str(age_group),
            "user_country": str(country),
            "user_gender": str(gender),
            # label
            "label": 1 if tid == gold_tid else 0,
        }
        # session-continuity features (29-31)
        row.update(session_match_features(m, played_meta))
        # NEW sasrec channel feature (32) — only when the sasrec sub was active.
        if "sasrec_rank" in c:
            row["sasrec_rank_inv"] = 1.0 / max(1, c["sasrec_rank"])
        # Lever 2: how many union channels surfaced this candidate (cross-channel
        # agreement). Carried on the candidate dict by the feature builder; only
        # emitted when present so legacy builds are unchanged.
        if "n_channels_hit" in c:
            row["n_channels_hit"] = int(c["n_channels_hit"])
        # NEW bge-v2 bi-encoder features (33, 34) — only when the build opted into
        # bge (--bge-model + --bge-cache-dir). Mirrors the ce_score/ce_rank_inv
        # pair EXACTLY for train/serve parity: bge_cos passes through unchanged,
        # bge_rank_inv = 1/max(1, rank). Omitted entirely when bge is off so
        # existing parquets/models are byte-unchanged.
        if with_bge:
            row["bge_cos"] = float(c.get("bge_cos", 0.0))
            row["bge_rank_inv"] = 1.0 / max(1, c.get("bge_rank", c["wrrf_rank"]))
        # Tier-2 #4.1 leak-free relevance features (pretrained Qwen3 dense cosine +
        # raw BM25). Passthrough from the candidate dict (the RelevanceScorer
        # injects them upstream); identical read at serve in lgbm_rerank ->
        # train/serve parity. Omitted unless the build opted in (--with-relevance).
        if with_relevance:
            row["qwen_meta_cos"] = float(c.get("qwen_meta_cos", 0.0))
            row["bm25_score"] = float(c.get("bm25_score", 0.0))
        # CLAP audio-similarity feature: candidate's acoustic similarity to the
        # session's played tracks. Only emitted when the clap lookup + played tids
        # are supplied (i.e. the build opted into clap), so legacy builds unchanged.
        if clap_lookup is not None and played_tids is not None:
            row["clap_session_sim"] = clap_session_similarity(
                tid, played_tids, clap_lookup, mean_vec=clap_mean)
            # Companion indicator: 1 if the candidate had a real CLAP vector, 0
            # if its similarity was mean-imputed. Lets the reranker discount
            # imputed rows instead of overloading the 0.0 sentinel.
            row["clap_has_vector"] = clap_has_vector(tid, clap_lookup)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ driver
def build(
    n_sessions: int,
    topk: int,
    seed: int,
    out_path: str,
    cache_dir: str,
    use_sasrec: bool = False,
    w_sasrec: float = 1.0,
    sasrec_model_dir: str = "sasrec_v1",
    oof_fold: int = None,
    oof_num_folds: int = None,
    use_clap: bool = False,
    bge_model: str = None,
    bge_cache_dir: str = None,
    with_relevance: bool = False,
    dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    use_segment_routing: bool = False,
    use_two_tower: bool = False,
    w_two_tower: float = 0.7,
    two_tower_model_dir: str = "two_tower_v1",
) -> None:
    # bge-v2 bi-encoder features are OFF unless BOTH flags are provided, so any
    # existing run (no bge args) is byte-identical to before.
    with_bge = bool(bge_model) and bool(bge_cache_dir)
    print(f"[lgbm-features] loading train split")
    tr = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(tr)), min(n_sessions, len(tr)))
    sessions = tr.select(indices).to_list()
    print(f"[lgbm-features] sampled {len(sessions)} sessions (seed={seed})")
    # OOF mode: keep ONLY sessions in fold `oof_fold`, scored by a SASRec that
    # held that fold OUT of training (--sasrec-model-dir must point at that
    # fold's model). Same (seed, n_sessions) across all K fold runs -> the K
    # per-fold parquets partition the full sample; concatenate them for the
    # leak-free OOF train parquet. See tests/test_oof_fold_assignment.py.
    if oof_fold is not None and oof_num_folds is not None:
        before = len(sessions)
        sessions = [s for s in sessions
                    if session_fold(s["session_id"], oof_num_folds) == oof_fold]
        print(f"[lgbm-features] OOF fold {oof_fold}/{oof_num_folds}: kept "
              f"{len(sessions)}/{before} sessions (model={sasrec_model_dir})")

    print(f"[lgbm-features] loading item_db + user_meta + cf-bpr + wRRF")
    item_db = MusicCatalogDB(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        ["all_tracks"],
        ["track_name", "artist_name", "album_name"],
    )
    track_meta = load_track_meta_lookup(item_db)
    user_meta = load_user_meta()
    cfbpr_tid_to_idx, cfbpr_track_mat, cfbpr_user_embs = load_track_cfbpr(cache_dir)
    pop_rank_pct = build_pop_rank_pct_map(track_meta)
    # CLAP audio lookup (Lever: clap_session_sim feature). Loaded only when
    # requested so non-clap builds don't pay the embedding download.
    #
    # GPU RE-TEST RUNBOOK (step 1, not yet run — Stage 22's CLAP regression was a
    # LEAK artifact, not a verdict that audio is useless). To evaluate CLAP (or
    # any reranker feature) HONESTLY, the train+val parquets must be built OOF and
    # scored on the held-out test split, NOT internal LGBM val (which shares the
    # full-train SASRec pool and is anti-correlated with dev):
    #   1. Train K OOF SASRec models, each holding one fold out (train_sasrec.py
    #      --oof-fold k --oof-num-folds K).
    #   2. Build K per-fold parquets here with --oof-fold k --oof-num-folds K
    #      --sasrec-model-dir <fold-k model> --use-clap; concatenate -> leak-free
    #      OOF train parquet. (clap_session_sim + clap_has_vector now mean-impute
    #      missing vectors instead of the old 0.0 sentinel.)
    #   3. Eval the retrained LGBM on the dev TEST split (nb74 cell 47), never the
    #      internal val nDCG. Only then is a CLAP verdict trustworthy.
    # See feedback_dont_chase_tiny_rerank_deltas + project_recall_levers_exhausted.
    clap_lookup = load_clap_lookup(cache_dir) if use_clap else None
    # Catalog-mean CLAP vector for imputing candidates with no audio embedding
    # (replaces the 0.0 sentinel). Computed once; None when clap is off.
    clap_mean = clap_mean_vector(clap_lookup) if clap_lookup is not None else None
    scorer = WRRFRunner(
        cache_dir=cache_dir,
        corpus_types=["track_name", "artist_name", "album_name"],
        use_sasrec=use_sasrec,
        w_sasrec=w_sasrec,
        sasrec_model_dir=sasrec_model_dir,
        use_segment_routing=use_segment_routing,
        use_two_tower=use_two_tower,
        w_two_tower=w_two_tower,
        two_tower_model_dir=two_tower_model_dir,
    )

    # bge-v2 bi-encoder feature (bge_cos + bge_rank). Loaded once when enabled.
    # The model was fine-tuned on bge_m3_structured queries, so we build a
    # PARALLEL structured query per turn (the main `queries` are RAW) and encode
    # against the bge catalog embeddings. cat_mat is L2-normalized so a dot
    # product is a cosine. See nb78 C1 for the parity-locked reference.
    bge_enc = None
    bge_cat_mat = None
    bge_cat_tid_to_idx: dict[str, int] = {}
    if with_bge:
        from sentence_transformers import SentenceTransformer  # noqa: E402
        import torch  # noqa: E402
        emb_path = os.path.join(bge_cache_dir, "track_embeddings.pkl")
        print(f"[lgbm-features] bge: loading catalog pickle {emb_path}")
        with open(emb_path, "rb") as f:
            _obj = pickle.load(f)
        bge_cat_tids = _obj["track_ids"]
        bge_cat_mat = _obj["track_mat"].astype(np.float32)
        # L2-normalize rows -> dot product is cosine (matches nb78 C1).
        _norms = np.linalg.norm(bge_cat_mat, axis=1, keepdims=True)
        _norms[_norms == 0] = 1.0
        bge_cat_mat = bge_cat_mat / _norms
        bge_cat_tid_to_idx = {t: i for i, t in enumerate(bge_cat_tids)}
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[lgbm-features] bge: loading model {bge_model} on {_dev}")
        bge_enc = SentenceTransformer(bge_model, device=_dev)

    # Tier-2 #4.1 relevance scorer (qwen_meta_cos + bm25_score). Loaded once when
    # --with-relevance is set; reuses the dense channel's shared Qwen3 encoder.
    relevance_scorer = None
    if with_relevance:
        from mcrs.rerankers.relevance_scorer import RelevanceScorer
        print("[lgbm-features] loading RelevanceScorer (qwen_meta_cos + bm25_score)")
        relevance_scorer = RelevanceScorer(
            dataset_name, ["all_tracks"],
            ["track_name", "artist_name", "album_name"], cache_dir)

    # Build (query, chat_history, gold_tid, session_meta) triples per music turn.
    print(f"[lgbm-features] assembling queries")
    queries: list[str] = []
    bge_queries: list[str] = []  # parallel bge_m3_structured queries (only when with_bge)
    metas: list[dict] = []
    golds: list[str] = []
    query_tokens_list: list[set[str]] = []
    played_tids_list: list[list[str]] = []
    # User-turns-only dialog per turn for the SASRec channel. Must match what the
    # model was trained on (train_sasrec._walk_split) AND what serve passes
    # (crs_baseline._sasrec_dialog_turns): build_user_dialog over `prior` (which
    # already includes the current-turn user request). Omitting this made
    # sasrec_seq fall back to the raw query -> the wrrf_rank / n_channels_hit
    # features the reranker trains on were computed from a context the model
    # never saw (train/serve skew on a weight-1.0 channel).
    user_dialog_list: list[str] = []
    for sess in tqdm(sessions, desc="sessions"):
        convos = sess["conversations"]
        df = pd.DataFrame(convos)
        for _, music in df[df["role"] == "music"].iterrows():
            turn_n = int(music["turn_number"])
            gold_tid = music["content"]
            # build retrieval_input: all turns STRICTLY before this music turn
            # at this turn_number + the user turn at this turn_number. Shared
            # slice with train_sasrec via prior_turns -> SASRec dialog parity.
            prior = prior_turns(df, turn_n)
            lines = []
            for _, t in prior.iterrows():
                role = "assistant" if t["role"] == "music" else t["role"]
                content = t["content"]
                if t["role"] == "music":
                    try:
                        content = item_db.id_to_metadata(content)
                    except Exception:
                        content = str(content)
                lines.append(f"{role}: {content}")
            # GOAL-LESS query (matches the shipped config 194 raw mode -> correct
            # train/serve parity AS-IS). NOTE: the nb74 cell-4 dev harness appends
            # `\ngoal: <listener_goal>`, so dev currently measures a goal-FUL
            # query while train+serve are goal-less (a small known mismatch).
            #
            # GOAL-EVERYWHERE RUNBOOK (step 2b, GPU — only if goal in the query
            # beats the goal-less baseline; Stage 12 measured just +0.0072 recall
            # and recall isn't the binding constraint, so prior is low). To ship
            # goal, change ALL THREE together or you get train/serve skew:
            #   (a) here: append `\ngoal: <listener_goal>` from
            #       sess["conversation_goal"]["listener_goal"] (mirror cell 4);
            #   (b) serve: set config query_preprocessing_mode: "raw_with_goal"
            #       (already implemented in crs_baseline.build_retrieval_query);
            #   (c) retrain lgbm_clean_full on the goal-ful parquets.
            # (The goal's bigger payoff is likely the RESPONDER prompt, not the
            # retrieval query — see the responder-goal follow-up.)
            retrieval_input = "\n".join(lines)
            queries.append(retrieval_input)
            # Parallel bge_m3_structured query for the bge-v2 feature. Built here
            # (aligned with `queries`) to match nb78 C0 EXACTLY: assistant-role
            # remap, music-turn metadata expansion, listener_goal + user_profile,
            # max_history_turns=6, state=None. Only when bge is enabled.
            if with_bge:
                sm = [
                    {"role": ("assistant" if t["role"] == "music" else t["role"]),
                     "content": (item_db.id_to_metadata(t["content"])
                                 if t["role"] == "music" else t["content"])}
                    for _, t in prior.iterrows()
                ]
                _goal = sess.get("conversation_goal") or {}
                _gtx = (_goal.get("listener_goal") or "").strip()
                _up = sess.get("user_profile") or {}
                bge_queries.append(build_retrieval_query(
                    sm, mode="bge_m3_structured", goal_text=_gtx,
                    user_profile=_up, max_history_turns=6, state=None))
            metas.append({
                "session_id": sess["session_id"],
                "user_id": sess["user_id"],
                "turn_number": turn_n,
                "conversation_goal": sess.get("conversation_goal"),
            })
            golds.append(gold_tid)
            query_tokens_list.append(_tokenize_simple(retrieval_input))
            # SASRec dialog: user-turns-only over `prior` (same slice train_sasrec
            # uses), i.e. prior turns + the current-turn user request.
            user_dialog_list.append(build_user_dialog(prior.to_dict("records")))
            # Collect track_ids played BEFORE this turn (for session-continuity features).
            prior_music = df[(df["role"] == "music") & (df["turn_number"] < turn_n)]
            played_tids_list.append(list(prior_music["content"]))

    # bge-v2: batch-encode all parallel structured queries once (q_mat aligned
    # with `queries`). Per-query full-catalog cos + rank are computed inside the
    # wRRF chunk loop (in blocks) to bound memory, then injected into candidate
    # dicts before extract_features (same wiring as sasrec_rank).
    bge_q_mat = None
    if with_bge:
        print(f"[lgbm-features] bge: encoding {len(bge_queries)} structured queries")
        bge_q_mat = bge_enc.encode(
            bge_queries, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float32)

    print(f"[lgbm-features] built {len(queries)} queries; running wRRF topk={topk}")
    # Batch through wRRF in chunks. A larger CHUNK feeds more queries per union
    # call so the dense channel encodes in big GPU batches (see SUB_B in
    # dense_precomputed). 256 is safe once JAX preallocation is disabled.
    CHUNK = 256
    # Accumulate as periodic compact DataFrames rather than one giant list of
    # dicts: at full scale (--n-sessions 999999 x topk candidates) the dict list
    # alone can blow host RAM (tens of GB) and crash the run. Flush the buffer to
    # a DataFrame every FLUSH_ROWS rows; the final pd.concat yields the identical
    # frame (same columns/order, since extract_features returns consistent keys).
    FLUSH_ROWS = 200_000
    all_frames: list[pd.DataFrame] = []
    row_buf: list[dict] = []
    total_rows = 0
    t_union = 0.0   # cumulative retrieval + GPU-encode time
    t_feat = 0.0    # cumulative python feature-extraction time
    for i in tqdm(range(0, len(queries), CHUNK), desc="wrrf batches"):
        chunk_queries = queries[i:i+CHUNK]
        # Feed the session-aware union channels the prior played track_ids +
        # per-query user_id. Mirror the existing prior_tids = played_tids_list[i+j]
        # indexing so batch_context[j] lines up with chunk_queries[j].
        chunk_n = len(chunk_queries)
        chunk_context = [
            {"history_tids": played_tids_list[i + j],
             "user_dialog": user_dialog_list[i + j]} for j in range(chunk_n)
        ]
        chunk_user_ids = [metas[i + j]["user_id"] for j in range(chunk_n)]
        _t0 = time.perf_counter()
        chunk_results = scorer.run(
            chunk_queries, topk=topk,
            batch_context=chunk_context, user_ids=chunk_user_ids,
        )
        t_union += time.perf_counter() - _t0
        # bge-v2: full-catalog sims for this chunk in one matmul (chunk_n x T).
        # Inject bge_cos + bge_rank into each candidate dict BEFORE
        # extract_features (mirrors how sasrec_rank rides on the candidate dict).
        # tid not in the bge catalog -> cos 0.0, rank sentinel 10000.
        if with_bge:
            chunk_sims = bge_q_mat[i:i + chunk_n] @ bge_cat_mat.T  # (chunk_n, T)
            for j, cand_list in enumerate(chunk_results):
                sims_i = chunk_sims[j]
                order = np.argsort(-sims_i)  # descending full-catalog ranking
                tid_to_rank = {bge_cat_tids[idx]: r + 1
                               for r, idx in enumerate(order)}
                for c in cand_list:
                    cti = bge_cat_tid_to_idx.get(c["tid"])
                    c["bge_cos"] = float(sims_i[cti]) if cti is not None else 0.0
                    c["bge_rank"] = tid_to_rank.get(c["tid"], 10000)
        # Tier-2 #4.1: inject qwen_meta_cos + bm25_score via the shared
        # RelevanceScorer (same instance serve uses -> parity). One batch call
        # per chunk; tids absent from the catalog/bm25-topk -> 0.0.
        if with_relevance:
            cand_tids_per_q = [[c["tid"] for c in cl] for cl in chunk_results]
            feats = relevance_scorer.feats_for_batch(chunk_queries, cand_tids_per_q)
            for j, cand_list in enumerate(chunk_results):
                for c, f in zip(cand_list, feats[j]):
                    c["qwen_meta_cos"] = f["qwen_meta_cos"]
                    c["bm25_score"] = f["bm25_score"]
        _t0 = time.perf_counter()
        for j, cand_list in enumerate(chunk_results):
            sess_info = metas[i + j]
            uid = sess_info["user_id"]
            uinfo = user_meta.get(uid, {})
            # Resolve prior played track_ids -> metadata dicts for session-continuity features.
            prior_tids = played_tids_list[i + j]
            played_meta = [track_meta[t] for t in prior_tids if t in track_meta]
            rows = extract_features(
                query=chunk_queries[j],
                candidates=cand_list,
                gold_tid=golds[i + j],
                session_info=sess_info,
                user_info=uinfo,
                track_meta=track_meta,
                cfbpr_tid_to_idx=cfbpr_tid_to_idx,
                cfbpr_track_mat=cfbpr_track_mat,
                cfbpr_user_embs=cfbpr_user_embs,
                query_tokens=query_tokens_list[i + j],
                pop_rank_pct=pop_rank_pct,
                played_meta=played_meta,
                clap_lookup=clap_lookup,
                played_tids=prior_tids,
                clap_mean=clap_mean,
                with_bge=with_bge,
                with_relevance=with_relevance,
            )
            row_buf.extend(rows)
        t_feat += time.perf_counter() - _t0
        if len(row_buf) >= FLUSH_ROWS:
            all_frames.append(pd.DataFrame(row_buf))
            total_rows += len(row_buf)
            row_buf = []

    if row_buf:
        all_frames.append(pd.DataFrame(row_buf))
        total_rows += len(row_buf)
    print(f"[lgbm-features] total rows: {total_rows}")
    print(f"[lgbm-features] timing: union(retrieval+encode)={t_union:.1f}s  features={t_feat:.1f}s")
    df_out = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    # Label distribution sanity
    pos = int(df_out["label"].sum())
    print(f"[lgbm-features] positives: {pos}  negatives: {len(df_out) - pos}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(f"[lgbm-features] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Build LGBM LambdaMART training features from train conversations.")
    p.add_argument("--n-sessions", type=int, default=2000,
                   help="Number of train sessions to sample (default 2000; use 15000 for full).")
    p.add_argument("--topk", type=int, default=100,
                   help="wRRF top-K candidates per query (default 100).")
    p.add_argument("--seed", type=int, default=42,
                   help="Session-sample seed (default 42).")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "lgbm_features.parquet"),
                   help="Output parquet path.")
    p.add_argument("--cache-dir", type=str, default=str(BASELINES_DIR / "../experiments/cache"),
                   help="Cache dir (shared with retrievers).")
    p.add_argument("--use-sasrec", action="store_true",
                   help="Include the SASRec channel and emit sasrec_rank_inv feature.")
    p.add_argument("--w-sasrec", type=float, default=1.0,
                   help="RRF weight for the SASRec sub (default 1.0).")
    p.add_argument("--sasrec-model-dir", type=str, default="sasrec_v1",
                   help="Model directory / HF repo for the SASRec weights (default 'sasrec_v1').")
    # OOF cross-fitting: build features ONLY for sessions in fold `--oof-fold`,
    # using a SASRec that held that fold out (--sasrec-model-dir -> fold model).
    # Run once per fold (same --seed/--n-sessions) then concat the parquets.
    p.add_argument("--oof-fold", type=int, default=None,
                   help="OOF fold to SELECT for feature-building [0, oof-num-folds).")
    p.add_argument("--oof-num-folds", type=int, default=None,
                   help="Total OOF folds (set together with --oof-fold).")
    p.add_argument("--use-clap", action="store_true",
                   help="Emit clap_session_sim (CLAP audio similarity to played tracks).")
    # bge-v2 bi-encoder feature (bge_cos + bge_rank_inv). OFF unless BOTH set,
    # so existing runs are byte-unchanged.
    p.add_argument("--bge-model", type=str, default=None,
                   help="HF repo / local path of the merged bge-v2 model "
                        "(e.g. OrRim123/recsys2026-bge-m3-music-v2-merged). "
                        "Enables bge_cos + bge_rank_inv when set together with --bge-cache-dir.")
    p.add_argument("--bge-cache-dir", type=str, default=None,
                   help="Dir containing the bge catalog pickle track_embeddings.pkl "
                        "(the {cache}/retrieval_v2/dense_local/{safe}/{label}/ dir).")
    p.add_argument("--with-relevance", action="store_true",
                   help="Emit Tier-2 #4.1 leak-free relevance features "
                        "(qwen_meta_cos + bm25_score) via the shared RelevanceScorer.")
    p.add_argument("--use-segment-routing", action="store_true",
                   help="Build the candidate pool + wrrf_rank with segment-aware "
                        "routing (cold->content, warm->session) so the reranker "
                        "trains on the same routed order it serves on.")
    p.add_argument("--use-two-tower", action="store_true",
                   help="Add the Tier-1 #3.3 two-tower content channel to the union "
                        "at build time so the reranker trains on the same pool it "
                        "serves (converts the validated wall-recall into nDCG).")
    p.add_argument("--w-two-tower", type=float, default=0.7,
                   help="RRF weight for the two-tower channel (matches # 4-tt best).")
    p.add_argument("--two-tower-model-dir", type=str, default="two_tower_v1",
                   help="Trained two-tower checkpoint dir under retrieval_v2/two_tower/.")
    p.add_argument("--dataset-name", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                   help="Catalog dataset for the RelevanceScorer dense/bm25 models.")
    args = p.parse_args()
    if (args.oof_fold is None) != (args.oof_num_folds is None):
        p.error("--oof-fold and --oof-num-folds must be set together")
    if args.oof_fold is not None and not (0 <= args.oof_fold < args.oof_num_folds):
        p.error("--oof-fold must be in [0, --oof-num-folds)")
    # Resolve --out + --cache-dir to absolute paths so the mid-run chdir into
    # BASELINES_DIR (required by the mcrs factory's relative cache lookups)
    # doesn't misplace the output. Previous bug: --out data/lgbm_features.parquet
    # landed at music-crs-baselines/data/ instead of repo-root data/.
    args.out = os.path.abspath(args.out)
    args.cache_dir = os.path.abspath(args.cache_dir)
    # bge requires BOTH flags; resolve the cache dir to absolute (survives the
    # mid-run chdir into BASELINES_DIR, like --cache-dir above).
    if (args.bge_model is None) != (args.bge_cache_dir is None):
        p.error("--bge-model and --bge-cache-dir must be set together")
    if args.bge_cache_dir is not None:
        args.bge_cache_dir = os.path.abspath(args.bge_cache_dir)

    origin_cwd = os.getcwd()
    os.chdir(BASELINES_DIR)
    try:
        build(args.n_sessions, args.topk, args.seed, args.out, args.cache_dir,
              use_sasrec=args.use_sasrec,
              w_sasrec=args.w_sasrec,
              sasrec_model_dir=args.sasrec_model_dir,
              oof_fold=args.oof_fold,
              oof_num_folds=args.oof_num_folds,
              use_clap=args.use_clap,
              bge_model=args.bge_model,
              bge_cache_dir=args.bge_cache_dir,
              with_relevance=args.with_relevance,
              dataset_name=args.dataset_name,
              use_segment_routing=args.use_segment_routing,
              use_two_tower=args.use_two_tower,
              w_two_tower=args.w_two_tower,
              two_tower_model_dir=args.two_tower_model_dir)
    finally:
        os.chdir(origin_cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
