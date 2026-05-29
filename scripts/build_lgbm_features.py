"""Build LightGBM LambdaMART training features from train conversations.

For each music turn in each (sampled) train session:
  1. Build retrieval_input = newline-joined chat history up to the user turn.
  2. Run wRRF retrieval -> top-N candidate tids (N=100 by default).
  3. For each candidate, compute a 14-feature vector:
       Numeric:
         - wrrf_score         : fusion score from RRF
         - bm25_score         : standalone BM25 score (proxy for name-match)
         - dense_meta_cos     : cosine vs metadata-qwen3 track embedding
         - dense_lyrics_cos   : cosine vs lyrics-qwen3 track embedding
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
import math
import os
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
class WRRFRunner:
    """Thin wrapper that runs the 021-champion wRRF stack and returns ranked
    candidates with fusion score + position. Inference-friendly: the LGBM
    reranker at inference only needs (tid, wrrf_rank) per candidate — no
    per-sub ranks — so feature computation is cheap and deterministic."""

    def __init__(self, cache_dir: str, corpus_types: list[str],
                 use_sasrec: bool = False,
                 w_sasrec: float = 1.0,
                 sasrec_model_dir: str = "sasrec_v1"):
        # wrrf_union_v1 is the 3-channel recall union (lexical + frozen-Qwen
        # semantic + same-artist session continuity); session_cf was dropped
        # after the G1 ablation. The same-artist channel needs
        # batch_context['history_tids'] to fire — see run().
        self.use_sasrec = use_sasrec
        if use_sasrec:
            self.wrrf = load_retrieval_module(
                "wrrf_union_v1",
                "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                ["all_tracks"],
                corpus_types,
                cache_dir,
                extra_config={
                    "use_sasrec": True,
                    "w_sasrec": w_sasrec,
                    "sasrec_model_dir": sasrec_model_dir,
                },
            )
        else:
            self.wrrf = load_retrieval_module(
                "wrrf_union_v1",
                "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                ["all_tracks"],
                corpus_types,
                cache_dir,
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
                result.append([
                    {
                        "tid": tid,
                        "wrrf_rank": r + 1,
                        "sasrec_rank": rankmap.get(tid, 10000),
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
            # NEW retrieval-rank features (19-21) — populated by caller; default to wrrf_rank.
            "bm25_rank_inv": 1.0 / max(1, c.get("bm25_rank", c["wrrf_rank"])),
            "dense_meta_rank_inv": 1.0 / max(1, c.get("dense_meta_rank", c["wrrf_rank"])),
            "dense_lyrics_rank_inv": 1.0 / max(1, c.get("dense_lyrics_rank", c["wrrf_rank"])),
            # NEW reranker-output features (22, 23) — caller supplies; default 0.
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
) -> None:
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
    scorer = WRRFRunner(
        cache_dir=cache_dir,
        corpus_types=["track_name", "artist_name", "album_name"],
        use_sasrec=use_sasrec,
        w_sasrec=w_sasrec,
        sasrec_model_dir=sasrec_model_dir,
    )

    # Build (query, chat_history, gold_tid, session_meta) triples per music turn.
    print(f"[lgbm-features] assembling queries")
    queries: list[str] = []
    metas: list[dict] = []
    golds: list[str] = []
    query_tokens_list: list[set[str]] = []
    played_tids_list: list[list[str]] = []
    for sess in tqdm(sessions, desc="sessions"):
        convos = sess["conversations"]
        df = pd.DataFrame(convos)
        for _, music in df[df["role"] == "music"].iterrows():
            turn_n = int(music["turn_number"])
            gold_tid = music["content"]
            # build retrieval_input: all turns STRICTLY before this music turn
            # at this turn_number + the user turn at this turn_number.
            prior = df[(df["turn_number"] < turn_n) |
                       ((df["turn_number"] == turn_n) & (df["role"] == "user"))]
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
            retrieval_input = "\n".join(lines)
            queries.append(retrieval_input)
            metas.append({
                "session_id": sess["session_id"],
                "user_id": sess["user_id"],
                "turn_number": turn_n,
                "conversation_goal": sess.get("conversation_goal"),
            })
            golds.append(gold_tid)
            query_tokens_list.append(_tokenize_simple(retrieval_input))
            # Collect track_ids played BEFORE this turn (for session-continuity features).
            prior_music = df[(df["role"] == "music") & (df["turn_number"] < turn_n)]
            played_tids_list.append(list(prior_music["content"]))

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
            {"history_tids": played_tids_list[i + j]} for j in range(chunk_n)
        ]
        chunk_user_ids = [metas[i + j]["user_id"] for j in range(chunk_n)]
        _t0 = time.perf_counter()
        chunk_results = scorer.run(
            chunk_queries, topk=topk,
            batch_context=chunk_context, user_ids=chunk_user_ids,
        )
        t_union += time.perf_counter() - _t0
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

    origin_cwd = os.getcwd()
    os.chdir(BASELINES_DIR)
    try:
        build(args.n_sessions, args.topk, args.seed, args.out, args.cache_dir,
              use_sasrec=args.use_sasrec,
              w_sasrec=args.w_sasrec,
              sasrec_model_dir=args.sasrec_model_dir,
              oof_fold=args.oof_fold,
              oof_num_folds=args.oof_num_folds)
    finally:
        os.chdir(origin_cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
