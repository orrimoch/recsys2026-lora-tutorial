"""LightGBM LambdaRank reranker (A1 / exp 027).

Trained via colab/Build_LGBM_Features_And_Train.ipynb on (query, wRRF
top-100 candidates) from 2000 train sessions with binary gold labels. Uses
11 features per candidate:

  Numeric:  wrrf_rank, cfbpr_score, pop_log, recency_years, tag_count, artist_in_query
  Categorical: goal_category, goal_specificity, user_age_group, user_country, user_gender

At inference:
  - Input: queries, candidate_tids_per_query (from wRRF), user_ids,
    goal_categories, goal_specificities, user_profiles_raw.
  - Computes features for each candidate (static track metadata cache +
    cfbpr lookup + query-artist match + session/user categoricals).
  - LGBM booster scores each (query, candidate) pair.
  - Returns top-K per query by descending score.

Loads model + metadata from a local directory:
  model_path/
    booster.txt      (LightGBM text-format model)
    metadata.json    (features, categorical levels, best_iter, val ndcg)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..retrieval_modules.session_history import session_match_features


_HELPERS_CACHE: Optional[dict] = None


def _lgbm_feature_helpers() -> dict:
    """Lazy import of feature helpers from scripts/build_lgbm_features.py.

    Cached after first call. Keeps module load cheap and avoids importing
    `lightgbm`-side dependencies when the reranker is constructed.
    """
    global _HELPERS_CACHE
    if _HELPERS_CACHE is not None:
        return _HELPERS_CACHE
    repo_root = Path(__file__).resolve().parents[3]
    scripts_dir = str(repo_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_lgbm_features import (
        compute_release_year_cyclical, compute_tag_overlap,
        last_turn_moved_toward_goal,
    )
    _HELPERS_CACHE = {
        "release_year_cyclical": compute_release_year_cyclical,
        "tag_overlap": compute_tag_overlap,
        "last_goal": last_turn_moved_toward_goal,
    }
    return _HELPERS_CACHE


FEATURES_NUMERIC = ["wrrf_rank", "cfbpr_score", "pop_log", "recency_years", "tag_count", "artist_in_query"]
FEATURES_CATEGORICAL = ["goal_category", "goal_specificity", "user_age_group", "user_country", "user_gender"]
ALL_FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _flatten_track_row(r: dict) -> dict:
    """Flatten a raw track-metadata dataset row to the fields the reranker and
    the shared session_match_features consume.

    MUST stay schema-aligned with build_lgbm_features.load_track_meta_lookup so
    train and serve compute identical features. In particular album_name is
    required: same_album is real during training but is silently pinned to 0 at
    inference if it is dropped here (train/serve skew)."""
    return {
        "artist_name": _first(r.get("artist_name")) or "",
        "album_name": _first(r.get("album_name")) or "",
        "tag_list": r.get("tag_list") or [],
        "popularity": float(r.get("popularity") or 0.0),
        "release_date": r.get("release_date"),
    }


def _tokenize_simple(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _parse_user_profile(raw: Any) -> dict[str, str]:
    """user_profile in Blind-A is a serialized dict (string). Sometimes already
    a dict, sometimes a JSON-ish string with single quotes. Parse defensively."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    # Try JSON (double quotes)
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try Python literal (single quotes)
    try:
        import ast
        return ast.literal_eval(raw)
    except Exception:
        return {}


class LGBM_RERANKER:
    def __init__(
        self,
        item_db_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        model_path: Optional[str] = None,
    ) -> None:
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"LGBM_RERANKER requires a local model dir; got {model_path!r}. "
                "Train via colab/Build_LGBM_Features_And_Train.ipynb, then unzip into "
                "models/lgbm_ranker/ on the M4."
            )
        import lightgbm as lgb
        self.model_path = model_path
        with open(os.path.join(model_path, "metadata.json")) as f:
            meta = json.load(f)
        # Tier-2 #4.3 multi-seed bagging: load N boosters when the model was bagged
        # (metadata n_bag>1 -> booster_0.txt..booster_{N-1}.txt); else the single
        # booster.txt. Predictions are averaged in _predict. Backward compatible.
        n_bag = int(meta.get("n_bag", 1))
        if n_bag > 1:
            self.boosters = [lgb.Booster(model_file=os.path.join(model_path, f"booster_{b}.txt"))
                             for b in range(n_bag)]
        else:
            self.boosters = [lgb.Booster(model_file=os.path.join(model_path, "booster.txt"))]
        self.booster = self.boosters[0]  # back-compat alias
        self.features: list[str] = meta["features"]
        self.categorical_features: list[str] = meta["categorical_features"]
        self.cat_levels: dict[str, list[str]] = meta["categorical_levels"]
        # Build category -> int code maps for fast encoding at inference.
        self.cat_index = {c: {v: i for i, v in enumerate(self.cat_levels[c])} for c in self.categorical_features}
        print(f"[lgbm-rerank] loaded {len(self.boosters)} booster(s) from {model_path} "
              f"(best_iter={meta.get('best_iteration')}, "
              f"val_ndcg20={meta.get('best_val_ndcg20'):.4f})")
        # Track metadata lookup + cf-bpr tables (lightweight, reused).
        self._load_track_meta(item_db_name, track_split_types)
        self._build_pop_rank_pct()
        self._load_cfbpr(cache_dir)
        # CLAP audio lookup — only when the model uses clap_session_sim (avoids
        # the embedding download for models without the feature).
        if "clap_session_sim" in self.features:
            from ..retrieval_modules.clap_similarity import (
                clap_mean_vector, load_clap_lookup,
            )
            self.clap_lookup = load_clap_lookup(cache_dir)
            # Catalog-mean vector for imputing missing candidates (train/serve
            # parity with build_lgbm_features). None when the lookup is empty.
            self.clap_mean = clap_mean_vector(self.clap_lookup)
        else:
            self.clap_lookup = None
            self.clap_mean = None
        # Lazy-load user metadata when first rerank() call arrives.
        self._user_meta: Optional[dict[str, dict]] = None

    def _load_track_meta(self, item_db_name: str, track_split_types: list[str]) -> None:
        from datasets import concatenate_datasets, load_dataset
        print(f"[lgbm-rerank] loading track metadata {item_db_name}")
        ds = load_dataset(item_db_name)
        concat = concatenate_datasets([ds[s] for s in track_split_types])
        self.tid_to_track: dict[str, dict] = {}
        for r in concat:
            self.tid_to_track[r["track_id"]] = _flatten_track_row(r)
        print(f"[lgbm-rerank] cached {len(self.tid_to_track)} track rows")

    def _build_pop_rank_pct(self) -> None:
        """Popularity percentile per track (0=most popular, 1=least), built from
        the SAME catalog the trainer used (self.tid_to_track). MUST match
        build_lgbm_features.build_pop_rank_pct_map: the pop_rank_pct feature was
        previously read from a caller-supplied dict that nothing passed, so it was
        a constant 0.5 at inference (a top-gain feature dead at serve -> the
        reranker couldn't reorder). Self-computing it fixes train/serve skew for
        every caller (dev eval + production)."""
        items = [(t, float(m.get("popularity") or 0.0))
                 for t, m in self.tid_to_track.items()]
        items.sort(key=lambda x: -x[1])
        n = max(1, len(items))
        self.pop_rank_pct: dict[str, float] = {}
        for rank, (t, pop) in enumerate(items):
            self.pop_rank_pct[t] = 0.5 if pop <= 0.0 else rank / n
        print(f"[lgbm-rerank] built pop_rank_pct for {len(self.pop_rank_pct)} tracks")

    def _load_cfbpr(self, cache_dir: str) -> None:
        # Reuse cf-bpr tables via the CF_BPR class so singleton caches hit.
        from ..retrieval_modules.cf_bpr import CF_BPR
        cf = CF_BPR(
            dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            split_types=["all_tracks"],
            corpus_types=["track_name"],
            cache_dir=cache_dir,
        )
        self.cfbpr_tid_to_idx = {t: i for i, t in enumerate(cf.track_ids)}
        self.cfbpr_track_mat = cf.track_mat
        self.cfbpr_user_embs = cf.user_embs

    def _load_user_meta_if_needed(self) -> None:
        if self._user_meta is not None:
            return
        from datasets import concatenate_datasets, load_dataset
        print("[lgbm-rerank] loading Challenge-User-Metadata")
        um = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Metadata")
        concat = concatenate_datasets([um[s] for s in um])
        self._user_meta = {
            r["user_id"]: {
                "age_group": r.get("age_group") or "unknown",
                "country_code": r.get("country_code") or "unknown",
                "gender": r.get("gender") or "unknown",
            }
            for r in concat
        }
        print(f"[lgbm-rerank] cached metadata for {len(self._user_meta)} users")

    def _encode_cat(self, col: str, value: Optional[str]) -> int:
        """Map a categorical string to the int code LightGBM expects. Unknown
        values map to -1 (LGBM treats as NaN/missing; booster handles)."""
        if value is None:
            return -1
        return self.cat_index[col].get(str(value), -1)

    def _compute_feature_matrix(
        self,
        query: str,
        candidate_tids: list[str],
        user_id: Optional[str],
        goal_category: Optional[str],
        goal_specificity: Optional[str],
        user_profile_raw: Any,
        extra_features_per_candidate: Optional[list[dict]] = None,
        extra_session_info: Optional[dict] = None,
    ) -> np.ndarray:
        """Build (N, F) feature matrix for N candidates.

        `extra_features_per_candidate[i]` (optional): per-candidate dict with any of
            bm25_rank / dense_meta_rank / dense_lyrics_rank / ce_score / ce_rank / sasrec_rank.
            Used only when the trained model lists those features in metadata.

        `extra_session_info` (optional): per-turn dict with any of
            goal_progress_assessments / prior_track_count / query_drift_score.
            Used only for matching trained features.
        """
        # User profile + metadata
        self._load_user_meta_if_needed()
        umeta = self._user_meta.get(user_id, {}) if user_id else {}
        uprof = _parse_user_profile(user_profile_raw)
        age = umeta.get("age_group") or uprof.get("age_group") or "unknown"
        country = umeta.get("country_code") or uprof.get("country_code") or "unknown"
        gender = umeta.get("gender") or uprof.get("gender") or "unknown"

        # cf-bpr user vector (None for cold)
        cfbpr_user_vec = self.cfbpr_user_embs.get(user_id) if user_id else None

        query_tokens = _tokenize_simple(query)
        query_joined = " ".join(query_tokens)

        # Pre-encode session-level categoricals (same for all candidates in this query).
        gc = self._encode_cat("goal_category", goal_category)
        gs = self._encode_cat("goal_specificity", goal_specificity)
        ag = self._encode_cat("user_age_group", age)
        cc = self._encode_cat("user_country", country)
        gn = self._encode_cat("user_gender", gender)

        n = len(candidate_tids)
        X = np.zeros((n, len(self.features)), dtype=np.float64)
        f_idx = {f: i for i, f in enumerate(self.features)}

        # Pull extended-feature helpers only if the trained model needs them.
        # Stage C session-continuity features are listed here for documentation,
        # but are computed below via session_match_features regardless of the
        # `helpers` bundle (they don't need it — only played_meta).
        _SESSION_MATCH_KEYS = (
            "same_artist", "same_album", "artist_in_session_count",
            "session_tag_overlap",
        )
        extended_keys = {
            "release_year_sin", "release_year_cos", "tag_overlap_count",
            "last_turn_moved_toward_goal", "bm25_rank_inv", "dense_meta_rank_inv",
            "dense_lyrics_rank_inv", "ce_score", "ce_rank_inv", "sasrec_rank_inv",
            "bge_cos", "bge_rank_inv",
            "qwen_meta_cos", "bm25_score",
            "n_channels_hit", "clap_session_sim",
            "turn_number_feat", "prior_track_count", "query_drift_score",
            "pop_rank_pct", "is_warm_user",
            *_SESSION_MATCH_KEYS,
        }
        need_helpers = any(
            k in f_idx for k in (extended_keys - set(_SESSION_MATCH_KEYS))
        )
        helpers = _lgbm_feature_helpers() if need_helpers else None
        sess = extra_session_info or {}
        last_goal = helpers["last_goal"](sess.get("goal_progress_assessments")) if helpers else -1

        # Session-continuity inputs: resolve the prior played track_ids to their
        # metadata dicts ONCE (shared across all candidates of this query).
        played_tids = (extra_session_info or {}).get("played_tids", []) or []
        played_meta = [self.tid_to_track[t] for t in played_tids if t in self.tid_to_track]
        need_session_match = any(k in f_idx for k in _SESSION_MATCH_KEYS)

        for rank, tid in enumerate(candidate_tids, start=1):
            m = self.tid_to_track.get(tid, {})
            artist = m.get("artist_name") or ""
            artist_in_q = 1 if artist and artist.lower() in query_joined else 0
            tag_count = len(m.get("tag_list") or [])
            pop = m.get("popularity") or 0.0
            rd = m.get("release_date")
            try:
                year = int(str(rd)[:4]) if rd else None
            except ValueError:
                year = None
            recency = (2026 - year) if year else 0.0

            # cfbpr_score is a model-derived feature; a leak-free ("clean")
            # model may omit it (see project_sasrec_lgbm_feature_leak). Guard
            # like the extended features so its absence doesn't KeyError.
            if "cfbpr_score" in f_idx:
                if cfbpr_user_vec is not None and tid in self.cfbpr_tid_to_idx:
                    tv = self.cfbpr_track_mat[self.cfbpr_tid_to_idx[tid]]
                    cfbpr_score = float(np.dot(cfbpr_user_vec, tv))
                else:
                    cfbpr_score = 0.0
                X[rank - 1, f_idx["cfbpr_score"]] = cfbpr_score

            X[rank - 1, f_idx["wrrf_rank"]] = rank
            X[rank - 1, f_idx["pop_log"]] = float(np.log1p(pop))
            X[rank - 1, f_idx["recency_years"]] = float(recency)
            X[rank - 1, f_idx["tag_count"]] = tag_count
            X[rank - 1, f_idx["artist_in_query"]] = artist_in_q
            X[rank - 1, f_idx["goal_category"]] = gc
            X[rank - 1, f_idx["goal_specificity"]] = gs
            X[rank - 1, f_idx["user_age_group"]] = ag
            X[rank - 1, f_idx["user_country"]] = cc
            X[rank - 1, f_idx["user_gender"]] = gn

            # Stage C session-continuity features (same_artist / same_album /
            # artist_in_session_count / session_tag_overlap). Computed via the
            # SHARED session_match_features (identical to training) — no helpers
            # bundle required, only played_meta. Guarded by f_idx so models that
            # don't list these columns are unaffected.
            if need_session_match:
                _sf = session_match_features(m, played_meta)
                for _k in _SESSION_MATCH_KEYS:
                    if _k in f_idx:
                        X[rank - 1, f_idx[_k]] = _sf[_k]

            # Extended (Stage C) features — guarded by f_idx so 11-col models
            # keep working unchanged.
            if helpers is not None:
                cand_extra = (extra_features_per_candidate[rank - 1]
                              if extra_features_per_candidate else {})
                if "release_year_sin" in f_idx or "release_year_cos" in f_idx:
                    rs_sin, rs_cos = helpers["release_year_cyclical"](rd)
                    if "release_year_sin" in f_idx:
                        X[rank - 1, f_idx["release_year_sin"]] = rs_sin
                    if "release_year_cos" in f_idx:
                        X[rank - 1, f_idx["release_year_cos"]] = rs_cos
                if "tag_overlap_count" in f_idx:
                    X[rank - 1, f_idx["tag_overlap_count"]] = helpers["tag_overlap"](
                        query, m.get("tag_list"))
                if "last_turn_moved_toward_goal" in f_idx:
                    X[rank - 1, f_idx["last_turn_moved_toward_goal"]] = last_goal
                if "bm25_rank_inv" in f_idx:
                    X[rank - 1, f_idx["bm25_rank_inv"]] = 1.0 / max(
                        1, cand_extra.get("bm25_rank", rank))
                if "dense_meta_rank_inv" in f_idx:
                    X[rank - 1, f_idx["dense_meta_rank_inv"]] = 1.0 / max(
                        1, cand_extra.get("dense_meta_rank", rank))
                if "dense_lyrics_rank_inv" in f_idx:
                    X[rank - 1, f_idx["dense_lyrics_rank_inv"]] = 1.0 / max(
                        1, cand_extra.get("dense_lyrics_rank", rank))
                if "ce_score" in f_idx:
                    X[rank - 1, f_idx["ce_score"]] = float(cand_extra.get("ce_score", 0.0))
                if "ce_rank_inv" in f_idx:
                    X[rank - 1, f_idx["ce_rank_inv"]] = 1.0 / max(
                        1, cand_extra.get("ce_rank", rank))
                if "sasrec_rank_inv" in f_idx:
                    X[rank - 1, f_idx["sasrec_rank_inv"]] = 1.0 / max(
                        1, cand_extra.get("sasrec_rank", rank))
                # bge-v2 bi-encoder features. Mirror ce_score/ce_rank_inv EXACTLY
                # (train/serve parity with build_lgbm_features.extract_features):
                # bge_cos passes through unchanged, bge_rank_inv = 1/max(1,rank).
                # The caller passes bge_cos/bge_rank via extra_features_per_candidate;
                # the reranker NEVER loads the bge model (live-serve wiring of the
                # bge query encode + full-catalog rank is a separate follow-up).
                # Tier-2 #4.1 leak-free relevance features. Passthrough (the
                # RelevanceScorer injects identical values at train + serve).
                if "qwen_meta_cos" in f_idx:
                    X[rank - 1, f_idx["qwen_meta_cos"]] = float(
                        cand_extra.get("qwen_meta_cos", 0.0))
                if "bm25_score" in f_idx:
                    X[rank - 1, f_idx["bm25_score"]] = float(
                        cand_extra.get("bm25_score", 0.0))
                if "bge_cos" in f_idx:
                    X[rank - 1, f_idx["bge_cos"]] = float(cand_extra.get("bge_cos", 0.0))
                if "bge_rank_inv" in f_idx:
                    X[rank - 1, f_idx["bge_rank_inv"]] = 1.0 / max(
                        1, cand_extra.get("bge_rank", rank))
                if "n_channels_hit" in f_idx:
                    # How many union channels surfaced this candidate. Default 1
                    # (a surfaced candidate was hit by >=1 channel).
                    X[rank - 1, f_idx["n_channels_hit"]] = float(
                        cand_extra.get("n_channels_hit", 1))
                if "clap_session_sim" in f_idx and self.clap_lookup is not None:
                    # CLAP audio similarity of this candidate to the session's
                    # played tracks (shared fn -> train/serve parity).
                    from ..retrieval_modules.clap_similarity import clap_session_similarity
                    X[rank - 1, f_idx["clap_session_sim"]] = clap_session_similarity(
                        tid, played_tids, self.clap_lookup, mean_vec=self.clap_mean)
                if "clap_has_vector" in f_idx and self.clap_lookup is not None:
                    # Companion indicator (train/serve parity with the builder).
                    from ..retrieval_modules.clap_similarity import clap_has_vector
                    X[rank - 1, f_idx["clap_has_vector"]] = clap_has_vector(
                        tid, self.clap_lookup)
                if "turn_number_feat" in f_idx:
                    X[rank - 1, f_idx["turn_number_feat"]] = int(sess.get("turn_number", 0))
                if "prior_track_count" in f_idx:
                    X[rank - 1, f_idx["prior_track_count"]] = int(sess.get("prior_track_count", 0))
                if "query_drift_score" in f_idx:
                    X[rank - 1, f_idx["query_drift_score"]] = float(sess.get("query_drift_score", 1.0))
                if "pop_rank_pct" in f_idx:
                    # Self-computed map (matches training); was a dead 0.5 before.
                    X[rank - 1, f_idx["pop_rank_pct"]] = float(self.pop_rank_pct.get(tid, 0.5))
                if "is_warm_user" in f_idx:
                    X[rank - 1, f_idx["is_warm_user"]] = int(cfbpr_user_vec is not None)

        return X

    def _predict(self, X) -> np.ndarray:
        """Score X with the booster(s). For a bagged model (n_bag>1) average the
        per-booster scores (variance reduction). Single booster -> its scores."""
        if len(self.boosters) == 1:
            return self.boosters[0].predict(X)
        return np.mean([b.predict(X) for b in self.boosters], axis=0)

    def rerank(
        self,
        queries: list[str],
        candidate_tids: list[list[str]],
        topk: int,
        user_ids: Optional[list[Optional[str]]] = None,
        goal_categories: Optional[list[Optional[str]]] = None,
        goal_specificities: Optional[list[Optional[str]]] = None,
        user_profiles_raw: Optional[list[Any]] = None,
        # Stage C extended-feature inputs (back-compat default None).
        extra_features_per_candidate: Optional[list[list[dict]]] = None,
        extra_session_info: Optional[list[dict]] = None,
    ) -> list[list[str]]:
        n = len(queries)
        # Default-fill side channels (support BGE-reranker-style calls that
        # pass just (queries, candidates, topk)).
        if user_ids is None:
            user_ids = [None] * n
        if goal_categories is None:
            goal_categories = [None] * n
        if goal_specificities is None:
            goal_specificities = [None] * n
        if user_profiles_raw is None:
            user_profiles_raw = [None] * n

        out: list[list[str]] = []
        for i in range(n):
            tids = candidate_tids[i]
            X = self._compute_feature_matrix(
                query=queries[i],
                candidate_tids=tids,
                user_id=user_ids[i],
                goal_category=goal_categories[i],
                goal_specificity=goal_specificities[i],
                user_profile_raw=user_profiles_raw[i],
                extra_features_per_candidate=(extra_features_per_candidate[i]
                                              if extra_features_per_candidate else None),
                extra_session_info=(extra_session_info[i]
                                    if extra_session_info else None),
            )
            scores = self._predict(X)
            order = np.argsort(-scores)[:topk]
            out.append([tids[j] for j in order])
        return out
