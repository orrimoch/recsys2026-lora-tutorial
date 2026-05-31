"""Shared CLAP audio-similarity helper for the reranker feature `clap_session_sim`.

The feature answers: "does this candidate SOUND like the tracks already played
this session?" — a content/audio relevance signal orthogonal to the lexical/
metadata features the reranker already has. CLAP (LAION audio embeddings) is the
`audio-laion_clap` column on the Track-Embeddings dataset; it currently only feeds
SASRec's item representation, never the reranker.

clap_session_similarity() is SHARED between training (scripts/build_lgbm_features.py)
and inference (mcrs/rerankers/lgbm_rerank.py) to guarantee identical train/serve
features (no skew), mirroring session_match_features.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

TRACK_EMB_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
CLAP_COL = "audio-laion_clap"

# Module-level shared cache so the train builder + the reranker (and repeated
# instantiations) share one {tid -> L2-normalized clap vec} map.
_SHARED_CLAP: dict[str, dict] = {}


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def clap_session_similarity(cand_tid: str, played_tids: list,
                            clap_lookup: dict,
                            mean_vec: Optional[np.ndarray] = None) -> float:
    """Mean cosine similarity of the candidate's CLAP vector to the session's
    played tracks' CLAP vectors. Vectors in clap_lookup are assumed L2-normalized
    (load_clap_lookup normalizes), so cosine == dot. Returns 0.0 when there is no
    history or no played track has a CLAP vector.

    mean_vec: optional L2-normalized catalog-mean CLAP vector (see
    clap_mean_vector). When supplied, a candidate that has NO CLAP vector is
    IMPUTED to mean_vec instead of returning a hard 0.0 — so "missing" no longer
    masquerades as "zero similarity" (the old 0.0 sentinel, which overloaded
    cold-candidate / no-audio with a genuine orthogonal score). When mean_vec is
    None the behaviour is unchanged (missing candidate -> 0.0), keeping legacy
    builds bit-identical. Missing PLAYED tracks are always skipped (never
    imputed) to avoid the degenerate mean-vs-mean == 1.0 artifact. Pair this with
    clap_has_vector() so the reranker can tell imputed rows apart."""
    cv = clap_lookup.get(cand_tid)
    if cv is None:
        if mean_vec is None:
            return 0.0
        cv = mean_vec
    if not played_tids:
        return 0.0
    sims = []
    for t in played_tids:
        pv = clap_lookup.get(t)
        if pv is not None:
            sims.append(float(np.dot(cv, pv)))
    if not sims:
        return 0.0
    return float(np.mean(sims))


def clap_has_vector(tid: str, clap_lookup: dict) -> int:
    """1 if the track has a real CLAP vector, else 0. Companion feature to the
    mean_vec imputation in clap_session_similarity() — lets the reranker
    distinguish a genuine low similarity from an imputed (missing-vector) row."""
    return 1 if clap_lookup.get(tid) is not None else 0


def clap_mean_vector(clap_lookup: dict) -> Optional[np.ndarray]:
    """L2-normalized catalog-mean CLAP vector, for imputing missing candidates.
    Returns None for an empty lookup.

    NOTE: this is the GLOBAL mean. An artist->category->global mean (per
    feedback_embedding_imputation) would need track metadata plumbed in here and
    is deferred; the global mean is the honest minimal fix for the 0.0-sentinel
    artifact."""
    if not clap_lookup:
        return None
    M = np.mean(np.stack(list(clap_lookup.values()), axis=0), axis=0)
    return _l2(np.asarray(M, dtype=np.float32))


def load_clap_lookup(cache_dir: str, split_types=("all_tracks",)) -> dict:
    """{track_id -> L2-normalized CLAP vector}. Cached to disk (mirrors cf_bpr).
    Shared in-process across callers via _SHARED_CLAP."""
    key = CLAP_COL
    if key in _SHARED_CLAP:
        return _SHARED_CLAP[key]
    cache_path = os.path.join(cache_dir, "clap", "clap_lookup.pkl")
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            lk = pickle.load(f)
        _SHARED_CLAP[key] = lk
        print(f"[clap] loaded lookup cache: {len(lk)} tracks from {cache_path}")
        return lk

    from datasets import concatenate_datasets, load_dataset

    print(f"[clap] loading {TRACK_EMB_DATASET}{list(split_types)} col={CLAP_COL}")
    ds = load_dataset(TRACK_EMB_DATASET)
    concat = concatenate_datasets([ds[s] for s in split_types])
    lk: dict[str, np.ndarray] = {}
    empty = 0
    for r in concat:
        emb = r.get(CLAP_COL)
        if not emb:
            empty += 1
            continue
        lk[r["track_id"]] = _l2(np.asarray(emb, dtype=np.float32))
    print(f"[clap] built lookup: {len(lk)} tracks (empty dropped={empty})")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(lk, f)
    _SHARED_CLAP[key] = lk
    return lk
