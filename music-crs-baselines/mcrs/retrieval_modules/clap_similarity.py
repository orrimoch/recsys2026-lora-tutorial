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
                            clap_lookup: dict) -> float:
    """Mean cosine similarity of the candidate's CLAP vector to the session's
    played tracks' CLAP vectors. Vectors in clap_lookup are assumed L2-normalized
    (load_clap_lookup normalizes), so cosine == dot. Returns 0.0 when the
    candidate has no CLAP vector, there is no history, or no played track has a
    CLAP vector."""
    cv = clap_lookup.get(cand_tid)
    if cv is None or not played_tids:
        return 0.0
    sims = []
    for t in played_tids:
        pv = clap_lookup.get(t)
        if pv is not None:
            sims.append(float(np.dot(cv, pv)))
    if not sims:
        return 0.0
    return float(np.mean(sims))


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
