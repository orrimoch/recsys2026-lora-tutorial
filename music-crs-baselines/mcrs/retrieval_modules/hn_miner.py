"""Hard-negative miner for bi-encoder fine-tuning.

Implements NV-Retriever's TopK-PercPos filtering [arXiv 2407.15831] adapted for
small catalogs (47K tracks): threshold defaults to 0.80 (not 0.95 — see plan
Task 5 + spec §6 reviewer finding).
"""
from __future__ import annotations

import random

import numpy as np


def percpos_filter(
    candidate_scores: list[float],
    positive_score: float,
    threshold: float = 0.80,
) -> list[bool]:
    """Return a boolean mask of which candidates pass the false-negative filter.

    A candidate is KEPT if candidate_score < threshold * positive_score.
    Candidates that score too close to the positive are likely false negatives
    (other valid answers we just didn't have labels for).
    """
    cutoff = threshold * positive_score
    return [s < cutoff for s in candidate_scores]


def sample_hard_negatives(
    filtered_ranks: list[int],
    k: int,
    rank_range: tuple[int, int] = (2, 200),
    seed: int = 42,
) -> list[int]:
    """Sample up to k negatives uniformly from the filtered list, within rank_range.

    If the filtered pool has fewer than k items in range, return all of them.
    """
    lo, hi = rank_range
    in_range = [r for r in filtered_ranks if lo <= r <= hi]
    if not in_range:
        return []
    if len(in_range) <= k:
        return list(in_range)
    rng = random.Random(seed)
    return sorted(rng.sample(in_range, k))


def mine_negatives_for_query(
    query_emb: np.ndarray,
    track_embs: np.ndarray,
    track_ids: list[str],
    gold_track_id: str,
    percpos_threshold: float = 0.80,
    k_negs: int = 15,
    pool_size: int = 200,
    seed: int = 42,
) -> list[str]:
    """Mine hard negatives for one query.

    1. Compute cosine similarities query × all tracks (assumes unit-normed).
    2. Identify the positive_score (sim to gold_track_id).
    3. Take top-`pool_size` candidates (excluding the gold).
    4. Apply PercPos filter at `percpos_threshold`.
    5. Sample up to k_negs from the filtered list.
    """
    if len(track_ids) != track_embs.shape[0]:
        raise ValueError("track_ids and track_embs length mismatch")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track_ids must be unique (duplicate IDs would leak gold as a negative)")
    track_norms = np.linalg.norm(track_embs, axis=1)
    if not np.allclose(track_norms, 1.0, atol=1e-3):
        raise ValueError("track_embs must be unit-normed (rows of L2-norm ≈ 1.0)")
    query_norm = float(np.linalg.norm(query_emb))
    if not (0.99 <= query_norm <= 1.01):
        raise ValueError(f"query_emb must be unit-normed (got L2-norm {query_norm:.4f})")
    sims = track_embs @ query_emb
    try:
        gold_idx = track_ids.index(gold_track_id)
    except ValueError:
        raise ValueError(f"gold_track_id {gold_track_id} not in track_ids")
    positive_score = float(sims[gold_idx])
    ranked_idxs = np.argsort(-sims)
    pool = [int(i) for i in ranked_idxs if i != gold_idx][:pool_size]
    pool_scores = [float(sims[i]) for i in pool]
    keep_mask = percpos_filter(pool_scores, positive_score, percpos_threshold)
    filtered_track_ids = [track_ids[pool[i]] for i, keep in enumerate(keep_mask) if keep]
    filtered_ranks_in_pool = [i + 2 for i, keep in enumerate(keep_mask) if keep]
    sampled_ranks = sample_hard_negatives(
        filtered_ranks_in_pool, k=k_negs, rank_range=(2, pool_size + 1), seed=seed,
    )
    rank_to_tid = dict(zip(filtered_ranks_in_pool, filtered_track_ids))
    return [rank_to_tid[r] for r in sampled_ranks]
