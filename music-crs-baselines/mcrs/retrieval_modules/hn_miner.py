"""Hard-negative miner for bi-encoder fine-tuning.

Two mining strategies are supported:

- "percpos" (default): NV-Retriever's TopK-PercPos filtering [arXiv 2407.15831].
  Drops candidates scoring within `(1 - threshold)` of the positive as likely
  false negatives, then uniformly samples from the surviving pool. Clean labels
  but excludes queries where the positive doesn't stand out — the structural-
  ceiling problem for sparse-label catalogs.

- "simans": SimANS sampling [Zhou et al. EMNLP 2022, arXiv 2210.11773]. No
  filter; samples negatives with Gaussian weight peaked at `s_pos - a` so
  near-positive (likely false-negative) candidates and far-positive (trivial)
  candidates both get low weight. Includes every query in training, addresses
  false negatives via per-sample down-weighting rather than a hard cutoff.
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


def sample_simans_negatives(
    pool_scores: list[float] | np.ndarray,
    positive_score: float,
    k: int,
    a: float = 0.1,
    b: float = 0.05,
    seed: int = 42,
) -> list[int]:
    """SimANS Gaussian-weighted negative sampling [Zhou et al. EMNLP 2022].

    Each pool candidate i receives weight
        w_i ∝ exp(- (s_i - s_pos + a)^2 / b)
    and `k` candidates are sampled without replacement from this distribution.

    The peak weight is at s_i = s_pos - a — i.e., the sampler targets
    candidates that score `a` BELOW the positive. `b` controls the spread:
    smaller b → narrower peak around target difficulty.

    Returns indices INTO pool_scores (not catalog indices).
    """
    scores = np.asarray(pool_scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("pool_scores must be 1-D")
    if len(scores) == 0:
        return []
    weights = np.exp(-((scores - positive_score + a) ** 2) / b)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        # Degenerate (e.g. all weights underflowed to 0): uniform fallback.
        weights = np.ones_like(scores)
        total = float(weights.sum())
    weights = weights / total
    take = min(k, len(scores))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(scores), size=take, replace=False, p=weights)
    return [int(i) for i in chosen]


def mine_negatives_for_query(
    query_emb: np.ndarray,
    track_embs: np.ndarray,
    track_ids: list[str],
    gold_track_id: str,
    percpos_threshold: float = 0.80,
    k_negs: int = 15,
    pool_size: int = 200,
    seed: int = 42,
    strategy: str = "percpos",
    simans_a: float = 0.1,
    simans_b: float = 0.05,
) -> list[str]:
    """Mine hard negatives for one query.

    strategy="percpos" (default):
      1. Top `pool_size` candidates (excluding gold).
      2. PercPos filter at `percpos_threshold`.
      3. Uniform sample up to k_negs from survivors.

    strategy="simans":
      1. Top `pool_size` candidates (excluding gold).
      2. SimANS-weighted sample of k_negs (no filter; near-positives are
         down-weighted, not dropped — see `sample_simans_negatives`).
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

    if strategy == "simans":
        chosen_pool_idxs = sample_simans_negatives(
            pool_scores, positive_score, k=k_negs,
            a=simans_a, b=simans_b, seed=seed,
        )
        return [track_ids[pool[i]] for i in chosen_pool_idxs]

    if strategy != "percpos":
        raise ValueError(f"unknown mining strategy: {strategy!r}")

    keep_mask = percpos_filter(pool_scores, positive_score, percpos_threshold)
    filtered_track_ids = [track_ids[pool[i]] for i, keep in enumerate(keep_mask) if keep]
    filtered_ranks_in_pool = [i + 2 for i, keep in enumerate(keep_mask) if keep]
    sampled_ranks = sample_hard_negatives(
        filtered_ranks_in_pool, k=k_negs, rank_range=(2, pool_size + 1), seed=seed,
    )
    rank_to_tid = dict(zip(filtered_ranks_in_pool, filtered_track_ids))
    return [rank_to_tid[r] for r in sampled_ranks]
