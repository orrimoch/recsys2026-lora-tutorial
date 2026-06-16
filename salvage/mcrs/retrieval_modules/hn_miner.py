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
from typing import Optional

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


def batch_mine_negatives(
    query_embs,
    track_embs,
    track_ids: list[str],
    gold_track_ids: list[str],
    percpos_threshold: float = 0.80,
    k_negs: int = 15,
    pool_size: int = 200,
    seed: int = 42,
    strategy: str = "percpos",
    simans_a: float = 0.1,
    simans_b: float = 0.05,
    use_gpu: bool = True,
    global_gold_ids: Optional[set] = None,
) -> list:
    """Vectorized HN mining for a BATCH of B queries.

    Same math as `mine_negatives_for_query` but the similarity matmul +
    top-K sort run ONCE per batch on GPU (instead of per-query on CPU).
    For BGE-M3-scale catalogs (~47K tracks), this is ~5-6× faster wall-
    clock at typical batch sizes (64-512) — the per-query CPU argsort +
    full-pool list comprehension dominated the previous mining loop.

    Args:
      query_embs: (B, D) unit-normalized. numpy.ndarray or torch.Tensor.
      track_embs: (N, D) unit-normalized. numpy.ndarray or torch.Tensor.
      track_ids: length N, unique.
      gold_track_ids: length B. Entries not in `track_ids` produce None in
        the output (no exception — matches the build script's per-row
        skip-when-no-gold semantics).
      seed: base seed. Row b uses `seed + b` so re-mining with the same
        base seed produces the same negs. Matches the per-query function's
        `seed=base + global_idx` convention when called with base=42+i
        and a batch starting at global index i.

    Returns:
      list of length B. Each entry is either:
        - a list of neg track_ids (success; length up to k_negs)
        - None (gold not in catalog, OR percpos filter left no candidates)

    Notes:
      - Uses torch on GPU when available (use_gpu=True). Falls back to
        CPU torch otherwise. Both produce identical results.
      - For bit-equivalence with the per-query function: row b's
        sampling uses seed = base_seed + b, so the same query at the
        same global position produces the same negs across re-runs.
    """
    import torch
    import numpy as _np

    # ---- Input validation -------------------------------------------------
    if hasattr(query_embs, "shape"):
        B, D = query_embs.shape[0], query_embs.shape[1]
    else:
        raise ValueError("query_embs must be an array with .shape")
    if hasattr(track_embs, "shape"):
        N, D2 = track_embs.shape[0], track_embs.shape[1]
    else:
        raise ValueError("track_embs must be an array with .shape")
    if D != D2:
        raise ValueError(
            f"query_embs and track_embs dim mismatch: D={D} vs {D2}"
        )
    if len(track_ids) != N:
        raise ValueError(
            f"track_ids length ({len(track_ids)}) != track_embs N ({N})"
        )
    if len(gold_track_ids) != B:
        raise ValueError(
            f"gold_track_ids length ({len(gold_track_ids)}) != query batch B ({B})"
        )
    if len(set(track_ids)) != N:
        raise ValueError("track_ids must be unique (duplicates would leak gold as negative)")

    # ---- Move tensors to device ------------------------------------------
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.from_numpy(_np.asarray(x, dtype=_np.float32)).to(device)

    q_t = _to_tensor(query_embs)        # (B, D)
    t_t = _to_tensor(track_embs)        # (N, D)

    # ---- Vectorized similarity + top-K ----------------------------------
    # sims: (B, N). Single GPU matmul handles all queries at once.
    sims_t = q_t @ t_t.T

    # Top-K with slack so we can exclude the gold and still get pool_size.
    top_k_eff = min(pool_size + 5, N)
    top_scores_t, top_idxs_t = torch.topk(sims_t, k=top_k_eff, dim=1)
    top_scores = top_scores_t.cpu().numpy()
    top_idxs = top_idxs_t.cpu().numpy()

    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    # Catalog indices that are SOME query's gold — excluded from every negative
    # pool so a true positive elsewhere can't be mined as a false negative here.
    global_gold_idxs = ({tid_to_idx[g] for g in global_gold_ids if g in tid_to_idx}
                        if global_gold_ids else set())

    # ---- Per-row filter + sample (cheap; pool is now ~205, not 47K) -----
    results: list = []
    for b in range(B):
        gold_tid = gold_track_ids[b]
        if gold_tid not in tid_to_idx:
            results.append(None)
            continue
        gold_idx = tid_to_idx[gold_tid]
        # positive_score: row b's dot with gold's catalog vector. Recompute
        # cheaply rather than indexing into sims_t (which we moved off GPU).
        if isinstance(query_embs, torch.Tensor):
            q_np = query_embs[b].detach().cpu().numpy()
        else:
            q_np = _np.asarray(query_embs[b], dtype=_np.float32)
        if isinstance(track_embs, torch.Tensor):
            t_np = track_embs[gold_idx].detach().cpu().numpy()
        else:
            t_np = _np.asarray(track_embs[gold_idx], dtype=_np.float32)
        positive_score = float(_np.dot(q_np, t_np))

        # Build pool: top-K minus the gold position, capped at pool_size.
        pool: list[int] = []
        pool_scores_b: list[float] = []
        for k in range(top_idxs.shape[1]):
            idx = int(top_idxs[b, k])
            if idx == gold_idx or idx in global_gold_idxs:
                continue
            pool.append(idx)
            pool_scores_b.append(float(top_scores[b, k]))
            if len(pool) >= pool_size:
                break

        row_seed = seed + b

        if strategy == "simans":
            chosen = sample_simans_negatives(
                pool_scores_b, positive_score, k=k_negs,
                a=simans_a, b=simans_b, seed=row_seed,
            )
            if not chosen:
                results.append([])
                continue
            results.append([track_ids[pool[i]] for i in chosen])
            continue

        if strategy != "percpos":
            raise ValueError(f"unknown mining strategy: {strategy!r}")

        keep_mask = percpos_filter(pool_scores_b, positive_score, percpos_threshold)
        filtered_track_ids = [track_ids[pool[i]] for i, keep in enumerate(keep_mask) if keep]
        filtered_ranks = [i + 2 for i, keep in enumerate(keep_mask) if keep]
        sampled_ranks = sample_hard_negatives(
            filtered_ranks, k=k_negs, rank_range=(2, pool_size + 1), seed=row_seed,
        )
        rank_to_tid = dict(zip(filtered_ranks, filtered_track_ids))
        results.append([rank_to_tid[r] for r in sampled_ranks])

    return results
