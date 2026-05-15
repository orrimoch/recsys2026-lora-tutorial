"""Pure functions for SID quantizer validation gates (per spec section 2.4)."""
from __future__ import annotations

import random


def validate_codebook_utilization(
    assignments: list[int],
    codebook_size: int,
    threshold: float = 0.80,
) -> tuple[bool, float]:
    """Gate 2: fraction of codebook entries used by >=1 track must be >= threshold.

    Returns (passed, utilization_fraction). Direct test of Sinkhorn regularization
    (collapsed codebooks have low utilization).
    """
    used = set(assignments)
    util = len(used) / codebook_size
    return (util >= threshold, util)


def validate_cluster_purity(
    buckets: dict[str, list[str]],
    tag_lookup: dict[str, list[str]],
    n_samples: int = 100,
    threshold: float = 0.60,
    seed: int = 42,
) -> tuple[bool, float]:
    """Gate 3: sample n_samples buckets; check fraction where tracks share >=1 tag.

    Singleton buckets count as pure. Tags compared case-insensitive,
    whitespace-stripped. Returns (passed, purity_fraction).
    """
    if not buckets:
        return (True, 1.0)
    rng = random.Random(seed)
    keys = list(buckets.keys())
    sampled_keys = rng.sample(keys, min(n_samples, len(keys)))

    pure_count = 0
    for k in sampled_keys:
        members = buckets[k]
        if len(members) <= 1:
            pure_count += 1
            continue
        tag_sets = [
            {t.strip().lower() for t in tag_lookup.get(tid, [])}
            for tid in members
        ]
        common = set.intersection(*tag_sets) if tag_sets else set()
        if common:
            pure_count += 1
    purity = pure_count / len(sampled_keys)
    return (purity >= threshold, purity)


def compute_relative_mse_gate(
    rqvae_mse: float,
    pca_mse: float,
    multiplier: float = 1.5,
    absolute_fallback: float = 0.01,
) -> tuple[bool, float]:
    """Gate 1: RQ-VAE reconstruction MSE <= multiplier * PCA-256 baseline MSE.

    Falls back to absolute threshold when PCA baseline is 0 (degenerate case).
    Returns (passed, ratio_or_absolute_value).
    """
    if pca_mse == 0.0:
        return (rqvae_mse < absolute_fallback, rqvae_mse)
    ratio = rqvae_mse / pca_mse
    return (ratio <= multiplier, ratio)
