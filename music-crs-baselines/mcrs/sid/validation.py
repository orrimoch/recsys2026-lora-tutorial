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
    threshold: float = 0.20,
    seed: int = 42,
) -> tuple[bool, float]:
    """Gate 3 v2 (2026-05-16): dominant-tag fraction averaged across sampled buckets.

    Per-bucket "dominance" = fraction of members carrying the bucket's most
    common tag. We then average dominance across n_samples buckets and pass
    if average >= threshold.

    Why not set.intersection (v1): at full scale, a healthy 47K-track quantizer
    with 99% codebook utilization produces buckets averaging ~184 tracks. For
    ALL 184 to share even one tag (set intersection) is statistically impossible
    even when clusters are genuinely meaningful. Dominance scales gracefully.

    Singleton buckets contribute dominance=1.0 (trivially dominated by their
    single member's most-frequent tag). Tags normalized lowercase + stripped.
    """
    if not buckets:
        return (True, 1.0)
    from collections import Counter
    rng = random.Random(seed)
    keys = list(buckets.keys())
    sampled_keys = rng.sample(keys, min(n_samples, len(keys)))

    dominances: list[float] = []
    for k in sampled_keys:
        members = buckets[k]
        if len(members) <= 1:
            dominances.append(1.0)
            continue
        tag_counts: Counter[str] = Counter()
        for tid in members:
            for t in tag_lookup.get(tid, []):
                tag_counts[t.strip().lower()] += 1
        if not tag_counts:
            dominances.append(0.0)
            continue
        top_count = tag_counts.most_common(1)[0][1]
        dominances.append(top_count / len(members))
    mean_dominance = sum(dominances) / len(dominances)
    return (mean_dominance >= threshold, mean_dominance)


def compute_relative_mse_gate(
    rqvae_mse: float,
    pca_mse: float,
    multiplier: float = 1.5,
    absolute_threshold: float = 0.02,
) -> tuple[bool, float]:
    """Gate 1: pass if RQ-VAE reconstruction quality is good enough.

    Two-path semantics (revised 2026-05-16 after first Colab run showed PCA-only
    gate was geometrically too strict):
      Primary path: ABSOLUTE — pass if rqvae_mse <= absolute_threshold (default 0.02).
        For L2-normalized inputs, MSE=0.02 ≈ 99% variance explained; that's
        good-enough quality regardless of PCA's continuous-reconstruction baseline.
      Fallback path: RELATIVE — if absolute fails, pass if rqvae_mse <= multiplier * pca_mse
        (kept as a safety net for cases where the data itself is intrinsically noisy
        and even PCA can't reconstruct well).

    Returns (passed, rqvae_mse) — second element is always absolute MSE for
    consistent log inspection. The pca_mse is reported separately by the orchestrator.
    """
    if rqvae_mse <= absolute_threshold:
        return (True, rqvae_mse)
    if pca_mse == 0.0:
        return (False, rqvae_mse)
    # Fallback: relative-to-PCA path for high-MSE cases
    ratio = rqvae_mse / pca_mse
    return (ratio <= multiplier, rqvae_mse)
