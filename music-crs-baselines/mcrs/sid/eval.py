"""SID generator eval (W3 gate): nDCG@20 + paired-bootstrap CI vs Phase 0 baseline.

The compute_ndcg_at_k function uses binary relevance with a single gold per query
— matches RecSys 2026 Music CRS turn-level evaluation. paired_bootstrap_ci is
imported from the existing scripts/compare_diagnostic_runs.py utility (don't
duplicate; use the version that's already battle-tested).
"""
from __future__ import annotations

import math
from typing import Iterable


def compute_ndcg_at_k(
    retrieved: list[str],
    gold: str,
    k: int = 20,
) -> float:
    """nDCG@k with binary relevance (single gold).

    DCG = 1 / log2(rank + 2) where rank is 0-indexed position of gold (if found).
    IDCG with one gold is always 1 (gold at rank 0 → 1/log2(2) = 1).
    """
    truncated = retrieved[:k]
    for rank, tid in enumerate(truncated):
        if tid == gold:
            return 1.0 / math.log2(rank + 2)
    return 0.0


def aggregate_ndcg(per_query_scores: Iterable[float]) -> float:
    """Mean nDCG across queries (defensive: empty input → 0)."""
    scores = list(per_query_scores)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
