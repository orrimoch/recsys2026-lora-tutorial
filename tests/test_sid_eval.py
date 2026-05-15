"""Tests for SID generator eval metric (W3)."""
import math


def test_ndcg_at_k_gold_at_rank_1_returns_1():
    """Gold appears at top → nDCG = 1.0."""
    from mcrs.sid.eval import compute_ndcg_at_k
    score = compute_ndcg_at_k(retrieved=["a", "b", "c"], gold="a", k=20)
    assert score == 1.0


def test_ndcg_at_k_gold_at_rank_2_uses_log_discount():
    """Gold at rank 2 → nDCG = 1/log2(3) ≈ 0.6309."""
    from mcrs.sid.eval import compute_ndcg_at_k
    score = compute_ndcg_at_k(retrieved=["x", "a", "y"], gold="a", k=20)
    expected = 1 / math.log2(2 + 1)
    assert abs(score - expected) < 1e-9


def test_ndcg_at_k_gold_not_in_top_k_returns_0():
    """Gold not in top-k → nDCG = 0."""
    from mcrs.sid.eval import compute_ndcg_at_k
    retrieved = [f"track_{i}" for i in range(20)]
    score = compute_ndcg_at_k(retrieved=retrieved, gold="not_here", k=20)
    assert score == 0.0


def test_ndcg_at_k_truncates_to_k():
    """Anything past position k is ignored even if gold is there."""
    from mcrs.sid.eval import compute_ndcg_at_k
    retrieved = [f"track_{i}" for i in range(25)]
    retrieved.append("gold")  # position 25 (0-indexed)
    score = compute_ndcg_at_k(retrieved=retrieved, gold="gold", k=20)
    assert score == 0.0


def test_aggregate_ndcg_averages_per_query():
    """aggregate_ndcg returns mean across per-query nDCGs."""
    from mcrs.sid.eval import aggregate_ndcg
    per_query = [1.0, 0.5, 0.0, 0.25]
    assert abs(aggregate_ndcg(per_query) - 0.4375) < 1e-9


def test_aggregate_ndcg_empty_returns_zero():
    """Empty list → 0.0 (defensive — should never happen but keep deterministic)."""
    from mcrs.sid.eval import aggregate_ndcg
    assert aggregate_ndcg([]) == 0.0
