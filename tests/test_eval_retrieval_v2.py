"""Tests for offline retrieval eval (nDCG@K / recall@K / MRR)."""
import pytest


def test_ndcg_at_k_perfect_top_1():
    """Gold at rank 1 → nDCG@20 = 1.0."""
    from scripts.eval_retrieval_v2 import compute_ndcg_at_k
    score = compute_ndcg_at_k(retrieved=["g", "a", "b", "c"], gold="g", k=20)
    assert score == 1.0


def test_ndcg_at_k_gold_outside_topk():
    """Gold below k → nDCG@20 = 0.0."""
    from scripts.eval_retrieval_v2 import compute_ndcg_at_k
    score = compute_ndcg_at_k(retrieved=["a", "b"] * 20, gold="g", k=20)
    assert score == 0.0


def test_recall_at_k_present():
    """Recall@K = 1 if gold in top-K."""
    from scripts.eval_retrieval_v2 import compute_recall_at_k
    assert compute_recall_at_k(retrieved=["a", "b", "g"], gold="g", k=3) == 1.0
    assert compute_recall_at_k(retrieved=["a", "b"], gold="g", k=3) == 0.0


def test_mrr_returns_reciprocal_of_first_correct_rank():
    """MRR for one query is 1 / rank_of_gold (or 0 if absent)."""
    from scripts.eval_retrieval_v2 import compute_mrr
    assert compute_mrr(retrieved=["a", "g", "c"], gold="g") == 0.5
    assert compute_mrr(retrieved=["a", "b", "c"], gold="g") == 0.0
