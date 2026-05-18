"""Tests for NV-Retriever-style hard-negative miner."""
import numpy as np
import pytest


def test_percpos_filter_drops_above_threshold():
    """Candidates scoring >= threshold * positive_score are dropped (false-neg guard)."""
    from mcrs.retrieval_modules.hn_miner import percpos_filter
    candidate_scores = [0.95, 0.82, 0.75, 0.60, 0.40]
    positive_score = 1.00
    kept = percpos_filter(candidate_scores, positive_score, threshold=0.80)
    assert kept == [False, False, True, True, True]  # 0.95 and 0.82 dropped


def test_percpos_filter_keeps_everything_at_threshold_1_0():
    """threshold=1.0 only drops candidates that exactly tie the positive."""
    from mcrs.retrieval_modules.hn_miner import percpos_filter
    kept = percpos_filter([0.99, 0.50], positive_score=1.00, threshold=1.0)
    assert kept == [True, True]


def test_sample_hard_negatives_respects_rank_range_and_cap():
    """sample_hard_negatives draws from the filtered top-K within rank_range."""
    from mcrs.retrieval_modules.hn_miner import sample_hard_negatives
    filtered_ranks = [5, 7, 10, 12, 15, 20, 30, 40, 50, 80]
    drawn = sample_hard_negatives(filtered_ranks, k=4, rank_range=(2, 200), seed=42)
    assert len(drawn) == 4
    for rank in drawn:
        assert 2 <= rank <= 200
        assert rank in filtered_ranks


def test_sample_hard_negatives_returns_all_when_pool_smaller_than_k():
    """If only 3 negatives survive the filter but k=10, return all 3 (no padding)."""
    from mcrs.retrieval_modules.hn_miner import sample_hard_negatives
    drawn = sample_hard_negatives([5, 10, 15], k=10, rank_range=(2, 200), seed=42)
    assert len(drawn) == 3


def test_mine_negatives_for_query_returns_negs_aligned_to_track_ids():
    """End-to-end mine: returns list of track_ids (strings), not indices."""
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    track_ids = ["t1", "t2", "t3", "t4", "t5"]
    query_emb = np.array([1.0, 0.0])
    track_embs = np.array([
        [1.0, 0.0],   # t1 — gold
        [0.95, 0.31], # t2 — hard near-gold
        [0.80, 0.60], # t3 — medium
        [0.50, 0.87], # t4 — far
        [0.0, 1.0],   # t5 — orthogonal
    ])
    negs = mine_negatives_for_query(
        query_emb=query_emb, track_embs=track_embs, track_ids=track_ids,
        gold_track_id="t1", percpos_threshold=0.97, k_negs=2, seed=42,
    )
    assert len(negs) <= 2
    assert "t1" not in negs
    for n in negs:
        assert n in track_ids
