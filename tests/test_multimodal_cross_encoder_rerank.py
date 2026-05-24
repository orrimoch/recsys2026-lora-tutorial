"""Unit tests for MULTIMODAL_RERANKER (Phase 7).

The model-backed scoring (``_score_candidates``) is integration-level (loads the
570M reranker + artifacts; runs on Colab). Here we test the pure ordering helper
and the ``rerank()`` ordering + caching via a subclass that overrides
``_score_candidates`` and bypasses the heavy ``__init__`` — no model load.
"""
import numpy as np

from mcrs.rerankers.multimodal_cross_encoder_rerank import (
    MULTIMODAL_RERANKER,
    rerank_by_scores,
)


def test_rerank_by_scores_orders_desc_and_truncates():
    tids = ["a", "b", "c", "d"]
    scores = [0.1, 0.9, 0.5, 0.3]
    assert rerank_by_scores(tids, scores, 2) == ["b", "c"]
    assert rerank_by_scores(tids, scores, 10) == ["b", "c", "d", "a"]


def test_rerank_by_scores_stable_on_ties():
    tids = ["x", "y", "z"]
    scores = [1.0, 1.0, 0.0]
    assert rerank_by_scores(tids, scores, 2) == ["x", "y"]  # ties keep input order


class _StubReranker(MULTIMODAL_RERANKER):
    """Bypasses the heavy __init__; scores by tid length as a deterministic stand-in."""

    def __init__(self):
        self._cache = {}
        self.score_calls = 0

    def _score_candidates(self, query, candidate_tids, user_id):
        self.score_calls += 1
        return np.array([len(t) for t in candidate_tids], dtype=float)


def test_rerank_orders_by_score_and_truncates():
    rr = _StubReranker()
    out = rr.rerank(["q"], [["aa", "b", "ccc"]], topk=2, user_ids=["u1"])
    assert out == [["ccc", "aa"]]  # scores=lengths [2,1,3] -> desc ccc,aa


def test_rerank_caches_by_query_user_candidates():
    rr = _StubReranker()
    rr.rerank(["q"], [["aa", "b", "ccc"]], topk=3, user_ids=["u1"])
    rr.rerank(["q"], [["aa", "b", "ccc"]], topk=3, user_ids=["u1"])
    assert rr.score_calls == 1  # second call served from cache


def test_rerank_cache_is_candidate_order_independent():
    rr = _StubReranker()
    out1 = rr.rerank(["q"], [["aa", "b", "ccc"]], topk=3, user_ids=["u1"])
    out2 = rr.rerank(["q"], [["ccc", "aa", "b"]], topk=3, user_ids=["u1"])
    assert rr.score_calls == 1   # same candidate set -> cache hit
    assert out1 == out2


def test_rerank_works_without_user_ids():
    rr = _StubReranker()
    out = rr.rerank(["q"], [["aa", "ccc", "b"]], topk=2)
    assert out == [["ccc", "aa"]]
