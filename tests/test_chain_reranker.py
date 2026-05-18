"""Tests for CHAIN_RERANKER."""
import pytest


class _StubReranker:
    """Records the call args + returns a fixed list (the first N per query)."""

    def __init__(self, name, topk):
        self.name = name
        self.topk = topk
        self.calls = []

    def rerank(self, queries, candidate_tids, topk, **kwargs):
        self.calls.append({"queries": queries, "cands": candidate_tids, "topk": topk, "kwargs": kwargs})
        # Mimic a real reranker: take the first `topk` per query.
        return [c[:topk] for c in candidate_tids]


def test_chain_reranker_runs_stages_in_order(tmp_path):
    from mcrs.rerankers.chain import CHAIN_RERANKER
    ce = _StubReranker("ce", 50)
    lg = _StubReranker("lgbm", 20)
    chain = CHAIN_RERANKER(stages=[("ce", 50, ce), ("lgbm", 20, lg)])
    out = chain.rerank(
        queries=["q1", "q2"],
        candidate_tids=[list(f"c{i}" for i in range(100)), list(f"d{i}" for i in range(100))],
        topk=20,
        user_ids=["u1", "u2"], goal_categories=[None, None],
        goal_specificities=[None, None], user_profiles_raw=[None, None],
    )
    # Stage 1 (CE) trims to topk=50; Stage 2 (LGBM) trims to topk=20.
    assert len(ce.calls) == 1 and ce.calls[0]["topk"] == 50
    assert len(lg.calls) == 1 and lg.calls[0]["topk"] == 20
    # Final shape matches caller's topk request (20).
    assert all(len(o) == 20 for o in out)


def test_chain_reranker_forwards_side_channel_kwargs():
    from mcrs.rerankers.chain import CHAIN_RERANKER
    ce = _StubReranker("ce", 50)
    chain = CHAIN_RERANKER(stages=[("ce", 50, ce)])
    chain.rerank(
        queries=["q"], candidate_tids=[["a", "b", "c"]], topk=2,
        user_ids=["u"], goal_categories=["cat"],
        goal_specificities=["spec"], user_profiles_raw=[{"age": 30}],
    )
    assert ce.calls[0]["kwargs"]["user_ids"] == ["u"]
    assert ce.calls[0]["kwargs"]["goal_categories"] == ["cat"]
    assert ce.calls[0]["kwargs"]["user_profiles_raw"] == [{"age": 30}]
