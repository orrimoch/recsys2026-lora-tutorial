"""R7 — weighted-RRF fusion → Candidate pool (F2 RetrievalChannel + fuse())."""
from __future__ import annotations

from mcrs.contracts import Candidate
from mcrs.retrieval.fusion import RRFFusion


class _Fake:
    def __init__(self, label, results):
        self.label = label
        self._results = results

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        return [r[:topk] for r in self._results]


class _RecChannel:
    """Records the query list it was handed (to assert per-channel routing)."""
    def __init__(self, label, query_key=None):
        self.label = label
        self.query_key = query_key
        self.seen = None

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        self.seen = list(queries)
        return [[f"{self.label}-{q}"] for q in queries]


def test_per_channel_query_routing():
    # ColBERT needs its focused query; other channels keep the full query (A1).
    dense = _RecChannel("dense")                        # no query_key -> main query
    cb = _RecChannel("colbert", query_key="colbert")   # routed to its per-channel query
    fus = RRFFusion([dense, cb], weights=[1.0, 1.0], k=60)
    fus.fuse(["FULL1", "FULL2"], topk=5, per_channel_queries={"colbert": ["foc1", "foc2"]})
    assert dense.seen == ["FULL1", "FULL2"]
    assert cb.seen == ["foc1", "foc2"]


def test_per_channel_routing_falls_back_to_main_when_key_absent():
    cb = _RecChannel("colbert", query_key="colbert")
    fus = RRFFusion([cb], k=60)
    fus.fuse(["Q1"], topk=5)                            # no per_channel_queries -> main
    assert cb.seen == ["Q1"]
    fus.fuse(["Q2"], topk=5, per_channel_queries={"other": ["x"]})  # key not for this channel
    assert cb.seen == ["Q2"]


def test_per_channel_routing_rejects_misaligned_length():
    # A routed query list must be 1:1 with the turns/batch_context, or rows silently cross-contaminate.
    import pytest
    cb = _RecChannel("colbert", query_key="colbert")
    fus = RRFFusion([cb], k=60)
    with pytest.raises(ValueError):
        fus.fuse(["Q1", "Q2"], topk=5, per_channel_queries={"colbert": ["only-one"]})


def test_fuse_per_sub_weighted_rrf_math():
    per_sub = [[["a", "b", "c"]], [["a", "c"]]]  # 2 channels, 1 query
    fused = RRFFusion.fuse_per_sub(per_sub, weights=[1.0, 1.0], k=60, topk=3)
    assert fused[0] == ["a", "c", "b"]  # a hit by both; c by both at deeper ranks; b once


def test_fusion_returns_candidates_with_score_and_ranks():
    ch1 = _Fake("bm25", [["a", "b", "c"]])
    ch2 = _Fake("dense", [["a", "c"]])
    fus = RRFFusion([ch1, ch2], weights=[1.0, 1.0], k=60)
    pools = fus.fuse(["q"], topk=3)
    assert len(pools) == 1
    pool = pools[0]
    assert [c.track_id for c in pool] == ["a", "c", "b"]
    assert all(isinstance(c, Candidate) for c in pool)
    top = pool[0]
    assert top.track_id == "a" and top.rrf_score > 0
    assert top.channel_ranks == {"bm25": 1, "dense": 1}


def test_fusion_channel_interface_returns_ids():
    fus = RRFFusion([_Fake("bm25", [["a", "b"]]), _Fake("dense", [["b"]])], k=60)
    out = fus.batch_text_to_item_retrieval(["q"], topk=2)
    assert out[0][0] == "b"  # b retrieved by both -> highest


def test_weights_shift_ranking():
    ch1 = _Fake("c1", [["a", "b"]])
    ch2 = _Fake("c2", [["b", "a"]])
    # upweight c2 -> its rank-1 (b) should win
    out = RRFFusion([ch1, ch2], weights=[1.0, 5.0], k=60).batch_text_to_item_retrieval(["q"], topk=2)
    assert out[0][0] == "b"


def test_zero_weight_channel_is_dropped():
    a = _Fake("A", [["a"]])
    b = _Fake("B", [["b"]])
    out = RRFFusion([a, b], weights=[1.0, 0.0], k=60).batch_text_to_item_retrieval(["q"], topk=5)
    assert out[0] == ["a"]                       # B zero-weighted -> contributes nothing


def test_segment_weights_select_per_query_channel_mix():
    a = _Fake("A", [["a"], ["a"]])               # both channels return for both queries
    b = _Fake("B", [["b"], ["b"]])
    fus = RRFFusion([a, b], weights=[1.0, 1.0], k=60,
                    segment_weights={"cold": [1.0, 0.0], "warm": [0.0, 1.0]})
    bc = [{"segment": "cold"}, {"segment": "warm"}]
    out = fus.batch_text_to_item_retrieval(["q1", "q2"], topk=5, batch_context=bc)
    assert out[0] == ["a"]                        # cold -> only A weighted
    assert out[1] == ["b"]                        # warm -> only B weighted


def test_unknown_or_missing_segment_falls_back_to_default_weights():
    a = _Fake("A", [["a"]])
    b = _Fake("B", [["b"]])
    fus = RRFFusion([a, b], weights=[1.0, 1.0], k=60, segment_weights={"cold": [1.0, 0.0]})
    out = fus.batch_text_to_item_retrieval(["q"], topk=5, batch_context=[{"segment": "warm"}])
    assert set(out[0]) == {"a", "b"}              # 'warm' not in map -> default [1,1] -> both kept
