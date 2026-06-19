"""K2 training-data driver — build (ctx, candidates, gold) groups by running fusion over turns."""
from __future__ import annotations

from mcrs.contracts import TurnContext, UserProfile
from mcrs.retrieval.fusion import RRFFusion
from mcrs.retrieval.query import QueryBuilder
from mcrs.rerank.train import build_rerank_groups


class _Fake:
    def __init__(self, label, results):
        self.label, self._results = label, results

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        return [r[:topk] for r in self._results]


def _turns():
    p = UserProfile("u", 1, "f", "US", [])
    return [TurnContext("s1", "u", 1, ["hi"], "g", p, [], "cold"),
            TurnContext("s1", "u", 2, ["hi", "more"], "g", p, ["a"], "warm")]


class _RecCh:
    """Records the query list it was handed (to assert per-channel routing)."""
    def __init__(self, label, query_key=None):
        self.label, self.query_key, self.seen = label, query_key, None

    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None):
        self.seen = list(queries)
        return [["a"] for _ in queries]


def test_build_groups_routes_focused_query_to_keyed_channel():
    # Train pools must route the focused query to the colbert channel exactly like InferenceHarness,
    # else K2 is trained on a different colbert pool than it meets at serve (train/serve skew).
    cb = _RecCh("colbert", query_key="colbert")
    dense = _RecCh("dense")
    fusion = RRFFusion([dense, cb], k=60)
    golds = {("s1", 1): "a", ("s1", 2): "a"}
    build_rerank_groups(QueryBuilder(), fusion, _turns(),
                        lambda t: golds[(t.session_id, t.turn_number)], topk=3,
                        per_channel_query_builders={"colbert": QueryBuilder(recency_window=1)})
    # turn1 utt ["hi"] goal "g" -> "hi g"; turn2 utt ["hi","more"] -> focused "more g"
    assert cb.seen == ["hi g", "more g"]
    assert dense.seen == ["hi g", "hi more g"]   # dense keeps the FULL query


def test_build_groups_raises_on_unmatched_routing_key():
    # A routing key that matches no channel query_key would SILENTLY fall back to the full query,
    # mistraining K2 on a full-vs-focused-skewed pool. Mirror InferenceHarness: fail loud.
    import pytest
    cb = _RecCh("colbert", query_key="colbert")
    fusion = RRFFusion([cb], k=60)
    with pytest.raises(ValueError):
        build_rerank_groups(QueryBuilder(), fusion, _turns(),
                            lambda t: "a", topk=3,
                            per_channel_query_builders={"colbert_ft": QueryBuilder(recency_window=1)})


def test_build_groups_runs_fusion_and_attaches_gold():
    fusion = RRFFusion([_Fake("bm25", [["a", "b", "c"], ["d", "e", "f"]])], k=60)
    golds = {("s1", 1): "b", ("s1", 2): "zzz"}  # turn 2 gold not in pool
    groups = build_rerank_groups(QueryBuilder(), fusion, _turns(),
                                 lambda t: golds[(t.session_id, t.turn_number)], topk=3)
    assert len(groups) == 2
    ctx, cands, gold = groups[0]
    assert ctx.turn_number == 1 and [c.track_id for c in cands] == ["a", "b", "c"] and gold == "b"
    assert groups[1][2] == "zzz"  # gold passed through; K2.build_training_data will skip it
