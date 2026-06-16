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


def test_build_groups_runs_fusion_and_attaches_gold():
    fusion = RRFFusion([_Fake("bm25", [["a", "b", "c"], ["d", "e", "f"]])], k=60)
    golds = {("s1", 1): "b", ("s1", 2): "zzz"}  # turn 2 gold not in pool
    groups = build_rerank_groups(QueryBuilder(), fusion, _turns(),
                                 lambda t: golds[(t.session_id, t.turn_number)], topk=3)
    assert len(groups) == 2
    ctx, cands, gold = groups[0]
    assert ctx.turn_number == 1 and [c.track_id for c in cands] == ["a", "b", "c"] and gold == "b"
    assert groups[1][2] == "zzz"  # gold passed through; K2.build_training_data will skip it
