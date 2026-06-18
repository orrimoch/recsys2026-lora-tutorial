"""Stage 1 — ColBERT fine-tune training-data builder (pure functions only).

Ports the original `scripts/build_colbert_train_data.py` tests (recall-union-lgbm branch)
onto the fresh-start contracts: positives now come from a `TurnContext` stream with an injected
`gold_fn` + `query_builder` (so train query == serve query), and `goal_progress` is read from the
raw HF row. The feedback signal stays the original way (contrastive triples), so the JSONL row
shape (query/positive/negatives + *_tids) is preserved for `losses.Contrastive`.
"""
from __future__ import annotations

from mcrs.contracts import Query, TurnContext, UserProfile
from mcrs.training.colbert_data import (
    build_colbert_triple,
    build_triples_from_pools,
    dev_eval_pack_from_pools,
    goal_progress_label,
    iter_colbert_positives,
    select_hard_negatives,
)

LABEL_POS = "MOVES_TOWARD_GOAL"


def _ctx(turn_number, *, sid="s1", uid="u1", goal="hard 90s hip hop", history=()):
    """Minimal causal TurnContext (utterances length == turn_number per the F2 guard)."""
    utts = [f"u{i}" for i in range(1, turn_number + 1)]
    prof = UserProfile(user_id=uid, age=None, gender=None, country=None, history_tids=list(history))
    seg = "warm" if history else "cold"
    return TurnContext(session_id=sid, user_id=uid, turn_number=turn_number, utterances=utts,
                       goal=goal, user_profile=prof, history_tids=list(history), segment=seg)


class _FakeQB:
    """Stand-in for QueryBuilder — recognizable text so we test the wiring, not QueryBuilder."""
    def build(self, ctx):
        return Query(text=f"Q:{ctx.session_id}:{ctx.turn_number}")


# raw HF session row (goal_progress lives here, NOT on TurnContext)
_ROW = {
    "session_id": "s1",
    "goal_progress_assessments": [
        {"turn_number": 1, "goal_progress_assessment": "MOVES_TOWARD_GOAL"},
        {"turn_number": 2, "goal_progress_assessment": "DOES_NOT_MOVE_TOWARD_GOAL"},
    ],
}


class TestGoalProgressLabel:
    def test_returns_label_for_labeled_turn(self):
        assert goal_progress_label(_ROW, 1) == "MOVES_TOWARD_GOAL"
        assert goal_progress_label(_ROW, 2) == "DOES_NOT_MOVE_TOWARD_GOAL"

    def test_returns_none_for_unlabeled_turn(self):
        assert goal_progress_label(_ROW, 3) is None

    def test_handles_missing_assessments_key(self):
        assert goal_progress_label({"session_id": "x"}, 1) is None


class TestSelectHardNegatives:
    def test_returns_top_k_non_gold_in_pool_order(self):
        pool = ["tA", "tGOLD1", "tB", "tC"]
        assert select_hard_negatives(pool, "tGOLD1", k=2) == ["tA", "tB"]

    def test_returns_none_when_gold_not_in_pool(self):
        # Recall miss -> the ranker can't recover it -> skip this training row.
        assert select_hard_negatives(["tA", "tB"], "tGOLD1", k=2) is None


class TestBuildColbertTriple:
    def test_triple_shape(self):
        triple = build_colbert_triple(
            query="q", pos_text="meta:tGOLD1", neg_texts=["meta:tA", "meta:tB"],
            pos_tid="tGOLD1", neg_tids=["tA", "tB"], session_id="s1", turn_number=1)
        assert triple["query"] == "q"
        assert triple["positive"] == "meta:tGOLD1"
        assert triple["negatives"] == ["meta:tA", "meta:tB"]
        assert triple["pos_tid"] == "tGOLD1"
        assert triple["neg_tids"] == ["tA", "tB"]
        assert triple["turn_number"] == 1


class TestIterColbertPositives:
    def _gold_fn(self):
        golds = {("s1", 1): "tGOLD1", ("s1", 2): "tGOLD2"}
        return lambda ctx: golds.get((ctx.session_id, ctx.turn_number))

    def _gp_fn(self, labels):
        return lambda ctx: labels.get((ctx.session_id, ctx.turn_number))

    def test_yields_query_gold_and_history(self):
        rows = iter_colbert_positives([_ctx(1, history=())], self._gold_fn(),
                                      _FakeQB(), self._gp_fn({("s1", 1): LABEL_POS}))
        assert len(rows) == 1
        r = rows[0]
        assert r["query"] == "Q:s1:1"
        assert r["gold_tid"] == "tGOLD1"
        assert r["turn_number"] == 1
        assert r["session_id"] == "s1"
        assert r["history_tids"] == []
        assert r["segment"] == "cold"  # carried through for segment-aware fusion (ML-review #4)

    def test_turn1_kept_even_without_label(self):
        # RCA #1: real turn-1 has NO goal_progress_assessment, yet the gate is 100% turn-1.
        rows = iter_colbert_positives([_ctx(1)], self._gold_fn(), _FakeQB(),
                                      self._gp_fn({}))  # no labels at all
        assert [r["turn_number"] for r in rows] == [1]
        assert rows[0]["gold_tid"] == "tGOLD1"

    def test_turn_gt1_requires_moves_label(self):
        turns = [_ctx(1), _ctx(2, history=("tGOLD1",))]
        labels = {("s1", 2): "DOES_NOT_MOVE_TOWARD_GOAL"}  # turn-2 off-goal -> dropped
        rows = iter_colbert_positives(turns, self._gold_fn(), _FakeQB(), self._gp_fn(labels))
        assert sorted(r["turn_number"] for r in rows) == [1]

    def test_turn_gt1_kept_when_moves(self):
        turns = [_ctx(1), _ctx(2, history=("tGOLD1",))]
        labels = {("s1", 1): LABEL_POS, ("s1", 2): LABEL_POS}
        rows = iter_colbert_positives(turns, self._gold_fn(), _FakeQB(), self._gp_fn(labels))
        assert sorted(r["turn_number"] for r in rows) == [1, 2]

    def test_skips_turn_without_gold(self):
        # gold_fn returns None (e.g. unmappable) -> no row (can't train without a positive).
        gold_fn = lambda ctx: None
        rows = iter_colbert_positives([_ctx(1)], gold_fn, _FakeQB(), self._gp_fn({}))
        assert rows == []


_DOC = lambda tid: f"meta:{tid}"


def _pos(gold, query="q", sid="s1", tn=1):
    return {"query": query, "gold_tid": gold, "session_id": sid, "turn_number": tn}


class TestBuildTriplesFromPools:
    def test_builds_triple_with_doc_texts(self):
        rows = [_pos("tGOLD1")]
        pools = [["tA", "tGOLD1", "tB", "tC"]]
        triples, stats = build_triples_from_pools(rows, pools, _DOC, k_negs=2)
        assert len(triples) == 1
        t = triples[0]
        assert t["query"] == "q"
        assert t["positive"] == "meta:tGOLD1"
        assert t["negatives"] == ["meta:tA", "meta:tB"]   # gold excluded, hardest-first, top-k
        assert t["pos_tid"] == "tGOLD1"
        assert t["neg_tids"] == ["tA", "tB"]
        assert stats["kept"] == 1 and stats["dropped_no_gold"] == 0 and stats["dropped_no_neg"] == 0

    def test_drops_row_when_gold_not_in_pool(self):
        triples, stats = build_triples_from_pools([_pos("tGOLD1")], [["tA", "tB"]], _DOC, k_negs=2)
        assert triples == []
        assert stats["dropped_no_gold"] == 1 and stats["kept"] == 0

    def test_drops_row_with_no_negatives(self):
        # gold in pool but nothing else -> no negative to form a contrastive pair.
        triples, stats = build_triples_from_pools([_pos("tGOLD1")], [["tGOLD1"]], _DOC, k_negs=5)
        assert triples == []
        assert stats["dropped_no_neg"] == 1 and stats["kept"] == 0

    def test_respects_k_negs_cap(self):
        pool = ["tGOLD1"] + [f"n{i}" for i in range(10)]
        triples, _ = build_triples_from_pools([_pos("tGOLD1")], [pool], _DOC, k_negs=3)
        assert len(triples[0]["negatives"]) == 3


def _drow(gold, query="q", tn=1):
    return {"query": query, "gold_tid": gold, "turn_number": tn}


class TestDevEvalPackFromPools:
    def test_builds_pack_fields(self):
        rows = [_drow("g1", "q1", 1), _drow("g2", "q2", 2)]
        pools = [["g1", "a"], ["b", "g2"]]
        pack = dev_eval_pack_from_pools(rows, pools, _DOC)
        assert pack["queries"] == ["q1", "q2"]
        assert pack["golds"] == ["g1", "g2"]
        assert pack["pools"] == [["g1", "a"], ["b", "g2"]]

    def test_wall_marks_turn1_only(self):
        rows = [_drow("g1", tn=1), _drow("g2", tn=2)]
        pack = dev_eval_pack_from_pools(rows, [["g1"], ["g2"]], _DOC)
        assert pack["wall"] == [True, False]

    def test_tid_to_text_covers_pool_union(self):
        rows = [_drow("g1"), _drow("g2", tn=1)]
        pack = dev_eval_pack_from_pools(rows, [["g1", "a"], ["a", "g2"]], _DOC)
        assert pack["tid_to_text"] == {"g1": "meta:g1", "a": "meta:a", "g2": "meta:g2"}

    def test_skips_rows_without_gold(self):
        # gold None -> drop the row AND its aligned pool (can't score recall without a gold).
        rows = [_drow(None, "q1"), _drow("g2", "q2")]
        pack = dev_eval_pack_from_pools(rows, [["x"], ["g2"]], _DOC)
        assert pack["queries"] == ["q2"] and pack["golds"] == ["g2"]
        assert pack["pools"] == [["g2"]]
