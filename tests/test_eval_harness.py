"""F3 — eval harness: official parity + diagnostics."""
from __future__ import annotations

import math

import pytest

from mcrs.contracts import SubmissionRow
from mcrs.eval.diagnostics import hit_rank, recall_at_k, score_diagnostic
from mcrs.eval.harness import GoldRow
from mcrs.eval.official import score_official

# Fixture: one session, two turns. t1 gold at rank 1; t2 gold at rank 2.
_GT = [
    GoldRow("s1", "u1", 1, "g1"),
    GoldRow("s1", "u1", 2, "g2"),
]
_PRED = [
    SubmissionRow("s1", "u1", 1, ["g1", "x", "y"], "resp one"),
    SubmissionRow("s1", "u1", 2, ["x", "g2", "z"], "resp two"),
]
_INV_LOG3 = 1.0 / math.log2(3)  # nDCG of a single gold at rank 2


# ---- official path (parity to evaluate_devset.py formulas) ----
def test_official_ndcg_matches_hand_computed_macro():
    s = score_official(_PRED, _GT, catalog_size=100)
    # turn1: (1,1,1); turn2: (0, 1/log2 3, 1/log2 3); macro = mean over the 2 turn buckets
    assert s["ndcg@1"] == pytest.approx(0.5)
    assert s["ndcg@10"] == pytest.approx((1.0 + _INV_LOG3) / 2)
    assert s["ndcg@20"] == pytest.approx((1.0 + _INV_LOG3) / 2)


def test_official_diversity_and_catalog_size():
    s = score_official(_PRED, _GT, catalog_size=100)
    # unique recommended {g1,x,y,g2,z} = 5 over catalog 100
    assert s["catalog_diversity"] == pytest.approx(0.05)
    # Distinct-2 over ["resp one","resp two"] -> {(resp,one),(resp,two)} / 2 = 1.0
    assert s["lexical_diversity"] == pytest.approx(1.0)
    assert s["total_catalog_size"] == 100


def test_official_keys_are_exactly_the_official_set():
    s = score_official(_PRED, _GT, catalog_size=100)
    assert set(s) == {"ndcg@1", "ndcg@10", "ndcg@20",
                      "catalog_diversity", "lexical_diversity", "total_catalog_size"}


def test_official_raises_on_duplicate_predictions():
    bad = [SubmissionRow("s1", "u1", 1, ["g1", "g1"], "r")]
    with pytest.raises(ValueError):
        score_official(bad, [GoldRow("s1", "u1", 1, "g1")], catalog_size=100)


def test_official_raises_on_missing_prediction_for_a_gold():
    with pytest.raises((KeyError, ValueError)):
        score_official([_PRED[0]], _GT, catalog_size=100)  # missing turn 2


# ---- diagnostic primitives ----
def test_recall_at_k_single_gold():
    assert recall_at_k(["a", "b", "g"], "g", 3) == 1.0
    assert recall_at_k(["a", "b", "g"], "g", 2) == 0.0


def test_recall_monotonic_in_k():
    r = ["a", "b", "g", "d"]
    assert recall_at_k(r, "g", 2) <= recall_at_k(r, "g", 3) <= recall_at_k(r, "g", 4)


def test_hit_rank_is_one_indexed_or_none():
    assert hit_rank(["a", "g", "c"], "g") == 2
    assert hit_rank(["a", "b"], "g") is None


def test_score_diagnostic_overall_and_segments():
    rep = score_diagnostic(
        _PRED, _GT, catalog_size=100,
        k_recall=(1, 20), k_ndcg=(20,),
        segment_of={"s1": "warm"},
    )
    assert rep.overall["recall@20"] == pytest.approx(1.0)   # both golds in top-20
    assert rep.overall["recall@1"] == pytest.approx(0.5)    # only t1 gold at rank 1
    assert "warm" in rep.by_segment
    assert rep.by_segment["warm"]["recall@20"] == pytest.approx(1.0)
