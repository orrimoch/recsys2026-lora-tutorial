"""Turn-stratified nDCG@20 reporting (Tier-0 #1).

Blind-A is 80 single-turn / zero-history sessions, so the turn-1 stratum is the
only honest proxy for Blind. The flat dev mean over all 8 turns over-credits the
session channels that are dead on Blind. ndcg_by_turn() exposes the per-turn
breakdown so go/no-go can gate on turn-1, not the average.

ndcg_at_k matches the official evaluator (music-crs-evaluator get_ndcg) for the
single-binary-gold case: DCG = sum rel/log2(rank+1), IDCG = 1 (one gold), so the
score is 1/log2(rank_1indexed+1) when the gold is in the top-k, else 0.
"""
import math

from mcrs.eval_ndcg import ndcg_at_k, ndcg_by_turn, recall_by_turn


def test_ndcg_gold_at_rank_one_is_one():
    assert ndcg_at_k("g", ["g", "a", "b"], k=20) == 1.0


def test_ndcg_gold_at_rank_two():
    assert ndcg_at_k("g", ["a", "g", "b"], k=20) == 1.0 / math.log2(3)


def test_ndcg_gold_absent_is_zero():
    assert ndcg_at_k("g", ["a", "b", "c"], k=20) == 0.0


def test_ndcg_gold_beyond_k_is_zero():
    ranked = ["a"] * 20 + ["g"]  # gold at 0-indexed pos 20 -> rank 21, beyond k=20
    assert ndcg_at_k("g", ranked, k=20) == 0.0


def test_ndcg_by_turn_overall_and_per_turn():
    # turn 1: one query, gold at rank1 -> 1.0 ; turn 2: one query, gold absent -> 0.0
    ranked = [["g"], ["x"]]
    golds = ["g", "g"]
    turns = [1, 2]
    rep = ndcg_by_turn(ranked, golds, turns, k=20)
    assert rep["per_turn"][1]["ndcg"] == 1.0
    assert rep["per_turn"][1]["n"] == 1
    assert rep["per_turn"][2]["ndcg"] == 0.0
    # overall flat mean over the 2 rows
    assert rep["overall"] == 0.5
    # turn-1 (cold / Blind proxy) surfaced explicitly
    assert rep["turn1"] == 1.0


def test_ndcg_by_turn_macro_averages_within_turn():
    # turn 1 has two queries (1.0 and 0.0) -> per-turn mean 0.5
    ranked = [["g"], ["x"], ["g"]]
    golds = ["g", "g", "g"]
    turns = [1, 1, 2]
    rep = ndcg_by_turn(ranked, golds, turns, k=20)
    assert rep["per_turn"][1]["ndcg"] == 0.5
    assert rep["per_turn"][1]["n"] == 2
    assert rep["per_turn"][2]["ndcg"] == 1.0


def test_recall_by_turn_hit_within_k_and_turn1():
    # turn 1: gold present in top-100 -> 1.0 ; turn 2: absent -> 0.0
    ranked = [["a", "g"], ["x", "y"]]
    golds = ["g", "g"]
    turns = [1, 2]
    rep = recall_by_turn(ranked, golds, turns, k=100)
    assert rep["per_turn"][1]["recall"] == 1.0
    assert rep["per_turn"][2]["recall"] == 0.0
    assert rep["overall"] == 0.5
    assert rep["turn1"] == 1.0


def test_recall_by_turn_respects_k_cutoff():
    ranked = [["a"] * 100 + ["g"]]  # gold at index 100, beyond k=100
    rep = recall_by_turn(ranked, ["g"], [1], k=100)
    assert rep["per_turn"][1]["recall"] == 0.0
