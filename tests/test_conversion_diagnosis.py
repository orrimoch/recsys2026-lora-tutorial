"""F3 — reranker conversion diagnosis (Option-0): decompose the nDCG gap into recall-bound
(gold never reached the pool) vs ranking-bound (gold in the pool but not ranked into the top-k)."""
from __future__ import annotations

import math

from mcrs.eval.diagnostics import conversion_diagnosis


def test_decomposes_into_recall_and_ranking_loss_with_verdict():
    # top_k=2, pool_k=4. ranked_ids[i] is the reranker's full ordered candidates for turn i.
    ranked = [
        ["g0", "a", "b", "c"],   # gold at rank 1 -> in top-2 (converted)
        ["a", "b", "g1", "c"],   # gold at rank 3 -> in pool, NOT in top-2 (ranking loss)
        ["a", "b", "c", "g2"],   # gold at rank 4 -> in pool, NOT in top-2 (ranking loss)
        ["a", "b", "c", "d"],    # gold absent -> not in pool (recall loss)
    ]
    golds = ["g0", "g1", "g2", "g3"]
    r = conversion_diagnosis(ranked, golds, pool_k=4, top_k=2)
    assert r["n"] == 4
    assert math.isclose(r["recall_at_pool"], 0.75)        # 3 of 4 golds reachable
    assert math.isclose(r["recall_at_top"], 0.25)         # 1 of 4 ranked into top-2
    assert math.isclose(r["conversion"], 0.25 / 0.75)     # of reachable golds, fraction converted
    assert math.isclose(r["recall_loss"], 0.25)           # unreachable -> ColBERT/recall's job
    assert math.isclose(r["ranking_loss"], 0.50)          # reachable-but-mis-ranked -> K2/CE's job
    assert r["verdict"] == "ranking-bound"                # ranking_loss (0.50) > recall_loss (0.25)


def test_recall_bound_verdict_when_pool_misses_dominate():
    ranked = [
        ["g0", "a"],     # converted
        ["a", "b"],      # gold absent -> recall loss
        ["a", "b"],      # gold absent -> recall loss
    ]
    golds = ["g0", "g1", "g2"]
    r = conversion_diagnosis(ranked, golds, pool_k=2, top_k=2)
    assert math.isclose(r["recall_loss"], 2 / 3)
    assert math.isclose(r["ranking_loss"], 0.0)
    assert r["verdict"] == "recall-bound"                 # in-pool golds all convert; gap is recall


def test_conversion_is_zero_when_no_gold_reaches_pool_no_div_by_zero():
    r = conversion_diagnosis([["a", "b"], ["c", "d"]], ["g0", "g1"], pool_k=2, top_k=1)
    assert r["recall_at_pool"] == 0.0
    assert r["conversion"] == 0.0                          # guarded, not NaN/ZeroDivisionError


def test_by_segment_decomposition():
    ranked = [
        ["g0", "a", "b", "c"],   # warm: converted
        ["a", "b", "g1", "c"],   # warm: ranking loss
        ["a", "b", "c", "d"],    # cold: recall loss
        ["g3", "a", "b", "c"],   # cold: converted
    ]
    golds = ["g0", "g1", "g2", "g3"]
    segs = ["warm", "warm", "cold", "cold"]
    r = conversion_diagnosis(ranked, golds, pool_k=4, top_k=2, segments=segs)
    assert math.isclose(r["by_segment"]["warm"]["recall_at_pool"], 1.0)   # both warm golds reachable
    assert math.isclose(r["by_segment"]["warm"]["ranking_loss"], 0.5)     # one warm mis-ranked
    assert math.isclose(r["by_segment"]["cold"]["recall_at_pool"], 0.5)   # one cold gold not in pool
    assert math.isclose(r["by_segment"]["cold"]["recall_at_top"], 0.5)
