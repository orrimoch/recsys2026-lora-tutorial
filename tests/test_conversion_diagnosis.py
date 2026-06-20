"""F3 — reranker conversion diagnosis (Option-0): decompose the nDCG gap into recall-bound
(gold never reached the pool) vs ranking-bound (gold in the pool but not ranked into the top-k)."""
from __future__ import annotations

import math

from mcrs.eval.diagnostics import (
    conversion_diagnosis,
    feature_rescue_analysis,
    gold_rank_distribution,
)


def test_feature_rescue_analysis_counts_rescues_and_demotions():
    # Does a candidate feature's OWN ranking rescue the base reranker's just-missed golds into its
    # top-k? top_k=2, band=(2,5): a gold the base ranks at 3-5 that the feature ranks <=2 is rescuable.
    base = [
        ["g0", "a", "b"],                       # g0 base rank 1 -> converted (base already has it)
        ["a", "b", "c", "d", "g1"],             # g1 base rank 5 -> in band (2,5]
        ["a", "b", "c", "d", "g2"],             # g2 base rank 5 -> in band (2,5]
        ["a", "b"],                             # g3 absent     -> not_in_pool
    ]
    feat = [
        ["g0", "a", "b"],                       # feature keeps g0 at rank 1
        ["g1", "a", "b", "c", "d"],             # feature ranks g1 at 1 (<=top_k) -> RESCUABLE
        ["a", "b", "c", "g2", "d"],             # feature ranks g2 at 4 (>top_k)  -> mid_missed
        ["a", "b"],
    ]
    golds = ["g0", "g1", "g2", "g3"]
    r = feature_rescue_analysis(base, feat, golds, top_k=2, band=(2, 5))
    assert r["counts"]["converted"] == 1
    assert r["counts"]["rescuable"] == 1        # g1
    assert r["counts"]["mid_missed"] == 1       # g2
    assert r["counts"]["not_in_pool"] == 1
    assert r["rescue_rate"] == 0.5              # 1 rescued of 2 band golds
    assert r["demotion_risk"] == 0.0            # no base-top-k gold pushed out by the feature


def test_feature_rescue_analysis_flags_demotion_risk():
    base = [["g0", "a", "b", "c"]]              # g0 base rank 1 -> converted
    feat = [["a", "b", "c", "g0"]]              # feature drops g0 to rank 4 (>top_k) -> demotion risk
    r = feature_rescue_analysis(base, feat, ["g0"], top_k=2, band=(2, 5))
    assert r["counts"]["converted"] == 1
    assert r["demotion_risk"] == 1.0           # the one converted gold would be demoted by the feature


def test_feature_rescue_analysis_raises_on_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        feature_rescue_analysis([["g"]], ["g"], ["g", "h"])


def test_gold_rank_distribution_buckets_by_hit_rank():
    # Where does each gold sit in the reranked list? Tells how far down a fix must reach to convert
    # the in-pool-but-not-top-20 golds. Buckets are half-open bands on the 1-based hit rank.
    ranked = [
        ["g0"] + [f"a{i}" for i in range(40)],     # gold rank 1   -> "1-20"
        [f"a{i}" for i in range(30)] + ["g1"],     # gold rank 31  -> "21-50"
        [f"a{i}" for i in range(80)] + ["g2"],     # gold rank 81  -> "51-100"
        [f"a{i}" for i in range(40)],              # gold absent   -> "not_in_pool"
    ]
    golds = ["g0", "g1", "g2", "g3"]
    d = gold_rank_distribution(ranked, golds, buckets=(20, 50, 100, 200, 500))
    assert d["n"] == 4
    assert d["labels"][0] == "1-20" and d["labels"][-1] == "not_in_pool"
    assert d["buckets"]["1-20"] == 1
    assert d["buckets"]["21-50"] == 1
    assert d["buckets"]["51-100"] == 1
    assert d["buckets"]["101-200"] == 0
    assert d["buckets"]["not_in_pool"] == 1
    assert sum(d["buckets"].values()) == 4         # every gold lands in exactly one band


def test_gold_rank_distribution_raises_on_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        gold_rank_distribution([["g0"], ["g1"]], ["g0"])


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


def test_raises_when_top_k_exceeds_pool_k():
    # top_k > pool_k is a misuse: recall@top would exceed recall@pool -> a negative, meaningless
    # ranking_loss. Fail loud instead of silently returning garbage.
    import pytest
    with pytest.raises(ValueError):
        conversion_diagnosis([["g"]], ["g"], pool_k=10, top_k=20)


def test_raises_on_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        conversion_diagnosis([["g0"], ["g1"]], ["g0"], pool_k=5, top_k=2)          # golds too short
    with pytest.raises(ValueError):
        conversion_diagnosis([["g0"], ["g1"]], ["g0", "g1"], pool_k=5, top_k=2,
                             segments=["warm"])                                      # segments too short


def test_empty_input_is_balanced_without_crashing():
    r = conversion_diagnosis([], [], pool_k=500, top_k=20)
    assert r["n"] == 0 and r["recall_at_pool"] == 0.0 and r["conversion"] == 0.0
    assert r["verdict"] == "balanced"                  # no headroom either way; must not divide-by-zero


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
