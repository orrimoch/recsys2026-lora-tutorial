"""R7 — per-segment fusion weight sweep (cheap re-fusion over cached per-channel lists)."""
from __future__ import annotations

from mcrs.eval.weight_sweep import fused_recall, segment_weight_sweep


def test_fused_recall_reflects_weights():
    # channel A finds the gold only for q0; channel B only for q1 (no shared distractor)
    per = {"A": [["g0"], []], "B": [[], ["g1"]]}
    golds = ["g0", "g1"]
    assert fused_recall(per, {"A": 1.0, "B": 0.0}, golds, ks=[1])[1] == 0.5   # only A -> only q0 hit
    assert fused_recall(per, {"A": 1.0, "B": 1.0}, golds, ks=[1])[1] == 1.0   # both -> both hit


def test_segment_weight_sweep_picks_best_weights_per_segment():
    per = {"content": [["g0"], ["x"]], "hist": [["x"], ["g1"]]}
    golds = ["g0", "g1"]
    segs = ["cold", "warm"]                                   # q0 cold (content-findable), q1 warm (hist)
    cands = [("content_only", {"content": 1.0, "hist": 0.0}),
             ("hist_only", {"content": 0.0, "hist": 1.0})]
    res = segment_weight_sweep(per, golds, segs, cands, ks=[1])
    assert res["cold"][0][0] == "content_only"               # best for cold
    assert res["warm"][0][0] == "hist_only"                  # best for warm
