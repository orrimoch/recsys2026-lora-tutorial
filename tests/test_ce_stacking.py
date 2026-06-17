"""Leak-safe CE-score stacking wiring (K3 → K2 feature).

The frozen pretrained cross-encoder never saw gold labels, so its (query, doc) score is a fixed
function — stacking it into K2 as a feature is leak-free with NO OOF (neural.py docstring). These
cover the two wiring helpers: the FeatureBuilder adapter and the batch lookup builder.
"""
from __future__ import annotations

from types import SimpleNamespace

from mcrs.rerank.neural import build_ce_score_lookup, make_ce_feature_fn


def _ctx(sid, turn):
    return SimpleNamespace(session_id=sid, turn_number=turn)


# ----- make_ce_feature_fn (FeatureBuilder adapter) -----
def test_ce_feature_fn_returns_lookup_value():
    fn = make_ce_feature_fn({("s1", 2, "tA"): 0.7})
    assert fn(_ctx("s1", 2), "tA") == 0.7


def test_ce_feature_fn_default_for_unscored_candidate():
    # Candidates outside the CE-scored top-K get the sentinel — SAME value at train and serve.
    fn = make_ce_feature_fn({("s1", 2, "tA"): 0.7}, default=-1.0)
    assert fn(_ctx("s1", 2), "tZ") == -1.0
    assert fn(_ctx("s9", 9), "tA") == -1.0          # different turn -> not scored


# ----- build_ce_score_lookup -----
class _FakeScorer:
    """Stands in for NeuralReranker.score(ctx, pool) -> {tid: raw_ce_score}."""
    def __init__(self, mapping):
        self.mapping = mapping

    def score(self, ctx, pool):
        return self.mapping.get((ctx.session_id, ctx.turn_number), {})


def test_build_lookup_normalizes_within_pool():
    turns = [_ctx("s1", 1)]
    scorer = _FakeScorer({("s1", 1): {"a": 0.0, "b": 5.0, "c": 10.0}})
    lk = build_ce_score_lookup(turns, [None], scorer, normalize=True)
    assert lk[("s1", 1, "a")] == 0.0
    assert lk[("s1", 1, "b")] == 0.5
    assert lk[("s1", 1, "c")] == 1.0


def test_build_lookup_raw_when_not_normalized():
    turns = [_ctx("s1", 1)]
    scorer = _FakeScorer({("s1", 1): {"a": 2.0, "b": 5.0}})
    lk = build_ce_score_lookup(turns, [None], scorer, normalize=False)
    assert lk[("s1", 1, "b")] == 5.0


def test_build_lookup_skips_empty_pools_and_keys_per_turn():
    turns = [_ctx("s1", 1), _ctx("s2", 1)]
    scorer = _FakeScorer({("s1", 1): {"a": 1.0}})   # s2 returns {} -> skipped
    lk = build_ce_score_lookup(turns, [None, None], scorer, normalize=True)
    assert set(lk) == {("s1", 1, "a")}
