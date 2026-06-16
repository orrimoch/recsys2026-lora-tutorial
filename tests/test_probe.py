"""P0 — recall-ceiling probe (pure aggregation over precomputed per-channel rankings)."""
from __future__ import annotations

import pytest

from mcrs.eval.probe import recall_ceiling

_PER_CHANNEL = {
    "bm25": [["g", "x"], ["y", "z"]],
    "cf":   [["a"],      ["g2", "g"]],
}
_GOLDS = ["g", "g2"]


def test_per_channel_recall_at_k():
    rep = recall_ceiling(_PER_CHANNEL, _GOLDS, ks=[1, 2])
    assert rep["per_channel"]["bm25"]["recall"][1] == pytest.approx(0.5)
    assert rep["per_channel"]["bm25"]["recall"][2] == pytest.approx(0.5)
    assert rep["per_channel"]["cf"]["recall"][1] == pytest.approx(0.5)


def test_fused_recall_beats_or_matches_channels():
    rep = recall_ceiling(_PER_CHANNEL, _GOLDS, ks=[1, 2], fusion_k=60)
    # fusion finds both golds by k=2 (one per channel)
    assert rep["fused"]["recall"][2] == pytest.approx(1.0)
    assert rep["fused"]["recall"][1] == pytest.approx(0.5)


def test_unique_recall_attributed_per_channel():
    rep = recall_ceiling(_PER_CHANNEL, _GOLDS, ks=[2])
    # each gold hit by exactly one channel => each contributes 0.5 unique recall
    assert rep["per_channel"]["bm25"]["unique_recall"] == pytest.approx(0.5)
    assert rep["per_channel"]["cf"]["unique_recall"] == pytest.approx(0.5)


def test_segment_breakdown():
    rep = recall_ceiling(_PER_CHANNEL, _GOLDS, ks=[2], segments=["cold", "warm"])
    assert rep["per_channel"]["bm25"]["by_segment"]["cold"][2] == pytest.approx(1.0)
    assert rep["per_channel"]["bm25"]["by_segment"]["warm"][2] == pytest.approx(0.0)
    assert rep["fused"]["by_segment"]["warm"][2] == pytest.approx(1.0)
