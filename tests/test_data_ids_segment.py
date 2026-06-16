"""F1 — id normalization + cold/warm segment (pure helpers, no data)."""
from __future__ import annotations

from mcrs.data.ids import canonical_track_id, canonical_track_ids
from mcrs.data.segment import segment_for


def test_canonical_passes_through_plain_uuid():
    tid = "97f5eeec-1ec7-4bb9-93e9-a948ee7466fc"
    assert canonical_track_id(tid) == tid


def test_canonical_strips_prefix_and_whitespace():
    assert canonical_track_id("  track_id: abc-123 ") == "abc-123"


def test_canonical_is_idempotent():
    raw = " track_id: abc-123 "
    once = canonical_track_id(raw)
    assert canonical_track_id(once) == once


def test_canonical_track_ids_maps_list():
    assert canonical_track_ids(["track_id: a", " b "]) == ["a", "b"]


def test_segment_cold_when_history_at_or_below_threshold():
    assert segment_for([], cold_threshold=1) == "cold"
    assert segment_for(["t1"], cold_threshold=1) == "cold"


def test_segment_warm_when_history_above_threshold():
    assert segment_for(["t1", "t2"], cold_threshold=1) == "warm"
