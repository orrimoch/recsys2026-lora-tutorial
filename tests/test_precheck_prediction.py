"""Tests for scripts/precheck_prediction.py."""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.precheck_prediction import precheck


def test_precheck_passes_clean_prediction(tmp_path):
    """All 80 entries valid → ok=True, no errors."""
    catalog = {"t1", "t2", "t3"}
    records = [
        {"session_id": f"S{i}", "user_id": f"U{i}", "turn_number": 1,
         "predicted_track_ids": ["t1", "t2", "t3"]}
        for i in range(80)
    ]
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(records))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert result["ok"]
    assert result["errors"] == []


def test_precheck_flags_wrong_entry_count(tmp_path):
    """Less than 80 entries → ok=False, count error."""
    catalog = {"t1"}
    p = tmp_path / "pred.json"
    p.write_text(json.dumps([
        {"session_id": "S0", "user_id": "U0", "turn_number": 1, "predicted_track_ids": ["t1"]}
    ]))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert not result["ok"]
    assert any("expected 80 entries, got 1" in e for e in result["errors"])


def test_precheck_flags_missing_required_fields(tmp_path):
    """Record missing 'predicted_track_ids' → ok=False with field error."""
    catalog = {"t1"}
    records = [
        {"session_id": f"S{i}", "user_id": f"U{i}", "turn_number": 1,
         "predicted_track_ids": ["t1"]}
        for i in range(79)
    ]
    records.append({"session_id": "S79", "user_id": "U79", "turn_number": 1})  # missing predicted_track_ids
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(records))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert not result["ok"]
    assert any("predicted_track_ids" in e for e in result["errors"])


def test_precheck_flags_hallucinated_track_ids(tmp_path):
    """Track IDs not in catalog → ok=False with hallucination error."""
    catalog = {"t1", "t2"}
    records = [
        {"session_id": f"S{i}", "user_id": f"U{i}", "turn_number": 1,
         "predicted_track_ids": ["t1", "HALLUCINATED_TRACK", "t2"]}
        for i in range(80)
    ]
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(records))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert not result["ok"]
    assert any("HALLUCINATED_TRACK" in e for e in result["errors"])


def test_precheck_flags_duplicate_tracks_per_record(tmp_path):
    """Same track id appears twice in a record's predicted_track_ids → ok=False."""
    catalog = {"t1", "t2"}
    records = [
        {"session_id": f"S{i}", "user_id": f"U{i}", "turn_number": 1,
         "predicted_track_ids": ["t1", "t2", "t1"]}  # duplicate t1
        for i in range(80)
    ]
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(records))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert not result["ok"]
    assert any("duplicate" in e.lower() for e in result["errors"])
