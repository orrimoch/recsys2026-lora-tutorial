"""Tests for scripts/compare_blind_predictions.py."""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_blind_predictions import _load_pred, ndcg_at_k


def test_load_pred_keys_by_session_and_turn(tmp_path):
    """Pred records keyed by (session_id, turn_number) so 8-turn sessions don't collapse."""
    pred_path = tmp_path / "pred.json"
    records = [
        {"session_id": "S1", "user_id": "U1", "turn_number": 1, "predicted_track_ids": ["t1", "t2", "t3"]},
        {"session_id": "S1", "user_id": "U1", "turn_number": 2, "predicted_track_ids": ["t4", "t5"]},
        {"session_id": "S2", "user_id": "U2", "turn_number": 1, "predicted_track_ids": ["t6"]},
    ]
    pred_path.write_text(json.dumps(records))
    out = _load_pred(pred_path)
    assert set(out.keys()) == {"S1__1", "S1__2", "S2__1"}
    assert out["S1__1"] == ["t1", "t2", "t3"]
    assert out["S1__2"] == ["t4", "t5"]


def test_load_pred_uses_predicted_track_ids_not_alternates(tmp_path):
    """Real prediction key is 'predicted_track_ids' (not 'predicted_items')."""
    pred_path = tmp_path / "pred.json"
    records = [
        {"session_id": "S1", "user_id": "U1", "turn_number": 1, "predicted_track_ids": ["real_t1"]},
    ]
    pred_path.write_text(json.dumps(records))
    out = _load_pred(pred_path)
    assert out["S1__1"] == ["real_t1"]


def test_load_pred_falls_back_to_alternate_keys(tmp_path):
    """If predicted_track_ids missing, try predicted_items / tracks / track_ids."""
    pred_path = tmp_path / "pred.json"
    records = [
        {"session_id": "S1", "turn_number": 1, "tracks": ["fallback_t1"]},
    ]
    pred_path.write_text(json.dumps(records))
    out = _load_pred(pred_path)
    assert out["S1__1"] == ["fallback_t1"]


def test_load_pred_skips_records_missing_session_or_turn(tmp_path):
    """Records without session_id or turn_number are silently skipped."""
    pred_path = tmp_path / "pred.json"
    records = [
        {"session_id": "S1", "turn_number": 1, "predicted_track_ids": ["t1"]},
        {"turn_number": 1, "predicted_track_ids": ["t2"]},  # missing session_id
        {"session_id": "S3", "predicted_track_ids": ["t3"]},  # missing turn_number
    ]
    pred_path.write_text(json.dumps(records))
    out = _load_pred(pred_path)
    assert set(out.keys()) == {"S1__1"}


def test_ndcg_at_k_gold_at_rank_1():
    """Gold at position 0 → nDCG = 1.0."""
    assert ndcg_at_k(["a", "b", "c"], "a", k=20) == 1.0


def test_ndcg_at_k_gold_at_rank_3():
    """Gold at position 2 (0-indexed) → nDCG = 1/log2(4) = 0.5."""
    import math
    score = ndcg_at_k(["x", "y", "a"], "a", k=20)
    assert abs(score - 1.0 / math.log2(4)) < 1e-9


def test_ndcg_at_k_gold_not_in_top_k_returns_zero():
    """Gold beyond k → 0."""
    retrieved = [f"t{i}" for i in range(20)]
    retrieved.append("gold")  # position 20
    assert ndcg_at_k(retrieved, "gold", k=20) == 0.0


def test_load_gold_extracts_per_turn_via_canonical_logic(monkeypatch):
    """Mirror make_ground_truth.py: gold = 2nd content row of each turn's conversations.

    Verifies the gold loader correctly extracts per-(session, turn) gold tracks
    from the nested conversations structure (NOT a non-existent top-level field).
    """
    from scripts.compare_blind_predictions import _load_gold

    fake_dataset = [
        {
            "session_id": "S1",
            "conversations": [
                # Turn 1: user→music(gold)→assistant_response
                {"turn_number": 1, "role": "user", "content": "play me indie"},
                {"turn_number": 1, "role": "music", "content": "gold_track_T1"},
                {"turn_number": 1, "role": "assistant", "content": "here you go"},
                # Turn 2
                {"turn_number": 2, "role": "user", "content": "something newer"},
                {"turn_number": 2, "role": "music", "content": "gold_track_T2"},
                {"turn_number": 2, "role": "assistant", "content": "ok"},
            ],
        },
    ]

    # Monkeypatch load_dataset to return our fake dataset (no network).
    def _fake_load_dataset(name, split):
        return fake_dataset

    # Patch via sys.modules so the deferred `from datasets import load_dataset`
    # inside _load_gold picks up the fake.
    import types
    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = _fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    out = _load_gold("fake-dataset", "test")
    assert out == {"S1__1": "gold_track_T1", "S1__2": "gold_track_T2"}
