"""Tests for scripts/responder_eval.py — unified composite eval.

The composite eval reads a `prediction.json` (output of
run_inference_devset.py / run_inference_blindset.py), joins to gold
ground-truth from the HF dataset, and computes the per-row + summary
reward components used by the §6.3 gates and the leaderboard composite.

CPU-testable pieces:
- Per-row composite computation (compose from reward_fns).
- Aggregation across N rows into a summary dict.
- Schema validation on the input prediction.json.
- IO: writes a results JSON with per-row + summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture
def prediction_rows():
    """Mirror of run_inference_blindset.py output schema."""
    return [
        {
            "session_id": "sess_a", "user_id": "u1", "turn_number": 1,
            "predicted_track_ids": ["uuid-gold-a", "uuid-other1", "uuid-other2"],
            "predicted_response": (
                "<user_state>\nmood: reflective\n</user_state>\n"
                "<response>Holocene by Bon Iver leans into a layered "
                "arrangement, matching the reflective mood.</response>"
            ),
        },
        {
            "session_id": "sess_b", "user_id": "u2", "turn_number": 1,
            "predicted_track_ids": ["uuid-decoy1", "uuid-decoy2"],
            "predicted_response": (
                "<user_state>\nmood: dark\n</user_state>\n"
                "<response>Heart-Shaped Box features grunge tempo.</response>"
            ),
        },
    ]


@pytest.fixture
def gold_lookup():
    """(session_id, turn_number) → gold_track_id."""
    return {
        ("sess_a", 1): "uuid-gold-a",   # gold IS at rank 1 → R_retr ≈ 1.0
        ("sess_b", 1): "uuid-gold-b",   # gold NOT in pred → R_retr = 0.0
    }


@pytest.fixture
def context_lookup():
    """(session_id, turn_number) → context dict for r_rule + r_user_prof."""
    return {
        ("sess_a", 1): {
            "track_name": "Holocene", "artist_name": "Bon Iver",
            "history_text": "user: chill folk",
            "user_profile": {"country_name": "Norway", "age_group": "30s", "gender": "F"},
        },
        ("sess_b", 1): {
            "track_name": "Heart-Shaped Box", "artist_name": "Nirvana",
            "history_text": "",
            "user_profile": None,
        },
    }


# ---------------------------------------------------------------------------
# Per-row composite
# ---------------------------------------------------------------------------

class TestComposeRow:
    def test_returns_full_component_dict(self, prediction_rows, gold_lookup, context_lookup):
        from responder_eval import compose_row
        row = prediction_rows[0]
        comps = compose_row(row, gold_lookup, context_lookup)
        for k in ("r_turn", "r_retr", "r_rule", "r_format", "r_judge", "r_user_prof"):
            assert k in comps

    def test_top_1_gold_yields_full_r_retr(self, prediction_rows, gold_lookup, context_lookup):
        from responder_eval import compose_row
        comps = compose_row(prediction_rows[0], gold_lookup, context_lookup)
        # gold at rank 1 → R_retr = 1.0
        assert comps["r_retr"] == 1.0

    def test_no_gold_in_pred_yields_zero_r_retr(self, prediction_rows, gold_lookup, context_lookup):
        from responder_eval import compose_row
        comps = compose_row(prediction_rows[1], gold_lookup, context_lookup)
        assert comps["r_retr"] == 0.0

    def test_envelope_format_passes(self, prediction_rows, gold_lookup, context_lookup):
        from responder_eval import compose_row
        comps = compose_row(prediction_rows[0], gold_lookup, context_lookup)
        assert comps["r_format"] == 1.0

    def test_missing_gold_returns_r_retr_zero_no_crash(self, prediction_rows, context_lookup):
        from responder_eval import compose_row
        # Empty gold_lookup → r_retr=0 but no exception.
        comps = compose_row(prediction_rows[0], {}, context_lookup)
        assert comps["r_retr"] == 0.0

    def test_missing_context_returns_zero_terms_no_crash(self, prediction_rows, gold_lookup):
        from responder_eval import compose_row
        comps = compose_row(prediction_rows[0], gold_lookup, {})
        # No context → r_rule + r_user_prof effectively get None inputs and
        # return their "no signal" values. r_retr is unaffected.
        assert comps["r_retr"] == 1.0
        # r_rule may still be > 0 from envelope/length checks (no top1_meta).
        assert 0.0 <= comps["r_rule"] <= 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_aggregate_returns_summary_dict(self, prediction_rows, gold_lookup, context_lookup):
        from responder_eval import aggregate
        per_row = []
        from responder_eval import compose_row
        for r in prediction_rows:
            per_row.append(compose_row(r, gold_lookup, context_lookup))
        summary = aggregate(per_row)
        assert "n" in summary
        assert "mean_r_turn" in summary
        assert "mean_r_retr" in summary
        assert "mean_r_judge" in summary

    def test_aggregate_mean_matches_manual(self, prediction_rows, gold_lookup, context_lookup):
        from responder_eval import compose_row, aggregate
        per_row = [compose_row(r, gold_lookup, context_lookup) for r in prediction_rows]
        summary = aggregate(per_row)
        assert summary["n"] == 2
        manual_mean = (per_row[0]["r_turn"] + per_row[1]["r_turn"]) / 2
        assert abs(summary["mean_r_turn"] - manual_mean) < 1e-9

    def test_aggregate_empty_returns_zero_n(self):
        from responder_eval import aggregate
        summary = aggregate([])
        assert summary["n"] == 0
        # All means default to 0 — no division by zero.
        assert summary["mean_r_turn"] == 0.0


# ---------------------------------------------------------------------------
# Recall@k summary
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_recall_counts_gold_in_top_k(self, prediction_rows, gold_lookup):
        from responder_eval import recall_at_k
        # sess_a gold IS in pred at rank 1; sess_b gold NOT in pred.
        rec = recall_at_k(prediction_rows, gold_lookup, k=10)
        # 1 of 2 prediction rows has gold in top-10.
        assert rec == 0.5

    def test_recall_at_k_handles_missing_gold(self, prediction_rows):
        from responder_eval import recall_at_k
        rec = recall_at_k(prediction_rows, {}, k=10)
        # No golds known → recall is 0 (no positives to find).
        assert rec == 0.0


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestValidatePredictionSchema:
    def test_passes_on_valid_input(self, prediction_rows):
        from responder_eval import validate_prediction_schema
        # Should not raise.
        validate_prediction_schema(prediction_rows)

    def test_raises_on_missing_required_field(self):
        from responder_eval import validate_prediction_schema
        bad = [{"session_id": "x", "predicted_track_ids": [], "predicted_response": "y"}]
        with pytest.raises(ValueError, match="missing required field"):
            validate_prediction_schema(bad)

    def test_raises_on_non_list_input(self):
        from responder_eval import validate_prediction_schema
        with pytest.raises(ValueError, match="list"):
            validate_prediction_schema({"not": "a list"})


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class TestMainCli:
    def test_main_writes_results_json(self, prediction_rows, gold_lookup, context_lookup, tmp_path):
        import responder_eval as mod
        # Build a synthetic prediction.json + a synthetic gold/context json
        # (the CLI reads gold from a parquet by default; for tests we go
        # through a JSON path the script supports for offline use).
        pred_path = tmp_path / "prediction.json"
        gold_path = tmp_path / "gold.json"
        ctx_path = tmp_path / "context.json"
        results_path = tmp_path / "results.json"

        with pred_path.open("w") as f:
            json.dump(prediction_rows, f)
        with gold_path.open("w") as f:
            json.dump([{"session_id": s, "turn_number": t, "gold_track_id": g}
                       for (s, t), g in gold_lookup.items()], f)
        with ctx_path.open("w") as f:
            ctx_rows = []
            for (s, t), c in context_lookup.items():
                ctx_rows.append({
                    "session_id": s, "turn_number": t,
                    **c,
                })
            json.dump(ctx_rows, f)

        rc = mod.main([
            "--predictions", str(pred_path),
            "--gold-json", str(gold_path),
            "--context-json", str(ctx_path),
            "--out", str(results_path),
        ])
        assert rc == 0
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert "summary" in data
        assert "per_row" in data
        assert data["summary"]["n"] == 2

    def test_main_returns_nonzero_on_missing_predictions(self, tmp_path):
        import responder_eval as mod
        rc = mod.main([
            "--predictions", str(tmp_path / "nope.json"),
            "--out", str(tmp_path / "out.json"),
        ])
        assert rc == 1
