"""Tests for scripts/train_distilled_judge.py.

The distilled judge takes (context, response) text pairs and predicts a
Gemini-aligned scalar score in [0, 1]. It replaces `r_judge_stub` in the
W6 GRPO reward closure (Option B refactor).

Training data: data/reward_calibration_anchors.parquet (307,600 rows, of
which ~210,000 carry a `judge_anchor` ∈ {1.0, 5.0}; source B rows lack a
target and are dropped).

The CPU-testable pieces are: input pair construction, target normalization,
session-level train/val split, NaN filtering, and the data prep CLI.
The actual training loop is GPU-only and tested by integration runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture
def anchor_df():
    """Mirror of data/reward_calibration_anchors.parquet schema (subset).

    Source A: GPA-derived (judge_anchor ∈ {1.0, 5.0}).
    Source B: distractors with NaN judge_anchor — must be filtered out.
    Source D: synthetic perturbations (judge_anchor=1.0) of POS gold.
    """
    rows = [
        # POS row from train_gpa
        {
            "source": "A", "exp_id": "train_gpa", "session_id": "s_a1",
            "user_id": "u1", "turn_number": 1,
            "user_query": "Looking for chill folk",
            "gold_track_id": "g1",
            "predicted_track_ids": ["g1", "x", "y"],
            "predicted_response": "Holocene by Bon Iver, layered + slow.",
            "track_name": "Holocene", "artist_name": "Bon Iver",
            "country_name": "Norway", "age_group": "30s", "gender": "F",
            "history_text": "user: chill mood",
            "goal_listener": "discover folk",
            "judge_anchor": 5.0, "perturb_variant": None,
        },
        # NEG row (same session)
        {
            "source": "A", "exp_id": "train_gpa", "session_id": "s_a1",
            "user_id": "u1", "turn_number": 2,
            "user_query": "More upbeat",
            "gold_track_id": "g2",
            "predicted_track_ids": ["wrong1", "wrong2", "wrong3"],
            "predicted_response": "Some banal answer.",
            "track_name": "Other", "artist_name": "Other Artist",
            "country_name": "Norway", "age_group": "30s", "gender": "F",
            "history_text": "user: chill mood\nassistant: Holocene",
            "goal_listener": "discover folk",
            "judge_anchor": 1.0, "perturb_variant": None,
        },
        # Source-B row (no anchor) — must be filtered
        {
            "source": "B", "exp_id": "distractor", "session_id": "s_b1",
            "user_id": "u9", "turn_number": 1,
            "user_query": "Distractor query",
            "gold_track_id": "?",
            "predicted_track_ids": [],
            "predicted_response": "no anchor here",
            "track_name": "?", "artist_name": "?",
            "country_name": "?", "age_group": "?", "gender": "?",
            "history_text": "",
            "goal_listener": "?",
            "judge_anchor": float("nan"), "perturb_variant": None,
        },
        # Synthetic perturbation (drop_track_name) — anchor=1.0
        {
            "source": "D", "exp_id": "perturb", "session_id": "s_a1",
            "user_id": "u1", "turn_number": 1,
            "user_query": "Looking for chill folk",
            "gold_track_id": "g1",
            "predicted_track_ids": ["g1"],
            "predicted_response": "this song fits the mood.",
            "track_name": "Holocene", "artist_name": "Bon Iver",
            "country_name": "Norway", "age_group": "30s", "gender": "F",
            "history_text": "user: chill mood",
            "goal_listener": "discover folk",
            "judge_anchor": 1.0, "perturb_variant": "drop_track_name",
        },
        # Another session's POS for split testing
        {
            "source": "A", "exp_id": "train_gpa", "session_id": "s_a2",
            "user_id": "u2", "turn_number": 1,
            "user_query": "90s grunge",
            "gold_track_id": "g3",
            "predicted_track_ids": ["g3"],
            "predicted_response": "Heart-Shaped Box has the right tempo.",
            "track_name": "Heart-Shaped Box", "artist_name": "Nirvana",
            "country_name": "USA", "age_group": "20s", "gender": "M",
            "history_text": "",
            "goal_listener": "rediscover 90s",
            "judge_anchor": 5.0, "perturb_variant": None,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Anchor normalization
# ---------------------------------------------------------------------------

class TestNormalizeAnchor:
    def test_neg_maps_to_zero(self):
        from train_distilled_judge import normalize_anchor
        assert normalize_anchor(1.0) == 0.0

    def test_pos_maps_to_one(self):
        from train_distilled_judge import normalize_anchor
        assert normalize_anchor(5.0) == 1.0

    def test_mid_value_linear(self):
        from train_distilled_judge import normalize_anchor
        # 3.0 → (3-1)/4 = 0.5 (judge_anchor is binary today, but the
        # normalization must remain linear so a future Gemini-API anchor
        # lands in the correct [0,1] band).
        assert abs(normalize_anchor(3.0) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Filtering — drop NaN-anchor rows (source B)
# ---------------------------------------------------------------------------

class TestFilterAnchors:
    def test_drops_source_b_no_anchor(self, anchor_df):
        from train_distilled_judge import filter_anchors
        out = filter_anchors(anchor_df)
        assert len(out) == 4  # was 5; source B row removed
        assert not out["judge_anchor"].isna().any()

    def test_preserves_source_a_d(self, anchor_df):
        from train_distilled_judge import filter_anchors
        out = filter_anchors(anchor_df)
        assert set(out["source"].unique()) == {"A", "D"}

    def test_returns_copy_not_view(self, anchor_df):
        # Mutating the result must NOT corrupt the original parquet roundtrip.
        from train_distilled_judge import filter_anchors
        out = filter_anchors(anchor_df)
        out["judge_anchor"] = -99.0
        assert anchor_df["judge_anchor"].iloc[0] == 5.0


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

class TestBuildPair:
    def test_returns_two_strings(self, anchor_df):
        from train_distilled_judge import build_pair
        row = anchor_df.iloc[0]
        ctx, resp = build_pair(row)
        assert isinstance(ctx, str) and isinstance(resp, str)
        assert ctx and resp

    def test_context_includes_user_query_and_track(self, anchor_df):
        from train_distilled_judge import build_pair
        row = anchor_df.iloc[0]
        ctx, _ = build_pair(row)
        assert "chill folk" in ctx
        assert "Holocene" in ctx
        assert "Bon Iver" in ctx

    def test_context_includes_personalization_signal(self, anchor_df):
        """Personalization is one of two Gemini judge axes — country/age/gender
        must be in the context so the cross-encoder can learn the signal."""
        from train_distilled_judge import build_pair
        row = anchor_df.iloc[0]
        ctx, _ = build_pair(row)
        assert "Norway" in ctx
        assert "30s" in ctx

    def test_response_is_predicted_response(self, anchor_df):
        from train_distilled_judge import build_pair
        row = anchor_df.iloc[0]
        _, resp = build_pair(row)
        assert resp == row["predicted_response"]

    def test_handles_missing_fields_gracefully(self):
        """Some real rows have None/empty profile fields. Pair builder must
        not raise — the cross-encoder will tokenize the empty placeholder."""
        from train_distilled_judge import build_pair
        row = pd.Series({
            "user_query": "test", "history_text": None,
            "track_name": None, "artist_name": "Artist",
            "country_name": None, "age_group": None, "gender": None,
            "predicted_response": "hi",
        })
        ctx, resp = build_pair(row)
        assert "Artist" in ctx
        assert resp == "hi"


# ---------------------------------------------------------------------------
# Session-level train/val split
# ---------------------------------------------------------------------------

class TestSplitTrainVal:
    def test_no_session_leakage(self, anchor_df):
        """Sessions in val MUST NOT appear in train. This is the
        feedback_no_data_leakage rule applied to the judge calibration step."""
        from train_distilled_judge import filter_anchors, split_train_val
        df = filter_anchors(anchor_df)
        train_df, val_df = split_train_val(df, val_frac=0.5, seed=42)
        train_sess = set(train_df["session_id"])
        val_sess = set(val_df["session_id"])
        assert not (train_sess & val_sess), \
            f"leakage: {train_sess & val_sess}"

    def test_split_seed_determinism(self, anchor_df):
        from train_distilled_judge import filter_anchors, split_train_val
        df = filter_anchors(anchor_df)
        t1, v1 = split_train_val(df, val_frac=0.5, seed=42)
        t2, v2 = split_train_val(df, val_frac=0.5, seed=42)
        pd.testing.assert_frame_equal(
            t1.sort_values(["session_id", "turn_number"]).reset_index(drop=True),
            t2.sort_values(["session_id", "turn_number"]).reset_index(drop=True),
        )

    def test_split_covers_all_rows(self, anchor_df):
        from train_distilled_judge import filter_anchors, split_train_val
        df = filter_anchors(anchor_df)
        train_df, val_df = split_train_val(df, val_frac=0.5, seed=42)
        assert len(train_df) + len(val_df) == len(df)


# ---------------------------------------------------------------------------
# CLI smoke (data-prep step only — training itself is GPU)
# ---------------------------------------------------------------------------

class TestPrepareCli:
    def test_prepare_writes_train_val_jsonl(self, anchor_df, tmp_path):
        """`--prepare` mode emits two JSONL files (train.jsonl + val.jsonl)
        with rows {context, response, label}. The actual GPU training
        consumes those files via a streaming dataset — keeps the script
        memory-light on Colab."""
        import train_distilled_judge as mod
        anchor_path = tmp_path / "anchors.parquet"
        anchor_df.to_parquet(anchor_path, index=False)
        out_dir = tmp_path / "out"
        rc = mod.main([
            "--mode", "prepare",
            "--anchors", str(anchor_path),
            "--out-dir", str(out_dir),
            "--val-frac", "0.5",
            "--seed", "42",
        ])
        assert rc == 0
        train_path = out_dir / "train.jsonl"
        val_path = out_dir / "val.jsonl"
        assert train_path.exists() and val_path.exists()
        # Schema sanity on the train file.
        first = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
        assert {"context", "response", "label"} <= set(first.keys())
        assert 0.0 <= first["label"] <= 1.0
        # No source-B rows leaked through (filter_anchors must have run).
        for line in train_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["context"]  # non-empty
