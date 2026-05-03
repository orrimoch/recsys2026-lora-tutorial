"""Tests for scripts/build_trl_datasets.py.

Verifies the TRL schema contracts (KTO/DPO/GRPO column names + dtypes) on
synthetic input — no dependency on the ~60k-row reward_train.parquet. Catches
the failure mode the huggingface-llm-trainer skill flags as 50%+ of training
failures: format drift between our reward dataset and TRL's expected schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_trl_datasets import (  # noqa: E402
    DPO_REQUIRED,
    GRPO_REQUIRED,
    KTO_REQUIRED,
    _validate_schema,
    build_dpo_random,
    build_grpo_prompts,
    build_kto,
)


# ---------------------------------------------------------------------------
# Synthetic input — small parquet that exercises both POS and NEG GPA labels.
# ---------------------------------------------------------------------------

@pytest.fixture
def reward_df():
    """Mirror of scripts/build_reward_dataset.py output schema."""
    rows = []
    for i in range(20):
        label = 1 if i % 2 == 0 else 0
        rows.append({
            "text_a": f"User query: q{i}\nGoal: g{i}\nRecommended: track{i}",
            "text_b": f"response number {i} with some words inside",
            "label": label,
            "session_id": f"sess_{i // 4}",  # 5 sessions × 4 turns
            "user_id": f"user_{i // 4}",
            "turn_number": (i % 4) + 1,
            "goal_category": "discovery",
            "split": "train" if i < 16 else "val",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_passes_on_correct_kto(self):
        df = pd.DataFrame({
            "prompt": ["a", "b"],
            "completion": ["c", "d"],
            "label": [True, False],
        })
        # Should not raise.
        _validate_schema(df, KTO_REQUIRED, "kto")

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"prompt": ["a"], "completion": ["c"]})
        with pytest.raises(ValueError, match="missing required TRL columns"):
            _validate_schema(df, KTO_REQUIRED, "kto")

    def test_raises_on_empty(self):
        df = pd.DataFrame({"prompt": [], "completion": [], "label": []})
        with pytest.raises(ValueError, match="empty"):
            _validate_schema(df, KTO_REQUIRED, "kto")

    def test_raises_on_int_label_not_bool(self):
        df = pd.DataFrame({
            "prompt": ["a"],
            "completion": ["c"],
            "label": [1],  # int, not bool — should fail
        })
        with pytest.raises(ValueError, match="dtype expected bool"):
            _validate_schema(df, KTO_REQUIRED, "kto")


# ---------------------------------------------------------------------------
# build_kto
# ---------------------------------------------------------------------------

class TestBuildKto:
    def test_columns_correct(self, reward_df):
        out = build_kto(reward_df)
        assert set(KTO_REQUIRED) <= set(out.columns)

    def test_label_is_boolean_dtype(self, reward_df):
        import numpy as np
        out = build_kto(reward_df)
        # First-row spot check — numpy bool scalars (np.bool_) are TRL-compatible.
        assert isinstance(out["label"].iloc[0], (bool, np.bool_))
        # Whole-column dtype must be bool.
        assert out["label"].dtype == bool

    def test_label_pos_neg_count_matches_input(self, reward_df):
        out = build_kto(reward_df)
        n_true = int(out["label"].sum())
        n_false = int((~out["label"]).sum())
        assert n_true == 10
        assert n_false == 10

    def test_one_to_one_row_count(self, reward_df):
        out = build_kto(reward_df)
        assert len(out) == len(reward_df)

    def test_carries_split_column(self, reward_df):
        out = build_kto(reward_df)
        assert "split" in out.columns
        assert (out["split"] == "train").sum() == 16
        assert (out["split"] == "val").sum() == 4

    def test_text_a_maps_to_prompt(self, reward_df):
        out = build_kto(reward_df)
        assert out["prompt"].iloc[0] == reward_df["text_a"].iloc[0]
        assert out["completion"].iloc[0] == reward_df["text_b"].iloc[0]

    def test_raises_on_missing_input_columns(self):
        bad = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match=r"text_a, text_b, label"):
            build_kto(bad)


# ---------------------------------------------------------------------------
# build_dpo_random
# ---------------------------------------------------------------------------

class TestBuildDpoRandom:
    def test_columns_correct(self, reward_df):
        out = build_dpo_random(reward_df, seed=42)
        assert set(DPO_REQUIRED) <= set(out.columns)

    def test_n_pairs_is_min_pos_neg(self, reward_df):
        out = build_dpo_random(reward_df, seed=42)
        # 10 POS + 10 NEG → 10 pairs
        assert len(out) == 10

    def test_chosen_and_rejected_distinct(self, reward_df):
        out = build_dpo_random(reward_df, seed=42)
        assert (out["chosen"] != out["rejected"]).all()

    def test_chosen_drawn_from_pos(self, reward_df):
        out = build_dpo_random(reward_df, seed=42)
        pos_text_b = set(reward_df[reward_df["label"] == 1]["text_b"])
        for chosen in out["chosen"]:
            assert chosen in pos_text_b

    def test_rejected_drawn_from_neg(self, reward_df):
        out = build_dpo_random(reward_df, seed=42)
        neg_text_b = set(reward_df[reward_df["label"] == 0]["text_b"])
        for rejected in out["rejected"]:
            assert rejected in neg_text_b

    def test_seed_determinism(self, reward_df):
        out1 = build_dpo_random(reward_df, seed=42)
        out2 = build_dpo_random(reward_df, seed=42)
        pd.testing.assert_frame_equal(out1, out2)

    def test_different_seeds_produce_different_pairings(self, reward_df):
        out1 = build_dpo_random(reward_df, seed=42)
        out2 = build_dpo_random(reward_df, seed=99)
        # At least one of the chosen↔rejected mappings should differ.
        assert not out1.equals(out2)

    def test_raises_when_no_negatives(self):
        bad = pd.DataFrame({
            "text_a": ["a"] * 5,
            "text_b": ["b"] * 5,
            "label": [1] * 5,
        })
        with pytest.raises(ValueError, match="POS.*NEG"):
            build_dpo_random(bad)

    def test_raises_when_no_positives(self):
        bad = pd.DataFrame({
            "text_a": ["a"] * 5,
            "text_b": ["b"] * 5,
            "label": [0] * 5,
        })
        with pytest.raises(ValueError, match="POS.*NEG"):
            build_dpo_random(bad)

    def test_no_cross_split_leakage(self, reward_df):
        """A train-tagged DPO pair must NEVER source chosen/rejected text from
        a val-split row. This is the no-leakage invariant — if it ever breaks,
        a val response string is being shown to the model at training time.
        """
        out = build_dpo_random(reward_df, seed=42)
        assert "split" in out.columns, "DPO output must carry split when input has one."
        val_text_b = set(reward_df[reward_df["split"] == "val"]["text_b"].astype(str))
        train_pairs = out[out["split"] == "train"]
        leaked_chosen = set(train_pairs["chosen"]) & val_text_b
        leaked_rejected = set(train_pairs["rejected"]) & val_text_b
        assert not leaked_chosen, f"val text_b leaked into train chosen: {leaked_chosen!r}"
        assert not leaked_rejected, f"val text_b leaked into train rejected: {leaked_rejected!r}"

    def test_within_split_pairing_preserves_total_pair_count(self, reward_df):
        """Sanity: for the fixture, train has 8 POS + 8 NEG and val has 2 POS +
        2 NEG. Within-split pairing should produce 8 + 2 = 10 pairs total — same
        as the old global-pool implementation. Catches the case where a fix
        accidentally drops one side of a small split.
        """
        out = build_dpo_random(reward_df, seed=42)
        assert len(out) == 10
        assert (out["split"] == "train").sum() == 8
        assert (out["split"] == "val").sum() == 2

    def test_skips_one_sided_split_with_warning(self, capsys):
        """A split with only POS (or only NEG) should be skipped with a warning,
        not abort the whole build — provided at least one other split is usable.
        """
        df = pd.DataFrame({
            "text_a": [f"q{i}" for i in range(8)],
            "text_b": [f"r{i}" for i in range(8)],
            "label":  [1, 0, 1, 0, 1, 0, 1, 1],  # val (last 2) is POS-only
            "split":  ["train"] * 6 + ["val"] * 2,
        })
        out = build_dpo_random(df, seed=42)
        captured = capsys.readouterr()
        assert "skipping split='val'" in captured.err
        assert (out["split"] == "train").any()
        assert not (out["split"] == "val").any()


# ---------------------------------------------------------------------------
# build_grpo_prompts
# ---------------------------------------------------------------------------

class TestBuildGrpoPrompts:
    def test_columns_correct(self, reward_df):
        out = build_grpo_prompts(reward_df)
        assert set(GRPO_REQUIRED) <= set(out.columns)

    def test_deduplicates_prompts(self):
        df = pd.DataFrame({
            "text_a": ["q1", "q1", "q2", "q3", "q3", "q3"],
            "text_b": ["a", "b", "c", "d", "e", "f"],
            "label": [1, 0, 1, 1, 0, 1],
        })
        out = build_grpo_prompts(df)
        assert len(out) == 3
        assert set(out["prompt"]) == {"q1", "q2", "q3"}

    def test_carries_split_column(self, reward_df):
        out = build_grpo_prompts(reward_df)
        assert "split" in out.columns

    def test_raises_on_missing_text_a(self):
        bad = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValueError, match="text_a"):
            build_grpo_prompts(bad)

    def test_no_cross_split_leakage(self):
        """A prompt that appears in train must NOT appear in val. Without this,
        GRPO val rollouts could come from prompts the model already trained on.
        Train-precedence: shared prompts stay in train, val keeps only unseen.
        """
        df = pd.DataFrame({
            # "shared_q" appears in BOTH splits — the bug case.
            "text_a": ["train_q1", "shared_q", "val_q1", "shared_q", "val_q2"],
            "text_b": ["a", "b", "c", "d", "e"],
            "label":  [1, 1, 1, 0, 0],
            "split":  ["train", "train", "val", "val", "val"],
        })
        out = build_grpo_prompts(df)
        train_prompts = set(out[out["split"] == "train"]["prompt"])
        val_prompts = set(out[out["split"] == "val"]["prompt"])
        assert train_prompts.isdisjoint(val_prompts), (
            f"Cross-split leakage: {train_prompts & val_prompts!r} in both."
        )
        assert "shared_q" in train_prompts, "Shared prompt should stay in train."
        assert "shared_q" not in val_prompts, "Shared prompt must be removed from val."
        assert val_prompts == {"val_q1", "val_q2"}


# ---------------------------------------------------------------------------
# Round-trip: write to parquet, re-read, verify contract preserved
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_kto_parquet_roundtrip(self, reward_df, tmp_path):
        out = build_kto(reward_df)
        p = tmp_path / "kto.parquet"
        out.to_parquet(p, index=False)
        re_read = pd.read_parquet(p)
        # Schema must survive the round-trip.
        _validate_schema(re_read, KTO_REQUIRED, "kto-roundtrip")
        assert re_read["label"].dtype == bool

    def test_dpo_parquet_roundtrip(self, reward_df, tmp_path):
        out = build_dpo_random(reward_df, seed=42)
        p = tmp_path / "dpo.parquet"
        out.to_parquet(p, index=False)
        re_read = pd.read_parquet(p)
        _validate_schema(re_read, DPO_REQUIRED, "dpo-roundtrip")
        assert (re_read["chosen"] != re_read["rejected"]).all()

    def test_grpo_parquet_roundtrip(self, reward_df, tmp_path):
        out = build_grpo_prompts(reward_df)
        p = tmp_path / "grpo_prompts.parquet"
        out.to_parquet(p, index=False)
        re_read = pd.read_parquet(p)
        _validate_schema(re_read, GRPO_REQUIRED, "grpo-roundtrip")
