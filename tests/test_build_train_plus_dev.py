"""Tests for scripts/build_train_plus_dev.py (W7 final-pass data prep).

W7 retrains the responder on `train + dev` combined per
`project_submission_prep.md`: iterate-on-train, retrain-on-train+dev for the
final Blind submission. This script is a thin wrapper that:
  1. Reads two reward parquets produced by `build_reward_dataset.py`
     (one with `--hf-split train`, one with `--hf-split test`).
  2. Tags each row with a `data_origin` column ("train" / "dev").
  3. Concatenates and writes one combined parquet.

The HF-side data fetching is exercised by `build_reward_dataset.py` itself
(no test there today; covered by integration runs). What's testable in
isolation is the concat + origin marker logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture
def train_parquet(tmp_path):
    p = tmp_path / "reward_train.parquet"
    pd.DataFrame([
        {"text_a": "tA1", "text_b": "tB1", "label": 1,
         "session_id": "s_train_a", "user_id": "u1",
         "turn_number": 1, "goal_category": "discovery", "split": "train"},
        {"text_a": "tA2", "text_b": "tB2", "label": 0,
         "session_id": "s_train_a", "user_id": "u1",
         "turn_number": 2, "goal_category": "discovery", "split": "train"},
    ]).to_parquet(p, index=False)
    return p


@pytest.fixture
def dev_parquet(tmp_path):
    p = tmp_path / "reward_dev.parquet"
    pd.DataFrame([
        {"text_a": "dA1", "text_b": "dB1", "label": 1,
         "session_id": "s_dev_a", "user_id": "u9",
         "turn_number": 1, "goal_category": "revisit", "split": "train"},
    ]).to_parquet(p, index=False)
    return p


class TestConcatTrainAndDev:
    def test_basic_concat_row_count(self, train_parquet, dev_parquet, tmp_path):
        from build_train_plus_dev import concat_train_and_dev
        out = tmp_path / "combined.parquet"
        stats = concat_train_and_dev(train_parquet, dev_parquet, out)
        df = pd.read_parquet(out)
        assert len(df) == 3  # 2 train rows + 1 dev row
        assert stats["n_train"] == 2
        assert stats["n_dev"] == 1
        assert stats["n_total"] == 3

    def test_origin_column_marks_each_row(self, train_parquet, dev_parquet, tmp_path):
        from build_train_plus_dev import concat_train_and_dev
        out = tmp_path / "combined.parquet"
        concat_train_and_dev(train_parquet, dev_parquet, out)
        df = pd.read_parquet(out)
        # Every row carries data_origin in {"train","dev"}.
        assert "data_origin" in df.columns
        train_rows = df[df["session_id"].str.startswith("s_train_")]
        dev_rows = df[df["session_id"].str.startswith("s_dev_")]
        assert all(train_rows["data_origin"] == "train")
        assert all(dev_rows["data_origin"] == "dev")

    def test_preserves_input_columns(self, train_parquet, dev_parquet, tmp_path):
        from build_train_plus_dev import concat_train_and_dev
        out = tmp_path / "combined.parquet"
        concat_train_and_dev(train_parquet, dev_parquet, out)
        df = pd.read_parquet(out)
        for col in ["text_a", "text_b", "label", "session_id",
                    "user_id", "turn_number"]:
            assert col in df.columns

    def test_split_column_unified_to_train(self, train_parquet, dev_parquet, tmp_path):
        """W7 retrains on ALL of train+dev → every row should be `split=train`
        from the downstream W4/W5/W6 builders' perspective. The original
        `split` column from the input parquets is overwritten; the original
        provenance is preserved in `data_origin` instead. This prevents the
        envelope augmenter or sdpo/grpo builders from accidentally treating
        the dev rows as a held-out val set.
        """
        from build_train_plus_dev import concat_train_and_dev
        out = tmp_path / "combined.parquet"
        concat_train_and_dev(train_parquet, dev_parquet, out)
        df = pd.read_parquet(out)
        assert all(df["split"] == "train")

    def test_raises_when_train_input_missing(self, dev_parquet, tmp_path):
        from build_train_plus_dev import concat_train_and_dev
        out = tmp_path / "combined.parquet"
        with pytest.raises(FileNotFoundError, match="train"):
            concat_train_and_dev(tmp_path / "nope.parquet", dev_parquet, out)

    def test_raises_when_dev_input_missing(self, train_parquet, tmp_path):
        from build_train_plus_dev import concat_train_and_dev
        out = tmp_path / "combined.parquet"
        with pytest.raises(FileNotFoundError, match="dev"):
            concat_train_and_dev(train_parquet, tmp_path / "nope.parquet", out)

    def test_raises_on_schema_mismatch(self, train_parquet, tmp_path):
        """If a future change to build_reward_dataset.py drops a column on one
        side but not the other, the concat must fail loudly — not silently
        produce a NaN-filled column that downstream builders would skip.
        """
        from build_train_plus_dev import concat_train_and_dev
        bad_dev = tmp_path / "bad_dev.parquet"
        # Missing several required columns vs. train_parquet's schema.
        pd.DataFrame([{"text_a": "x", "text_b": "y", "label": 1}]).to_parquet(bad_dev, index=False)
        out = tmp_path / "combined.parquet"
        with pytest.raises(ValueError, match="schema mismatch"):
            concat_train_and_dev(train_parquet, bad_dev, out)


class TestMainCli:
    def test_main_writes_parquet(self, train_parquet, dev_parquet, tmp_path):
        import build_train_plus_dev as mod
        out_path = tmp_path / "out.parquet"
        rc = mod.main([
            "--train", str(train_parquet),
            "--dev", str(dev_parquet),
            "--out", str(out_path),
        ])
        assert rc == 0
        assert out_path.exists()
        df = pd.read_parquet(out_path)
        assert len(df) == 3
        assert "data_origin" in df.columns

    def test_main_returns_nonzero_on_missing_input(self, tmp_path):
        import build_train_plus_dev as mod
        rc = mod.main([
            "--train", str(tmp_path / "nope.parquet"),
            "--dev", str(tmp_path / "also_nope.parquet"),
            "--out", str(tmp_path / "out.parquet"),
        ])
        assert rc == 1
