"""Tests for scripts/build_grpo_dataset.py (W6 Rank-GRPO main loop).

W6 trains a TRL GRPOTrainer with `compose_r_turn` as the reward function.
The dataset must carry side-channel columns (predicted_track_ids,
gold_track_id, top1_meta_json, user_state_json, history_text) per row so
the reward fn can compute R_retr / R_rule deterministically when called by
TRL with `**kwargs` from the dataset.

Design constraint: the retriever is FROZEN — `predicted_track_ids` is
computed once outside this script (in the colab notebook via CMQR+ProRank)
and passed in as a separate parquet keyed by (session_id, turn_number).
This script joins the two parquets, validates the join, and writes the
TRL-ready GRPO parquet.
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


# ---------------------------------------------------------------------------
# Module import (Task 1)
# ---------------------------------------------------------------------------

class TestImports:
    def test_module_imports(self):
        import build_grpo_dataset  # noqa: F401


# ---------------------------------------------------------------------------
# Rationales block formatter (Task 2)
# ---------------------------------------------------------------------------

class TestRationalesBlock:
    def test_block_shape(self):
        from build_grpo_dataset import format_rationales_block
        out = format_rationales_block(["matches mood + low energy", "1990s era pref"])
        assert out.startswith("<reranker_rationales>")
        assert out.rstrip().endswith("</reranker_rationales>")
        assert "1. matches mood + low energy" in out
        assert "2. 1990s era pref" in out

    def test_empty_list_emits_valid_block(self):
        from build_grpo_dataset import format_rationales_block
        out = format_rationales_block([])
        assert "<reranker_rationales>" in out
        assert "</reranker_rationales>" in out
        body = out.split("<reranker_rationales>", 1)[1].split("</reranker_rationales>", 1)[0]
        assert body.strip() == ""

    def test_truncates_to_top_k(self):
        from build_grpo_dataset import format_rationales_block
        many = [f"r{i}" for i in range(50)]
        out = format_rationales_block(many, top_k=5)
        assert "1. r0" in out
        assert "5. r4" in out
        assert "6. r5" not in out

    def test_strips_internal_newlines_in_rationale(self):
        from build_grpo_dataset import format_rationales_block
        out = format_rationales_block(["good\nmatch", "ok"])
        assert "1. good match" in out
        assert "2. ok" in out


# ---------------------------------------------------------------------------
# Envelope state extraction (Task 3)
# ---------------------------------------------------------------------------

class TestExtractUserState:
    def test_extracts_known_keys(self):
        from build_grpo_dataset import extract_user_state
        text_b = (
            "<user_state>\n"
            "mood: reflective\n"
            "energy: low\n"
            "</user_state>\n"
            "<response>Holocene by Bon Iver fits the mood.</response>"
        )
        state = extract_user_state(text_b)
        assert state == {"mood": "reflective", "energy": "low"}

    def test_returns_empty_when_envelope_missing(self):
        from build_grpo_dataset import extract_user_state
        assert extract_user_state("just plain text, no tags") == {}

    def test_returns_empty_when_state_block_empty(self):
        from build_grpo_dataset import extract_user_state
        text = "<user_state></user_state>\n<response>x</response>"
        assert extract_user_state(text) == {}

    def test_ignores_unknown_keys(self):
        from build_grpo_dataset import extract_user_state
        text = (
            "<user_state>\n"
            "mood: calm\n"
            "color: blue\n"
            "</user_state>\n<response>x</response>"
        )
        assert extract_user_state(text) == {"mood": "calm"}


# ---------------------------------------------------------------------------
# Schema validators (Task 4)
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_envelope_missing_required_raises(self):
        from build_grpo_dataset import validate_envelope_schema
        bad = pd.DataFrame({"text_a": ["x"], "text_b": ["y"]})
        with pytest.raises(ValueError, match="envelope parquet missing"):
            validate_envelope_schema(bad)

    def test_envelope_complete_passes(self):
        from build_grpo_dataset import validate_envelope_schema
        ok = pd.DataFrame({
            "text_a": ["x"], "text_b": ["y"], "label": [1],
            "session_id": ["s"], "turn_number": [1], "split": ["train"],
        })
        validate_envelope_schema(ok)

    def test_retrieval_missing_required_raises(self):
        from build_grpo_dataset import validate_retrieval_schema
        bad = pd.DataFrame({"session_id": ["s"], "turn_number": [1]})
        with pytest.raises(ValueError, match="retrieval parquet missing"):
            validate_retrieval_schema(bad)

    def test_retrieval_complete_passes(self):
        from build_grpo_dataset import validate_retrieval_schema
        ok = pd.DataFrame({
            "session_id": ["s"], "turn_number": [1],
            "gold_track_id": ["g"], "predicted_track_ids": [["a", "b"]],
            "top1_track_name": ["Holocene"], "top1_artist_name": ["Bon Iver"],
            "reranker_rationales": [["matches mood"]],
        })
        validate_retrieval_schema(ok)


# ---------------------------------------------------------------------------
# Core build (Task 5)
# ---------------------------------------------------------------------------

@pytest.fixture
def envelope_df():
    """Mirror of the W4/W5 envelope-augmented reward parquet."""
    def env(state, response):
        return f"<user_state>\n{state}\n</user_state>\n<response>\n{response}\n</response>"

    rows = [
        {
            "text_a": ("User query: I want chill folk\n"
                       "Goal category: discovery\n"
                       "Recommended track: Holocene by Bon Iver [folk]\n"
                       "Prior dialog: looking for winding-down music"),
            "text_b": env("mood: reflective\nenergy: low",
                          "Holocene by Bon Iver leans into a layered arrangement."),
            "label": 1, "session_id": "sess_a", "user_id": "u1",
            "turn_number": 1, "split": "train",
        },
        {
            "text_a": ("User query: 90s grunge\n"
                       "Goal category: revisit\n"
                       "Recommended track: Heart-Shaped Box by Nirvana"),
            "text_b": env("mood: dark", "Heart-Shaped Box features grunge tempo."),
            "label": 1, "session_id": "sess_b", "user_id": "u2",
            "turn_number": 1, "split": "train",
        },
        {
            "text_a": ("User query: more upbeat\n"
                       "Goal category: discovery\n"
                       "Recommended track: SomeTrack by SomeArtist"),
            "text_b": env("mood: low", "SomeTrack vibes."),
            "label": 0, "session_id": "sess_a", "user_id": "u1",
            "turn_number": 2, "split": "train",
        },
        {
            "text_a": ("User query: missing retrieval\n"
                       "Recommended track: NoRetrieve by Nobody"),
            "text_b": env("mood: calm", "NoRetrieve fits."),
            "label": 1, "session_id": "sess_c", "user_id": "u3",
            "turn_number": 1, "split": "val",
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def retrieval_df():
    rows = [
        {
            "session_id": "sess_a", "turn_number": 1,
            "gold_track_id": "uuid-holocene",
            "predicted_track_ids": ["uuid-holocene", "uuid-other1", "uuid-other2"],
            "top1_track_name": "Holocene", "top1_artist_name": "Bon Iver",
            "reranker_rationales": ["matches reflective mood",
                                    "low energy fit",
                                    "indie folk genre"],
        },
        {
            "session_id": "sess_b", "turn_number": 1,
            "gold_track_id": "uuid-heart",
            # gold NOT in top-3 — exercises the gold_in_predicted=0 counter
            "predicted_track_ids": ["uuid-decoy1", "uuid-decoy2", "uuid-decoy3"],
            "top1_track_name": "Decoy Track", "top1_artist_name": "Decoy Artist",
            "reranker_rationales": ["weak", "match"],
        },
    ]
    return pd.DataFrame(rows)


class TestBuildGrpoDataset:
    def test_filters_to_pos_rows_only(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, stats = build_grpo_dataset(envelope_df, retrieval_df)
        assert stats["pos_rows_in"] == 3
        assert stats["joined_rows"] == 2
        assert stats["unjoined_pos"] == 1
        assert len(out) == 2

    def test_output_columns(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        expected = {
            "prompt", "gold_track_id", "predicted_track_ids",
            "top1_meta_json", "user_state_json", "history_text",
            "session_id", "turn_number", "split",
        }
        assert expected.issubset(set(out.columns))

    def test_prompt_contains_rationales_block(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        sess_a = out[out["session_id"] == "sess_a"].iloc[0]
        assert "<reranker_rationales>" in sess_a["prompt"]
        assert "1. matches reflective mood" in sess_a["prompt"]
        assert "Recommended track: Holocene by Bon Iver" in sess_a["prompt"]

    def test_top1_meta_json_roundtrip(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        sess_a = out[out["session_id"] == "sess_a"].iloc[0]
        meta = json.loads(sess_a["top1_meta_json"])
        assert meta == {"track_name": "Holocene", "artist_name": "Bon Iver"}

    def test_user_state_json_roundtrip(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        sess_a = out[out["session_id"] == "sess_a"].iloc[0]
        state = json.loads(sess_a["user_state_json"])
        assert state == {"mood": "reflective", "energy": "low"}

    def test_history_text_extracted_when_present(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        sess_a = out[out["session_id"] == "sess_a"].iloc[0]
        assert "winding-down music" in sess_a["history_text"]

    def test_history_text_empty_when_absent(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        sess_b = out[out["session_id"] == "sess_b"].iloc[0]
        assert sess_b["history_text"] == ""

    def test_predicted_ids_carried_as_list(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        sess_a = out[out["session_id"] == "sess_a"].iloc[0]
        assert isinstance(sess_a["predicted_track_ids"], list)
        assert sess_a["predicted_track_ids"][0] == "uuid-holocene"

    def test_split_carried(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        assert all(out["split"] == "train")

    def test_gold_in_predicted_counter(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out, stats = build_grpo_dataset(envelope_df, retrieval_df)
        assert stats["gold_in_predicted"] == 1
        assert 0.0 < stats["recall_at_n"] < 1.0

    def test_raises_on_invalid_envelope_schema(self, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="envelope parquet missing"):
            build_grpo_dataset(bad, retrieval_df)

    def test_raises_on_invalid_retrieval_schema(self, envelope_df):
        from build_grpo_dataset import build_grpo_dataset
        bad = pd.DataFrame({"session_id": ["s"]})
        with pytest.raises(ValueError, match="retrieval parquet missing"):
            build_grpo_dataset(envelope_df, bad)

    def test_seed_determinism(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        out1, _ = build_grpo_dataset(envelope_df, retrieval_df, seed=42)
        out2, _ = build_grpo_dataset(envelope_df, retrieval_df, seed=42)
        cols = ["session_id", "turn_number", "prompt", "gold_track_id"]
        pd.testing.assert_frame_equal(
            out1[cols].sort_values(cols).reset_index(drop=True),
            out2[cols].sort_values(cols).reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Parquet roundtrip + reward fn compatibility (Task 6)
# ---------------------------------------------------------------------------

class TestParquetRoundtrip:
    def test_roundtrip_preserves_columns(self, envelope_df, retrieval_df, tmp_path):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        p = tmp_path / "grpo.parquet"
        out.to_parquet(p, index=False)
        re_read = pd.read_parquet(p)
        for col in [
            "prompt", "gold_track_id", "predicted_track_ids",
            "top1_meta_json", "user_state_json", "history_text",
            "session_id", "turn_number", "split",
        ]:
            assert col in re_read.columns, col

    def test_roundtrip_preserves_predicted_ids_list(self, envelope_df, retrieval_df, tmp_path):
        from build_grpo_dataset import build_grpo_dataset
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        p = tmp_path / "grpo.parquet"
        out.to_parquet(p, index=False)
        re_read = pd.read_parquet(p)
        sess_a = re_read[re_read["session_id"] == "sess_a"].iloc[0]
        ids = list(sess_a["predicted_track_ids"])
        assert ids[0] == "uuid-holocene"
        assert len(ids) == 3


class TestRewardFnCompat:
    """Smoke that compose_r_turn accepts the columns we emit, end-to-end."""
    def test_compose_r_turn_consumes_emitted_columns(self, envelope_df, retrieval_df):
        from build_grpo_dataset import build_grpo_dataset
        from reward_fns import compose_r_turn
        out, _ = build_grpo_dataset(envelope_df, retrieval_df)
        sess_a = out[out["session_id"] == "sess_a"].iloc[0]

        completion = (
            "<user_state>\nmood: reflective\nenergy: low\n</user_state>\n"
            "<response>Holocene by Bon Iver leans into a layered arrangement "
            "with a slow tempo, matching the reflective mood you described. "
            "Want a sparser version next?</response>"
        )
        comps = compose_r_turn(
            predicted_track_ids=list(sess_a["predicted_track_ids"]),
            gold_track_id=sess_a["gold_track_id"],
            response_text=completion,
            valid_catalog={"uuid-holocene", "uuid-other1", "uuid-other2"},
            top1_meta=json.loads(sess_a["top1_meta_json"]),
            user_state=json.loads(sess_a["user_state_json"]),
            history_text=sess_a["history_text"],
        )
        assert "r_turn" in comps
        assert comps["r_retr"] == 1.0
        assert comps["r_turn"] >= 0.70


class TestMainCli:
    def test_main_writes_parquet(self, envelope_df, retrieval_df, tmp_path):
        import build_grpo_dataset as mod
        env_path = tmp_path / "env.parquet"
        ret_path = tmp_path / "ret.parquet"
        out_path = tmp_path / "grpo.parquet"
        envelope_df.to_parquet(env_path, index=False)
        retrieval_df.to_parquet(ret_path, index=False)
        rc = mod.main([
            "--envelope", str(env_path),
            "--retrieval", str(ret_path),
            "--out", str(out_path),
        ])
        assert rc == 0
        assert out_path.exists()
        re_read = pd.read_parquet(out_path)
        assert len(re_read) == 2

    def test_main_returns_nonzero_on_missing_envelope(self, tmp_path):
        import build_grpo_dataset as mod
        rc = mod.main([
            "--envelope", str(tmp_path / "nonexistent.parquet"),
            "--retrieval", str(tmp_path / "also_missing.parquet"),
            "--out", str(tmp_path / "out.parquet"),
        ])
        assert rc == 1
