"""Tests for scripts/augment_envelope.py (W4 prerequisite).

The envelope wrapper is critical: without it, B1 (KTO) trains the responder
to produce raw text and B3 (Rank-GRPO) immediately fails its R_format
hard-zero check on those outputs. These tests verify:

  - The envelope format matches scripts/reward_fns.ENVELOPE regex exactly
  - Allowed/disallowed state keys are filtered correctly
  - Empty / unknown / missing states fall back to `(unknown)` placeholder
  - Round-trip through pd.read_parquet preserves the envelope intact
  - Round-trip through scripts/build_trl_datasets produces a valid KTO file
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from augment_envelope import (  # noqa: E402
    ALLOWED_STATE_KEYS,
    augment,
    augment_row,
    format_state_block,
    load_state,
    wrap_envelope,
)
from reward_fns import ENVELOPE  # noqa: E402


# ---------------------------------------------------------------------------
# format_state_block
# ---------------------------------------------------------------------------

class TestFormatStateBlock:
    def test_full_state(self):
        state = {
            "mood": "calm", "intent": "explore", "energy": "low",
            "sonic_pref": "folk", "era_pref": "2010s", "familiarity": "familiar",
        }
        out = format_state_block(state)
        for k in ALLOWED_STATE_KEYS:
            assert f"{k}:" in out

    def test_unknown_skipped(self):
        state = {"mood": "calm", "energy": "unknown", "sonic_pref": ""}
        out = format_state_block(state)
        assert "mood: calm" in out
        assert "energy" not in out
        assert "sonic_pref" not in out

    def test_disallowed_keys_filtered(self):
        state = {"mood": "calm", "song_title": "leaked"}  # song_title not in ALLOWED
        out = format_state_block(state)
        assert "mood: calm" in out
        assert "song_title" not in out

    def test_none_returns_unknown(self):
        assert format_state_block(None) == "(unknown)"

    def test_empty_dict_returns_unknown(self):
        assert format_state_block({}) == "(unknown)"

    def test_all_unknown_values_returns_unknown(self):
        assert format_state_block({"mood": "unknown", "energy": ""}) == "(unknown)"

    def test_partial_state_renders_only_known(self):
        out = format_state_block({"mood": "happy"})
        assert out == "mood: happy"


# ---------------------------------------------------------------------------
# wrap_envelope + reward_fns.ENVELOPE regex parity
# ---------------------------------------------------------------------------

class TestWrapEnvelope:
    def test_envelope_parses_through_reward_fns_regex(self):
        text = wrap_envelope("Glad you liked that one!", "mood: calm\nenergy: low")
        m = ENVELOPE.search(text)
        assert m is not None, f"envelope did not match ENVELOPE regex: {text!r}"
        # The regex captures (state_block, response_block).
        state_block = m.group(1).strip()
        response_block = m.group(2).strip()
        assert "mood: calm" in state_block
        assert "Glad you liked" in response_block

    def test_envelope_with_unknown_state_still_parses(self):
        text = wrap_envelope("Some response", "(unknown)")
        m = ENVELOPE.search(text)
        assert m is not None
        # state block parses but contains only the placeholder.
        assert "(unknown)" in m.group(1)

    def test_envelope_preserves_response_text_verbatim(self):
        original = 'Multi-line response\nwith "quotes" and\nnewlines.'
        text = wrap_envelope(original, "mood: calm")
        m = ENVELOPE.search(text)
        assert m.group(2).strip() == original


# ---------------------------------------------------------------------------
# load_state
# ---------------------------------------------------------------------------

class TestLoadState:
    def test_returns_none_when_missing(self, tmp_path):
        s = load_state(tmp_path, "sess_x", 1)
        assert s is None

    def test_returns_dict_when_present(self, tmp_path):
        cache_file = tmp_path / "sess_x__1.json"
        with cache_file.open("w") as f:
            json.dump({"mood": "calm"}, f)
        s = load_state(tmp_path, "sess_x", 1)
        assert s == {"mood": "calm"}

    def test_strips_fallback_marker(self, tmp_path):
        # StateTracker writes `_fallback: prior_turn_state` for fallback rows;
        # the augmenter should ignore that marker.
        cache_file = tmp_path / "sess_x__2.json"
        with cache_file.open("w") as f:
            json.dump({"mood": "calm", "_fallback": "prior_turn_state"}, f)
        s = load_state(tmp_path, "sess_x", 2)
        assert s == {"mood": "calm"}

    def test_handles_session_id_with_special_chars(self, tmp_path):
        # State tracker sanitizes non-alnum chars to `_`. The loader must
        # apply the same transformation.
        sid = "sess-with-dashes_and_special"
        # Sanitized → same as input here (only - and _ allowed).
        cache_file = tmp_path / f"{sid}__1.json"
        with cache_file.open("w") as f:
            json.dump({"mood": "calm"}, f)
        s = load_state(tmp_path, sid, 1)
        assert s is not None
        assert s["mood"] == "calm"

    def test_invalid_json_returns_none(self, tmp_path):
        cache_file = tmp_path / "sess_y__1.json"
        cache_file.write_text("not valid json {{{")
        s = load_state(tmp_path, "sess_y", 1)
        assert s is None


# ---------------------------------------------------------------------------
# augment_row + augment (end-to-end on a small DataFrame)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Mirrors scripts/build_reward_dataset.py output schema."""
    return pd.DataFrame([
        {
            "text_a": "User query: q1\nGoal: g1\nRecommended: t1",
            "text_b": "Glad you liked that pop track!",
            "label": 1, "session_id": "sess_a", "user_id": "u1",
            "turn_number": 1, "goal_category": "discovery", "split": "train",
        },
        {
            "text_a": "User query: q2\nGoal: g2\nRecommended: t2",
            "text_b": "Try this jazz piece next.",
            "label": 0, "session_id": "sess_a", "user_id": "u1",
            "turn_number": 2, "goal_category": "discovery", "split": "train",
        },
        {
            "text_a": "User query: q3\nGoal: g3\nRecommended: t3",
            "text_b": "Here is something mellow.",
            "label": 1, "session_id": "sess_b", "user_id": "u2",
            "turn_number": 1, "goal_category": "discovery", "split": "val",
        },
    ])


class TestAugmentRow:
    def test_text_b_replaced_with_envelope(self, synthetic_df):
        row = synthetic_df.iloc[0].to_dict()
        out = augment_row(row, state={"mood": "happy"})
        assert "<user_state>" in out["text_b"]
        assert "</user_state>" in out["text_b"]
        assert "<response>" in out["text_b"]
        assert "</response>" in out["text_b"]
        assert "Glad you liked that pop track!" in out["text_b"]
        assert "mood: happy" in out["text_b"]

    def test_other_columns_preserved(self, synthetic_df):
        row = synthetic_df.iloc[0].to_dict()
        out = augment_row(row, state={"mood": "happy"})
        for col in ["text_a", "label", "session_id", "user_id",
                    "turn_number", "goal_category", "split"]:
            assert out[col] == row[col]

    def test_no_state_uses_unknown(self, synthetic_df):
        row = synthetic_df.iloc[0].to_dict()
        out = augment_row(row, state=None)
        assert "(unknown)" in out["text_b"]


class TestAugmentEndToEnd:
    def test_stub_mode_produces_unknown_envelopes(self, synthetic_df, tmp_path):
        out_df, stats = augment(synthetic_df, state_cache_dir=None, stub=True)
        assert len(out_df) == 3
        assert stats["state_hit"] == 0
        assert stats["state_miss"] == 3
        for text in out_df["text_b"]:
            assert "<user_state>" in text
            assert "(unknown)" in text

    def test_with_state_cache_uses_cached_states(self, synthetic_df, tmp_path):
        # Pre-populate cache for two of the three turns.
        cache_dir = tmp_path / "state"
        cache_dir.mkdir()
        with (cache_dir / "sess_a__1.json").open("w") as f:
            json.dump({"mood": "happy", "energy": "high"}, f)
        with (cache_dir / "sess_a__2.json").open("w") as f:
            json.dump({"mood": "reflective"}, f)
        # sess_b__1 left missing → should use (unknown).

        out_df, stats = augment(synthetic_df, state_cache_dir=cache_dir, stub=False)
        assert stats["state_hit"] == 2
        assert stats["state_miss"] == 1

        # Row 0: cached state present
        assert "mood: happy" in out_df.iloc[0]["text_b"]
        assert "energy: high" in out_df.iloc[0]["text_b"]
        # Row 1: cached state present (different mood)
        assert "mood: reflective" in out_df.iloc[1]["text_b"]
        # Row 2: cache miss → (unknown)
        assert "(unknown)" in out_df.iloc[2]["text_b"]

    def test_schema_preserved(self, synthetic_df):
        out_df, _ = augment(synthetic_df, state_cache_dir=None, stub=True)
        assert set(out_df.columns) == set(synthetic_df.columns)

    def test_raises_on_missing_required_column(self):
        bad = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            augment(bad, state_cache_dir=None, stub=True)


# ---------------------------------------------------------------------------
# Round-trip through build_trl_datasets — KTO path must accept envelope output
# ---------------------------------------------------------------------------

class TestRoundTripThroughTRL:
    def test_envelope_then_kto_format(self, synthetic_df, tmp_path):
        # 1. Envelope-augment
        out_df, _ = augment(synthetic_df, state_cache_dir=None, stub=True)
        # 2. Run through build_trl_datasets.build_kto
        from build_trl_datasets import KTO_REQUIRED, _validate_schema, build_kto
        kto = build_kto(out_df)
        # 3. Schema must still pass TRL validation
        _validate_schema(kto, KTO_REQUIRED, "kto-after-envelope")
        # 4. The completion column must contain envelope tags
        for c in kto["completion"]:
            assert "<user_state>" in c
            assert "<response>" in c

    def test_parquet_roundtrip_preserves_envelope(self, synthetic_df, tmp_path):
        out_df, _ = augment(synthetic_df, state_cache_dir=None, stub=True)
        p = tmp_path / "envelope.parquet"
        out_df.to_parquet(p, index=False)
        re_read = pd.read_parquet(p)
        for original, re_read_text in zip(out_df["text_b"], re_read["text_b"]):
            assert original == re_read_text
        # Envelope still parses through reward_fns.ENVELOPE
        for text in re_read["text_b"]:
            assert ENVELOPE.search(text) is not None
