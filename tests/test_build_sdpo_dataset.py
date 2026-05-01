"""Tests for scripts/build_sdpo_dataset.py (W5 conditional).

Verifies the (1-pos, N-neg) → N-pair expansion + perturbation correctness:
  - Envelope structure preserved across both chosen and rejected (W4 P0 #5
    requirement: r_format must hold on BOTH sides of the DPO pair, else the
    model learns "no envelope is preferable").
  - chosen ≠ rejected always (degenerate pairs would be wasted gradient).
  - Perturbations actually mutate response (not silent no-ops on edge cases).
  - Cross-session GPA NEG fallback works when same-session has no NEG turn.
  - TRL DPO schema (prompt, chosen, rejected) is producible.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_sdpo_dataset import (  # noqa: E402
    BANNED_INJECTION,
    ENVELOPE_RE,
    PERTURBATION_VARIANTS,
    _build_neg_inner,
    _perturb_drop_track_name,
    _perturb_drop_why,
    _perturb_inject_banned,
    _perturb_truncate_5,
    _rewrap,
    _split_envelope,
    _track_artist_from_text_a,
    build_sdpo_pairs,
)
from reward_fns import ENVELOPE  # noqa: E402  — same regex used by W6 r_format


# ---------------------------------------------------------------------------
# Envelope split + rewrap
# ---------------------------------------------------------------------------

class TestEnvelopeSplit:
    def test_split_extracts_inner(self):
        text = "<user_state>\nmood: calm\n</user_state>\n<response>\nGreat track!\n</response>"
        prefix, inner, suffix = _split_envelope(text)
        assert "<response>" in prefix
        assert inner == "Great track!"
        assert "</response>" in suffix

    def test_rewrap_roundtrip(self):
        original = "<user_state>\nmood: calm\n</user_state>\n<response>\nOriginal text.\n</response>"
        prefix, inner, suffix = _split_envelope(original)
        rewrapped = _rewrap(prefix, inner, suffix)
        # Re-parse must succeed
        m = ENVELOPE_RE.search(rewrapped)
        assert m is not None
        assert m.group(2).strip() == "Original text."

    def test_split_failure_returns_full_text(self):
        text = "no envelope here"
        prefix, inner, suffix = _split_envelope(text)
        assert prefix == ""
        assert inner == "no envelope here"
        assert suffix == ""

    def test_rewrap_with_empty_prefix_returns_inner(self):
        # Fallback path when envelope split failed — rewrap returns just the inner.
        out = _rewrap("", "new text", "")
        assert out == "new text"


# ---------------------------------------------------------------------------
# Perturbation correctness
# ---------------------------------------------------------------------------

class TestPerturbations:
    def test_drop_track_name_replaces_both(self):
        inner = 'Try "Holocene" by Bon Iver, it has a dreamy vibe.'
        out = _perturb_drop_track_name(inner, "Holocene", "Bon Iver")
        assert "Holocene" not in out
        assert "Bon Iver" not in out
        assert "this song" in out
        assert "the artist" in out

    def test_drop_track_name_handles_missing_args(self):
        inner = "Just a generic response with no names."
        out = _perturb_drop_track_name(inner, "", "")
        assert out == inner  # nothing to replace

    def test_drop_track_name_word_boundary_p0_3_regression(self):
        """W5 P0 #3 regression: track named "track" must NOT corrupt the
        word "soundtrack" via substring match. \\b boundaries required."""
        inner = "this soundtrack is great, the track Holocene works."
        out = _perturb_drop_track_name(inner, "track", "")
        assert "soundtrack" in out, f"substring corruption: {out!r}"
        assert "soundthis song" not in out, (
            f"P0 #3 regression: substring 'track' inside 'soundtrack' was "
            f"replaced. Got: {out!r}"
        )
        # The standalone word "track" should still be replaced.
        assert "this song Holocene" in out

    def test_drop_track_name_skips_too_short_names(self):
        """Names < 3 chars (e.g. "I", "You") would over-match common English.
        Skip them entirely rather than risk shredding the response."""
        inner = "I love this Holocene song!"
        out = _perturb_drop_track_name(inner, "I", "")
        assert out == inner, f"too-short name should be skipped, got {out!r}"
        # But 3+ char names work normally
        out2 = _perturb_drop_track_name(inner, "Holocene", "")
        assert "this song" in out2

    def test_inject_banned_appends(self):
        inner = "Cool tune you might like."
        out = _perturb_inject_banned(inner)
        assert out.startswith(inner)
        assert "absolutely fantastic" in out
        assert "perfectly amazing" in out

    def test_truncate_5_keeps_only_first_5_words(self):
        inner = "This is a very long response with many words after this point too."
        out = _perturb_truncate_5(inner)
        assert len(out.split()) == 5
        assert out == "This is a very long"

    def test_truncate_5_handles_empty(self):
        # Empty input → "(empty)" placeholder, never empty string.
        assert _perturb_truncate_5("") == "(empty)"

    def test_drop_why_replaces_musical_vocab(self):
        inner = "It features a layered tempo because of the rhythm and groove."
        out = _perturb_drop_why(inner)
        # Each "why" word becomes "[X]"
        assert out.count("[X]") >= 4  # features, tempo, because, rhythm, groove
        assert "features" not in out.lower()


# ---------------------------------------------------------------------------
# Track/artist extraction
# ---------------------------------------------------------------------------

class TestTrackArtistExtraction:
    def test_extracts_basic(self):
        text_a = ("User query: I want jazz\n"
                  "Listener goal: discover\n"
                  "Goal category: discovery\n"
                  "Recommended track: Holocene by Bon Iver [folk, indie]\n"
                  "Prior dialog: ")
        name, artist = _track_artist_from_text_a(text_a)
        assert name == "Holocene"
        assert artist == "Bon Iver"

    def test_no_recommended_track_section(self):
        text_a = "Just plain text with no recommended-track field."
        name, artist = _track_artist_from_text_a(text_a)
        assert name == ""
        assert artist == ""

    def test_no_tag_brackets_p0_1_regression(self):
        """W5 P0 #1 regression: empty tag_list means text_a has no `[...]`
        suffix. The regex must still extract via newline anchor.

        This is the EXACT shape that was silently dropping drop_track_name
        on ~30% of real production rows."""
        text_a = ("User query: I want jazz\n"
                  "Goal category: discovery\n"
                  "Recommended track: Holocene by Bon Iver\n"
                  "Prior dialog: foo")
        name, artist = _track_artist_from_text_a(text_a)
        assert name == "Holocene", f"P0 #1 regression: got name={name!r}"
        assert artist == "Bon Iver", f"P0 #1 regression: got artist={artist!r}"

    def test_no_tag_brackets_no_prior_dialog(self):
        """Even tighter: no tags AND no Prior dialog line, just trailing newline."""
        text_a = "Recommended track: Holocene by Bon Iver\n"
        name, artist = _track_artist_from_text_a(text_a)
        assert name == "Holocene"
        assert artist == "Bon Iver"

    def test_no_artist(self):
        """Some tracks don't have an artist field; regex should still get name."""
        text_a = "Recommended track: Some Title\nPrior dialog: foo"
        name, artist = _track_artist_from_text_a(text_a)
        assert name == "Some Title"
        assert artist == ""


# ---------------------------------------------------------------------------
# build_neg_inner dispatch
# ---------------------------------------------------------------------------

class TestBuildNegInner:
    def test_dispatches_to_correct_variant(self):
        # Inner needs to contain `why`-vocab tokens (because/features/tempo/etc.)
        # for drop_why to register a change.
        inner = "Holocene by Bon Iver features grunge tempo because of its rhythm."
        # drop_track_name removes names
        out_dt = _build_neg_inner("drop_track_name", inner, "Holocene", "Bon Iver")
        assert "Holocene" not in out_dt
        # inject_banned adds banned phrase
        out_ib = _build_neg_inner("inject_banned", inner, "Holocene", "Bon Iver")
        assert "absolutely fantastic" in out_ib
        # truncate_5 limits length
        out_t5 = _build_neg_inner("truncate_5", inner, "Holocene", "Bon Iver")
        assert len(out_t5.split()) == 5
        # drop_why replaces vocab
        out_dw = _build_neg_inner("drop_why", inner, "Holocene", "Bon Iver")
        assert out_dw != inner
        assert "[X]" in out_dw  # vocab tokens were replaced

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="unknown perturbation"):
            _build_neg_inner("nonexistent", "any", "any", "any")


# ---------------------------------------------------------------------------
# build_sdpo_pairs end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def envelope_df():
    """Mirror of `data/reward_train_envelope.parquet` schema."""
    def env(state_block, response):
        return f"<user_state>\n{state_block}\n</user_state>\n<response>\n{response}\n</response>"

    rows = [
        # POS turn 1 of session_a — has cross-session NEG (turn 2)
        {
            "text_a": "User query: q1\nGoal: g1\nRecommended track: Holocene by Bon Iver",
            "text_b": env("mood: calm", "Holocene by Bon Iver leans into folk vocals."),
            "label": 1, "session_id": "sess_a", "user_id": "u1",
            "turn_number": 1, "split": "train",
        },
        {
            "text_a": "User query: q2\nGoal: g2\nRecommended track: track2 by artist2",
            "text_b": env("mood: low", "Track2 has bad vibes really."),
            "label": 0, "session_id": "sess_a", "user_id": "u1",
            "turn_number": 2, "split": "train",
        },
        # POS turn 1 of session_b — NO cross-session NEG (forces drop_why fallback)
        {
            "text_a": "User query: q3\nGoal: g3\nRecommended track: Heartshaped by Nirvana",
            "text_b": env("mood: dark", "Heartshaped by Nirvana features grunge tempo."),
            "label": 1, "session_id": "sess_b", "user_id": "u2",
            "turn_number": 1, "split": "train",
        },
    ]
    return pd.DataFrame(rows)


class TestBuildSdpoPairs:
    def test_produces_4_pairs_per_pos_turn(self, envelope_df):
        out, stats = build_sdpo_pairs(envelope_df, seed=42)
        # 2 POS rows × 4 negatives = 8 pairs (assuming no degenerates).
        assert len(out) == 8
        assert stats["pos_turns_processed"] == 2

    def test_columns_correct_for_trl_dpo(self, envelope_df):
        out, _ = build_sdpo_pairs(envelope_df)
        for col in ["prompt", "chosen", "rejected"]:
            assert col in out.columns

    def test_chosen_and_rejected_distinct(self, envelope_df):
        out, _ = build_sdpo_pairs(envelope_df)
        assert (out["chosen"] != out["rejected"]).all()

    def test_envelope_preserved_on_perturbation_negs(self, envelope_df):
        out, _ = build_sdpo_pairs(envelope_df)
        # Perturbation negatives must keep envelope structure (only inner mutated)
        for _, row in out.iterrows():
            if row["neg_type"] in PERTURBATION_VARIANTS or row["neg_type"] == "drop_why":
                # rejected is envelope-wrapped (perturbation kept tags)
                assert ENVELOPE.search(row["rejected"]) is not None, (
                    f"perturbation neg lost envelope: type={row['neg_type']} "
                    f"rejected={row['rejected'][:200]!r}"
                )
            assert ENVELOPE.search(row["chosen"]) is not None

    def test_cross_session_neg_used_when_available(self, envelope_df):
        out, stats = build_sdpo_pairs(envelope_df)
        # session_a has 1 NEG row → POS turn 1 should use it as 4th neg.
        sess_a_pairs = out[out["prompt"].str.contains("Recommended track: Holocene")]
        types = set(sess_a_pairs["neg_type"])
        assert "cross_session_neg" in types, types

    def test_drop_why_fallback_when_no_session_neg(self, envelope_df):
        out, stats = build_sdpo_pairs(envelope_df)
        # session_b has no NEG → POS turn 1 falls back to drop_why.
        sess_b_pairs = out[out["prompt"].str.contains("Recommended track: Heartshaped")]
        types = set(sess_b_pairs["neg_type"])
        assert "drop_why" in types
        assert stats["drop_why_fallbacks"] >= 1

    def test_seed_determinism(self, envelope_df):
        out1, _ = build_sdpo_pairs(envelope_df, seed=42)
        out2, _ = build_sdpo_pairs(envelope_df, seed=42)
        # Sort to avoid order-sensitive comparison
        cols = ["prompt", "chosen", "rejected", "neg_type"]
        pd.testing.assert_frame_equal(
            out1[cols].sort_values(cols).reset_index(drop=True),
            out2[cols].sort_values(cols).reset_index(drop=True),
        )

    def test_raises_on_missing_required_columns(self):
        bad = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_sdpo_pairs(bad)

    def test_only_pos_rows_become_anchors(self, envelope_df):
        out, _ = build_sdpo_pairs(envelope_df)
        # NEG rows in input should NOT appear as `chosen` in output.
        # (They might appear as `rejected` via cross_session_neg.)
        neg_text_b = envelope_df[envelope_df["label"] == 0]["text_b"].iloc[0]
        # No row's `chosen` should equal a NEG row's text_b.
        assert (out["chosen"] != neg_text_b).all()

    def test_split_column_carried(self, envelope_df):
        out, _ = build_sdpo_pairs(envelope_df)
        assert "split" in out.columns
        assert all(out["split"] == "train")

    def test_silent_noop_counters_fire_when_track_missing_p1_7(self):
        """W5 P1 #7 regression: silent-noop counters must increment when
        a perturbation does nothing (track_name not in response). Without
        these counters, the data builder masks regex bugs as
        `degenerate_pairs_skipped` with no diagnostic.
        """
        def env(state, response):
            return f"<user_state>\n{state}\n</user_state>\n<response>\n{response}\n</response>"
        # POS row where the gold response does NOT contain the recommended
        # track name → drop_track_name should silent-noop.
        df = pd.DataFrame([
            {
                "text_a": "Recommended track: ObscureTrack by ObscureArtist\nPrior dialog: foo",
                "text_b": env("mood: calm", "Some generic acoustic vibe answer."),
                "label": 1, "session_id": "sess_x", "user_id": "u",
                "turn_number": 1, "split": "train",
            },
        ])
        _, stats = build_sdpo_pairs(df)
        # The track name "ObscureTrack" is NOT in the gold response →
        # drop_track_name produces unchanged inner → counter fires.
        assert stats["drop_track_name_silent_noop"] >= 1, stats

    def test_track_name_extraction_failures_counted(self):
        """W5 P1 #7: when text_a doesn't contain a Recommended-track line at
        all, track_name_extraction_failures must increment."""
        def env(state, response):
            return f"<user_state>\n{state}\n</user_state>\n<response>\n{response}\n</response>"
        df = pd.DataFrame([
            {
                "text_a": "User query: foo\nGoal: bar\n",  # NO Recommended track field
                "text_b": env("mood: calm", "Some response with content here for the test."),
                "label": 1, "session_id": "sess_y", "user_id": "u",
                "turn_number": 1, "split": "train",
            },
        ])
        _, stats = build_sdpo_pairs(df)
        assert stats["track_name_extraction_failures"] == 1, stats


# ---------------------------------------------------------------------------
# Round-trip with TRL DPO schema validator from build_trl_datasets
# ---------------------------------------------------------------------------

class TestTrlSchemaCompat:
    def test_dpo_schema_passes_validator(self, envelope_df, tmp_path):
        out, _ = build_sdpo_pairs(envelope_df)
        # Drop the diagnostic columns we don't ship to TRL
        trl_only = out[["prompt", "chosen", "rejected"]]
        from build_trl_datasets import DPO_REQUIRED, _validate_schema
        _validate_schema(trl_only, DPO_REQUIRED, "sdpo-dpo-schema")

    def test_parquet_roundtrip_preserves_envelopes(self, envelope_df, tmp_path):
        out, _ = build_sdpo_pairs(envelope_df)
        p = tmp_path / "sdpo.parquet"
        out.to_parquet(p, index=False)
        re_read = pd.read_parquet(p)
        # P1 #8 fix (W5 review): drop the prior cross_session_neg opt-out.
        # The augment_envelope.py pipeline wraps ALL rows including label==0
        # in the envelope, so cross_session_neg `text_b` is also envelope-
        # wrapped. Asymmetric envelope between chosen + rejected = DPO learns
        # "no envelope is preferable." Assert envelope on BOTH sides.
        for _, row in re_read.iterrows():
            assert ENVELOPE.search(row["chosen"]) is not None, (
                f"chosen lost envelope: {row['chosen'][:200]!r}"
            )
            assert ENVELOPE.search(row["rejected"]) is not None, (
                f"rejected ({row['neg_type']}) lost envelope: "
                f"{row['rejected'][:200]!r}"
            )
