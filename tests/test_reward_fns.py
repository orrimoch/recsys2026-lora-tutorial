"""Tests for scripts/reward_fns.py — Component-B reward terms (W1, Gap 8).

Covers:
  - parse_envelope / parse_user_state (strict + permissive cases)
  - dedupe_keep_first
  - filter_catalog_membership
  - r_retr (rank-1 hit, rank-2 hit, miss)
  - r_rule sub-scores (track-name, why-clause, banned phrases, length)
  - r_format envelope strictness
  - compose_r_turn hard-zero behavior (catalog miss + missing format)
  - compose_r_session smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import reward_fns as rf  # noqa: E402


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParseEnvelope:
    def test_strict_envelope(self):
        text = "<user_state>\nmood: calm\n</user_state>\n<response>Hello.</response>"
        parsed = rf.parse_envelope(text)
        assert parsed is not None
        state_block, resp = parsed
        assert "mood: calm" in state_block
        assert resp == "Hello."

    def test_no_envelope_returns_none(self):
        assert rf.parse_envelope("just plain text") is None

    def test_envelope_must_have_both_tags(self):
        # Has <user_state> but no <response>
        assert rf.parse_envelope("<user_state>x</user_state>") is None


class TestParseUserState:
    def test_basic_keys(self):
        block = "mood: calm\nintent: explore\nenergy: low"
        state = rf.parse_user_state(block)
        assert state == {"mood": "calm", "intent": "explore", "energy": "low"}

    def test_unknown_keys_dropped(self):
        block = "mood: calm\nfoo: bar\nnot_a_key: value"
        state = rf.parse_user_state(block)
        assert state == {"mood": "calm"}

    def test_empty_lines_tolerated(self):
        block = "\nmood: calm\n\n\nfamiliarity: high\n"
        state = rf.parse_user_state(block)
        assert state == {"mood": "calm", "familiarity": "high"}

    def test_garbage_lines_skipped(self):
        block = "mood: calm\nnot a kv\nintent: explore"
        state = rf.parse_user_state(block)
        assert state == {"mood": "calm", "intent": "explore"}

    def test_case_insensitive_keys(self):
        block = "MOOD: calm\nIntent: explore"
        state = rf.parse_user_state(block)
        assert state == {"mood": "calm", "intent": "explore"}


# ---------------------------------------------------------------------------
# Dedupe + catalog filter
# ---------------------------------------------------------------------------

class TestDedupeAndFilter:
    def test_dedupe_keeps_first_seen(self):
        assert rf.dedupe_keep_first(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_dedupe_empty(self):
        assert rf.dedupe_keep_first([]) == []

    def test_dedupe_no_duplicates_passthrough(self):
        assert rf.dedupe_keep_first(["a", "b", "c"]) == ["a", "b", "c"]

    def test_catalog_filter_drops_invalid(self):
        valid = {"a", "b", "c", "d"}
        out = rf.filter_catalog_membership(
            ["zzz", "a", "yyy", "b"], valid, ["c", "d", "a"], target_len=3,
        )
        assert out == ["a", "b", "c"]

    def test_catalog_filter_padding_from_pool(self):
        valid = {"a", "b", "c", "d", "e"}
        out = rf.filter_catalog_membership(
            ["a"], valid, ["b", "c", "d"], target_len=3,
        )
        assert out == ["a", "b", "c"]

    def test_catalog_filter_no_dupes_within_output(self):
        valid = {"a", "b", "c"}
        out = rf.filter_catalog_membership(
            ["a", "a", "b"], valid, ["b", "c"], target_len=3,
        )
        assert out == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# r_retr
# ---------------------------------------------------------------------------

class TestRRetr:
    def test_hit_rank_1_is_one(self):
        assert abs(rf.r_retr(["a", "b", "c"], "a") - 1.0) < 1e-9

    def test_hit_rank_2_in_band(self):
        # 0.5*nDCG@20 + 0.3*nDCG@10 + 0.2*0  → ≈ 0.504 (1/log2(3) ≈ 0.6309)
        rr = rf.r_retr(["b", "a", "c"], "a")
        assert 0.4 < rr < 0.7

    def test_miss_is_zero(self):
        assert rf.r_retr(["b", "c"], "a") == 0.0

    def test_empty_gold_is_zero(self):
        assert rf.r_retr(["a", "b"], "") == 0.0


# ---------------------------------------------------------------------------
# r_rule
# ---------------------------------------------------------------------------

class TestRRule:
    def test_full_match_high_score(self):
        meta = {"track_name": "Holocene", "artist_name": "Bon Iver"}
        state = {"mood": "reflective", "energy": "low"}
        resp = (
            "Bon Iver's Holocene leans into a layered arrangement and slow tempo, "
            "matching the reflective mood you described. Want a sparser version next?"
        )
        score = rf.r_rule(resp, top1_meta=meta, user_state=state, history_text="winding down dreamy")
        assert score >= 0.7

    def test_banned_phrase_penalises(self):
        meta = {"track_name": "Holocene", "artist_name": "Bon Iver"}
        state = {"mood": "reflective"}
        good = "Bon Iver's Holocene leans into a layered arrangement."
        bad = good + " Absolutely fantastic!"
        s_good = rf.r_rule(good, top1_meta=meta, user_state=state)
        s_bad = rf.r_rule(bad, top1_meta=meta, user_state=state)
        assert s_bad < s_good

    def test_empty_response_zero(self):
        assert rf.r_rule("", top1_meta=None) == 0.0

    def test_no_meta_just_lower_signal(self):
        # No top1_meta or state → fewer sub-scores, but length + sentence checks pass
        resp = "This song features a layered arrangement and atmospheric tempo. Enjoy!"
        score = rf.r_rule(resp)
        # Should still get partial credit for why-clause + length-band + sentences + no-banned
        assert score > 0.2

    def test_score_bounded_at_one(self):
        meta = {"track_name": "Holocene", "artist_name": "Bon Iver"}
        state = {"mood": "reflective", "energy": "low", "intent": "deepen", "sonic_pref": "folk"}
        resp = (
            "Bon Iver's Holocene leans into a layered arrangement, slow tempo, dreamy "
            "atmosphere — features the reflective mood and low energy you mentioned, "
            "with a folk sonic_pref texture from the 2010s era. Want a sparser version next?"
        )
        score = rf.r_rule(resp, top1_meta=meta, user_state=state, history_text="winding down dreamy")
        assert score <= 1.0


# ---------------------------------------------------------------------------
# r_format
# ---------------------------------------------------------------------------

class TestRFormat:
    def test_well_formed_envelope(self):
        text = "<user_state>\nmood: calm\nenergy: low\n</user_state>\n<response>Hello.</response>"
        assert rf.r_format(text) == 1.0

    def test_no_envelope(self):
        assert rf.r_format("just plain text") == 0.0

    def test_envelope_but_empty_state(self):
        text = "<user_state></user_state><response>x</response>"
        assert rf.r_format(text) == 0.0

    def test_envelope_with_only_unknown_keys(self):
        # Keys not in ALLOWED_STATE_KEYS → state empty → score 0.0
        text = "<user_state>foo: bar\nbaz: qux</user_state><response>x</response>"
        assert rf.r_format(text) == 0.0


# ---------------------------------------------------------------------------
# r_user_prof
# ---------------------------------------------------------------------------

class TestRUserProf:
    def test_country_match(self):
        prof = {"country_name": "Japan", "age_group": "25-34", "gender": "F"}
        assert rf.r_user_prof("This Japan-released track suits you", prof) >= 0.5

    def test_no_profile_zero(self):
        assert rf.r_user_prof("Anything goes", None) == 0.0

    def test_empty_response_zero(self):
        prof = {"country_name": "Japan"}
        assert rf.r_user_prof("", prof) == 0.0


# ---------------------------------------------------------------------------
# compose_r_turn — hard-zero behavior
# ---------------------------------------------------------------------------

class TestComposeRTurn:
    def test_happy_path(self):
        meta = {"track_name": "Holocene", "artist_name": "Bon Iver"}
        state = {"mood": "reflective", "energy": "low"}
        resp = (
            "<user_state>\nmood: reflective\nenergy: low\n</user_state>\n"
            "<response>Bon Iver's Holocene leans into a layered arrangement and slow tempo, "
            "matching the reflective mood. Enjoy!</response>"
        )
        comps = rf.compose_r_turn(
            predicted_track_ids=["track1", "track2"],
            gold_track_id="track1",
            response_text=resp,
            top1_meta=meta,
            user_state=state,
            history_text="winding down dreamy",
            valid_catalog={"track1", "track2"},
        )
        assert comps["catalog_ok"] == 1.0
        assert comps["r_format"] == 1.0
        assert comps["r_retr"] == 1.0
        assert 0.0 < comps["r_turn"] <= 1.0

    def test_hard_zero_on_hallucinated_track(self):
        comps = rf.compose_r_turn(
            predicted_track_ids=["hallucinated"],
            gold_track_id="track1",
            response_text="<user_state>mood: calm</user_state><response>x</response>",
            valid_catalog={"track1"},
        )
        assert comps["r_turn"] == 0.0
        assert comps["catalog_ok"] == 0.0

    def test_hard_zero_on_missing_format(self):
        # No envelope → r_format=0 → hard-zero r_turn (when include_format=True)
        comps = rf.compose_r_turn(
            predicted_track_ids=["track1"],
            gold_track_id="track1",
            response_text="just plain text, no envelope",
            valid_catalog={"track1"},
        )
        assert comps["r_format"] == 0.0
        assert comps["r_turn"] == 0.0

    def test_no_format_check_skips_hard_zero(self):
        # include_format=False mirrors the W1 LHS test path (gold has no envelope)
        comps = rf.compose_r_turn(
            predicted_track_ids=["track1"],
            gold_track_id="track1",
            response_text="just plain text, no envelope",
            valid_catalog={"track1"},
            include_format=False,
        )
        # Still gets R_retr-driven score
        assert comps["r_turn"] > 0.0

    def test_no_judge_subtraction_consistency(self):
        meta = {"track_name": "Holocene", "artist_name": "Bon Iver"}
        state = {"mood": "reflective"}
        resp = (
            "<user_state>\nmood: reflective\n</user_state>\n"
            "<response>Bon Iver's Holocene features a layered arrangement.</response>"
        )
        comps = rf.compose_r_turn(
            predicted_track_ids=["track1"],
            gold_track_id="track1",
            response_text=resp,
            top1_meta=meta,
            user_state=state,
            valid_catalog={"track1"},
        )
        rtnj = rf.compose_r_turn_no_judge(comps)
        # judge_score=None means r_judge=0; no-judge equals r_turn here
        assert abs(rtnj - comps["r_turn"]) < 1e-9


# ---------------------------------------------------------------------------
# Session shaping
# ---------------------------------------------------------------------------

class TestComposeRSession:
    def test_monotonic_session(self):
        sess = rf.compose_r_session(
            r_turns=[0.4, 0.5, 0.55, 0.55, 0.6, 0.6, 0.6, 0.65],
            r_judge_per_turn=[0.3, 0.4, 0.5, 0.5, 0.55, 0.6, 0.6, 0.65],
            responses=["Track has groove and warmth."] * 8,
            track_ids_per_turn=[["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"],
                                ["i", "j"], ["k", "l"], ["m", "n"], ["o", "p"]],
            catalog_size=50000,
        )
        assert sess["mean_r_turn"] > 0
        assert sess["monotonicity"] == 1.0

    def test_empty_session(self):
        sess = rf.compose_r_session(
            r_turns=[], r_judge_per_turn=[], responses=[], track_ids_per_turn=[],
            catalog_size=50000,
        )
        assert sess["r_session"] == 0.0


# ---------------------------------------------------------------------------
# Reward weights — Option B
# ---------------------------------------------------------------------------

class TestWeights:
    def test_weights_match_option_b(self):
        # Plan §6.1 Option B revision (W1 fallback).
        assert rf.W_RETR == 0.70
        assert rf.W_JUDGE == 0.10
        assert rf.W_RULE == 0.15
        assert rf.W_FORMAT == 0.05
        assert rf.W_USER_PROF == 0.00

    def test_weights_sum_to_one(self):
        total = rf.W_RETR + rf.W_JUDGE + rf.W_RULE + rf.W_FORMAT + rf.W_USER_PROF
        assert abs(total - 1.0) < 1e-9
