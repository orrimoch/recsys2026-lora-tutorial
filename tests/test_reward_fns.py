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
    def test_weights_match_option_b_v4(self):
        # Gap-analysis Step 3: data path piped end-to-end (build_reward_dataset
        # extracts user_profile_json from User-Metadata DB, build_grpo_dataset
        # carries through, reward closures pass user_profile=). Restored
        # W_USER_PROF=0.05 with W_RULE 0.20→0.15.
        assert rf.W_RETR == 0.40
        assert rf.W_JUDGE == 0.30
        assert rf.W_RULE == 0.15
        assert rf.W_FORMAT == 0.10
        assert rf.W_USER_PROF == 0.05

    def test_weights_sum_to_one(self):
        total = rf.W_RETR + rf.W_JUDGE + rf.W_RULE + rf.W_FORMAT + rf.W_USER_PROF
        assert abs(total - 1.0) < 1e-9


class TestRTurnClamp:
    """W1-W8 review P0-3: r_turn must be clamped to [0, 1].

    Without the clamp, the +0.05 lex_div bonus added on top of a maxed-out
    base r_turn (R_retr=1, R_judge=1, R_rule=1, R_format=1, R_user_prof=1
    with weights summing to 1.0) yields r_turn = 1.05, breaking the gate
    threshold semantics in colab/32 cell 14 (Δ R_turn ≥ +0.03).
    """
    def test_r_turn_never_exceeds_one_under_max_bonus(self):
        envelope = (
            "<user_state>\nmood: calm\nenergy: low\n</user_state>\n"
            "<response>"
            "Holocene by Bon Iver leans into the layered arrangement, slow "
            "tempo, calm mood, with a vibe that is both reflective and warm. "
            "Want a sparser version next? It has groove and atmosphere."
            "</response>"
        )
        # Maxed-out reward with peer rollouts that yield high diversity.
        comps = rf.compose_r_turn(
            predicted_track_ids=["a"], gold_track_id="a",
            response_text=envelope,
            top1_meta={"track_name": "Holocene", "artist_name": "Bon Iver"},
            user_state={"mood": "calm", "energy": "low"},
            user_profile={"country_name": "Japan", "age_group": "25-34", "gender": "F"},
            history_text="winding down dreamy",
            judge_score=1.0,  # max judge
            judge_trust=1.0,
            group_responses=[
                "first wildly different rollout response",
                "second alternate phrasing entirely",
                "third unique response with novel terms",
                "fourth distinct generation w/ varied tokens",
            ],
        )
        assert comps["r_turn"] <= 1.0, (
            f"r_turn must be clamped to [0,1] but got {comps['r_turn']:.4f}"
        )


class TestGroupResponsesBonus:
    """W6-review Option B refactor: compose_r_turn accepts an optional
    `group_responses` kwarg so the reward closure can pass the G peer
    rollouts of the same prompt and get a diversity bonus.

    This is the conversation-data signal we couldn't extract before —
    `compose_r_session.lex_div_distinct2` was defined but never wired into
    training. Now wired as a small additive bonus (max +0.05 to r_turn).
    """
    def test_no_group_responses_no_bonus(self):
        comps = rf.compose_r_turn(
            predicted_track_ids=["a"], gold_track_id="a",
            response_text="<user_state>mood: calm</user_state><response>x</response>",
        )
        # Without group_responses, no bonus key OR key is 0.0.
        assert comps.get("r_lex_div_group", 0.0) == 0.0

    def test_high_diversity_yields_bonus(self):
        envelope = lambda body: f"<user_state>mood: calm</user_state><response>{body}</response>"
        diverse_group = [
            envelope("Holocene by Bon Iver leans into a layered arrangement"),
            envelope("Skinny Love builds slowly with sparse instrumentation"),
            envelope("Re: Stacks features stripped-back acoustic guitar"),
            envelope("Wash your house in flooding atmospheric layers"),
        ]
        comps = rf.compose_r_turn(
            predicted_track_ids=["a"], gold_track_id="a",
            response_text=diverse_group[0],
            group_responses=diverse_group,
        )
        assert comps["r_lex_div_group"] > 0.5  # very diverse → close to 1.0
        # Bonus is added on top of the base r_turn (which is non-zero
        # because format passes and r_rule fires on at least envelope-mention).

    def test_identical_group_zero_diversity(self):
        """W1-W8 review P1-1: identical rollouts MUST give zero bonus.

        Old implementation used `lex_div_distinct2` over the JOINED text
        of all responses, which gave a non-zero floor (~0.013) even when
        all rollouts were identical (because internal within-text bigrams
        vary). New implementation uses pairwise across-rollout distinct-2
        — identical rollouts → 0 bonus, divergent rollouts → high bonus.
        """
        envelope = "<user_state>mood: calm</user_state><response>same text always</response>"
        comps = rf.compose_r_turn(
            predicted_track_ids=["a"], gold_track_id="a",
            response_text=envelope,
            group_responses=[envelope] * 4,
        )
        # 4 identical rollouts → all pairwise overlaps == 1.0 → diversity == 0.0.
        assert comps["r_lex_div_group"] == 0.0, (
            f"identical rollouts should give 0 bonus; got {comps['r_lex_div_group']:.4f}"
        )

    def test_singleton_group_no_bonus(self):
        # group_responses with 1 element → no peers → no bonus.
        envelope = "<user_state>mood: calm</user_state><response>only one</response>"
        comps = rf.compose_r_turn(
            predicted_track_ids=["a"], gold_track_id="a",
            response_text=envelope,
            group_responses=[envelope],
        )
        assert comps.get("r_lex_div_group", 0.0) == 0.0


class TestDistilledJudge:
    """W6-review Option B: DistilledJudge replaces r_judge_stub.

    Tests the back-compat (no-checkpoint) path because GPU/torch isn't
    available in CI. The actual model-load path is exercised by
    integration runs after a checkpoint is trained on Colab.
    """

    def test_no_checkpoint_returns_zero(self):
        judge = rf.DistilledJudge(checkpoint=None)
        assert judge.score("any context", "any response") == 0.0

    def test_no_checkpoint_batch_returns_zeros(self):
        judge = rf.DistilledJudge(checkpoint=None)
        scores = judge.score_batch(["c1", "c2", "c3"], ["r1", "r2", "r3"])
        assert scores == [0.0, 0.0, 0.0]

    def test_score_returns_float_in_unit_interval(self):
        # With no checkpoint: 0.0 (a valid float in [0, 1]).
        judge = rf.DistilledJudge()
        s = judge.score("ctx", "resp")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_score_batch_length_matches_input(self):
        judge = rf.DistilledJudge()
        ctxs = ["a"] * 5
        resps = ["b"] * 5
        out = judge.score_batch(ctxs, resps)
        assert len(out) == len(ctxs)

    def test_cache_does_not_explode_no_checkpoint(self):
        # Sanity: repeated calls don't allocate the cache when degrading.
        judge = rf.DistilledJudge()
        for _ in range(50):
            judge.score("x", "y")
        assert len(judge._cache) == 0  # cache only populated when model loads

    def test_compose_r_turn_accepts_distilled_judge_score(self):
        # Wire-through smoke: the closure pattern in colab/32 cell 11 will
        # call judge.score(ctx, completion) and pass the result into
        # compose_r_turn(judge_score=...). Verify compose_r_turn handles a
        # plain float (back-compat) and that the weight refactor is intact.
        judge = rf.DistilledJudge()
        score = judge.score("ctx", "resp")
        comps = rf.compose_r_turn(
            predicted_track_ids=["a"], gold_track_id="a",
            response_text="<user_state>mood: calm</user_state><response>x</response>",
            judge_score=score,
        )
        assert "r_turn" in comps
        assert comps["r_judge"] == 0.0  # no checkpoint → judge contributes 0

