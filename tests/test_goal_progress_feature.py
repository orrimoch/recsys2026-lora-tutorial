"""Tests for mcrs/rerank/goal_progress_feature.py.

TDD: these tests are written BEFORE the implementation.  Run with:
    ./recsys26/bin/python -m pytest tests/test_goal_progress_feature.py -v
"""
from __future__ import annotations

import pytest

from mcrs.rerank.goal_progress_feature import (
    build_prior_goal_progress_rate,
    make_prior_goal_progress_score_fn,
)
from mcrs.contracts import TurnContext, UserProfile


# ---------------------------------------------------------------------------
# Minimal fake Conversations-like object
# ---------------------------------------------------------------------------

def _profile(uid: str = "u1") -> UserProfile:
    return UserProfile(user_id=uid, age=None, gender=None, country=None, history_tids=[])


def _ctx(session_id: str, turn_number: int, uid: str = "u1") -> TurnContext:
    """Build a minimal TurnContext with the given session and turn number."""
    return TurnContext(
        session_id=session_id,
        user_id=uid,
        turn_number=turn_number,
        utterances=["q"] * turn_number,  # len must equal turn_number (F2 causal guard)
        goal=None,
        user_profile=_profile(uid),
        history_tids=[],
        segment="warm",
    )


class FakeConversations:
    """Minimal stand-in that exposes the same interface `build_prior_goal_progress_rate` uses:
      - iteration yields raw row dicts with `session_id` and `goal_progress_assessments`
      - `.turns()` yields TurnContexts
    """

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def turns(self):
        for row in self._rows:
            sid = row["session_id"]
            uid = row.get("user_id", "u1")
            # derive distinct turn numbers from conversations entries
            turn_nums = sorted(
                {int(e["turn_number"]) for e in row.get("conversations", [])}
            )
            for t in turn_nums:
                yield _ctx(session_id=sid, turn_number=t, uid=uid)


def _make_row(session_id: str, turns: list[int],
              assessments: list[tuple[int, str | None]]) -> dict:
    """Build a minimal raw row.

    turns: list of turn numbers that exist in conversations.
    assessments: list of (turn_number, assessment_string_or_None) — None means the
                 entry exists but assessment is null.
    """
    conversations = [
        {"turn_number": str(t), "role": "user", "content": "q"} for t in turns
    ]
    gpa = []
    for tn, label in assessments:
        entry: dict = {"turn_number": str(tn)}
        if label is not None:
            entry["goal_progress_assessment"] = label
        else:
            entry["goal_progress_assessment"] = None  # explicit null in the entry
        gpa.append(entry)
    return {
        "session_id": session_id,
        "user_id": "u1",
        "conversations": conversations,
        "goal_progress_assessments": gpa,
    }


# ---------------------------------------------------------------------------
# Tests for build_prior_goal_progress_rate
# ---------------------------------------------------------------------------

class TestBuildPriorGoalProgressRate:

    def test_turn1_always_returns_neutral_default(self):
        """Turn 1 has no prior turns, so the rate must be the neutral default (0.5)."""
        row = _make_row("s1", turns=[1], assessments=[(1, "MOVES_TOWARD_GOAL")])
        convs = FakeConversations([row])
        rate = build_prior_goal_progress_rate(convs)
        assert rate[("s1", 1)] == pytest.approx(0.5)

    def test_after_one_moves_one_does_not_rate_is_0_5(self):
        """After prior turns [MOVES, DOES_NOT], fraction = 1/2 = 0.5."""
        row = _make_row(
            "s1",
            turns=[1, 2, 3],
            assessments=[
                (1, "MOVES_TOWARD_GOAL"),
                (2, "DOES_NOT_MOVE_TOWARD_GOAL"),
                (3, "MOVES_TOWARD_GOAL"),  # turn 3 itself — must NOT affect turn 3's rate
            ],
        )
        convs = FakeConversations([row])
        rate = build_prior_goal_progress_rate(convs)
        # Turn 3 looks at turns 1 and 2: 1 MOVES + 1 DOES_NOT = 0.5
        assert rate[("s1", 3)] == pytest.approx(0.5)

    def test_after_two_moves_rate_is_1_0(self):
        """After prior turns [MOVES, MOVES], fraction = 2/2 = 1.0."""
        row = _make_row(
            "s1",
            turns=[1, 2, 3],
            assessments=[
                (1, "MOVES_TOWARD_GOAL"),
                (2, "MOVES_TOWARD_GOAL"),
                (3, "DOES_NOT_MOVE_TOWARD_GOAL"),
            ],
        )
        convs = FakeConversations([row])
        rate = build_prior_goal_progress_rate(convs)
        assert rate[("s1", 3)] == pytest.approx(1.0)

    def test_null_assessments_excluded_from_denominator(self):
        """Null assessments (missing label) must not count toward denominator or numerator."""
        row = _make_row(
            "s1",
            turns=[1, 2, 3, 4],
            assessments=[
                (1, "MOVES_TOWARD_GOAL"),
                (2, None),          # null — excluded
                (3, "MOVES_TOWARD_GOAL"),
                (4, "DOES_NOT_MOVE_TOWARD_GOAL"),
            ],
        )
        convs = FakeConversations([row])
        rate = build_prior_goal_progress_rate(convs)
        # Turn 4 sees priors 1..3: turn 1 MOVES (1), turn 2 null (skip), turn 3 MOVES (1)
        # rate = 2/2 = 1.0  (denominator is 2, not 3)
        assert rate[("s1", 4)] == pytest.approx(1.0)

    def test_all_null_priors_fall_back_to_default(self):
        """When ALL prior assessments are null, denominator=0 -> fall back to 0.5."""
        row = _make_row(
            "s1",
            turns=[1, 2, 3],
            assessments=[
                (1, None),
                (2, None),
                (3, "MOVES_TOWARD_GOAL"),
            ],
        )
        convs = FakeConversations([row])
        rate = build_prior_goal_progress_rate(convs)
        assert rate[("s1", 3)] == pytest.approx(0.5)

    def test_multiple_sessions_are_independent(self):
        """Different sessions should not bleed into each other."""
        row_a = _make_row(
            "sA",
            turns=[1, 2],
            assessments=[(1, "MOVES_TOWARD_GOAL"), (2, "MOVES_TOWARD_GOAL")],
        )
        row_b = _make_row(
            "sB",
            turns=[1, 2],
            assessments=[(1, "DOES_NOT_MOVE_TOWARD_GOAL"), (2, "MOVES_TOWARD_GOAL")],
        )
        convs = FakeConversations([row_a, row_b])
        rate = build_prior_goal_progress_rate(convs)
        # sA turn 2: prior = [MOVES] -> 1.0
        assert rate[("sA", 2)] == pytest.approx(1.0)
        # sB turn 2: prior = [DOES_NOT] -> 0.0
        assert rate[("sB", 2)] == pytest.approx(0.0)

    def test_causality_turn_t_assessment_never_affects_its_own_rate(self):
        """Proving causality: assessment at turn t must not change rate[(sid, t)].

        We build a session where turn 3 flips between MOVES and DOES_NOT_MOVE.
        The rate for turn 3 must be identical in both cases because it depends only
        on turns 1 and 2.
        """
        # Version A: turn 3 = MOVES
        row_a = _make_row(
            "s1",
            turns=[1, 2, 3],
            assessments=[
                (1, "MOVES_TOWARD_GOAL"),
                (2, "DOES_NOT_MOVE_TOWARD_GOAL"),
                (3, "MOVES_TOWARD_GOAL"),
            ],
        )
        # Version B: turn 3 = DOES_NOT_MOVE
        row_b = _make_row(
            "s1",
            turns=[1, 2, 3],
            assessments=[
                (1, "MOVES_TOWARD_GOAL"),
                (2, "DOES_NOT_MOVE_TOWARD_GOAL"),
                (3, "DOES_NOT_MOVE_TOWARD_GOAL"),
            ],
        )
        rate_a = build_prior_goal_progress_rate(FakeConversations([row_a]))
        rate_b = build_prior_goal_progress_rate(FakeConversations([row_b]))
        # Both must give the same rate for turn 3 (1 MOVES / 2 non-null = 0.5)
        assert rate_a[("s1", 3)] == pytest.approx(0.5)
        assert rate_b[("s1", 3)] == pytest.approx(0.5)
        assert rate_a[("s1", 3)] == pytest.approx(rate_b[("s1", 3)])


# ---------------------------------------------------------------------------
# Tests for make_prior_goal_progress_score_fn
# ---------------------------------------------------------------------------

class TestMakePriorGoalProgressScoreFn:

    def test_score_fn_returns_rate_map_value(self):
        """score_fn should return the precomputed rate for (session_id, turn_number)."""
        rate_map = {("s1", 2): 0.75}
        fn = make_prior_goal_progress_score_fn(rate_map, default=0.5)
        ctx = _ctx("s1", 2)
        assert fn(ctx, "any_track_id") == pytest.approx(0.75)

    def test_score_fn_uses_default_for_unknown_key(self):
        """Unknown (session_id, turn_number) key should return the configured default."""
        rate_map = {}
        fn = make_prior_goal_progress_score_fn(rate_map, default=0.5)
        ctx = _ctx("s_unknown", 5)
        assert fn(ctx, "t1") == pytest.approx(0.5)

    def test_score_fn_is_tid_independent(self):
        """The score must be identical regardless of which track_id is passed (session-level signal)."""
        rate_map = {("s1", 3): 0.8}
        fn = make_prior_goal_progress_score_fn(rate_map, default=0.5)
        ctx = _ctx("s1", 3)
        assert fn(ctx, "track_aaa") == pytest.approx(0.8)
        assert fn(ctx, "track_bbb") == pytest.approx(0.8)
        assert fn(ctx, "totally_different_id") == pytest.approx(0.8)

    def test_score_fn_custom_default(self):
        """A custom default (not 0.5) is respected for unknown keys."""
        rate_map = {}
        fn = make_prior_goal_progress_score_fn(rate_map, default=0.0)
        ctx = _ctx("s_missing", 1)
        assert fn(ctx, "t1") == pytest.approx(0.0)

    def test_score_fn_uses_session_and_turn_not_user(self):
        """Look-up key is (session_id, turn_number), not user_id."""
        rate_map = {("sess_x", 4): 0.333}
        fn = make_prior_goal_progress_score_fn(rate_map, default=0.5)
        ctx = _ctx("sess_x", 4, uid="any_user")
        assert fn(ctx, "t1") == pytest.approx(0.333)
