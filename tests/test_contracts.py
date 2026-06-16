"""F2 — data-contract tests (round-trip serialization + causal guards).

Per .claude/documents/features/11_F2_interfaces_contracts_config.md §6-7.
"""
from __future__ import annotations

import pytest

from mcrs.contracts import (
    Candidate,
    Query,
    RankedList,
    SubmissionRow,
    TurnContext,
    UserProfile,
)


def _profile() -> UserProfile:
    return UserProfile(user_id="u1", age=30, gender="f", country="US",
                       history_tids=["t1", "t2"])


def _ctx(turn_number: int = 2, n_utt: int = 2) -> TurnContext:
    return TurnContext(
        session_id="u1__2024-01-01",
        user_id="u1",
        turn_number=turn_number,
        utterances=[f"u{i}" for i in range(n_utt)],
        goal="discover upbeat pop",
        user_profile=_profile(),
        history_tids=["t1", "t2"],
        segment="warm",
    )


def test_submission_row_roundtrip():
    row = SubmissionRow(
        session_id="u1__2024-01-01", user_id="u1", turn_number=2,
        predicted_track_ids=["t9", "t8", "t7"], predicted_response="Try these.",
    )
    assert SubmissionRow.from_dict(row.to_dict()) == row


def test_userprofile_roundtrip():
    p = _profile()
    assert UserProfile.from_dict(p.to_dict()) == p


def test_turncontext_roundtrip_with_nested_profile():
    ctx = _ctx()
    restored = TurnContext.from_dict(ctx.to_dict())
    assert restored == ctx
    assert isinstance(restored.user_profile, UserProfile)


def test_candidate_roundtrip():
    c = Candidate(track_id="t1", channel_scores={"bm25": 1.2},
                  channel_ranks={"bm25": 3}, rrf_score=0.05,
                  features={"pop": 0.9})
    assert Candidate.from_dict(c.to_dict()) == c


def test_query_defaults_are_independent():
    q1, q2 = Query(text="a"), Query(text="b")
    q1.per_channel["bm25"] = "x"
    assert q2.per_channel == {}  # no shared mutable default


def test_turncontext_rejects_noncausal_utterance_count():
    # turn_number must equal len(utterances) (turns 1..t only) — F2 §4 causal guard
    with pytest.raises(ValueError):
        _ctx(turn_number=3, n_utt=2)


def test_turncontext_accepts_matching_utterance_count():
    ctx = _ctx(turn_number=2, n_utt=2)  # must not raise
    assert ctx.turn_number == len(ctx.utterances)


def test_rankedlist_holds_candidates_in_order():
    ctx = _ctx()
    items = [Candidate(track_id="t9"), Candidate(track_id="t8")]
    rl = RankedList(turn=ctx, items=items)
    assert [c.track_id for c in rl.items] == ["t9", "t8"]
