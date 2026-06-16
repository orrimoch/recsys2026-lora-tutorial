"""R1 — causal query construction from TurnContext."""
from __future__ import annotations

from mcrs.contracts import TurnContext, UserProfile
from mcrs.retrieval.query import QueryBuilder


def _ctx(utterances, goal="find upbeat pop"):
    return TurnContext(
        session_id="s", user_id="u", turn_number=len(utterances),
        utterances=list(utterances), goal=goal,
        user_profile=UserProfile("u", 20, "f", "US", []),
        history_tids=[], segment="cold",
    )


def test_build_concatenates_utterances_and_appends_goal():
    q = QueryBuilder().build(_ctx(["play heart shaped box", "more like that"]))
    assert "play heart shaped box" in q.text
    assert "more like that" in q.text
    assert "find upbeat pop" in q.text  # goal appended


def test_recency_window_keeps_only_last_n_utterances():
    q = QueryBuilder(recency_window=1).build(_ctx(["old one", "newest msg"]))
    assert "newest msg" in q.text and "find upbeat pop" in q.text
    assert "old one" not in q.text


def test_context_cap_drops_oldest_but_keeps_latest_and_goal():
    q = QueryBuilder(context_cap=3).build(
        _ctx(["aaa bbb ccc", "mid", "latest"], goal="g")
    )
    assert "latest" in q.text and "g" in q.text   # latest utterance + goal never dropped
    assert "aaa bbb ccc" not in q.text            # oldest dropped to fit the cap


def test_no_goal_is_handled():
    q = QueryBuilder().build(_ctx(["just this"], goal=None))
    assert q.text.strip() == "just this"
