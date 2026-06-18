"""F1 — Conversations (causal TurnContext + separate gold) and integrity helpers."""
from __future__ import annotations

import os

import pytest

from mcrs.contracts import TurnContext
from mcrs.data.catalog import Catalog
from mcrs.data.conversations import Conversations
from mcrs.data.integrity import unresolved_ids

_ROW = {
    "session_id": "u1__d", "user_id": "u1",
    "user_profile": {"age": 19, "gender": "female", "country_name": "Bulgaria"},
    "conversation_goal": {"listener_goal": "find upbeat pop"},
    "conversations": [
        {"role": "user", "content": "play heart-shaped box", "turn_number": 1, "thought": ""},
        {"role": "music", "content": "track-1", "turn_number": 1, "thought": "secret"},
        {"role": "assistant", "content": "here you go", "turn_number": 1, "thought": ""},
        {"role": "user", "content": "more like that", "turn_number": 2, "thought": ""},
        {"role": "music", "content": "track-2", "turn_number": 2, "thought": ""},
        {"role": "assistant", "content": "sure", "turn_number": 2, "thought": ""},
    ],
}


# Blind-A shape: the trailing entry is a USER query with no `music` gold (the prediction target).
_BLIND_ROW = {
    "session_id": "u9__b", "user_id": "u9",
    "user_profile": {"age": 30, "gender": "male", "country_code": "US"},
    "conversation_goal": {"listener_goal": "chill evening"},
    "conversations": [
        {"role": "user", "content": "play something jazzy", "turn_number": 1, "thought": ""},
        {"role": "music", "content": "track-1", "turn_number": 1, "thought": ""},
        {"role": "assistant", "content": "ok", "turn_number": 1, "thought": ""},
        {"role": "user", "content": "more upbeat", "turn_number": 2, "thought": ""},
        {"role": "music", "content": "track-2", "turn_number": 2, "thought": ""},
        {"role": "assistant", "content": "sure", "turn_number": 2, "thought": ""},
        {"role": "user", "content": "now something to dance to", "turn_number": 3, "thought": ""},  # TARGET (no gold)
    ],
}


def test_target_turns_predicts_the_trailing_user_turn_per_session():
    # turns() only emits music-bearing turns, so it MISSES the blind target. target_turns() must emit
    # exactly one TurnContext per session at the trailing turn (mirrors run_inference_blindset.py).
    c = Conversations([_BLIND_ROW], cold_threshold=0)
    assert len(list(c.turns())) == 2          # the two history turns — NOT the target
    tgts = list(c.target_turns())
    assert len(tgts) == 1
    (t,) = tgts
    assert t.turn_number == 3
    assert t.utterances == ["play something jazzy", "more upbeat", "now something to dance to"]
    assert t.history_tids == ["track-1", "track-2"] and t.segment == "warm"
    assert t.goal == "chill evening" and t.user_profile.country == "US"
    assert len(t.utterances) == t.turn_number  # causal guard holds


def test_target_turns_one_row_per_session():
    c = Conversations([_BLIND_ROW, {**_ROW, "session_id": "u1__d2"}])
    tgts = list(c.target_turns())
    assert len(tgts) == 2
    assert len({t.session_id for t in tgts}) == 2


def test_gold_is_the_music_entry_per_turn():
    c = Conversations([_ROW])
    assert c.gold("u1__d", 1) == "track-1"
    assert c.gold("u1__d", 2) == "track-2"


def test_gold_unknown_session_returns_none_not_keyerror():
    # callers probe gold across multiple Conversations splits (train+dev); a session not in THIS split
    # must return None, not raise — else the cross-split lookup crashes (blind K2 train-gold).
    c = Conversations([_ROW])
    assert c.gold("not-a-session", 1) is None


def test_turns_are_causal_and_carry_session_history():
    c = Conversations([_ROW], cold_threshold=0)
    turns = list(c.turns())
    assert len(turns) == 2

    t1, t2 = turns
    assert isinstance(t1, TurnContext)
    # turn 1: only its own user utterance, no prior history
    assert t1.turn_number == 1 and t1.utterances == ["play heart-shaped box"]
    assert t1.history_tids == [] and t1.segment == "cold"
    assert t1.goal == "find upbeat pop" and t1.user_profile.age == 19
    # turn 2: user utterances 1..2, prior gold in history, warm at threshold 0
    assert t2.utterances == ["play heart-shaped box", "more like that"]
    assert t2.history_tids == ["track-1"] and t2.segment == "warm"


def test_no_gold_or_future_or_thought_leaks_into_turn():
    turns = list(Conversations([_ROW]).turns())
    for t in turns:
        gold = "track-1" if t.turn_number == 1 else "track-2"
        assert gold not in t.utterances and gold not in t.history_tids
        assert "secret" not in t.utterances          # thought never enters utterances
    # turn 1 must not see turn 2's content
    assert "more like that" not in turns[0].utterances


def test_unresolved_ids_flags_out_of_catalog():
    cat = Catalog([{"track_id": "track-1", "track_name": ["A"], "artist_name": ["X"],
                    "album_name": ["Al"], "release_date": "2001"}])
    assert unresolved_ids(["track-1", "track-1"], cat) == set()
    assert unresolved_ids(["track-1", "ghost"], cat) == {"ghost"}


@pytest.mark.skipif(not os.path.isdir("data/TalkPlayData-Challenge-Dataset"),
                    reason="dataset not on disk")
def test_from_disk_gold_matches_official_parse():
    c = Conversations.from_disk("data/TalkPlayData-Challenge-Dataset", split="test")
    turns = list(c.turns())
    assert turns and all(len(t.utterances) == t.turn_number for t in turns[:50])
    # every gold resolves to a non-empty id
    sid, tn = turns[0].session_id, turns[0].turn_number
    assert c.gold(sid, tn)
