"""Tests for SID training data builders + query formatter."""
from typing import Any


def test_format_query_for_sid_input_includes_user_profile_when_present():
    """User profile keys (age, country_code, preferred_musical_culture) appear in output."""
    from mcrs.sid.training_data import format_query_for_sid_input

    out = format_query_for_sid_input(
        chat_history=[],
        current_user_query="play me something dreamy",
        user_profile={"age": 36, "country_code": "MX", "preferred_musical_culture": "Anglo-American Rock"},
        conversation_goal=None,
    )
    assert "age=36" in out
    assert "country=MX" in out
    assert "Anglo-American Rock" in out
    assert "dreamy" in out


def test_format_query_for_sid_input_includes_conversation_goal_listener_goal():
    """conversation_goal.listener_goal text appears in output."""
    from mcrs.sid.training_data import format_query_for_sid_input

    out = format_query_for_sid_input(
        chat_history=[],
        current_user_query="next track",
        user_profile=None,
        conversation_goal={"category": "F", "listener_goal": "find energetic 90s rock"},
    )
    assert "find energetic 90s rock" in out
    assert "next track" in out


def test_format_query_for_sid_input_omits_user_profile_block_when_none():
    """When user_profile is None, no [USER] line in output."""
    from mcrs.sid.training_data import format_query_for_sid_input

    out = format_query_for_sid_input(
        chat_history=[],
        current_user_query="anything",
        user_profile=None,
        conversation_goal=None,
    )
    assert "[USER]" not in out
    assert "anything" in out


def test_format_query_for_sid_input_includes_chat_history_with_role_labels():
    """chat_history messages render with role + content."""
    from mcrs.sid.training_data import format_query_for_sid_input

    chat = [
        {"role": "user", "content": "I want indie rock"},
        {"role": "assistant", "content": "track_name: Mr Brightside, artist_name: The Killers"},
    ]
    out = format_query_for_sid_input(
        chat_history=chat,
        current_user_query="something newer",
        user_profile=None,
        conversation_goal=None,
    )
    assert "user: I want indie rock" in out
    assert "Mr Brightside" in out
    assert "something newer" in out


def test_format_query_for_sid_input_truncates_long_history_messages():
    """Each history message is truncated to 200 chars to keep prompt tractable."""
    from mcrs.sid.training_data import format_query_for_sid_input

    long_msg = "x" * 300
    chat = [{"role": "user", "content": long_msg}]
    out = format_query_for_sid_input(
        chat_history=chat,
        current_user_query="ok",
        user_profile=None,
        conversation_goal=None,
    )
    # The assistant's content is truncated to 200 chars max
    assert "x" * 300 not in out
    assert "x" * 200 in out


def test_windowed_chat_history_returns_last_n_turn_pairs():
    """N=3 keeps the last 6 messages (3 user + 3 assistant pairs)."""
    from mcrs.sid.training_data import windowed_chat_history

    chat = []
    for i in range(5):  # 5 user-assistant turn-pairs = 10 messages
        chat.append({"role": "user", "content": f"u{i}"})
        chat.append({"role": "assistant", "content": f"a{i}"})

    out = windowed_chat_history(chat, n_turns=3)
    assert len(out) == 6
    # Should keep the last 6 (i=2 onwards)
    assert out[0]["content"] == "u2"
    assert out[-1]["content"] == "a4"


def test_windowed_chat_history_no_op_when_n_turns_is_none():
    """n_turns=None preserves full history."""
    from mcrs.sid.training_data import windowed_chat_history

    chat = [{"role": "user", "content": str(i)} for i in range(20)]
    out = windowed_chat_history(chat, n_turns=None)
    assert len(out) == 20


def test_windowed_chat_history_no_op_when_history_short():
    """When len(chat) <= 2*n_turns, no truncation needed."""
    from mcrs.sid.training_data import windowed_chat_history

    chat = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    out = windowed_chat_history(chat, n_turns=3)
    assert out == chat


def test_windowed_chat_history_handles_empty_input():
    from mcrs.sid.training_data import windowed_chat_history
    assert windowed_chat_history([], n_turns=3) == []
