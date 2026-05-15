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


def test_build_metadata_as_query_pairs_one_pair_per_track():
    """Returns one (query, sid) pair for each track present in the SID lookup."""
    from mcrs.sid.training_data import build_metadata_as_query_pairs

    track_metadata = [
        {"track_id": "t1", "track_name": ["Bohemian Rhapsody"], "artist_name": ["Queen"], "album_name": ["A Night at the Opera"], "tag_list": ["rock"], "release_date": "1975"},
        {"track_id": "t2", "track_name": ["Yesterday"], "artist_name": ["The Beatles"], "album_name": ["Help!"], "tag_list": ["pop"], "release_date": "1965"},
    ]
    track_to_sid = {"t1": (1, 2, 3), "t2": (4, 5, 6)}

    pairs = build_metadata_as_query_pairs(track_metadata, track_to_sid)
    assert len(pairs) == 2
    by_tid = {p["track_id"]: p for p in pairs}
    assert "Bohemian Rhapsody" in by_tid["t1"]["query"]
    assert "Queen" in by_tid["t1"]["query"]
    assert by_tid["t1"]["code_1"] == 1
    assert by_tid["t1"]["code_2"] == 2
    assert by_tid["t1"]["code_3"] == 3
    assert by_tid["t1"]["source"] == "metadata"


def test_build_metadata_as_query_pairs_skips_tracks_not_in_sid_lookup():
    """If a track has no SID assignment, skip it (don't crash, don't synthesize)."""
    from mcrs.sid.training_data import build_metadata_as_query_pairs

    track_metadata = [
        {"track_id": "t1", "track_name": ["A"], "artist_name": ["B"], "album_name": [""], "tag_list": [], "release_date": ""},
        {"track_id": "t_missing", "track_name": ["C"], "artist_name": ["D"], "album_name": [""], "tag_list": [], "release_date": ""},
    ]
    track_to_sid = {"t1": (1, 2, 3)}  # t_missing absent

    pairs = build_metadata_as_query_pairs(track_metadata, track_to_sid)
    assert len(pairs) == 1
    assert pairs[0]["track_id"] == "t1"


def test_build_metadata_as_query_pairs_handles_empty_metadata_fields():
    """Tracks with empty/missing metadata fields still produce a pair (using whatever's available)."""
    from mcrs.sid.training_data import build_metadata_as_query_pairs

    track_metadata = [
        {"track_id": "t1", "track_name": ["Only Title"], "artist_name": [], "album_name": [], "tag_list": [], "release_date": ""},
    ]
    track_to_sid = {"t1": (1, 2, 3)}

    pairs = build_metadata_as_query_pairs(track_metadata, track_to_sid)
    assert len(pairs) == 1
    assert "Only Title" in pairs[0]["query"]


def test_build_raw_conversation_pairs_one_pair_per_music_turn():
    """For each music-role turn in conversations, emit one (query, SID) pair."""
    from mcrs.sid.training_data import build_raw_conversation_pairs

    sessions = [
        {
            "session_id": "s1", "user_id": "u1",
            "user_profile": {"age": 30, "country_code": "US", "preferred_musical_culture": "Pop"},
            "conversation_goal": {"category": "F", "listener_goal": "find rock"},
            "conversations": [
                {"turn_number": 1, "role": "user", "content": "play rock"},
                {"turn_number": 1, "role": "music", "content": "t1"},
                {"turn_number": 2, "role": "user", "content": "more"},
                {"turn_number": 2, "role": "music", "content": "t2"},
            ],
        },
    ]
    track_to_sid = {"t1": (1, 2, 3), "t2": (4, 5, 6)}

    pairs = build_raw_conversation_pairs(sessions, track_to_sid, n_turns_window=3)
    assert len(pairs) == 2
    # First pair: user_query="play rock", target=t1's SID=(1,2,3)
    assert "play rock" in pairs[0]["query"]
    assert pairs[0]["code_1"] == 1
    assert pairs[0]["track_id"] == "t1"
    assert pairs[0]["source"] == "raw"
    # Second pair: user_query="more", target=t2's SID=(4,5,6) — should include t1 in chat history
    assert pairs[1]["code_1"] == 4
    assert "more" in pairs[1]["query"]


def test_build_raw_conversation_pairs_skips_music_turns_with_no_sid():
    """If a track in conversations isn't in track_to_sid (e.g. test-only track), skip its pair."""
    from mcrs.sid.training_data import build_raw_conversation_pairs

    sessions = [
        {
            "session_id": "s1", "user_id": "u1",
            "user_profile": None, "conversation_goal": None,
            "conversations": [
                {"turn_number": 1, "role": "user", "content": "play rock"},
                {"turn_number": 1, "role": "music", "content": "t_missing"},
            ],
        },
    ]
    track_to_sid: dict = {}  # t_missing not in lookup

    pairs = build_raw_conversation_pairs(sessions, track_to_sid, n_turns_window=3)
    assert pairs == []


def test_build_raw_conversation_pairs_handles_session_without_user_profile_or_goal():
    """Sessions with None profile/goal still produce pairs (just no [USER]/[GOAL] sections)."""
    from mcrs.sid.training_data import build_raw_conversation_pairs

    sessions = [
        {
            "session_id": "s1", "user_id": "u1",
            "user_profile": None, "conversation_goal": None,
            "conversations": [
                {"turn_number": 1, "role": "user", "content": "anything"},
                {"turn_number": 1, "role": "music", "content": "t1"},
            ],
        },
    ]
    track_to_sid = {"t1": (1, 2, 3)}

    pairs = build_raw_conversation_pairs(sessions, track_to_sid, n_turns_window=3)
    assert len(pairs) == 1
    assert "[USER]" not in pairs[0]["query"]
    assert "anything" in pairs[0]["query"]
