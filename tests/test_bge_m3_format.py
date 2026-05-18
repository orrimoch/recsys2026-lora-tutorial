"""Tests for BGE-M3 query/track text formatters used in fine-tune + inference."""
import pytest


def test_format_track_text_includes_all_5_fields():
    """Track text concatenates the 5 corpus fields in the documented order."""
    from mcrs.retrieval_modules.bge_m3_format import format_track_text
    text = format_track_text(
        track_name="Suite: Judy Blue Eyes",
        artist_name="Crosby Stills Nash",
        album_name="Crosby Stills & Nash",
        release_date="1969-05-29",
        tag_list=["folk rock", "harmony"],
    )
    assert "Suite: Judy Blue Eyes" in text
    assert "Crosby Stills Nash" in text
    assert "1969" in text
    assert "folk rock" in text


def test_format_track_text_handles_empty_fields():
    """Missing fields render as 'unknown' or empty, not crash."""
    from mcrs.retrieval_modules.bge_m3_format import format_track_text
    text = format_track_text(
        track_name="Unknown Track", artist_name=None, album_name=None,
        release_date=None, tag_list=None,
    )
    assert "Unknown Track" in text


def test_format_query_text_uses_inference_pipeline_format():
    """Query format must match what crs_baseline.batch_chat produces at inference.

    Since format_query_text now delegates to build_retrieval_query in
    mode='raw' (default), output is newline-joined "role: content" across
    all turns including the appended current_user_query.
    """
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    text = format_query_text(
        chat_history=[
            {"role": "user", "content": "I like 70s rock"},
            {"role": "assistant", "content": "How about CSN?"},
        ],
        current_user_query="Yes, more like that",
        user_profile={"age": 35, "country_code": "US"},
        conversation_goal={"listener_goal": "find 70s folk-rock"},
    )
    # Chat history + current query are all present (mode='raw' joins them).
    assert "I like 70s rock" in text
    assert "How about CSN?" in text
    assert "Yes, more like that" in text
    # mode='raw' uses "role: content" lines.
    assert "user: I like 70s rock" in text
    assert "assistant: How about CSN?" in text
    assert "user: Yes, more like that" in text


def test_format_query_handles_empty_history():
    """First-turn case: chat_history is empty list."""
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    text = format_query_text(
        chat_history=[],
        current_user_query="Play me something upbeat",
        user_profile=None,
        conversation_goal=None,
    )
    assert "Play me something upbeat" in text


# ---------------------------------------------------------------------------
# Contract-pinning tests: exact equality against build_retrieval_query.
# These prevent silent drift between fine-tune and inference query strings.
# ---------------------------------------------------------------------------


def test_format_query_text_mode_raw_matches_build_retrieval_query():
    """In mode='raw', the output must equal what production's build_retrieval_query produces."""
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    from mcrs.crs_baseline import build_retrieval_query

    chat = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
    current = "play rock"
    expected = build_retrieval_query(
        chat + [{"role": "user", "content": current}],
        mode="raw", goal_text="",
    )
    actual = format_query_text(
        chat_history=chat, current_user_query=current,
        user_profile=None, conversation_goal=None, mode="raw",
    )
    assert actual == expected


def test_format_query_text_mode_last_user_with_goal_uses_goal_text():
    """Mode 'last_user_with_goal' picks up conversation_goal.listener_goal."""
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    from mcrs.crs_baseline import build_retrieval_query

    chat = [{"role": "user", "content": "x"}]
    current = "more please"
    goal = {"listener_goal": "find 70s folk"}
    expected = build_retrieval_query(
        chat + [{"role": "user", "content": current}],
        mode="last_user_with_goal", goal_text="find 70s folk",
    )
    actual = format_query_text(
        chat_history=chat, current_user_query=current,
        user_profile=None, conversation_goal=goal, mode="last_user_with_goal",
    )
    assert actual == expected


def test_format_query_text_truncates_history_to_max_turns():
    """max_history_turns truncates chat_history before delegation."""
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    long_hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    text = format_query_text(long_hist, "q", max_history_turns=3, mode="raw")
    # Last 3 chat messages (m7, m8, m9) + final q should appear; m0–m6 should not.
    assert "m0" not in text
    assert "m9" in text
    assert "q" in text


def test_format_track_text_pins_separator_and_field_order():
    """Exact-string pin for the 5-field track corpus format."""
    from mcrs.retrieval_modules.bge_m3_format import format_track_text
    text = format_track_text("T", "A", "Alb", "2020-01-01", ["t1", "t2"])
    assert text == (
        "track_name: T | artist_name: A | album_name: Alb | "
        "release_date: 2020-01-01 | tag_list: t1, t2"
    )
