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


def test_format_query_text_mode_bge_m3_structured_emits_4_blocks():
    """ML-reviewer B1: structured 4-block format ([USER]/[GOAL]/[HISTORY]/[QUERY])
    is the spec'd train+inference format for the BGE-M3 fine-tune."""
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    text = format_query_text(
        chat_history=[
            {"role": "user", "content": "I like 70s rock"},
            {"role": "assistant", "content": "How about CSN?"},
        ],
        current_user_query="Yes, more like that",
        user_profile={"age_group": "35-44", "country_code": "US", "gender": "F"},
        conversation_goal={"listener_goal": "find 70s folk-rock"},
        mode="bge_m3_structured",
    )
    # Four ordered blocks present.
    assert "[USER]:" in text
    assert "[GOAL]:" in text
    assert "[HISTORY]:" in text
    assert "[QUERY]:" in text
    # User-block fields rendered.
    assert "age=35-44" in text
    assert "country=US" in text
    assert "gender=F" in text
    # Goal block carries listener_goal.
    assert "find 70s folk-rock" in text
    # History uses U:/A: turn markers and " | " separator (not raw "role: content").
    assert "U: I like 70s rock" in text
    assert "A: How about CSN?" in text
    # Final user turn lands in [QUERY], NOT [HISTORY].
    assert "[QUERY]: Yes, more like that" in text
    # The current query must NOT appear in the history section.
    history_block = text.split("[HISTORY]:")[1].split("[QUERY]:")[0]
    assert "Yes, more like that" not in history_block


def test_format_query_text_bge_m3_structured_missing_fields_render_unknown():
    """Missing user_profile / goal must still produce all 4 blocks (positional
    stability lets the encoder learn section boundaries)."""
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    text = format_query_text(
        chat_history=[],
        current_user_query="play me jazz",
        user_profile=None,
        conversation_goal=None,
        mode="bge_m3_structured",
    )
    assert "[USER]: age=unknown country=unknown gender=unknown" in text
    assert "[GOAL]: \n" in text or text.rstrip().count("[GOAL]: ") == 1
    assert "[HISTORY]: " in text
    assert "[QUERY]: play me jazz" in text


def test_build_retrieval_query_bge_m3_structured_caps_history():
    """[HISTORY] block keeps only the last `max_history_turns` turns BEFORE the
    final user turn (which becomes [QUERY])."""
    from mcrs.crs_baseline import build_retrieval_query
    sm = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
          for i in range(10)]
    sm.append({"role": "user", "content": "current"})
    text = build_retrieval_query(
        sm, mode="bge_m3_structured", goal_text="g",
        user_profile={"age_group": "25-34"}, max_history_turns=3,
    )
    # Only the last 3 PRIOR turns appear in history; earlier ones don't.
    assert "m7" in text and "m8" in text and "m9" in text
    assert "m0" not in text and "m1" not in text
    # Current query is in [QUERY], not [HISTORY].
    assert "[QUERY]: current" in text
    assert "current" not in text.split("[HISTORY]:")[1].split("[QUERY]:")[0]


def test_build_retrieval_query_bge_m3_structured_sanitizes_pipe_in_content():
    """User-supplied pipe chars in content must not collide with the ` | ` separator."""
    from mcrs.crs_baseline import build_retrieval_query
    sm = [
        {"role": "user", "content": "I want a | b | c"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "go"},
    ]
    text = build_retrieval_query(sm, mode="bge_m3_structured")
    history = text.split("[HISTORY]:")[1].split("[QUERY]:")[0]
    # The user content's `|` chars are replaced; only the section separator
    # ` | ` remains between turns. Count of ` | ` between U: and A:.
    assert "U: I want a / b / c" in history
    assert "A: ok" in history
    # Exactly one ` | ` separator between the two history turns.
    assert history.count(" | ") == 1


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


def test_format_track_text_unwraps_single_element_lists():
    """Sub 2 fix: HF catalog returns single-element lists for some fields
    (e.g. track_name=['A Forbidden Dance']). Must render as plain string,
    NOT as Python list-repr. Brackets in track text add ~6-8 noise tokens
    per track that the encoder has to learn to ignore."""
    from mcrs.retrieval_modules.bge_m3_format import format_track_text
    text = format_track_text(
        track_name=["A Forbidden Dance"],
        artist_name=["Alesana"],
        album_name=["A Place Where The Sun Is Silent"],
        release_date=["2011-10-18"],
        tag_list=["metalcore", "post-hardcore"],
    )
    # No Python list brackets in the output.
    assert "[" not in text and "]" not in text, \
        f"track text has list-repr brackets: {text!r}"
    assert "track_name: A Forbidden Dance" in text
    assert "artist_name: Alesana" in text
    assert "tag_list: metalcore, post-hardcore" in text


def test_format_track_text_handles_multi_element_artist_list():
    """For genuine multi-element lists (rare but real: collaborations),
    join with comma rather than rendering list-repr."""
    from mcrs.retrieval_modules.bge_m3_format import format_track_text
    text = format_track_text(
        track_name="Collab Song",
        artist_name=["Artist A", "Artist B"],
    )
    assert "artist_name: Artist A, Artist B" in text
    assert "[" not in text and "]" not in text


def test_format_query_text_default_max_history_is_4():
    """Sub 2 fix: max_history_turns 6 -> 4. Long [HISTORY]: blocks at 6 turns
    push p90 query length to 1772 chars, truncating [QUERY]: at the end with
    max_query_len=384. 4 turns keeps p90 below 1100 chars (within budget)."""
    import inspect
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    sig = inspect.signature(format_query_text)
    assert sig.parameters['max_history_turns'].default == 4, \
        f"format_query_text max_history_turns default should be 4 (Sub 2), got {sig.parameters['max_history_turns'].default}"


def test_format_track_text_pins_separator_and_field_order():
    """Exact-string pin for the 5-field track corpus format."""
    from mcrs.retrieval_modules.bge_m3_format import format_track_text
    text = format_track_text("T", "A", "Alb", "2020-01-01", ["t1", "t2"])
    assert text == (
        "track_name: T | artist_name: A | album_name: Alb | "
        "release_date: 2020-01-01 | tag_list: t1, t2"
    )
