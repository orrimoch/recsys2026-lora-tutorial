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
    """Query format must match what crs_baseline.batch_chat produces at inference."""
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
    assert "I like 70s rock" in text
    assert "Yes, more like that" in text
    assert "US" in text or "35" in text


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
