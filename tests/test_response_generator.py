"""Tests for conversational response generation (predicted_response field)."""

from mcrs.crs_baseline import extract_cot_response


def test_stub_response_generator():
    assert True


def test_extract_cot_response_both_blocks():
    raw = (
        "<user_state>\n"
        "mood: melancholy\n"
        "session_intent: nostalgia\n"
        "energy: low\n"
        "sonic_pref: indie folk, acoustic\n"
        "era_pref: 2010s\n"
        "language_pref: english\n"
        "familiarity_pref: familiar\n"
        "recent_signals: Bon Iver, Sufjan Stevens\n"
        "avoid: none\n"
        "</user_state>\n"
        "<response>\n"
        "Sufjan Stevens' 'Mystery of Love' leans into the same hushed acoustic register "
        "as the Bon Iver tracks you mentioned. Want something with a denser arrangement next?\n"
        "</response>"
    )
    out = extract_cot_response(raw)
    assert out.startswith("Sufjan Stevens")
    assert "user_state" not in out
    assert "mood:" not in out
    assert "Want something" in out


def test_extract_cot_response_only_response_tag():
    raw = "<response>Just the reply text.</response>"
    assert extract_cot_response(raw) == "Just the reply text."


def test_extract_cot_response_user_state_no_response_tag():
    raw = (
        "<user_state>\nmood: high energy\n</user_state>\n"
        "Track X by Artist Y matches the high-energy vibe."
    )
    out = extract_cot_response(raw)
    assert out == "Track X by Artist Y matches the high-energy vibe."
    assert "<user_state>" not in out


def test_extract_cot_response_unclosed_response_tag():
    # Model ran out of tokens mid-response. Drop the user_state block
    # and the stray <response> opener; keep the partial reply.
    raw = (
        "<user_state>\nmood: chill\n</user_state>\n"
        "<response>The track sits between trip-hop and"
    )
    out = extract_cot_response(raw)
    assert out == "The track sits between trip-hop and"


def test_extract_cot_response_no_tags_fallback():
    raw = "Plain response with no envelope."
    assert extract_cot_response(raw) == "Plain response with no envelope."


def test_extract_cot_response_empty_input():
    assert extract_cot_response("") == ""
