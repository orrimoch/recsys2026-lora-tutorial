"""EXP-015 — artist-hypothesis generative recall (free local Qwen).

Pure-logic tests (no model load): the LLM names candidate ARTISTS for a listener,
we ground them by exact artist_name -> catalog tracks. These test the prompt build,
robust artist-list parsing, name normalization (accents / 'the' / punctuation), and
the gold-artist hit check that the feasibility probe and the channel both rely on.
"""
from mcrs.query_rewriters.artist_hypothesis import (
    build_artist_prompt,
    parse_artist_list,
    normalize_artist,
    artist_hit,
)


# ---- prompt -------------------------------------------------------------------
def test_prompt_carries_goal_culture_liked_artists_and_n():
    sys, user = build_artist_prompt(
        conversation="user: something dreamy and slow",
        goal="more shoegaze like Slowdive",
        liked_artists=["Slowdive", "Cocteau Twins"],
        culture="Anglo-American Rock",
        n=15)
    assert "shoegaze like Slowdive" in user
    assert "Cocteau Twins" in user            # liked artists seeded
    assert "Anglo-American Rock" in user       # culture seeded
    assert "15" in user                        # the requested count
    assert isinstance(sys, str) and sys        # a non-empty system instruction


def test_prompt_degrades_without_optional_signals():
    sys, user = build_artist_prompt(conversation="user: play something", goal="")
    assert isinstance(user, str) and user      # no crash with no goal/liked/culture


# ---- parse --------------------------------------------------------------------
def test_parse_numbered_list():
    assert parse_artist_list("1. Ride\n2. Lush\n3. Chapterhouse") == ["Ride", "Lush", "Chapterhouse"]


def test_parse_comma_and_bullets_and_quotes():
    assert parse_artist_list("- Ride\n- \"Lush\"\n- Chapterhouse") == ["Ride", "Lush", "Chapterhouse"]
    assert parse_artist_list("Ride, Lush, Chapterhouse") == ["Ride", "Lush", "Chapterhouse"]


def test_parse_dedupes_case_insensitively_and_caps():
    out = parse_artist_list("1. Ride\n2. ride\n3. RIDE\n4. Lush", max_n=2)
    assert out == ["Ride", "Lush"]            # dedupe (keep first casing) + cap to max_n


def test_parse_strips_trailing_notes_and_empty():
    assert parse_artist_list("1. Ride - dreamy guitars\n2.\n3. Lush") == ["Ride", "Lush"]


# ---- normalize ----------------------------------------------------------------
def test_normalize_drops_accents_the_and_punctuation():
    assert normalize_artist("The Beatles") == "beatles"
    assert normalize_artist("Sigur Rós") == "sigur ros"
    assert normalize_artist("Beyoncé") == "beyonce"
    assert normalize_artist("  Radiohead  ") == "radiohead"


# ---- artist_hit ---------------------------------------------------------------
def test_artist_hit_normalized_match():
    gen = ["Ride", "My Bloody Valentine", "The Lush"]
    assert artist_hit("Lush", gen) is True            # "The Lush" -> "lush" == "lush"
    assert artist_hit("Slowdive", gen) is False


def test_artist_hit_accent_and_case_insensitive():
    assert artist_hit("Sigur Rós", ["sigur ros", "mum"]) is True


def test_artist_hit_handles_list_valued_gold():
    # catalog artist_name fields are sometimes list-valued
    assert artist_hit(["Radiohead"], ["radiohead"]) is True
    assert artist_hit(None, ["radiohead"]) is False
