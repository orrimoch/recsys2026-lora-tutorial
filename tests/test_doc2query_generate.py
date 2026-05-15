"""Tests for the doc2query pure-function units (prompt + parser).

The generation loop itself (LLM call, batching) is wiring — not TDD'd here.
We test the two parts most likely to bug:
  - build_doc2query_prompt: given a track metadata dict, returns the prompt
  - parse_generated_queries: given the raw LLM completion, extracts the 5 queries
"""


# ---- build_doc2query_prompt ----

def test_build_doc2query_prompt_includes_track_name_and_artist():
    from scripts.doc2query_generate import build_doc2query_prompt

    metadata = {
        "track_name": ["Bohemian Rhapsody"],
        "artist_name": ["Queen"],
        "album_name": ["A Night at the Opera"],
        "tag_list": ["rock", "classic rock"],
        "release_date": "1975",
    }
    prompt = build_doc2query_prompt(metadata, n_queries=5)

    assert "Bohemian Rhapsody" in prompt
    assert "Queen" in prompt
    assert "5" in prompt  # asks for 5 queries
    assert "conversational" in prompt.lower()


def test_build_doc2query_prompt_handles_missing_fields():
    from scripts.doc2query_generate import build_doc2query_prompt

    metadata = {"track_name": ["Yesterday"], "artist_name": ["The Beatles"]}
    prompt = build_doc2query_prompt(metadata, n_queries=3)

    assert "Yesterday" in prompt
    assert "The Beatles" in prompt
    assert "3" in prompt


def test_build_doc2query_prompt_skips_empty_lists():
    from scripts.doc2query_generate import build_doc2query_prompt

    metadata = {"track_name": ["Imagine"], "tag_list": []}
    prompt = build_doc2query_prompt(metadata, n_queries=5)

    # The empty tag_list shouldn't render as "Tags: " (empty value)
    assert "Imagine" in prompt
    assert "Tags: \n" not in prompt and "tag_list: \n" not in prompt


# ---- parse_generated_queries ----

def test_parse_generated_queries_extracts_numbered_list():
    from scripts.doc2query_generate import parse_generated_queries

    completion = (
        "1. play me something melancholy for a rainy day\n"
        "2. a sad acoustic ballad from the 60s\n"
        "3. classic Beatles songs about regret\n"
        "4. something I can cry to on Sunday morning\n"
        "5. melancholy Paul McCartney solo work"
    )
    queries = parse_generated_queries(completion, n_expected=5)

    assert len(queries) == 5
    assert queries[0] == "play me something melancholy for a rainy day"
    assert queries[-1] == "melancholy Paul McCartney solo work"


def test_parse_generated_queries_strips_dash_or_bullet_prefixes():
    from scripts.doc2query_generate import parse_generated_queries

    completion = (
        "- play something upbeat\n"
        "- music for a workout\n"
        "* high-energy pop hits\n"
    )
    queries = parse_generated_queries(completion, n_expected=3)

    assert queries == [
        "play something upbeat",
        "music for a workout",
        "high-energy pop hits",
    ]


def test_parse_generated_queries_handles_preamble_before_list():
    """LLM often emits 'Here are 5 queries:' before the actual list — strip it."""
    from scripts.doc2query_generate import parse_generated_queries

    completion = (
        "Here are 5 conversational queries:\n\n"
        "1. play me indie rock from the 2000s\n"
        "2. moody alternative songs\n"
        "3. driving music for late night\n"
        "4. introspective post-rock\n"
        "5. atmospheric instrumental tracks"
    )
    queries = parse_generated_queries(completion, n_expected=5)

    assert len(queries) == 5
    assert queries[0] == "play me indie rock from the 2000s"


def test_parse_generated_queries_drops_blank_lines():
    from scripts.doc2query_generate import parse_generated_queries

    completion = "1. a\n\n2. b\n\n  \n3. c\n"
    queries = parse_generated_queries(completion, n_expected=3)
    assert queries == ["a", "b", "c"]


def test_parse_generated_queries_returns_at_most_n_expected():
    """Even if the LLM over-generates, we cap at n_expected."""
    from scripts.doc2query_generate import parse_generated_queries

    completion = "1. q1\n2. q2\n3. q3\n4. q4\n5. q5\n6. extra\n7. another"
    queries = parse_generated_queries(completion, n_expected=5)
    assert len(queries) == 5


def test_parse_generated_queries_returns_what_it_can_if_under_target():
    """If LLM emits fewer than n_expected, return what's there (don't error)."""
    from scripts.doc2query_generate import parse_generated_queries

    completion = "1. only one"
    queries = parse_generated_queries(completion, n_expected=5)
    assert queries == ["only one"]
