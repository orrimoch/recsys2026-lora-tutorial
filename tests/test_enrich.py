"""A1 — catalog enrichment / doc2query (pure helpers + injected-generator driver)."""
from __future__ import annotations

import threading

from mcrs.enrich.doc2query import (
    build_enrich_prompt,
    clean_enrichment,
    enrich_catalog,
    enrich_catalog_concurrent,
    enriched_document,
    meta_text,
)

_META = {"track_id": "a", "track_name": ["Heart Shaped Box"], "artist_name": ["Nirvana"],
         "album_name": ["In Utero"], "tag_list": ["grunge", "90s"], "release_date": "1993-09-21"}


def test_enrich_prompt_has_hallucination_grounding_clause():
    # Spec §4.1 hallucination guard: the system prompt must force grounding strictly in the provided
    # metadata + tags, and (for unknown tracks) invent no specific facts — else fabricated eras/
    # collaborators/dates get baked into the indexed, gold-bearing doc text (ML-review T2 #3).
    system, _ = build_enrich_prompt(_META)
    s = system.lower()
    assert "ground" in s                       # ground strictly in the given metadata/tags
    assert "invent" in s                        # invent no specific facts
    assert "date" in s and "collaborat" in s    # bans fabricated dates/collaborators (and labels/charts)


def test_meta_text_flattens_list_fields_and_year():
    t = meta_text(_META)
    assert "Heart Shaped Box" in t and "Nirvana" in t and "grunge" in t and "1993" in t


def test_clean_enrichment_strips_markdown_and_newlines():
    raw = "1. upbeat grunge anthem\n- 90s alternative rock\n* songs like Nirvana\n"
    out = clean_enrichment(raw)
    assert "\n" not in out and "-" not in out.split()[0]
    assert "upbeat grunge anthem" in out and "90s alternative rock" in out and "songs like Nirvana" in out


def test_enriched_document_is_base_plus_cleaned_expansion():
    doc = enriched_document(_META, "moody 90s grunge for angsty afternoons")
    assert "Nirvana" in doc and "moody 90s grunge" in doc


def test_build_prompt_returns_system_and_user_with_metadata():
    system, user = build_enrich_prompt(_META, n_requests=4)
    assert isinstance(system, str) and "Nirvana" in user


def test_prompt_steers_toward_attributes_intent_not_title_restatement():
    system, _ = build_enrich_prompt(_META, n_requests=4)
    s = system.lower()
    assert "attribute" in s or "intent" in s           # focus on attributes/intent
    assert "title" in s                                # explicitly mentions not relying on the title


def test_build_prompt_few_shot_examples_anchor_the_style():
    examples = ["something upbeat for a workout", "more songs like Radiohead but mellower"]
    system, user = build_enrich_prompt(_META, n_requests=4, examples=examples)
    prompt = system + user
    assert "something upbeat for a workout" in prompt
    assert "more songs like Radiohead but mellower" in prompt


def test_enrich_catalog_with_injected_generator_covers_rows():
    rows = [_META, {"track_id": "track_id: b", "track_name": ["Take Five"],
                    "artist_name": ["Brubeck"], "album_name": ["Time Out"], "release_date": "1959"}]
    gen = lambda system, user: "query one\nquery two"      # fake LLM
    docs = enrich_catalog(rows, gen, n_requests=2)
    assert set(docs) == {"a", "b"}                          # canonical ids (prefix stripped)
    assert "query one" in docs["a"] and "Nirvana" in docs["a"]


_ROWS = [
    _META,
    {"track_id": "track_id: b", "track_name": ["Take Five"], "artist_name": ["Brubeck"],
     "album_name": ["Time Out"], "release_date": "1959"},
    {"track_id": "c", "track_name": ["Clair de Lune"], "artist_name": ["Debussy"]},
]


def test_enrich_catalog_concurrent_matches_serial_result():
    gen = lambda system, user: "query one\nquery two"       # fake LLM
    serial = enrich_catalog(_ROWS, gen, n_requests=2)
    concurrent = enrich_catalog_concurrent(_ROWS, gen, n_requests=2, max_workers=4)
    assert concurrent == serial                             # same ids, same enriched docs


def test_enrich_catalog_concurrent_resumes_from_done():
    gen = lambda system, user: "fresh"
    done = {"a": "already enriched doc"}                    # canonical id already present
    docs = enrich_catalog_concurrent(_ROWS, gen, n_requests=2, max_workers=4, done=done)
    assert docs["a"] == "already enriched doc"             # untouched, not regenerated
    assert set(docs) == {"a", "b", "c"} and "fresh" in docs["b"]


def test_enrich_catalog_concurrent_runs_requests_in_parallel():
    # Barrier of 3 only releases when 3 requests are in flight at once; a serial loop would
    # never get a 2nd request started, so wait() would time out and break the barrier.
    barrier = threading.Barrier(3, timeout=5)

    def gen(system, user):
        barrier.wait()                                      # raises BrokenBarrierError if serial
        return "ok"

    docs = enrich_catalog_concurrent(_ROWS, gen, n_requests=2, max_workers=3)
    assert set(docs) == {"a", "b", "c"}
