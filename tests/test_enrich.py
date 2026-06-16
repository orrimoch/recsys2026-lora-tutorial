"""A1 — catalog enrichment / doc2query (pure helpers + injected-generator driver)."""
from __future__ import annotations

from mcrs.enrich.doc2query import (
    build_enrich_prompt,
    clean_enrichment,
    enrich_catalog,
    enriched_document,
    meta_text,
)

_META = {"track_id": "a", "track_name": ["Heart Shaped Box"], "artist_name": ["Nirvana"],
         "album_name": ["In Utero"], "tag_list": ["grunge", "90s"], "release_date": "1993-09-21"}


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


def test_enrich_catalog_with_injected_generator_covers_rows():
    rows = [_META, {"track_id": "track_id: b", "track_name": ["Take Five"],
                    "artist_name": ["Brubeck"], "album_name": ["Time Out"], "release_date": "1959"}]
    gen = lambda system, user: "query one\nquery two"      # fake LLM
    docs = enrich_catalog(rows, gen, n_requests=2)
    assert set(docs) == {"a", "b"}                          # canonical ids (prefix stripped)
    assert "query one" in docs["a"] and "Nirvana" in docs["a"]
