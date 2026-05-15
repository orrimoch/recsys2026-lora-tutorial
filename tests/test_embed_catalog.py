"""Tests for the catalog-embedding doc-text builder.

The embedding step builds one text string per track from the catalog metadata.
Test the construction logic since it directly affects what BGE-M3 / Qwen3-4B
sees and what they retrieve against.
"""


def test_build_doc_text_concatenates_named_fields():
    """Default fields concatenate as 'field: value' separated by ' | '."""
    from scripts.embed_catalog import build_doc_text

    metadata = {
        "track_name": ["Bohemian Rhapsody"],
        "artist_name": ["Queen"],
        "album_name": ["A Night at the Opera"],
        "tag_list": ["rock", "classic rock", "1970s"],
    }
    text = build_doc_text(metadata, fields=["track_name", "artist_name", "album_name", "tag_list"])

    assert text == (
        "track_name: Bohemian Rhapsody | "
        "artist_name: Queen | "
        "album_name: A Night at the Opera | "
        "tag_list: rock, classic rock, 1970s"
    )


def test_build_doc_text_skips_missing_fields_silently():
    """If a requested field is absent from the row, skip it (don't crash)."""
    from scripts.embed_catalog import build_doc_text

    metadata = {"track_name": ["Yesterday"], "artist_name": ["The Beatles"]}
    text = build_doc_text(metadata, fields=["track_name", "artist_name", "album_name"])

    assert text == "track_name: Yesterday | artist_name: The Beatles"


def test_build_doc_text_skips_empty_lists():
    """Empty lists shouldn't render as 'field: ' (empty value); skip the field."""
    from scripts.embed_catalog import build_doc_text

    metadata = {"track_name": ["Imagine"], "tag_list": []}
    text = build_doc_text(metadata, fields=["track_name", "tag_list"])

    assert text == "track_name: Imagine"


def test_build_doc_text_handles_string_fields_as_well_as_lists():
    """release_date is often a scalar string (not a list) — should still work."""
    from scripts.embed_catalog import build_doc_text

    metadata = {"track_name": ["Smells Like Teen Spirit"], "release_date": "1991"}
    text = build_doc_text(metadata, fields=["track_name", "release_date"])

    assert text == "track_name: Smells Like Teen Spirit | release_date: 1991"
