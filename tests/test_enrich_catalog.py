"""EXP-016 — doc2query catalog enrichment (document expansion).

Pure-logic tests (no API, no model): the metadata->text, the enrichment prompt
(rich description + example listener-requests, with anti-hallucination guard), the
cleanup of the model's output, the APPEND (expansion, not replacement) of enrichment
onto the original doc, and the resumable "which tracks are still pending" selection.
The Gemini call + the 47k loop + parquet checkpointing are exercised in the notebook.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import enrich_catalog as ec  # noqa: E402

META = {"track_id": "t1", "track_name": "Around the World", "artist_name": "Daft Punk",
        "album_name": "Homework", "tag_list": ["house", "electronic", "dance", "90s"],
        "release_date": "1997-01-20", "popularity": 82}


def test_meta_text_has_core_fields_and_tags():
    s = ec.meta_text(META)
    assert "Daft Punk" in s and "Around the World" in s and "Homework" in s
    assert "house" in s and "1997" in s


def test_meta_text_tolerates_missing_and_list_fields():
    s = ec.meta_text({"track_id": "x", "track_name": ["Solo"], "artist_name": None,
                      "album_name": "", "tag_list": []})
    assert "Solo" in s  # list-valued name flattened; no crash on None/empty


def test_build_enrich_prompt_carries_metadata_asks_for_requests_and_guards_hallucination():
    system, user = ec.build_enrich_prompt(META, n_requests=4)
    assert "Daft Punk" in user and "house" in user
    assert "4" in user                                  # n example requests requested
    assert system                                       # non-empty system
    assert any(w in system.lower() for w in ("do not invent", "not invent", "don't invent",
                                             "no fake", "without inventing"))


def test_clean_enrichment_strips_markdown_collapses_and_caps():
    raw = "**Description:**\n- energetic French house\n* dance-party anthem\n\n## Requests\n"
    out = ec.clean_enrichment(raw)
    assert "**" not in out and "##" not in out and "\n" not in out
    assert out.strip()
    assert len(ec.clean_enrichment("x " * 5000, max_chars=1500)) <= 1500


def test_enriched_document_appends_keeps_original():
    enr = "energetic French house, late-90s; for a dance party; similar to Stardust, Cassius"
    doc = ec.enriched_document(META, enr)
    assert "Daft Punk" in doc and "Around the World" in doc   # original preserved (exact match kept)
    assert "dance party" in doc                                # enrichment appended
    assert doc.index("Daft Punk") < doc.index("dance party")  # original first, expansion after


def test_enriched_document_empty_enrichment_is_just_metadata():
    assert ec.enriched_document(META, "") == ec.meta_text(META)


def test_select_pending_skips_already_done():
    rows = [{"track_id": "a"}, {"track_id": "b"}, {"track_id": "c"}]
    assert [r["track_id"] for r in ec.select_pending(rows, {"b"})] == ["a", "c"]
    assert ec.select_pending(rows, {"a", "b", "c"}) == []
