import pytest
from mcrs.data.catalog import Catalog
from mcrs.training.ce_data import build_doc

ROW = {"track_id": "t1", "track_name": "A", "artist_name": "B", "album_name": "C",
       "release_date": "2011", "tag_list": ["x"]}

def test_build_doc_uses_enriched_and_char_caps():
    cat = Catalog([ROW], enriched_docs={"t1": "E" * 5000})
    assert build_doc(cat, "t1", max_doc_chars=2000) == "E" * 2000

def test_build_doc_hard_fails_on_enriched_missing():
    cat = Catalog([ROW])  # no enriched_docs
    with pytest.raises(KeyError):
        build_doc(cat, "t1")


# ---------------------------------------------------------------------------
# Hardening tests
# ---------------------------------------------------------------------------

def test_build_doc_short_doc_not_padded():
    """An enriched doc shorter than max_doc_chars must be returned verbatim (not padded)."""
    cat = Catalog([ROW], enriched_docs={"t1": "short doc"})
    result = build_doc(cat, "t1", max_doc_chars=2000)
    assert result == "short doc"              # shorter than cap, no padding


def test_build_doc_exactly_at_cap():
    """A doc whose length equals max_doc_chars must be returned in full (no off-by-one trim)."""
    doc = "X" * 2000
    cat = Catalog([ROW], enriched_docs={"t1": doc})
    assert build_doc(cat, "t1", max_doc_chars=2000) == doc


def test_build_doc_train_equals_serve():
    """build_doc called twice for the same track must return the same string (train==serve)."""
    cat = Catalog([ROW], enriched_docs={"t1": "enriched content " * 200})
    r1 = build_doc(cat, "t1")
    r2 = build_doc(cat, "t1")
    assert r1 == r2


def test_build_doc_uses_enriched_not_raw():
    """build_doc must return the enriched doc, not the raw metadata string."""
    cat = Catalog([ROW], enriched_docs={"t1": "enriched-specific-content"})
    result = build_doc(cat, "t1")
    # The enriched doc is returned, not the raw metadata (which would contain "track_name: A")
    assert result == "enriched-specific-content"
    assert "track_name" not in result
