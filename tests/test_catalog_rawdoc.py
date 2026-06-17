# tests/test_catalog_rawdoc.py
from mcrs.data.catalog import Catalog

ROW = {"track_id": "t1", "track_name": "Holocene", "artist_name": "Bon Iver",
       "album_name": "Bon Iver", "release_date": "2011-06-17", "tag_list": ["indie folk", "atmospheric"]}

def test_raw_doc_includes_tags():
    cat = Catalog([ROW])
    doc = cat.id_to_metadata("t1", enriched=False)
    assert "tags: indie folk, atmospheric" in doc
    assert "track_name: Holocene" in doc

def test_is_enriched():
    cat = Catalog([ROW], enriched_docs={"t1": "enriched..."})
    assert cat.is_enriched("t1") is True
    assert cat.is_enriched("t2") is False


# ── new edge-case / property / failure-mode tests ──────────────────────────

def test_list_field_renders_comma_joined():
    """A list-valued field (tag_list) is comma-joined in the raw doc."""
    cat = Catalog([ROW])
    doc = cat.id_to_metadata("t1", enriched=False)
    assert "indie folk, atmospheric" in doc


def test_none_field_renders_empty():
    """A field present in corpus_types but set to None renders as empty string (not 'None')."""
    row = {"track_id": "t2", "track_name": "Test", "artist_name": None,
           "album_name": "Album", "release_date": "2020-01-01", "tag_list": []}
    cat = Catalog([row])
    doc = cat.id_to_metadata("t2", enriched=False)
    assert "artist_name: " in doc
    assert "None" not in doc


def test_missing_key_renders_empty():
    """A corpus_types field entirely absent from the row renders as empty string."""
    row = {"track_id": "t3", "track_name": "Ghost Track"}  # no artist_name, album_name, etc.
    cat = Catalog([row], corpus_types=["track_name", "artist_name"])
    doc = cat.id_to_metadata("t3", enriched=False)
    # artist_name is missing from the row; .get() returns None -> rendered as empty
    assert "artist_name: " in doc
    assert "None" not in doc


def test_custom_corpus_types_excludes_tag_list():
    """corpus_types without 'tag_list' produces no 'tags:' field in the raw doc."""
    cat = Catalog([ROW], corpus_types=["track_name", "artist_name"])
    doc = cat.id_to_metadata("t1", enriched=False)
    assert "tags:" not in doc
    assert "track_name: Holocene" in doc
    assert "artist_name: Bon Iver" in doc


def test_is_enriched_unknown_id_returns_false():
    """is_enriched returns False for a track_id not in enriched_docs."""
    cat = Catalog([ROW])  # no enriched_docs
    assert cat.is_enriched("nonexistent") is False


def test_is_enriched_only_for_present_ids():
    """is_enriched returns True only for ids explicitly in enriched_docs."""
    cat = Catalog([ROW], enriched_docs={"t1": "enriched t1", "other": "enriched other"})
    assert cat.is_enriched("t1") is True
    assert cat.is_enriched("other") is True
    assert cat.is_enriched("t2") is False


def test_id_to_metadata_enriched_true_falls_back_to_raw_when_not_enriched():
    """id_to_metadata(enriched=True) returns the raw doc when the id has no enriched entry."""
    row2 = {"track_id": "t2", "track_name": "Re: Stacks", "artist_name": "Bon Iver",
            "album_name": "For Emma", "release_date": "2008-02-19", "tag_list": ["folk"]}
    # t1 has an enriched entry; t2 does not
    cat = Catalog([ROW, row2], enriched_docs={"t1": "enriched t1 doc"})
    assert cat.id_to_metadata("t1", enriched=True) == "enriched t1 doc"
    # t2 has no enriched entry — must fall back gracefully to raw doc
    raw_t2 = cat.id_to_metadata("t2", enriched=False)
    assert cat.id_to_metadata("t2", enriched=True) == raw_t2
    assert "Re: Stacks" in cat.id_to_metadata("t2", enriched=True)


def test_duplicate_rows_do_not_duplicate_index():
    """Inserting the same track_id twice keeps len() stable (no duplicate index entry)."""
    row_a = {"track_id": "dup", "track_name": "A", "artist_name": "X",
             "album_name": "Y", "release_date": "2020", "tag_list": []}
    row_b = {**row_a, "track_name": "B"}  # same track_id, different data
    cat = Catalog([row_a, row_b])
    assert len(cat) == 1
    assert len(cat.index_to_id) == 1
    # the second row's data should overwrite (latest wins)
    assert "B" in cat.id_to_metadata("dup", enriched=False)
