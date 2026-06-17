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
