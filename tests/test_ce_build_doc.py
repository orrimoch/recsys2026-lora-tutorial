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
