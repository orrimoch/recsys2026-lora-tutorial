"""R3 — BM25 sparse channel (F2 RetrievalChannel)."""
from __future__ import annotations

import os

import pytest

from mcrs.data.catalog import Catalog
from mcrs.retrieval.bm25_channel import BM25Channel

_ROWS = [
    {"track_id": "a", "track_name": ["Heart Shaped Box"], "artist_name": ["Nirvana"],
     "album_name": ["In Utero"], "release_date": "1993"},
    {"track_id": "b", "track_name": ["Take Five"], "artist_name": ["Dave Brubeck"],
     "album_name": ["Time Out"], "release_date": "1959"},
    {"track_id": "c", "track_name": ["So What"], "artist_name": ["Miles Davis"],
     "album_name": ["Kind of Blue"], "release_date": "1959"},
]


def test_bm25_implements_channel_and_returns_canonical_ids():
    ch = BM25Channel(Catalog(_ROWS))
    out = ch.batch_text_to_item_retrieval(["nirvana heart shaped box"], topk=3)
    assert isinstance(out, list) and len(out) == 1
    assert out[0][0] == "a"                       # best match ranked first
    assert set(out[0]) <= {"a", "b", "c"}         # canonical catalog ids only


def test_bm25_topk_respected_and_one_list_per_query():
    ch = BM25Channel(Catalog(_ROWS))
    out = ch.batch_text_to_item_retrieval(["miles davis", "dave brubeck take five"], topk=2)
    assert len(out) == 2 and all(len(r) <= 2 for r in out)
    assert out[0][0] == "c" and out[1][0] == "b"


@pytest.mark.skipif(not os.path.isdir("data/TalkPlayData-Challenge-Track-Metadata"),
                    reason="catalog not on disk")
def test_bm25_on_real_catalog_slice_returns_valid_ids():
    from datasets import load_from_disk
    rows = load_from_disk("data/TalkPlayData-Challenge-Track-Metadata")["all_tracks"].select(range(300))
    cat = Catalog(rows)
    ch = BM25Channel(cat)
    q = cat.id_to_metadata(cat.index_to_id[0])  # query with a real doc → itself should rank high
    out = ch.batch_text_to_item_retrieval([q], topk=10)
    assert len(out[0]) == 10 and set(out[0]) <= cat.track_ids
    assert cat.index_to_id[0] in out[0]
