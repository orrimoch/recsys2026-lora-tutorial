"""F1 — Catalog: id universe, metadata, doc text, enriched layering."""
from __future__ import annotations

import os

import pytest

from mcrs.data.catalog import Catalog

_ROWS = [
    {"track_id": "a", "track_name": ["Song A"], "artist_name": ["Artist X"],
     "album_name": ["Alb"], "release_date": "2001", "tag_list": ["pop", "rock"]},
    {"track_id": "b", "track_name": ["Song B"], "artist_name": ["Artist Y"],
     "album_name": ["Alb2"], "release_date": "2010", "tag_list": ["jazz"]},
]


def _cat(**kw):
    return Catalog(_ROWS, corpus_types=["track_name", "artist_name", "album_name", "release_date"], **kw)


def test_track_ids_is_the_universe():
    c = _cat()
    assert c.track_ids == frozenset({"a", "b"})
    assert len(c) == 2
    assert "a" in c and "zzz" not in c


def test_id_index_is_a_bijection():
    c = _cat()
    assert {c.index_to_id[c.id_to_index[t]] for t in c.track_ids} == {"a", "b"}
    assert len(c.id_to_index) == len(c.index_to_id) == 2


def test_metadata_returns_raw_row():
    assert _cat().metadata("a")["artist_name"] == ["Artist X"]


def test_id_to_metadata_joins_list_fields_over_corpus_types():
    doc = _cat().id_to_metadata("a")
    assert "Song A" in doc and "Artist X" in doc and "2001" in doc
    assert "Song B" not in doc


def test_id_to_metadata_enriched_falls_back_to_raw_when_uncovered():
    c = _cat()  # no enrichment provided
    assert c.id_to_metadata("a", enriched=True) == c.id_to_metadata("a", enriched=False)


def test_enriched_keys_are_canonicalized():
    # enriched_docs keyed by a raw 'track_id:'-prefixed / whitespaced id must still resolve under the
    # canonical id, else id_to_metadata silently falls back to the raw doc (ML-review finding #3).
    c = _cat(enriched_docs={"track_id: a ": "ENRICHED-A"})
    assert c.is_enriched("a") is True
    assert c.id_to_metadata("a", enriched=True) == "ENRICHED-A"


def test_id_to_metadata_enriched_uses_enrichment_when_present():
    c = _cat(enriched_docs={"a": "Song A — a soaring pop anthem about X"})
    out = c.id_to_metadata("a", enriched=True)
    assert "soaring pop anthem" in out
    # uncovered track still falls back gracefully
    assert c.id_to_metadata("b", enriched=True) == c.id_to_metadata("b", enriched=False)


@pytest.mark.skipif(
    not os.path.isdir("data/TalkPlayData-Challenge-Track-Metadata"),
    reason="catalog data not on disk",
)
def test_from_disk_loads_real_all_tracks():
    c = Catalog.from_disk("data/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    assert len(c) == 47071
    assert len(c.track_ids) == 47071  # no duplicate track_ids
