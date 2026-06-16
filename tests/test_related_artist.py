"""R6 — related-artist channel: cross-session artist co-occurrence → NEW artists' tracks."""
from __future__ import annotations

from mcrs.data.catalog import Catalog
from mcrs.retrieval.related_artist import (
    RelatedArtistChannel,
    build_artist_cooc,
    tid_to_artists_from_catalog,
)

# catalog: A has 2 tracks (pop-ordered), B/C/D one each
_ROWS = [
    {"track_id": "t1", "artist_name": ["A"], "popularity": 5},
    {"track_id": "t2", "artist_name": ["A"], "popularity": 1},
    {"track_id": "t3", "artist_name": ["B"], "popularity": 9},
    {"track_id": "t4", "artist_name": ["C"], "popularity": 2},
    {"track_id": "t5", "artist_name": ["D"], "popularity": 3},
]


def _sess(*tids):
    return {"conversations": [{"role": "music", "content": t} for t in tids]}


def test_build_cooc_counts_shared_sessions_ordered_pairs():
    t2a = {"t1": ["a"], "t3": ["b"], "t4": ["c"]}
    sessions = [_sess("t1", "t3"), _sess("t1", "t4"), _sess("t1", "t3")]  # A-B twice, A-C once
    cooc = build_artist_cooc(sessions, t2a)
    assert cooc["a"]["b"] == 2 and cooc["a"]["c"] == 1
    assert cooc["b"]["a"] == 2                          # symmetric counting


def test_build_cooc_dedups_within_a_session():
    t2a = {"t1": ["a"], "t3": ["b"]}
    cooc = build_artist_cooc([_sess("t1", "t1", "t3")], t2a)   # A played twice in one session
    assert cooc["a"]["b"] == 1                          # not 2 — de-dup per session


def test_expands_to_new_cooccurring_artists_ranked_by_count():
    cat = Catalog(_ROWS)
    cooc = {"a": {"b": 2, "c": 1}}                       # from A: B stronger than C
    ch = RelatedArtistChannel(cat, cooc)
    out = ch.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[{"history_tids": ["t1"]}])
    assert out[0] == ["t3", "t4"]                        # B(2) before C(1); A's own tracks excluded
    assert "t1" not in out[0] and "t2" not in out[0]    # seen artist A not re-emitted
    assert "t5" not in out[0]                            # D has no co-occurrence with A


def test_empty_history_returns_empty_list():
    ch = RelatedArtistChannel(Catalog(_ROWS), {"a": {"b": 1}})
    assert ch.batch_text_to_item_retrieval(["q"], topk=5, batch_context=[{"history_tids": []}]) == [[]]


def test_topk_truncates_and_excludes_played():
    cat = Catalog(_ROWS)
    ch = RelatedArtistChannel(cat, {"a": {"b": 2, "c": 1}})
    out = ch.batch_text_to_item_retrieval(["q"], topk=1, batch_context=[{"history_tids": ["t1"]}])
    assert out[0] == ["t3"]                              # only the top co-occurring artist's track


def test_tid_to_artists_from_catalog_normalizes():
    t2a = tid_to_artists_from_catalog(Catalog(_ROWS))
    assert t2a["t1"] == ["a"] and t2a["t3"] == ["b"] and t2a["t5"] == ["d"]


def test_returns_canonical_ids_from_prefixed_history():
    cat = Catalog(_ROWS)
    ch = RelatedArtistChannel(cat, {"a": {"b": 1}})
    out = ch.batch_text_to_item_retrieval(["q"], topk=5, batch_context=[{"history_tids": ["track_id: t1"]}])
    assert out[0] == ["t3"]                              # prefixed history canonicalized -> artist A seen
