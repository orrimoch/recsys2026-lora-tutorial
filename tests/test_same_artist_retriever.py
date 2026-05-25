from mcrs.retrieval_modules.same_artist import SameArtistRetriever

class _StubCatalog:
    meta = {
        "t1": {"artist_name": "A", "popularity": 5.0},
        "t2": {"artist_name": "A", "popularity": 9.0},
        "t3": {"artist_name": "B", "popularity": 1.0},
        "t4": {"artist_name": "C", "popularity": 1.0},
    }
    metadata_dict = meta

def _build():
    r = SameArtistRetriever.__new__(SameArtistRetriever)
    r._build_index(_StubCatalog())
    return r

def test_returns_unplayed_tracks_by_session_artists_excludes_played():
    r = _build()
    ctx = {"history_tids": ["t1"]}
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[ctx])[0]
    assert out == ["t2"]

def test_ranks_by_artist_session_count_then_popularity():
    r = _build()
    ctx = {"history_tids": ["t1", "t3"]}
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[ctx])[0]
    assert out == ["t2"]

def test_no_history_returns_empty():
    r = _build()
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[{}])[0]
    assert out == []
