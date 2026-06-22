"""R5 — content-kNN (history), CF, same-artist channels. Cold => empty (0 RRF contribution)."""
from __future__ import annotations

from mcrs.data.catalog import Catalog
from mcrs.data.embeddings import TrackEmbeddings, UserEmbeddings
from mcrs.retrieval.personalization import CFChannel, ContentKNNChannel, SameArtistChannel


def test_content_knn_returns_neighbors_of_history_and_empty_when_cold():
    te = TrackEmbeddings([
        {"track_id": "a", "m": [1.0, 0.0]},
        {"track_id": "b", "m": [0.0, 1.0]},
        {"track_id": "c", "m": [0.9, 0.1]},
    ])
    ch = ContentKNNChannel(te, modality="m")
    warm = ch.batch_text_to_item_retrieval(["x"], topk=3, batch_context=[{"history_tids": ["a"]}])
    assert warm[0].index("c") < warm[0].index("b")     # c (near a) ranks above b
    cold = ch.batch_text_to_item_retrieval(["x"], topk=3, batch_context=[{"history_tids": []}])
    assert cold[0] == []


def test_content_knn_label_override_for_multiple_modalities():
    te = TrackEmbeddings([{"track_id": "a", "m": [1.0, 0.0]}])
    assert ContentKNNChannel(te, modality="m").label == "content_knn"          # default
    assert ContentKNNChannel(te, modality="m", label="cknn_audio").label == "cknn_audio"


def test_colisten_cf_channel_retrieves_session_neighbors_in_cf_space():
    """Idea 1: ContentKNNChannel on cf-bpr = co-listen item->item from the in-session plays (pool the
    played tracks' cf-bpr vectors, retrieve their behavioral neighbors) — distinct from CFChannel's
    user->item. A track near the session in cf-bpr space ranks above an unrelated one; cold -> []."""
    te = TrackEmbeddings([
        {"track_id": "p1", "cf-bpr": [1.0, 0.0]},     # played
        {"track_id": "p2", "cf-bpr": [0.8, 0.2]},     # played (similar direction)
        {"track_id": "near", "cf-bpr": [0.9, 0.1]},   # behavioral neighbor of the session
        {"track_id": "far", "cf-bpr": [0.0, 1.0]},    # unrelated
    ])
    ch = ContentKNNChannel(te, modality="cf-bpr", label="colisten_cf")
    assert ch.label == "colisten_cf"
    out = ch.batch_text_to_item_retrieval(["x"], topk=4, batch_context=[{"history_tids": ["p1", "p2"]}])
    assert out[0].index("near") < out[0].index("far")   # co-listen neighbor above the unrelated track
    assert ch.batch_text_to_item_retrieval(["x"], topk=4, batch_context=[{"history_tids": []}])[0] == []


def test_cf_uses_user_vector_and_empty_for_missing_user():
    ue = UserEmbeddings([{"user_id": "u1", "cf-bpr": [1.0, 0.0]}])
    te = TrackEmbeddings([{"track_id": "a", "cf-bpr": [1.0, 0.0]},
                          {"track_id": "b", "cf-bpr": [0.0, 1.0]}])
    ch = CFChannel(ue, te)
    assert ch.batch_text_to_item_retrieval(["x"], topk=2, user_ids=["u1"])[0][0] == "a"
    assert ch.batch_text_to_item_retrieval(["x"], topk=2, user_ids=["ghost"])[0] == []


def test_same_artist_returns_other_tracks_by_history_artists():
    cat = Catalog([
        {"track_id": "a", "artist_name": ["Nirvana"]},
        {"track_id": "b", "artist_name": ["Nirvana"]},
        {"track_id": "c", "artist_name": ["Other"]},
    ])
    ch = SameArtistChannel(cat)
    out = ch.batch_text_to_item_retrieval(["x"], topk=5, batch_context=[{"history_tids": ["a"]}])
    assert out[0] == ["b"]                              # other Nirvana track; excludes history "a" and "c"
    assert ch.batch_text_to_item_retrieval(["x"], topk=5, batch_context=[{"history_tids": []}])[0] == []
