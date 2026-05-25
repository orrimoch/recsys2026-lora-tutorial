"""session_match_features now lives in the shared session_history module so
training (scripts/build_lgbm_features.py) and inference
(mcrs/rerankers/lgbm_rerank.py) use the identical implementation. These tests
exercise the function via its canonical (shared) import path."""
from mcrs.retrieval_modules.session_history import session_match_features


def test_same_artist_and_album_flags_and_counts():
    played_meta = [{"artist_name": "A", "album_name": "X"},
                   {"artist_name": "A", "album_name": "Y"}]
    cand = {"artist_name": "A", "album_name": "Y"}
    f = session_match_features(cand, played_meta)
    assert f["same_artist"] == 1
    assert f["same_album"] == 1
    assert f["artist_in_session_count"] == 2


def test_no_match():
    f = session_match_features({"artist_name": "Z", "album_name": "Q", "tag_list": ["jazz"]},
                               [{"artist_name": "A", "album_name": "X", "tag_list": ["rock"]}])
    assert f == {"same_artist": 0, "same_album": 0,
                 "artist_in_session_count": 0, "session_tag_overlap": 0}


def test_session_tag_overlap_counts_shared_tags():
    cand = {"artist_name": "Z", "album_name": "Q", "tag_list": ["Rock", "Indie", "Chill"]}
    played = [{"artist_name": "A", "album_name": "X", "tag_list": ["rock"]},
              {"artist_name": "B", "album_name": "Y", "tag_list": ["indie", "pop"]}]
    f = session_match_features(cand, played)
    assert f["session_tag_overlap"] == 2   # rock + indie shared (case-insensitive)


def test_case_insensitive_and_whitespace_trimmed():
    cand = {"artist_name": "  Radiohead ", "album_name": "OK Computer"}
    played = [{"artist_name": "RADIOHEAD", "album_name": "ok computer"}]
    f = session_match_features(cand, played)
    assert f["same_artist"] == 1
    assert f["same_album"] == 1
    assert f["artist_in_session_count"] == 1


def test_empty_played_meta_returns_zeros():
    f = session_match_features({"artist_name": "A", "album_name": "X", "tag_list": ["rock"]}, [])
    assert f == {"same_artist": 0, "same_album": 0,
                 "artist_in_session_count": 0, "session_tag_overlap": 0}


def test_reexport_from_build_script_is_same_object():
    # The training script must re-export the SAME function object (DRY guarantee).
    from scripts.build_lgbm_features import session_match_features as smf_script
    assert smf_script is session_match_features
