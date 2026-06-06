from scripts.build_lgbm_features import session_match_features


def test_extract_features_omits_degenerate_rank_inv_columns():
    # bm25_rank_inv / dense_meta_rank_inv / dense_lyrics_rank_inv were silent
    # duplicates of 1/wrrf_rank: WRRFRunner never populated the per-channel
    # ranks, so each collapsed to 1/wrrf_rank and fed the LGBM ranker three
    # collinear copies of its dominant feature. They must no longer be emitted.
    # (The serve mirror in lgbm_rerank.py stays f_idx-gated, so it auto-disables
    # them once the model is retrained without these columns.)
    import inspect
    from scripts import build_lgbm_features as blf
    src = inspect.getsource(blf.extract_features)
    # Check the emit form ("name":) so an explanatory comment that mentions the
    # names in prose doesn't trip the guard — only an actual emitted key fails.
    for dead in ("bm25_rank_inv", "dense_meta_rank_inv", "dense_lyrics_rank_inv"):
        assert f'"{dead}":' not in src, (
            f"{dead} is a degenerate 1/wrrf_rank duplicate — must not be emitted")


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
