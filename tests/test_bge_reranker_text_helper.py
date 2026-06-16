from mcrs.rerankers.bge_reranker import build_tid_text_map

def test_build_tid_text_map_pipe_joins_and_flattens_tags():
    md = {"t1": {"track_name": "Song A", "artist_name": "Artist X",
                 "album_name": "Alb", "tag_list": ["indie", "melancholic"]},
          "t2": {"track_name": "Song B", "artist_name": "Artist Y",
                 "album_name": None, "tag_list": []}}
    out = build_tid_text_map(md, ["track_name", "artist_name", "album_name", "tag_list"])
    assert out["t1"] == "Song A | Artist X | Alb | indie, melancholic"
    assert out["t2"] == "Song B | Artist Y"   # empty/None dropped
