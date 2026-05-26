"""Regression guard: the LGBM reranker's inference-side track-metadata loader
must carry EVERY field that the shared session_match_features consumes —
crucially album_name. Otherwise same_album is computed (correctly, from real
album names) during training in scripts/build_lgbm_features.py but is silently
ALWAYS 0 at inference, because _load_track_meta dropped album_name. That is the
exact train/serve feature skew FIX 0 + FIX C2 set out to eliminate.

These tests exercise the pure row-flattener that _load_track_meta uses, so they
run without loading HF datasets or a LightGBM booster.
"""
import numpy as np

from mcrs.rerankers.lgbm_rerank import LGBM_RERANKER, _flatten_track_row
from mcrs.retrieval_modules.session_history import session_match_features


# Every metadata field session_match_features reads off a candidate / played
# track. If the inference loader omits any of these, the matching feature is
# dead at serve time.
SESSION_MATCH_META_KEYS = {"artist_name", "album_name", "tag_list"}


def test_flatten_track_row_carries_all_session_match_keys():
    # HF rows often store these fields as single-element lists.
    raw = {
        "track_id": "t1",
        "artist_name": ["Radiohead"],
        "album_name": ["OK Computer"],
        "tag_list": ["rock", "alt"],
        "popularity": 10.0,
        "release_date": "1997",
    }
    flat = _flatten_track_row(raw)
    missing = SESSION_MATCH_META_KEYS - set(flat)
    assert not missing, f"inference track meta missing {missing} -> dead session features"
    assert flat["album_name"] == "OK Computer"   # list flattened to first
    assert flat["artist_name"] == "Radiohead"


def test_flatten_track_row_handles_missing_album():
    flat = _flatten_track_row({"track_id": "t2", "artist_name": "X"})
    assert flat["album_name"] == ""              # absent -> empty, never KeyError


def test_pop_rank_pct_built_and_nonconstant():
    """Regression: pop_rank_pct (a top-gain feature) was read from a caller dict
    nothing supplied -> constant 0.5 at inference -> dead feature -> reranker
    couldn't reorder. The reranker must self-build the percentile map from
    tid_to_track, matching build_lgbm_features.build_pop_rank_pct_map semantics
    (0 = most popular, 1 = least, missing/zero popularity -> 0.5)."""
    r = LGBM_RERANKER.__new__(LGBM_RERANKER)
    r.tid_to_track = {
        "hot": {"popularity": 100.0},
        "mid": {"popularity": 10.0},
        "cold": {"popularity": 1.0},
        "nopop": {"popularity": 0.0},
    }
    r._build_pop_rank_pct()
    assert r.pop_rank_pct["hot"] == 0.0          # most popular -> 0/N
    assert r.pop_rank_pct["nopop"] == 0.5         # zero popularity -> neutral
    # strictly increasing pct as popularity drops (not a dead constant)
    assert r.pop_rank_pct["hot"] < r.pop_rank_pct["mid"] < r.pop_rank_pct["cold"]
    assert len({r.pop_rank_pct[t] for t in ("hot", "mid", "cold")}) == 3


def test_same_album_fires_through_inference_loader_path():
    """Build tid_to_track exactly as production does (via _flatten_track_row from
    raw rows) and confirm same_album == 1 when a candidate shares the played
    track's album. Pre-fix this was impossible: album_name never survived the
    loader, so same_album was pinned to 0 regardless of the catalog."""
    raw_rows = [
        {"track_id": "p1", "artist_name": ["Radiohead"], "album_name": ["In Rainbows"],
         "tag_list": ["rock"], "popularity": 8.0, "release_date": "2007"},
        {"track_id": "c_same_album", "artist_name": ["Radiohead"], "album_name": ["In Rainbows"],
         "tag_list": ["rock"], "popularity": 5.0, "release_date": "2007"},
    ]
    tid_to_track = {r["track_id"]: _flatten_track_row(r) for r in raw_rows}

    # Sanity: the shared feature fn sees a real album match off these dicts.
    f = session_match_features(tid_to_track["c_same_album"], [tid_to_track["p1"]])
    assert f["same_album"] == 1

    # And it flows through the reranker's feature matrix.
    features = ["wrrf_rank", "cfbpr_score", "pop_log", "recency_years", "tag_count",
                "artist_in_query", "goal_category", "goal_specificity",
                "user_age_group", "user_country", "user_gender",
                "same_artist", "same_album", "artist_in_session_count",
                "session_tag_overlap"]
    r = LGBM_RERANKER.__new__(LGBM_RERANKER)
    r.features = features
    r.categorical_features = ["goal_category", "goal_specificity",
                              "user_age_group", "user_country", "user_gender"]
    r.cat_levels = {c: [] for c in r.categorical_features}
    r.cat_index = {c: {} for c in r.categorical_features}
    r.tid_to_track = tid_to_track
    r.cfbpr_user_embs = {}
    r.cfbpr_tid_to_idx = {}
    r.cfbpr_track_mat = np.zeros((0, 8), dtype=np.float64)
    r._user_meta = {}
    f_idx = {f: i for i, f in enumerate(features)}

    X = r._compute_feature_matrix(
        query="anything", candidate_tids=["c_same_album"], user_id=None,
        goal_category=None, goal_specificity=None, user_profile_raw=None,
        extra_session_info={"played_tids": ["p1"]},
    )
    assert X[0, f_idx["same_album"]] == 1
