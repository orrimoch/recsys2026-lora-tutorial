"""FIX C2: the LGBM reranker must compute the 4 Stage-C session-continuity
features at inference (same_artist / same_album / artist_in_session_count /
session_tag_overlap) from extra_session_info['played_tids'], using the SHARED
session_match_features (identical to the training feature builder).

We construct a minimal LGBM_RERANKER via __new__ (skips heavy __init__ that
loads boosters + HF datasets) and set just the attributes that
_compute_feature_matrix touches, then assert the matrix columns are populated.
"""
import numpy as np
import pytest

from mcrs.rerankers.lgbm_rerank import LGBM_RERANKER


def _make_stub(features):
    r = LGBM_RERANKER.__new__(LGBM_RERANKER)
    r.features = features
    # _compute_feature_matrix always pre-encodes these 5 session-level
    # categoricals via _encode_cat, so cat_index must hold their keys.
    r.categorical_features = ["goal_category", "goal_specificity",
                              "user_age_group", "user_country", "user_gender"]
    r.cat_levels = {c: [] for c in r.categorical_features}
    r.cat_index = {c: {} for c in r.categorical_features}
    # Track catalog: cand1 by a session artist, cand2 by a stranger.
    r.tid_to_track = {
        "p1": {"artist_name": "Radiohead", "album_name": "OK Computer",
               "tag_list": ["rock", "alt"], "popularity": 10.0, "release_date": "1997"},
        "c_same": {"artist_name": "Radiohead", "album_name": "In Rainbows",
                   "tag_list": ["rock"], "popularity": 5.0, "release_date": "2007"},
        "c_other": {"artist_name": "Beyonce", "album_name": "Lemonade",
                    "tag_list": ["pop"], "popularity": 99.0, "release_date": "2016"},
    }
    r.cfbpr_user_embs = {}
    r.cfbpr_tid_to_idx = {}
    r.cfbpr_track_mat = np.zeros((0, 8), dtype=np.float64)
    r._user_meta = {}
    return r


# The reranker loop always writes the base-11 columns unconditionally, so any
# trained model's feature list contains them. Mirror that here.
BASE_FEATURES = ["wrrf_rank", "cfbpr_score", "pop_log", "recency_years",
                 "tag_count", "artist_in_query", "goal_category",
                 "goal_specificity", "user_age_group", "user_country",
                 "user_gender"]
SESSION_FEATURES = ["same_artist", "same_album", "artist_in_session_count",
                    "session_tag_overlap"]


def test_session_features_populated_from_played_tids():
    # Model whose feature list is base-11 + the 4 session features.
    features = BASE_FEATURES + SESSION_FEATURES
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}

    X = r._compute_feature_matrix(
        query="something upbeat",
        candidate_tids=["c_same", "c_other"],
        user_id=None,
        goal_category=None,
        goal_specificity=None,
        user_profile_raw=None,
        extra_session_info={"played_tids": ["p1"]},  # session played a Radiohead track
    )

    # Row 0 = c_same (Radiohead, shares the rock tag with p1).
    assert X[0, f_idx["same_artist"]] == 1
    assert X[0, f_idx["same_album"]] == 0      # different album
    assert X[0, f_idx["artist_in_session_count"]] == 1
    assert X[0, f_idx["session_tag_overlap"]] == 1  # "rock" shared

    # Row 1 = c_other (Beyonce, pop) — no overlap with the session.
    assert X[1, f_idx["same_artist"]] == 0
    assert X[1, f_idx["same_album"]] == 0
    assert X[1, f_idx["artist_in_session_count"]] == 0
    assert X[1, f_idx["session_tag_overlap"]] == 0


def test_no_played_tids_yields_zero_session_features():
    features = BASE_FEATURES + SESSION_FEATURES
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}
    X = r._compute_feature_matrix(
        query="q", candidate_tids=["c_same"], user_id=None,
        goal_category=None, goal_specificity=None, user_profile_raw=None,
        extra_session_info={"played_tids": []},
    )
    for k in SESSION_FEATURES:
        assert X[0, f_idx[k]] == 0


def test_legacy_11col_model_untouched_no_session_columns():
    # A model that does NOT list the session features must run without error
    # and produce only the base-11 layout (no crash, no extra columns).
    features = list(BASE_FEATURES)
    r = _make_stub(features)
    X = r._compute_feature_matrix(
        query="q", candidate_tids=["c_same"], user_id=None,
        goal_category=None, goal_specificity=None, user_profile_raw=None,
        extra_session_info={"played_tids": ["p1"]},
    )
    assert X.shape == (1, len(features))


# ---------------------------------------------------------------------------
# sasrec_rank_inv tests (TDD: written before implementation)
# ---------------------------------------------------------------------------

def test_sasrec_rank_inv_set_from_extra_dict():
    """When 'sasrec_rank' is present in the per-candidate extra dict,
    X[0, f_idx['sasrec_rank_inv']] must equal 1.0 / sasrec_rank."""
    features = BASE_FEATURES + ["sasrec_rank_inv"]
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}

    X = r._compute_feature_matrix(
        query="some query",
        candidate_tids=["c_same"],
        user_id=None,
        goal_category=None,
        goal_specificity=None,
        user_profile_raw=None,
        extra_features_per_candidate=[{"sasrec_rank": 4}],
    )

    assert X.shape == (1, len(features))
    assert X[0, f_idx["sasrec_rank_inv"]] == pytest.approx(1.0 / 4)


def test_sasrec_rank_inv_fallback_to_wrrf_rank_when_key_absent():
    """When 'sasrec_rank' is absent from the extra dict, the value falls back
    to 1.0 / max(1, rank), where rank is the 1-indexed wRRF position.
    For the first (only) candidate rank == 1, so the result is 1.0."""
    features = BASE_FEATURES + ["sasrec_rank_inv"]
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}

    X = r._compute_feature_matrix(
        query="some query",
        candidate_tids=["c_same"],
        user_id=None,
        goal_category=None,
        goal_specificity=None,
        user_profile_raw=None,
        extra_features_per_candidate=[{}],  # no "sasrec_rank" key
    )

    assert X[0, f_idx["sasrec_rank_inv"]] == pytest.approx(1.0)  # 1/max(1,1) = 1.0


# ---------------------------------------------------------------------------
# leak-test: a "clean" model may omit cfbpr_score entirely (it's a leaked
# model-derived feature). The reranker must NOT KeyError when cfbpr_score is
# absent from the feature list. (TDD: written before guarding cfbpr_score.)
# ---------------------------------------------------------------------------

def test_clean_model_without_cfbpr_score_does_not_crash():
    """A model whose feature list omits cfbpr_score must rerank without error.
    Mirrors the lgbm_clean_v1 leak-test build (cfbpr_score + sasrec_rank_inv
    dropped). Before the guard fix this raised KeyError('cfbpr_score')."""
    features = [f for f in BASE_FEATURES if f != "cfbpr_score"] + SESSION_FEATURES
    assert "cfbpr_score" not in features
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}

    X = r._compute_feature_matrix(
        query="some query",
        candidate_tids=["c_same", "c_other"],
        user_id=None,
        goal_category=None,
        goal_specificity=None,
        user_profile_raw=None,
        extra_session_info={"played_tids": ["p1"]},
    )

    assert X.shape == (2, len(features))
    # the session-continuity columns still populate (sanity the matrix is real)
    assert X[0, f_idx["same_artist"]] == 1
