"""Train/serve parity tests for the bge-v2 bi-encoder LGBM features
(bge_cos + bge_rank_inv).

Two per-candidate features wire the fine-tuned bge-v2 signal into the LGBM
reranker:
  - bge_cos       : cosine(bge query emb, candidate track emb) — passes through
                    unchanged on both sides.
  - bge_rank_inv  : 1 / max(1, rank-in-the-full-catalog-bge-ranking).

The values MUST be identical between the training feature builder
(scripts/build_lgbm_features.extract_features with with_bge=True) and the serve
reranker (mcrs/rerankers/lgbm_rerank._compute_feature_matrix). These tests pin
that parity plus the off-by-default guard so existing parquets/models are
unchanged.
"""
import inspect
import re

import numpy as np
import pytest

from mcrs.rerankers.lgbm_rerank import LGBM_RERANKER


# ---------------------------------------------------------------------------
# Serve-side stub (same construction style as test_lgbm_rerank_session_features)
# ---------------------------------------------------------------------------
def _make_stub(features):
    r = LGBM_RERANKER.__new__(LGBM_RERANKER)
    r.features = features
    r.categorical_features = ["goal_category", "goal_specificity",
                              "user_age_group", "user_country", "user_gender"]
    r.cat_levels = {c: [] for c in r.categorical_features}
    r.cat_index = {c: {} for c in r.categorical_features}
    r.tid_to_track = {
        "c1": {"artist_name": "Radiohead", "album_name": "OK Computer",
               "tag_list": ["rock"], "popularity": 10.0, "release_date": "1997"},
        "c2": {"artist_name": "Beyonce", "album_name": "Lemonade",
               "tag_list": ["pop"], "popularity": 99.0, "release_date": "2016"},
    }
    r.cfbpr_user_embs = {}
    r.cfbpr_tid_to_idx = {}
    r.cfbpr_track_mat = np.zeros((0, 8), dtype=np.float64)
    r._user_meta = {}
    return r


BASE_FEATURES = ["wrrf_rank", "cfbpr_score", "pop_log", "recency_years",
                 "tag_count", "artist_in_query", "goal_category",
                 "goal_specificity", "user_age_group", "user_country",
                 "user_gender"]


def _train_side_row(c, with_bge):
    """Run the train feature builder for ONE candidate dict `c` and return the
    emitted row. Heavy mcrs deps are stubbed so extract_features runs in-process
    without loading any datasets/models."""
    from scripts.build_lgbm_features import extract_features
    session_info = {"session_id": "s", "user_id": "u", "turn_number": 1,
                    "conversation_goal": {}, "prior_track_count": 0}
    rows = extract_features(
        query="some upbeat query",
        candidates=[c],
        gold_tid="zzz",
        session_info=session_info,
        user_info={},
        track_meta={c["tid"]: {"artist_name": "X", "album_name": "Y",
                               "tag_list": [], "popularity": 1.0,
                               "release_date": "2000"}},
        cfbpr_tid_to_idx={},
        cfbpr_track_mat=np.zeros((0, 8), dtype=np.float64),
        cfbpr_user_embs={},
        query_tokens=set(),
        with_bge=with_bge,
    )
    return rows[0]


# ---------------------------------------------------------------------------
# (1) parity: train-side and serve-side agree for bge_cos + bge_rank_inv
# ---------------------------------------------------------------------------
def test_bge_train_serve_parity():
    """Identical candidate (bge_cos, bge_rank) -> identical feature values on
    the train builder and the serve reranker."""
    bge_cos = 0.7321
    bge_rank = 5
    c = {"tid": "c1", "wrrf_rank": 1, "bge_cos": bge_cos, "bge_rank": bge_rank}

    # train side
    train_row = _train_side_row(c, with_bge=True)
    assert train_row["bge_cos"] == pytest.approx(bge_cos)
    assert train_row["bge_rank_inv"] == pytest.approx(1.0 / bge_rank)

    # serve side
    features = BASE_FEATURES + ["bge_cos", "bge_rank_inv"]
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}
    X = r._compute_feature_matrix(
        query="some upbeat query",
        candidate_tids=["c1"],
        user_id=None, goal_category=None, goal_specificity=None,
        user_profile_raw=None,
        extra_features_per_candidate=[{"bge_cos": bge_cos, "bge_rank": bge_rank}],
    )
    serve_cos = X[0, f_idx["bge_cos"]]
    serve_rank_inv = X[0, f_idx["bge_rank_inv"]]

    # the load-bearing assertion: both sides compute the SAME numbers
    assert serve_cos == pytest.approx(train_row["bge_cos"])
    assert serve_rank_inv == pytest.approx(train_row["bge_rank_inv"])
    assert serve_rank_inv == pytest.approx(1.0 / bge_rank)


def test_bge_rank_inv_fallback_to_wrrf_rank_when_absent():
    """When bge_rank is missing the value falls back to 1/max(1, rank) on BOTH
    sides (train uses wrrf_rank; serve uses the 1-indexed candidate position)."""
    # train: wrrf_rank=3, no bge_rank -> 1/3
    c = {"tid": "c1", "wrrf_rank": 3, "bge_cos": 0.0}
    train_row = _train_side_row(c, with_bge=True)
    assert train_row["bge_rank_inv"] == pytest.approx(1.0 / 3)

    # serve: candidate at rank 1 (only one), no bge_rank -> 1/1 = 1.0
    features = BASE_FEATURES + ["bge_cos", "bge_rank_inv"]
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}
    X = r._compute_feature_matrix(
        query="q", candidate_tids=["c1"], user_id=None,
        goal_category=None, goal_specificity=None, user_profile_raw=None,
        extra_features_per_candidate=[{}],
    )
    assert X[0, f_idx["bge_rank_inv"]] == pytest.approx(1.0)
    assert X[0, f_idx["bge_cos"]] == pytest.approx(0.0)  # default 0.0


# ---------------------------------------------------------------------------
# (2) feature off by default: no bge columns unless with_bge
# ---------------------------------------------------------------------------
def test_bge_off_by_default_no_columns():
    """Without with_bge the builder must NOT emit bge_cos/bge_rank_inv, even if
    the candidate happens to carry the keys — existing parquet schema unchanged."""
    c = {"tid": "c1", "wrrf_rank": 1, "bge_cos": 0.9, "bge_rank": 2}
    row = _train_side_row(c, with_bge=False)
    assert "bge_cos" not in row
    assert "bge_rank_inv" not in row


def test_bge_default_with_bge_is_false():
    """extract_features defaults with_bge=False so all existing callers are
    byte-unchanged."""
    from scripts.build_lgbm_features import extract_features
    sig = inspect.signature(extract_features)
    assert sig.parameters["with_bge"].default is False


# ---------------------------------------------------------------------------
# (3) source-check: serve reranker guards both columns by `in f_idx`
# ---------------------------------------------------------------------------
def test_serve_guards_bge_by_in_f_idx():
    """Old models without the bge feature must still work: the reranker only
    writes bge_cos/bge_rank_inv when they are in the model's f_idx."""
    src = inspect.getsource(LGBM_RERANKER._compute_feature_matrix)
    assert re.search(r'if\s+"bge_cos"\s+in\s+f_idx', src)
    assert re.search(r'if\s+"bge_rank_inv"\s+in\s+f_idx', src)
    # and both are in the extended_keys set so need_helpers fires
    assert '"bge_cos"' in src and '"bge_rank_inv"' in src


def test_legacy_model_without_bge_columns_untouched():
    """A model whose feature list omits the bge columns reranks without error
    and produces only its declared layout (no crash, no extra columns)."""
    features = list(BASE_FEATURES)
    r = _make_stub(features)
    X = r._compute_feature_matrix(
        query="q", candidate_tids=["c1", "c2"], user_id=None,
        goal_category=None, goal_specificity=None, user_profile_raw=None,
        extra_features_per_candidate=[{"bge_cos": 0.5, "bge_rank": 3},
                                      {"bge_cos": 0.1, "bge_rank": 9}],
    )
    assert X.shape == (2, len(features))
