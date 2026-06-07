"""Tier-2 #4.1: leak-free relevance features qwen_meta_cos + bm25_score.

Both are PASSTHROUGH per-candidate features (like bge_cos): a shared
RelevanceScorer computes them upstream and injects them into candidate dicts;
extract_features (train) and lgbm_rerank (serve) just read them, so the values
are identical by construction. These tests pin the pure computation, the
train/serve parity, and the off-by-default guard.
"""
import numpy as np
import pytest

from mcrs.rerankers.lgbm_rerank import LGBM_RERANKER
from mcrs.rerankers.relevance_scorer import relevance_feats_for_candidates


# ---- pure core ----

def test_relevance_feats_cosine_and_bm25():
    catalog = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)  # already unit-norm
    tid_to_idx = {"a": 0, "b": 1}
    q = np.array([1.0, 0.0])
    bm25_map = {"a": 3.5}
    out = relevance_feats_for_candidates(q, catalog, tid_to_idx, bm25_map, ["a", "b"])
    assert out[0]["qwen_meta_cos"] == pytest.approx(1.0)   # q aligned with a
    assert out[1]["qwen_meta_cos"] == pytest.approx(0.0)   # q orthogonal to b
    assert out[0]["bm25_score"] == pytest.approx(3.5)
    assert out[1]["bm25_score"] == pytest.approx(0.0)      # missing -> 0


def test_relevance_feats_unknown_tid_zero_cos():
    catalog = np.array([[1.0, 0.0]], dtype=np.float64)
    out = relevance_feats_for_candidates(
        np.array([1.0, 0.0]), catalog, {"a": 0}, {}, ["zzz"])
    assert out[0]["qwen_meta_cos"] == 0.0 and out[0]["bm25_score"] == 0.0


# ---- train/serve parity (mirror test_lgbm_bge_feature) ----

BASE_FEATURES = ["wrrf_rank", "cfbpr_score", "pop_log", "recency_years",
                 "tag_count", "artist_in_query", "goal_category",
                 "goal_specificity", "user_age_group", "user_country", "user_gender"]


def _make_stub(features):
    r = LGBM_RERANKER.__new__(LGBM_RERANKER)
    r.features = features
    r.categorical_features = ["goal_category", "goal_specificity",
                              "user_age_group", "user_country", "user_gender"]
    r.cat_levels = {c: [] for c in r.categorical_features}
    r.cat_index = {c: {} for c in r.categorical_features}
    r.tid_to_track = {"c1": {"artist_name": "X", "album_name": "Y",
                             "tag_list": [], "popularity": 1.0, "release_date": "2000"}}
    r.cfbpr_user_embs = {}
    r.cfbpr_tid_to_idx = {}
    r.cfbpr_track_mat = np.zeros((0, 8), dtype=np.float64)
    r._user_meta = {}
    return r


def _train_row(c, with_relevance):
    from scripts.build_lgbm_features import extract_features
    rows = extract_features(
        query="q", candidates=[c], gold_tid="zzz",
        session_info={"session_id": "s", "user_id": "u", "turn_number": 1,
                      "conversation_goal": {}, "prior_track_count": 0},
        user_info={},
        track_meta={c["tid"]: {"artist_name": "X", "album_name": "Y",
                               "tag_list": [], "popularity": 1.0, "release_date": "2000"}},
        cfbpr_tid_to_idx={}, cfbpr_track_mat=np.zeros((0, 8), dtype=np.float64),
        cfbpr_user_embs={}, query_tokens=set(), with_relevance=with_relevance,
    )
    return rows[0]


def test_relevance_train_serve_parity():
    cos, bm = 0.6123, 4.2
    c = {"tid": "c1", "wrrf_rank": 1, "qwen_meta_cos": cos, "bm25_score": bm}
    train_row = _train_row(c, with_relevance=True)
    assert train_row["qwen_meta_cos"] == pytest.approx(cos)
    assert train_row["bm25_score"] == pytest.approx(bm)

    features = BASE_FEATURES + ["qwen_meta_cos", "bm25_score"]
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}
    X = r._compute_feature_matrix(
        query="q", candidate_tids=["c1"], user_id=None,
        goal_category=None, goal_specificity=None, user_profile_raw=None,
        extra_features_per_candidate=[{"qwen_meta_cos": cos, "bm25_score": bm}])
    assert X[0, f_idx["qwen_meta_cos"]] == pytest.approx(train_row["qwen_meta_cos"])
    assert X[0, f_idx["bm25_score"]] == pytest.approx(train_row["bm25_score"])


def test_relevance_serve_defaults_zero_when_absent():
    features = BASE_FEATURES + ["qwen_meta_cos", "bm25_score"]
    r = _make_stub(features)
    f_idx = {f: i for i, f in enumerate(features)}
    X = r._compute_feature_matrix(
        query="q", candidate_tids=["c1"], user_id=None,
        goal_category=None, goal_specificity=None, user_profile_raw=None,
        extra_features_per_candidate=[{}])
    assert X[0, f_idx["qwen_meta_cos"]] == 0.0
    assert X[0, f_idx["bm25_score"]] == 0.0


def test_relevance_off_by_default_no_columns():
    c = {"tid": "c1", "wrrf_rank": 1, "qwen_meta_cos": 0.9, "bm25_score": 9.0}
    row = _train_row(c, with_relevance=False)
    assert "qwen_meta_cos" not in row and "bm25_score" not in row


# ---- RelevanceScorer.feats_for_batch wrapper (stubbed dense + bm25) ----

def _stub_scorer():
    from mcrs.rerankers.relevance_scorer import RelevanceScorer, _l2_normalize_rows
    r = RelevanceScorer.__new__(RelevanceScorer)

    class _FakeDense:
        track_ids = ["a", "b"]
        track_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        def _encode_queries(self, qs):
            # every query points at track "a"
            return np.array([[1.0, 0.0]] * len(qs))

    r.dense = _FakeDense()
    r.catalog_norm = _l2_normalize_rows(r.dense.track_mat)
    r.tid_to_idx = {"a": 0, "b": 1}
    r.bm25_topk = 500
    r._bm25_maps = lambda qs: [{"a": 2.0} for _ in qs]  # bm25 favors "a"
    return r


def test_feats_for_batch_aligns_and_scores():
    r = _stub_scorer()
    out = r.feats_for_batch(["q1", "q2"], [["a", "b"], ["b"]])
    assert len(out) == 2 and len(out[0]) == 2 and len(out[1]) == 1
    assert out[0][0]["qwen_meta_cos"] == pytest.approx(1.0)   # q aligned with "a"
    assert out[0][1]["qwen_meta_cos"] == pytest.approx(0.0)   # orthogonal to "b"
    assert out[0][0]["bm25_score"] == pytest.approx(2.0)
    assert out[0][1]["bm25_score"] == pytest.approx(0.0)      # "b" not in bm25 map
    assert out[1][0]["qwen_meta_cos"] == pytest.approx(0.0)   # q2 cand "b"


# ---- serve/train alignment guards (parity is by construction via RelevanceScorer) ----

def test_serve_relevance_scorer_uses_served_dataset_and_corpus():
    # crs_baseline must build the scorer with the SAME item_db_name + corpus_types
    # the served union uses, so serve magnitudes match the train feature builder.
    import inspect
    from mcrs.crs_baseline import CRS_BASELINE
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert "RelevanceScorer(" in src
    assert "self.item_db_name" in src and "self.corpus_types" in src


def test_build_relevance_scorer_matches_union_dataset():
    # build_lgbm_features must build the scorer over the same catalog the union
    # uses (Track-Metadata) + the 3-field corpus, so build == serve.
    import inspect
    from scripts.build_lgbm_features import build
    src = inspect.getsource(build)
    assert "RelevanceScorer(" in src
    assert '"track_name", "artist_name", "album_name"' in src
