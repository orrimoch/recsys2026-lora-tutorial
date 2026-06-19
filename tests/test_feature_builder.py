"""K1 — rerank feature builder (causal, pure, leak-free)."""
from __future__ import annotations

import math

from mcrs.contracts import Candidate, TurnContext, UserProfile
from mcrs.data.catalog import Catalog
from mcrs.rerank.features import FeatureBuilder

_CAT = Catalog([
    {"track_id": "a", "popularity": 50.0, "release_date": "1993-01-01", "artist_name": ["Nirvana"]},
    {"track_id": "b", "popularity": 10.0, "release_date": "1959", "artist_name": ["Miles Davis"]},
    {"track_id": "c", "popularity": 5.0, "release_date": "1994", "artist_name": ["Nirvana"]},
], corpus_types=[])


def _ctx():
    return TurnContext("s", "u", 2, ["hello there", "more music"], "goal",
                       UserProfile("u", 1, "f", "US", []), ["a"], "warm")


def _cands():
    return [Candidate("c", channel_ranks={"bm25": 1, "cf": 2}, rrf_score=0.04),
            Candidate("b", channel_ranks={"bm25": 4}, rrf_score=0.01)]


def test_feature_names_stable_and_cover_groups():
    fb = FeatureBuilder(_CAT, channel_labels=["bm25", "cf"])
    for key in ("rrf_score", "n_channels_hit", "best_rank_inv", "rank_inv__bm25",
                "rank_inv__cf", "turn_number", "is_cold", "artist_in_history", "log_popularity"):
        assert key in fb.feature_names


def test_features_computed_correctly():
    fb = FeatureBuilder(_CAT, channel_labels=["bm25", "cf"])
    c, b = fb.build(_ctx(), _cands())
    assert c.features["rrf_score"] == 0.04
    assert c.features["n_channels_hit"] == 2.0
    assert c.features["best_rank_inv"] == 1.0
    assert c.features["rank_inv__bm25"] == 1.0 and c.features["rank_inv__cf"] == 0.5
    assert c.features["is_cold"] == 0.0
    assert c.features["artist_in_history"] == 1.0          # c is Nirvana, history "a" is Nirvana
    assert c.features["log_popularity"] == math.log1p(5.0)
    assert c.features["release_year"] == 1994.0
    assert c.features["turn_number"] == 2.0 and c.features["history_len"] == 1.0
    # b: only bm25, different artist
    assert b.features["rank_inv__cf"] == 0.0 and b.features["artist_in_history"] == 0.0


def test_score_fns_add_relevance_features():
    # injected per-candidate relevance scorer (e.g. dense query<->doc cosine), GPU-free in tests
    fb = FeatureBuilder(_CAT, channel_labels=["bm25", "cf"],
                        score_fns={"dense_cos": lambda ctx, tid: {"c": 0.9, "b": 0.2}[tid]})
    assert "dense_cos" in fb.feature_names
    c, b = fb.build(_ctx(), _cands())
    assert c.features["dense_cos"] == 0.9 and b.features["dense_cos"] == 0.2
    assert set(c.features) == set(fb.feature_names)        # matrix() stays consistent


def test_no_score_fns_keeps_default_feature_set():
    fb = FeatureBuilder(_CAT, channel_labels=["bm25", "cf"])
    assert not any(n.startswith("dense") for n in fb.feature_names)


def test_build_is_pure_and_keys_match_feature_names():
    fb = FeatureBuilder(_CAT, channel_labels=["bm25", "cf"])
    once = fb.build(_ctx(), _cands())[0].features
    twice = fb.build(_ctx(), _cands())[0].features
    assert once == twice
    assert set(once) == set(fb.feature_names)


# ----- T3.1: interaction / consensus features + per-turn score calibration -----
def test_interaction_consensus_features():
    fb = FeatureBuilder(_CAT, channel_labels=["bm25", "dense", "cf"])
    # gold-ish cand hit top-5 by both bm25 AND dense, and by 3 channels within top-10
    cand = Candidate("c", channel_ranks={"bm25": 1, "dense": 3, "cf": 8}, rrf_score=0.05)
    other = Candidate("b", channel_ranks={"bm25": 40}, rrf_score=0.01)
    c, b = fb.build(_ctx(), [cand, other])
    assert c.features["n_channels_top10"] == 3.0
    assert c.features["consensus_3plus"] == 1.0
    assert c.features["top5_bm25_and_dense"] == 1.0
    assert b.features["n_channels_top10"] == 0.0       # rank 40 is outside top-10
    assert b.features["consensus_3plus"] == 0.0
    assert b.features["top5_bm25_and_dense"] == 0.0


def test_per_turn_score_calibration_minmax():
    fb = FeatureBuilder(_CAT, channel_labels=["bm25"],
                        score_fns={"dense_cos": lambda ctx, tid: {"c": 0.9, "b": 0.1}[tid]})
    assert "dense_cos_norm" in fb.feature_names
    c, b = fb.build(_ctx(), _cands())
    assert c.features["dense_cos"] == 0.9 and b.features["dense_cos"] == 0.1   # raw preserved
    assert c.features["dense_cos_norm"] == 1.0 and b.features["dense_cos_norm"] == 0.0  # min-max in pool


# ----- T3.4: popularity percentile + recency -----
def test_popularity_percentile_and_recency():
    fb = FeatureBuilder(_CAT, channel_labels=["bm25"])
    c, b = fb.build(_ctx(), _cands())
    # catalog pops: a=50, b=10, c=5 -> c (5.0) is the least popular -> low percentile; b (10.0) higher
    assert 0.0 < c.features["popularity_percentile"] <= b.features["popularity_percentile"]
    # recency: 1994 is more recent than 1959 -> higher recency, both in [0,1]
    assert 0.0 <= b.features["recency"] <= c.features["recency"] <= 1.0


def test_features_handle_no_catalog():
    # catalog=None path: percentile defaults to 0.5, no crash, *_norm present
    fb = FeatureBuilder(None, channel_labels=["bm25"],
                        score_fns={"s": lambda ctx, tid: 1.0})
    cands = [Candidate("x", channel_ranks={"bm25": 1}, rrf_score=0.1)]
    out = fb.build(_ctx(), cands)
    assert out[0].features["popularity_percentile"] == 0.5
    assert set(out[0].features) == set(fb.feature_names)
