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
