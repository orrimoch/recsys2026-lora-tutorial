"""K3 — neural (cross-encoder) reranker: re-score only the top cross_encoder_k, stack via score()."""
from __future__ import annotations

from mcrs.contracts import Candidate, TurnContext, UserProfile
from mcrs.data.catalog import Catalog
from mcrs.retrieval.query import QueryBuilder
from mcrs.rerank.neural import NeuralReranker

# each track's doc carries its id (via artist_name) so the fake scorer can key on it
_CAT = Catalog([{"track_id": t, "artist_name": [t]} for t in ("t1", "t2", "t3", "t4")],
               corpus_types=["artist_name"])
_NEUR = {"t1": 0.2, "t2": 0.1, "t3": 0.9, "t4": 0.5}


def _ctx():
    return TurnContext("s", "u", 1, ["mellow jazz please"], "goal",
                       UserProfile("u", 1, "f", "US", []), [], "cold")


def _cands(ids):
    return [Candidate(t, rrf_score=1.0 / (i + 1)) for i, t in enumerate(ids)]


def _fake_scorer(pairs):
    out = []
    for _q, doc in pairs:
        out.append(next((v for tid, v in _NEUR.items() if tid in doc), 0.0))
    return out


def test_rerank_reorders_top_k_by_neural_score():
    k3 = NeuralReranker(_CAT, QueryBuilder(), _fake_scorer, cross_encoder_k=3)
    out = k3.rerank(_ctx(), _cands(["t1", "t2", "t3"]))   # K2 order t1,t2,t3
    assert [c.track_id for c in out.items] == ["t3", "t1", "t2"]   # by neural score 0.9,0.2,0.1


def test_only_top_k_rescored_rest_keep_k2_order():
    seen = []
    def scorer(pairs):
        seen.extend(d for _q, d in pairs)
        return _fake_scorer(pairs)
    k3 = NeuralReranker(_CAT, QueryBuilder(), scorer, cross_encoder_k=2)
    out = k3.rerank(_ctx(), _cands(["t2", "t1", "t3", "t4"]))   # K2 order t2,t1,t3,t4
    assert [c.track_id for c in out.items] == ["t1", "t2", "t3", "t4"]  # top2 flipped (t1 0.2 > t2 0.1); t3,t4 fixed
    assert len(seen) == 2 and all("t3" not in d and "t4" not in d for d in seen)  # only top2 scored


def test_rerank_adds_no_ids_and_drops_none():
    k3 = NeuralReranker(_CAT, QueryBuilder(), _fake_scorer, cross_encoder_k=10)
    ids_in = ["t1", "t2", "t3", "t4"]
    out = k3.rerank(_ctx(), _cands(ids_in))
    assert sorted(c.track_id for c in out.items) == sorted(ids_in)


def test_score_returns_neural_scores_for_top_k_only():
    k3 = NeuralReranker(_CAT, QueryBuilder(), _fake_scorer, cross_encoder_k=2)
    sc = k3.score(_ctx(), _cands(["t1", "t2", "t3"]))
    assert set(sc) == {"t1", "t2"} and sc["t1"] == 0.2 and sc["t2"] == 0.1


def test_pairs_use_query_text_and_enriched_doc():
    cat = Catalog([{"track_id": "t1", "artist_name": ["t1"]}],
                  corpus_types=["artist_name"], enriched_docs={"t1": "ENRICHED mellow jazz blurb t1"})
    captured = []
    def scorer(pairs):
        captured.extend(pairs)
        return [1.0] * len(pairs)
    NeuralReranker(cat, QueryBuilder(), scorer, cross_encoder_k=1, enriched=True).score(_ctx(), _cands(["t1"]))
    q, doc = captured[0]
    assert "mellow jazz" in q and "ENRICHED" in doc           # query text + enriched doc side
