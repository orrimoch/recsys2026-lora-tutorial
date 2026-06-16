"""K2 — LightGBM LambdaMART reranker (F2 Reranker)."""
from __future__ import annotations

from mcrs.contracts import Candidate, RankedList, TurnContext, UserProfile
from mcrs.rerank.features import FeatureBuilder
from mcrs.rerank.lgbm import LGBMReranker


def _ctx():
    return TurnContext("s", "u", 1, ["q"], None, UserProfile("u", 1, "f", "US", []), [], "cold")


def _group(gold_id, ids):
    """gold gets the strongest retrieval signal; distractors weaker."""
    cands = []
    for j, t in enumerate(ids):
        if t == gold_id:
            cands.append(Candidate(t, channel_ranks={"bm25": 1, "dense": 1}, rrf_score=1.0))
        else:
            cands.append(Candidate(t, channel_ranks={"bm25": j + 5}, rrf_score=0.05))
    return (_ctx(), cands, gold_id)


def _fb():
    return FeatureBuilder(catalog=None, channel_labels=["bm25", "dense"])


def test_build_training_data_skips_groups_without_gold_in_pool():
    rk = LGBMReranker(_fb())
    groups = [_group("a", ["a", "b", "c"]),                       # gold in pool
              (_ctx(), [Candidate("x"), Candidate("y")], "zzz")]  # gold NOT in pool -> skip
    X, y, gsizes = rk.build_training_data(groups)
    assert gsizes == [3]                 # only the first group kept
    assert int(y.sum()) == 1             # exactly one positive in the kept group
    assert X.shape[0] == 3


def test_reranker_learns_to_put_gold_first():
    rk = LGBMReranker(_fb(), n_estimators=50)
    train = [_group("g", ["g", f"d{i}", f"e{i}", f"f{i}"]) for i in range(40)]
    rk.fit(train)
    ctx, cands, gold = _group("g", ["d99", "e99", "g", "f99"])   # gold not first in input
    ranked = rk.rerank(ctx, cands)
    assert isinstance(ranked, RankedList)
    assert ranked.items[0].track_id == "g"                       # learned high-rrf == relevant
    assert {c.track_id for c in ranked.items} == {c.track_id for c in cands}  # same set, reordered


def test_rerank_requires_a_fitted_model():
    import pytest
    with pytest.raises(RuntimeError):
        LGBMReranker(_fb()).rerank(_ctx(), [Candidate("a")])


def test_save_load_roundtrip_preserves_ranking(tmp_path):
    rk = LGBMReranker(_fb(), n_estimators=30)
    rk.fit([_group("g", ["g", f"d{i}", f"e{i}"]) for i in range(30)])
    ctx, cands, _ = _group("g", ["d1", "g", "e1"])
    before = [c.track_id for c in rk.rerank(ctx, list(cands)).items]
    p = str(tmp_path / "k2.txt")
    rk.save(p)
    loaded = LGBMReranker(_fb()).load(p)
    after = [c.track_id for c in loaded.rerank(ctx, list(cands)).items]
    assert before == after
