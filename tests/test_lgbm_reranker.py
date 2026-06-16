"""K2 — LightGBM LambdaMART reranker (F2 Reranker)."""
from __future__ import annotations

from mcrs.contracts import Candidate, RankedList, TurnContext, UserProfile
from mcrs.rerank.features import FeatureBuilder
from mcrs.rerank.lgbm import LGBMReranker


def _ctx(session="s"):
    return TurnContext(session, "u", 1, ["q"], None, UserProfile("u", 1, "f", "US", []), [], "cold")


def _group(gold_id, ids, session="s"):
    """gold gets the strongest retrieval signal; distractors weaker."""
    cands = []
    for j, t in enumerate(ids):
        if t == gold_id:
            cands.append(Candidate(t, channel_ranks={"bm25": 1, "dense": 1}, rrf_score=1.0))
        else:
            cands.append(Candidate(t, channel_ranks={"bm25": j + 5}, rrf_score=0.05))
    return (_ctx(session), cands, gold_id)


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


def test_negative_cap_limits_group_size_keeping_the_gold():
    rk = LGBMReranker(_fb(), neg_cap=2)
    X, y, gsizes = rk.build_training_data([_group("g", ["g", "d", "e", "f", "h"])])
    assert gsizes == [3]                 # gold + 2 negatives (was 1 + 4)
    assert int(y.sum()) == 1             # gold retained


def test_session_split_is_disjoint():
    rk = LGBMReranker(_fb(), val_fraction=0.3, seed=1)
    groups = [_group("g", ["g", "x", "y"], session=f"s{i}") for i in range(10)]
    tr, va = rk._session_split(groups)
    tr_s = {g[0].session_id for g in tr}
    va_s = {g[0].session_id for g in va}
    assert tr_s and va_s and not (tr_s & va_s)      # disjoint, both non-empty
    assert len(va_s) == 3                            # 30% of 10 sessions


def test_fit_uses_validation_split_and_still_ranks_gold_first():
    rk = LGBMReranker(_fb(), n_estimators=80, val_fraction=0.25,
                      min_val_groups=5, early_stopping_rounds=10)
    rk.fit([_group("g", ["g", f"d{i}", f"e{i}", f"f{i}"], session=f"s{i}") for i in range(40)])
    assert rk.n_val_groups_ > 0 and rk.n_train_groups_ > 0   # session-disjoint val used
    ranked = rk.rerank(*_group("g", ["d9", "e9", "g", "f9"])[:2])
    assert ranked.items[0].track_id == "g"


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


def test_rerank_raises_on_feature_spec_mismatch(tmp_path):
    # trained with bm25+dense+a score feature; reloaded with a DIFFERENT feature set -> must fail loud
    fb_train = FeatureBuilder(catalog=None, channel_labels=["bm25", "dense"],
                              score_fns={"dense_cos": lambda ctx, tid: 0.0})
    rk = LGBMReranker(fb_train, n_estimators=20)
    rk.fit([_group("g", ["g", f"d{i}", f"e{i}"]) for i in range(20)])
    p = str(tmp_path / "k2.txt"); rk.save(p)
    ctx, cands, _ = _group("g", ["d1", "g", "e1"])
    mismatched = LGBMReranker(FeatureBuilder(catalog=None, channel_labels=["bm25", "dense"])).load(p)
    import pytest
    with pytest.raises(ValueError):
        mismatched.rerank(ctx, list(cands))      # feature set differs from the trained model
    matched = LGBMReranker(FeatureBuilder(catalog=None, channel_labels=["bm25", "dense"],
                                          score_fns={"dense_cos": lambda ctx, tid: 0.0})).load(p)
    assert matched.rerank(ctx, list(cands)).items                      # same spec -> works
