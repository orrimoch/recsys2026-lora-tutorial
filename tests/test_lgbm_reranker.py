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


def _cat(rows):
    from mcrs.data.catalog import Catalog
    return Catalog(rows)


# ---- graded labels (win #1: tier the LambdaRank labels so near-misses get gradient) ----
def test_graded_labels_assign_gold_then_same_artist_then_rest():
    """gold -> 2, same-artist-as-gold (non-gold) -> 1, unrelated -> 0. The extra tier creates
    graded pairs LambdaRank can learn from where binary labels gave none (one positive vs 499 zeros)."""
    cat = _cat([{"track_id": "g", "artist_name": ["A"]},
                {"track_id": "sa", "artist_name": ["A"]},     # same artist as the gold
                {"track_id": "x", "artist_name": ["B"]}])      # different artist
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]), graded_labels=True)
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa", "x"]]
    _, y, _ = rk._xy([(_ctx(), cands, "g")], cap=False, graded=True)
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 2, "sa": 1, "x": 0}


def test_graded_labels_off_by_default_keeps_binary():
    cat = _cat([{"track_id": "g", "artist_name": ["A"]},
                {"track_id": "sa", "artist_name": ["A"]},
                {"track_id": "x", "artist_name": ["B"]}])
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]))   # graded off
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa", "x"]]
    _, y, _ = rk.build_training_data([(_ctx(), cands, "g")])
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 1, "sa": 0, "x": 0}


def test_graded_labels_without_catalog_degrade_to_gold_tier_only():
    """No catalog (unit tests / no metadata) => can't compute same-artist; still emits the graded
    gold tier (2) without crashing, every non-gold falls to 0."""
    rk = LGBMReranker(_fb(), graded_labels=True)       # catalog=None
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa", "x"]]
    _, y, _ = rk._xy([(_ctx(), cands, "g")], cap=False, graded=True)
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 2, "sa": 0, "x": 0}


def test_validation_pool_stays_binary_even_when_graded_on():
    """Early stopping must optimize the TRUE single-gold nDCG@20, so the val pool is scored binary
    even when training labels are graded (caller passes graded=False for val)."""
    cat = _cat([{"track_id": "g", "artist_name": ["A"]},
                {"track_id": "sa", "artist_name": ["A"]},
                {"track_id": "x", "artist_name": ["B"]}])
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]), graded_labels=True)
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa", "x"]]
    _, y, _ = rk._xy([(_ctx(), cands, "g")], cap=False, graded=False)
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 1, "sa": 0, "x": 0}


def test_graded_label_matches_any_shared_artist_on_multi_artist_tracks():
    """artist_name is list-valued; a candidate tiers to 1 if it shares ANY artist with the gold
    (set intersection), not only on an exact artist-list match."""
    cat = _cat([{"track_id": "g", "artist_name": ["A", "B"]},     # gold: two artists
                {"track_id": "sa", "artist_name": ["B", "C"]},    # shares B -> tier 1
                {"track_id": "x", "artist_name": ["D"]}])          # shares none -> 0
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]), graded_labels=True)
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa", "x"]]
    _, y, _ = rk._xy([(_ctx(), cands, "g")], cap=False, graded=True)
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 2, "sa": 1, "x": 0}


def test_graded_gold_absent_from_catalog_still_labels_gold_tier():
    """If the gold isn't in the catalog (no artist to match against), no candidate can tier to 1,
    but the gold itself still gets the graded gold tier (2) — no crash, no fabricated tier-1s."""
    cat = _cat([{"track_id": "sa", "artist_name": ["A"]},          # gold 'g' NOT in catalog
                {"track_id": "x", "artist_name": ["B"]}])
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]), graded_labels=True)
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa", "x"]]
    _, y, _ = rk._xy([(_ctx(), cands, "g")], cap=False, graded=True)
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 2, "sa": 0, "x": 0}


def test_graded_gold_in_catalog_but_missing_artist_yields_no_tier_one():
    """Distinct branch from gold-absent: the gold IS in the catalog but its row has no/empty
    artist_name (_artists does `.get('artist_name') or []`). gold_artists is empty, so nothing tiers
    to 1, yet the gold still gets 2 — locks the `or []` behavior."""
    cat = _cat([{"track_id": "g"},                                  # gold in catalog, NO artist_name
                {"track_id": "sa", "artist_name": ["A"]},
                {"track_id": "x", "artist_name": []}])               # empty artist list
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]), graded_labels=True)
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa", "x"]]
    _, y, _ = rk._xy([(_ctx(), cands, "g")], cap=False, graded=True)
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 2, "sa": 0, "x": 0}


def test_graded_labels_tier_every_same_artist_negative_not_just_the_first():
    """Two distinct non-gold tracks by the gold's artist must BOTH tier to 1 — guards against a bug
    that labels only the first same-artist match (every other graded test has exactly one tier-1)."""
    cat = _cat([{"track_id": "g", "artist_name": ["A"]},
                {"track_id": "sa1", "artist_name": ["A"]},          # same artist
                {"track_id": "sa2", "artist_name": ["A"]},          # same artist (second)
                {"track_id": "x", "artist_name": ["B"]}])
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]), graded_labels=True)
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa1", "sa2", "x"]]
    _, y, _ = rk._xy([(_ctx(), cands, "g")], cap=False, graded=True)
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 2, "sa1": 1, "sa2": 1, "x": 0}


def test_graded_labels_computed_on_capped_pool_and_keep_the_gold():
    """neg_cap drops negatives BEFORE labelling; the label array must match the capped group size and
    always retain the gold (tier 2). A same-artist neg may or may not survive the random cap — the
    invariant is len(y)==group_size and exactly one tier-2 (the gold)."""
    cat = _cat([{"track_id": "g", "artist_name": ["A"]}]
               + [{"track_id": f"n{i}", "artist_name": [f"art{i}"]} for i in range(20)])
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]), neg_cap=3, graded_labels=True, seed=42)
    cands = [Candidate("g", channel_ranks={"bm25": 1}, rrf_score=1.0)] + [
        Candidate(f"n{i}", channel_ranks={"bm25": i + 2}, rrf_score=0.1) for i in range(20)]
    _, y, gsizes = rk._xy([(_ctx(), cands, "g")], cap=True, graded=True)
    assert gsizes == [4]                      # gold + neg_cap(3)
    assert len(y) == 4                         # label array matches the capped group
    assert int((y == 2).sum()) == 1            # exactly one gold tier, retained through capping


def test_build_training_data_emits_graded_labels_when_enabled():
    """Integration through the PUBLIC training entrypoint (not just _xy): with graded_labels=True the
    plain-fit data path yields {0,1,2}, with the gold at 2 and a same-artist neg at 1."""
    cat = _cat([{"track_id": "g", "artist_name": ["A"]},
                {"track_id": "sa", "artist_name": ["A"]},
                {"track_id": "x", "artist_name": ["B"]}])
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25"]), graded_labels=True)
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=0.5) for t in ["g", "sa", "x"]]
    X, y, _ = rk.build_training_data([(_ctx(), cands, "g")])
    assert set(int(v) for v in y) == {0, 1, 2}
    assert y.dtype.kind == "i"                  # LightGBM lambdarank needs integer labels
    assert {c.track_id: int(v) for c, v in zip(cands, y)} == {"g": 2, "sa": 1, "x": 0}


def test_graded_fit_runs_end_to_end_and_ranks_gold_first():
    cat = _cat([{"track_id": t, "artist_name": [f"art_{t}"]}
                for t in ["g"] + [f"d{i}" for i in range(40)]
                + [f"e{i}" for i in range(40)] + [f"f{i}" for i in range(40)]])
    rk = LGBMReranker(FeatureBuilder(cat, ["bm25", "dense"]), n_estimators=50, graded_labels=True)
    rk.fit([_group("g", ["g", f"d{i}", f"e{i}", f"f{i}"]) for i in range(40)])
    ranked = rk.rerank(*_group("g", ["d9", "e9", "g", "f9"])[:2])
    assert ranked.items[0].track_id == "g"


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


def test_cap_negative_sampling_is_per_group_and_deterministic():
    """K1 fix: _cap mixes a per-group salt into the seed, so different (session, turn) groups draw
    DIFFERENT negative subsets (a single fixed seed would draw the identical positional slice every
    time), while a given group stays reproducible run-to-run."""
    rk = LGBMReranker(_fb(), neg_cap=3, seed=42)
    gold = "g"
    cands = [Candidate("g", channel_ranks={"bm25": 1}, rrf_score=1.0)] + [
        Candidate(f"n{i}", channel_ranks={"bm25": i + 2}, rrf_score=0.1) for i in range(20)
    ]
    # deterministic for a fixed (seed, salt)
    a1 = [c.track_id for c in rk._cap(cands, gold, salt=("s1", 1))]
    a2 = [c.track_id for c in rk._cap(cands, gold, salt=("s1", 1))]
    assert a1 == a2
    assert a1[0] == "g" and len([t for t in a1 if t != "g"]) == 3   # gold kept + neg_cap negatives
    # decorrelated across groups: many distinct salts -> more than one distinct negative draw
    draws = {tuple(sorted(c.track_id for c in rk._cap(cands, gold, salt=(f"s{i}", i))))
             for i in range(8)}
    assert len(draws) > 1   # a single fixed seed (the old bug) would collapse this to exactly 1


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


def test_xy_normalizes_injected_scores_over_full_pool_not_capped_subset():
    """Train/serve skew fix: the per-turn `*_norm` calibration must use the FULL pool's min/max
    (what val/serve sees), not the random neg-capped subset's. Otherwise a candidate's `_norm`
    is computed at a different scale at train than at serve, and the GBDT splits don't transfer.

    Setup: full pool spans s=0..19 on the negatives; the gold's raw s=5.0. With neg_cap=3 the
    capped pool is {gold + 3 random negs}, whose min/max almost never coincide with the full
    pool's 0/19 — so the gold's full-pool norm (5/19) differs from any capped-pool norm."""
    scores = {"g": 5.0, **{f"n{i}": float(i) for i in range(20)}}   # full pool: min 0, max 19
    fb = FeatureBuilder(catalog=None, channel_labels=["bm25"],
                        score_fns={"s": lambda ctx, tid: scores[tid]})
    rk = LGBMReranker(fb, neg_cap=3, seed=42)
    ids = ["g"] + [f"n{i}" for i in range(20)]
    cands = [Candidate(t, channel_ranks={"bm25": 1}, rrf_score=1.0) for t in ids]
    rk._xy([(_ctx(), cands, "g")], cap=True)
    g = next(c for c in cands if c.track_id == "g")
    assert abs(g.features["s_norm"] - (5.0 / 19.0)) < 1e-9   # full-pool norm, not capped-pool


def test_assert_feature_parity_names_the_missing_and_extra_features():
    """Early, descriptive load-time guard: when a FeatureBuilder is reconstructed without the same
    score_fns/channels the model was trained on, assert_feature_parity() must fail BEFORE any rerank
    (not deep inside the harness) and NAME the missing/extra columns so the caller knows what to add.
    Simulates loading a K2 trained with ce_score + a colbert channel into a stripped FeatureBuilder."""
    import pytest
    fb = FeatureBuilder(catalog=None, channel_labels=["bm25", "dense"],
                        score_fns={"dense_cos": lambda ctx, tid: 0.0})
    rk = LGBMReranker(fb)
    # pretend we loaded a model trained on the FULL spine (extra colbert channel + ce_score feature)
    rk.feature_names_ = list(fb.feature_names) + ["rank_inv__colbert", "ce_score", "ce_score_norm"]
    with pytest.raises(ValueError) as e:
        rk.assert_feature_parity()
    msg = str(e.value)
    assert "rank_inv__colbert" in msg and "ce_score" in msg     # names what's MISSING from the builder


def test_assert_feature_parity_passes_when_specs_match():
    fb = FeatureBuilder(catalog=None, channel_labels=["bm25", "dense"],
                        score_fns={"dense_cos": lambda ctx, tid: 0.0})
    rk = LGBMReranker(fb)
    rk.feature_names_ = list(fb.feature_names)                  # exact match
    assert rk.assert_feature_parity() is rk                     # no raise, chainable


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
