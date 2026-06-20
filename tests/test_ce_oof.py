from mcrs.training.ce_data import normalize_within_pool
from mcrs.training.ce_finetune import build_ce_ft_lookup, oof_ce_scores

def test_normalize_within_pool_minmax():
    out = normalize_within_pool({"a": 0.0, "b": 5.0, "c": 10.0})
    assert out["a"] == 0.0 and out["c"] == 1.0 and out["b"] == 0.5

def test_oof_scores_use_only_held_out_fold_model():
    # Fake fine-tune fn returns a model tagged with the set of folds it trained on.
    # A row's score must come from a model that did NOT train on that row's fold.
    turns = [("s%d" % i, i) for i in range(6)]   # 6 sessions
    def fake_fit(train_rows, **kw):
        trained_folds = frozenset(f for (_sid, _t, f) in train_rows)
        return ("model", trained_folds)
    def fake_score(model, row):
        _name, trained_folds = model
        assert row["fold"] not in trained_folds   # INVARIANT: never score a row with a model that saw its fold
        return {"cand_a": 1.0, "cand_b": 0.5}     # per-candidate scores for this turn's pool
    scores = oof_ce_scores(turns, folds=3, seed=0, fit_fn=fake_fit, score_fn=fake_score)
    assert len(scores) == 12                      # 6 turns x 2 candidates, each scored exactly once
    assert all(len(k) == 3 for k in scores)       # keys are (session_id, turn_number, track_id)
    assert scores[("s0", 0, "cand_a")] == 1.0


# ----- Step 2b: leak-safe fine-tuned-CE feature lookup (OOF train + all-train serve) -----
def _fake_fit(train_rows, **kw):
    return frozenset(f for (_s, _t, f) in train_rows)   # the folds this model trained on


def _fake_score(model, row):
    assert row["fold"] not in model                     # never score a row a model saw (None for serve)
    return {"a": 1.0, "b": 0.5}                          # per-candidate pool scores


def test_build_ce_ft_lookup_oof_train_plus_disjoint_serve():
    # TRAIN turns get OOF scores (held-out-fold model); SERVE turns (disjoint sessions) get scored by
    # one all-train model. Both end up in one (sid, turn, tid) -> score lookup for make_ce_feature_fn.
    train = [("s%d" % i, 1) for i in range(6)]
    serve = [("t%d" % i, 5) for i in range(2)]           # disjoint sessions
    look = build_ce_ft_lookup(train, serve, folds=3, seed=0, fit_fn=_fake_fit, score_fn=_fake_score)
    assert len(look) == 16                                # (6 train + 2 serve) turns x 2 candidates
    assert look[("s0", 1, "a")] == 1.0                   # a train turn scored (leak-free via OOF)
    assert look[("t0", 5, "b")] == 0.5                   # a serve turn scored (all-train model)
    assert all(len(k) == 3 for k in look)


def test_build_ce_ft_lookup_raises_when_train_and_serve_share_a_session():
    import pytest
    with pytest.raises(ValueError):                       # overlap would leak the serve model into K2 train
        build_ce_ft_lookup([("s0", 1)], [("s0", 5)], folds=2, seed=0,
                            fit_fn=_fake_fit, score_fn=_fake_score)
