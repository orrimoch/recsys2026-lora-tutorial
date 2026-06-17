from mcrs.training.ce_data import normalize_within_pool
from mcrs.training.ce_finetune import oof_ce_scores

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
        return 1.0
    scores = oof_ce_scores(turns, folds=3, seed=0, fit_fn=fake_fit, score_fn=fake_score)
    assert len(scores) == 6                       # every train row scored exactly once
