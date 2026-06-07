"""Temporal third selection set (Tier-0 #3).

~26 nb74 stages were all selected on the same official test split with sub-0.005
deltas = overfitting the holdout. Reserve the LATEST fraction of train sessions
(by session_date) as a third selection set, build LGBM features only from the
EARLIER sessions, and use the reserved tail for config selection so split='test'
stays a rarely-touched honest final check. The carve must be deterministic and
yield disjoint train / holdout id sets that cover the input.
"""
from scripts.carve_temporal_selection_set import select_temporal_holdout


def _sess(i, date):
    return {"session_id": f"s{i}", "session_date": date}


def test_holdout_is_the_latest_fraction_by_date():
    sessions = [_sess(i, f"201{i}-01-01") for i in range(10)]  # s9 latest
    train, holdout = select_temporal_holdout(sessions, frac=0.2)
    assert holdout == {"s9", "s8"}
    assert "s0" in train and "s9" not in train


def test_train_and_holdout_are_disjoint_and_cover_all():
    sessions = [_sess(i, f"20{10+i}-06-01") for i in range(8)]
    train, holdout = select_temporal_holdout(sessions, frac=0.25)
    assert train.isdisjoint(holdout)
    assert train | holdout == {f"s{i}" for i in range(8)}
    assert len(holdout) == 2


def test_ties_on_date_break_deterministically_by_session_id():
    # all same date -> ordering falls back to session_id; latest ids go to holdout
    sessions = [_sess(i, "2018-12-31") for i in range(5)]  # s0..s4
    train, holdout = select_temporal_holdout(sessions, n=2)
    assert holdout == {"s4", "s3"}  # highest session_ids by string order


def test_n_overrides_frac():
    sessions = [_sess(i, f"2018-12-{i+1:02d}") for i in range(10)]
    train, holdout = select_temporal_holdout(sessions, frac=0.9, n=3)
    assert len(holdout) == 3
    assert len(train) == 7
