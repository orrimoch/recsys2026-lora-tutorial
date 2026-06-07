"""Tier-2 follow-up: early-stop on the temporal holdout, not the leaky internal
val. The Stage-11 retrain overfit (best_iter 998/1000) because early stopping used
the internal val, which is leaky and anti-correlated with dev. split_by_holdout
lets the trainer carve the held-out (latest-by-date) sessions out of the train
parquet and early-stop on them instead — a leak-free selection signal.
"""
import pandas as pd
import pytest

from scripts.train_lgbm_ranker import split_by_holdout


def _df(session_ids):
    return pd.DataFrame({
        "session_id": session_ids,
        "turn_number": [1] * len(session_ids),
        "candidate_tid": [f"t{i}" for i in range(len(session_ids))],
        "label": [0] * len(session_ids),
    })


def test_split_by_holdout_partitions_rows():
    df = _df(["s1", "s2", "s3", "s4"])
    train, hold = split_by_holdout(df, {"s3", "s4"})
    assert set(train["session_id"]) == {"s1", "s2"}
    assert set(hold["session_id"]) == {"s3", "s4"}
    assert len(train) + len(hold) == len(df)


def test_split_by_holdout_disjoint_and_complete():
    df = _df(["a", "a", "b", "c", "c", "c"])  # multi-row sessions
    train, hold = split_by_holdout(df, {"c"})
    assert set(train["session_id"]) == {"a", "b"}
    assert set(hold["session_id"]) == {"c"}
    assert len(hold) == 3  # all rows of session c


def test_split_by_holdout_empty_holdout_raises():
    df = _df(["s1", "s2"])
    with pytest.raises(ValueError, match="holdout"):
        split_by_holdout(df, {"not-present"})
