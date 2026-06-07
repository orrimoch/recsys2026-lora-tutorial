"""Tier-0 #5: combine the K per-fold OOF feature parquets into one leak-free
train parquet. The K folds must PARTITION the sample (disjoint by session_id);
if any session appears in two folds the OOF guarantee is broken, so combine must
refuse rather than silently double-count / reintroduce the leak.
"""
import pandas as pd
import pytest

from scripts.concat_oof_features import combine_oof_parquets


def _frame(session_ids):
    return pd.DataFrame({
        "session_id": session_ids,
        "turn_number": [1] * len(session_ids),
        "candidate_tid": [f"t{i}" for i in range(len(session_ids))],
        "label": [0] * len(session_ids),
    })


def test_combines_disjoint_folds():
    a = _frame(["s1", "s2"])
    b = _frame(["s3", "s4"])
    out = combine_oof_parquets([a, b])
    assert len(out) == 4
    assert set(out["session_id"]) == {"s1", "s2", "s3", "s4"}
    assert list(out.columns) == list(a.columns)


def test_rejects_session_overlap_across_folds():
    a = _frame(["s1", "s2"])
    b = _frame(["s2", "s3"])  # s2 leaks across folds
    with pytest.raises(ValueError, match="overlap"):
        combine_oof_parquets([a, b])


def test_requires_at_least_one_frame():
    with pytest.raises(ValueError):
        combine_oof_parquets([])
