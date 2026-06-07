"""Tier-2 #4.3: lambdarank alignment + drop-all-negative groups + multi-seed
bagging.

- truncation_level=20 aligns the lambdarank objective with the nDCG@20 metric.
- drop_all_negative_groups removes (session,turn) groups with no positive (with
  recall@100 ~0.5, ~half the groups have zero gold rows -> zero gradient anyway).
- multi-seed bagging trains N boosters with different seeds and averages their
  scores at serve (variance reduction; cannot worsen the in-sample leak since it
  adds no new feature).
"""
import inspect

import numpy as np
import pandas as pd

from scripts.train_lgbm_ranker import drop_all_negative_groups, main as train_main
from mcrs.rerankers.lgbm_rerank import LGBM_RERANKER


def _df(rows):
    # rows: list of (session_id, turn, label)
    return pd.DataFrame({
        "session_id": [r[0] for r in rows],
        "turn_number": [r[1] for r in rows],
        "candidate_tid": [f"t{i}" for i in range(len(rows))],
        "label": [r[2] for r in rows],
    })


def test_drop_all_negative_groups_removes_zero_positive_groups():
    df = _df([("s1", 1, 0), ("s1", 1, 1),   # group has a positive -> kept
             ("s2", 1, 0), ("s2", 1, 0)])   # all-negative -> dropped
    out = drop_all_negative_groups(df)
    assert set(out["session_id"]) == {"s1"}
    assert len(out) == 2


def test_drop_all_negative_groups_keeps_when_positive_present():
    df = _df([("s1", 1, 1), ("s1", 2, 0), ("s1", 2, 1)])  # both groups have a pos
    out = drop_all_negative_groups(df)
    assert len(out) == 3


def test_truncation_level_aligned_to_20():
    src = inspect.getsource(train_main)
    assert "lambdarank_truncation_level" in src and "20" in src


# ---- bagging: averaged prediction at serve ----

class _FakeBooster:
    def __init__(self, vals):
        self.vals = np.asarray(vals, dtype=float)

    def predict(self, X):
        return self.vals


def test_predict_averages_over_boosters():
    r = LGBM_RERANKER.__new__(LGBM_RERANKER)
    r.boosters = [_FakeBooster([1.0, 2.0, 3.0]), _FakeBooster([3.0, 2.0, 1.0])]
    out = r._predict(np.zeros((3, 1)))
    assert np.allclose(out, [2.0, 2.0, 2.0])


def test_predict_single_booster_unchanged():
    r = LGBM_RERANKER.__new__(LGBM_RERANKER)
    r.boosters = [_FakeBooster([0.1, 0.9, 0.5])]
    assert np.allclose(r._predict(np.zeros((3, 1))), [0.1, 0.9, 0.5])
