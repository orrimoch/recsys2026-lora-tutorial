"""Tier-0 #4: the internal LGBM val is leaky (98.6% session overlap with train +
shares the full-train SASRec/cf-bpr pool), which is exactly why it has behaved as
an anti-correlated 'trap'. It is acceptable ONLY for early stopping. The trainer
must surface this every run so val_ndcg@20 is never used for model selection;
selection happens on the temporal holdout / split='test'.
"""
import inspect

from scripts.train_lgbm_ranker import internal_val_warning, main


def test_internal_val_warning_flags_early_stopping_only():
    w = internal_val_warning().lower()
    assert "early stop" in w or "early-stop" in w
    assert "do not select" in w or "not for selection" in w or "never select" in w


def test_internal_val_warning_points_to_honest_metric():
    w = internal_val_warning().lower()
    assert "holdout" in w or "split='test'" in w or "split=test" in w or "temporal" in w


def test_trainer_emits_the_warning():
    assert "internal_val_warning" in inspect.getsource(main), \
        "main() must print internal_val_warning() so val is never used for selection"
