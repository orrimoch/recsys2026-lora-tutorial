"""OOF (out-of-fold) cross-fitting for the SASRec-derived LGBM feature.

The leak fix requires that build_lgbm_features scores each train session with a
SASRec model that did NOT train on that session. That guarantee holds ONLY if
train_sasrec (which EXCLUDES fold k from training) and build_lgbm_features
(which SELECTS fold k for feature-building) compute the SAME fold for every
session. If the two ever drift, the OOF guarantee silently breaks and the leak
returns undetected — so this test pins the exact, shared, deterministic mapping.
"""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _session_fold_fns():
    """Import session_fold from BOTH producers. Skips cleanly until they exist
    (TDD: this test is written before the functions are added)."""
    fns = []
    for modname, rel in [("train_sasrec", "scripts/train_sasrec.py"),
                         ("build_lgbm_features", "scripts/build_lgbm_features.py")]:
        try:
            mod = _load(modname, rel)
        except Exception as e:  # heavy import deps may be missing locally
            pytest.skip(f"could not import {rel}: {e!r}")
        if not hasattr(mod, "session_fold"):
            pytest.skip(f"{rel} has no session_fold yet (TDD pre-impl)")
        fns.append(mod.session_fold)
    return fns


def test_both_producers_agree_on_every_fold():
    """train_sasrec.session_fold and build_lgbm_features.session_fold must be
    byte-for-byte identical for a wide range of ids and fold counts."""
    f_train, f_feat = _session_fold_fns()
    ids = [f"sess_{i}" for i in range(500)] + ["abc-123", "00000", "", "x"]
    for k in (2, 3, 5, 10):
        for sid in ids:
            assert f_train(sid, k) == f_feat(sid, k), (sid, k)


def test_fold_in_range_and_deterministic():
    f_train, _ = _session_fold_fns()
    for k in (2, 5, 10):
        for i in range(200):
            sid = f"s{i}"
            v = f_train(sid, k)
            assert 0 <= v < k
            assert v == f_train(sid, k)  # deterministic across calls


def test_folds_roughly_balanced():
    """Hash partition should spread sessions across folds (no degenerate fold)."""
    f_train, _ = _session_fold_fns()
    K = 5
    counts = [0] * K
    for i in range(5000):
        counts[f_train(f"session-{i}", K)] += 1
    # every fold gets a healthy share (expected 1000 each; allow wide slack)
    assert all(c > 700 for c in counts), counts
