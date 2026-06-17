"""K3b — cross-encoder fine-tune training loop + OOF orchestration."""
from __future__ import annotations

from mcrs.training.ce_data import assign_session_folds


def oof_ce_scores(turns, *, folds, seed, fit_fn, score_fn):
    """Leak-free per-row CE scores via k-fold session-disjoint cross-fitting.

    turns: list of (session_id, turn_number). fit_fn(train_rows, **)->model.
    score_fn(model, row)->float. Each row is scored by a model trained on the OTHER folds only.
    """
    sids = [s for s, _ in turns]
    fold_of = assign_session_folds(sids, k=folds, seed=seed)
    rows = [{"session_id": s, "turn": t, "fold": f} for (s, t), f in zip(turns, fold_of)]
    out = {}
    for held in range(folds):
        train_rows = [(r["session_id"], r["turn"], r["fold"]) for r in rows if r["fold"] != held]
        model = fit_fn(train_rows, fold=held)
        for r in rows:
            if r["fold"] == held:
                out[(r["session_id"], r["turn"])] = score_fn(model, r)
    return out
