"""Carve a temporal third selection set from the train split (Tier-0 #3).

Why: every model-selection decision (~26 nb74 stages) was made on the SAME
official test split, deltas often < inter-seed noise -> overfitting the holdout.
Reserve the LATEST fraction of train sessions (by session_date) as an internal
selection set. Build LGBM features only from the EARLIER sessions (pass the
emitted train-id / holdout-id files to the feature builder), select configs on
the reserved tail, and touch split='test' rarely so it stays an honest final
check. A time-based (not random) split also avoids leaking future artists into
the selection set -- the documented failure mode where every signal looked flat
on dev.

Usage:
    python scripts/carve_temporal_selection_set.py --frac 0.15 \
        --out data/temporal_selection_split.json

Writes JSON: {train_ids: [...], holdout_ids: [...], cutoff_date, counts,
overlap_with_test}. Downstream: build_lgbm_features over train_ids only; evaluate
candidate configs on holdout_ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def select_temporal_holdout(sessions, frac: float = 0.15, n: int | None = None):
    """Reserve the latest `n` (or `frac`) sessions by date as the holdout set.

    Args:
        sessions: iterable of dicts with 'session_id' and 'session_date'
            (date string sortable as YYYY-MM-DD).
        frac: fraction reserved as holdout when `n` is None.
        n: explicit holdout size; overrides `frac` when given.

    Returns:
        (train_ids, holdout_ids): two disjoint sets of session_id covering the
        input. Holdout = the latest sessions, ordered by (session_date,
        session_id) so ties on date break deterministically by id.
    """
    items = [(str(s["session_date"]), str(s["session_id"])) for s in sessions]
    items.sort()  # ascending by (date, id); the tail is the latest
    total = len(items)
    if n is None:
        n = round(frac * total)
    n = max(0, min(n, total))
    holdout = {sid for _, sid in items[total - n:]} if n else set()
    train = {sid for _, sid in items[: total - n]} if n else {sid for _, sid in items}
    return train, holdout


def _cutoff_date(sessions, holdout_ids):
    """Earliest session_date among the holdout = the temporal cutoff."""
    dates = [str(s["session_date"]) for s in sessions
             if str(s["session_id"]) in holdout_ids]
    return min(dates) if dates else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dataset", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    ap.add_argument("--out", default="data/temporal_selection_split.json")
    args = ap.parse_args()

    from datasets import load_dataset
    train = load_dataset(args.dataset, split="train")
    sessions = [{"session_id": r["session_id"], "session_date": r["session_date"]}
                for r in train]
    train_ids, holdout_ids = select_temporal_holdout(
        sessions, frac=args.frac, n=args.n)

    # Sanity: the official test split must not overlap the selection set.
    test = load_dataset(args.dataset, split="test")
    test_ids = {r["session_id"] for r in test}
    overlap = sorted(holdout_ids & test_ids)

    out = {
        "train_ids": sorted(train_ids),
        "holdout_ids": sorted(holdout_ids),
        "cutoff_date": _cutoff_date(sessions, holdout_ids),
        "counts": {"train": len(train_ids), "holdout": len(holdout_ids),
                   "total": len(sessions), "test": len(test_ids)},
        "overlap_with_test": overlap,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[carve] train={out['counts']['train']} holdout={out['counts']['holdout']} "
          f"cutoff>={out['cutoff_date']} overlap_with_test={len(overlap)}")
    print(f"[carve] wrote {args.out}")
    if overlap:
        print("[carve] WARNING: holdout overlaps the official test split!", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
