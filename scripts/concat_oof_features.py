"""Combine K per-fold OOF feature parquets into one leak-free train parquet (Tier-0 #5).

The OOF cross-fit (the fix for the in-sample SASRec/cf-bpr feature leak that made
internal val anti-correlated with dev) works like this:

  1. Train K SASRec models, each EXCLUDING fold k:
       python scripts/train_sasrec.py --oof-fold k --oof-num-folds K ...   (x K, GPU)
  2. Build K per-fold feature parquets, each SELECTING fold k and scoring it with
     the fold-k model that never saw it (same --seed/--n-sessions across all K):
       python scripts/build_lgbm_features.py --oof-fold k --oof-num-folds K \
           --sasrec-model-dir <fold-k model> ... --out fold_k.parquet           (x K)
  3. THIS script: concatenate the K parquets into the leak-free OOF train parquet.

The K folds partition the sample, so every session's SASRec-derived features were
produced by a model that did not train on it. combine_oof_parquets refuses if the
folds overlap by session (which would silently reintroduce the leak).

Usage:
    python scripts/concat_oof_features.py \
        --inputs fold_0.parquet fold_1.parquet ... fold_4.parquet \
        --out experiments/cache/retrieval_v2/lgbm/lgbm_train_oof.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def combine_oof_parquets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-fold OOF feature frames after verifying they partition the
    sample (disjoint by session_id). Raises ValueError on empty input or any
    cross-fold session overlap (a broken OOF guarantee)."""
    if not frames:
        raise ValueError("combine_oof_parquets requires at least one frame")

    seen: set = set()
    for i, df in enumerate(frames):
        ids = set(df["session_id"].unique())
        overlap = seen & ids
        if overlap:
            sample = sorted(map(str, overlap))[:5]
            raise ValueError(
                f"fold {i} session_id overlap with earlier folds "
                f"({len(overlap)} sessions, e.g. {sample}); OOF guarantee broken")
        seen |= ids

    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="per-fold OOF feature parquets (one per fold)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = [pd.read_parquet(p) for p in args.inputs]
    combined = combine_oof_parquets(frames)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)
    n_sess = combined["session_id"].nunique()
    pos = int(combined["label"].sum()) if "label" in combined.columns else -1
    print(f"[oof-concat] {len(args.inputs)} folds -> {len(combined)} rows, "
          f"{n_sess} sessions, positives={pos}")
    print(f"[oof-concat] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
