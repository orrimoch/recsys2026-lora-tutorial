"""Combine the train and dev reward parquets into one for the W7 retrain pass.

Per `project_submission_prep.md`: two-phase submission protocol.
  1. Iterate on TRAIN-ONLY for W4/W5/W6 (already done — `data/reward_train.parquet`).
  2. For the final Blind-B submission, retrain Component-B on TRAIN+DEV combined.

This script is the data-prep step for phase 2. It expects two reward parquets
already produced by `build_reward_dataset.py`:

    python scripts/build_reward_dataset.py --hf-split train --out data/reward_train.parquet
    python scripts/build_reward_dataset.py --hf-split test  --out data/reward_dev.parquet --split-val 0.0

Then concatenates them with a `data_origin` marker column ("train" / "dev")
and writes one combined parquet at `data/reward_train_plus_dev.parquet`.

The downstream pipeline (augment_envelope.py → build_grpo_dataset.py) is
unchanged: it sees a single parquet with the same schema. The `split` column
is unified to "train" so the envelope augmenter / SDPO / GRPO builders don't
accidentally hold out the dev rows as a val set — provenance lives only in
the new `data_origin` column.

Usage:
    python scripts/build_train_plus_dev.py \\
        --train data/reward_train.parquet \\
        --dev   data/reward_dev.parquet \\
        --out   data/reward_train_plus_dev.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_IN = REPO_ROOT / "data" / "reward_train.parquet"
DEFAULT_DEV_IN = REPO_ROOT / "data" / "reward_dev.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "reward_train_plus_dev.parquet"


def concat_train_and_dev(
    train_path: "Path | str",
    dev_path: "Path | str",
    out_path: "Path | str",
) -> dict:
    """Concat the train and dev reward parquets with a data_origin marker.

    Args:
        train_path: parquet from `build_reward_dataset.py --hf-split train`.
        dev_path:   parquet from `build_reward_dataset.py --hf-split test`.
        out_path:   destination parquet.

    Returns:
        dict with `n_train`, `n_dev`, `n_total` row counts.

    Raises:
        FileNotFoundError: when either input parquet is missing.
        ValueError:         when the two inputs have mismatched schemas.
    """
    train_p = Path(train_path)
    dev_p = Path(dev_path)
    if not train_p.exists():
        raise FileNotFoundError(f"train parquet not found: {train_p}")
    if not dev_p.exists():
        raise FileNotFoundError(f"dev parquet not found: {dev_p}")

    train_df = pd.read_parquet(train_p)
    dev_df = pd.read_parquet(dev_p)

    # Strict schema check — prevents NaN-padded columns from silently slipping
    # through into the W7 retrain dataset.
    train_cols = set(train_df.columns)
    dev_cols = set(dev_df.columns)
    if train_cols != dev_cols:
        only_train = sorted(train_cols - dev_cols)
        only_dev = sorted(dev_cols - train_cols)
        raise ValueError(
            f"schema mismatch between train and dev reward parquets. "
            f"only-in-train: {only_train}; only-in-dev: {only_dev}. "
            f"Re-run build_reward_dataset.py for both splits with the same code."
        )

    train_df = train_df.copy()
    dev_df = dev_df.copy()
    train_df["data_origin"] = "train"
    dev_df["data_origin"] = "dev"

    combined = pd.concat([train_df, dev_df], ignore_index=True)
    # Unify `split` so the downstream envelope/SDPO/GRPO builders see one
    # cohesive train slice. Original provenance lives in `data_origin`.
    combined["split"] = "train"

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_p, index=False)

    return {
        "n_train": len(train_df),
        "n_dev": len(dev_df),
        "n_total": len(combined),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=str(DEFAULT_TRAIN_IN),
                   help="reward parquet built from HF split=train (W1-W5 already produced).")
    p.add_argument("--dev", default=str(DEFAULT_DEV_IN),
                   help="reward parquet built from HF split=test (the dev set; build with --split-val 0.0).")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="combined output parquet for the W7 retrain pass.")
    args = p.parse_args(argv)

    try:
        stats = concat_train_and_dev(args.train, args.dev, args.out)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[train+dev] wrote {stats['n_total']:,} rows → {out_path} ({size_mb:.1f} MB)")
    print(f"[train+dev] origin breakdown:")
    print(f"  train: {stats['n_train']:,}")
    print(f"  dev:   {stats['n_dev']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
