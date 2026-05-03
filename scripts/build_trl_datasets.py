"""Convert the GPA-labeled reward dataset into TRL-validated training parquets.

Per the huggingface-llm-trainer skill: "50%+ of training failures are due to
dataset format issues" — DPO especially strict about column names. This script
takes the existing `data/reward_train.parquet` (built by
`scripts/build_reward_dataset.py`, schema (text_a, text_b, label, session_id,
user_id, turn_number, goal_category, split)) and emits three TRL-shape
parquets under `data/trl/`:

  data/trl/kto.parquet
      KTO (Kahneman-Tversky Optimization) — TRL `KTOTrainer` expects:
          prompt:    str   (the context the response is judged against)
          completion: str  (the response being scored)
          label:     bool  (True = desirable, False = undesirable)
      One-to-one map from reward_train: text_a → prompt, text_b → completion,
      label==1 → True. Used for B1 (W4) — cheapest first-pass post-training.

  data/trl/dpo.parquet
      DPO (Direct Preference Optimization) — TRL `DPOTrainer` expects:
          prompt:   str
          chosen:   str
          rejected: str
      Random POS↔NEG pairing within seed. Each POS row pairs with a NEG row
      and the POS's `text_a` becomes the shared `prompt`. Note: this is the
      *vanilla DPO* format. **S-DPO (W5, plan §6.3 B2 conditional) needs a
      richer 1-pos-N-negs construction with hard-track and response-mutation
      negatives** — that's a separate dataset built later (or via on-the-fly
      collation inside the S-DPO trainer).

  data/trl/grpo_prompts.parquet
      GRPO (Group Relative Policy Optimization) — TRL `GRPOTrainer` expects:
          prompt: str
      Prompt-only — the policy generates completions online, our reward
      functions in `scripts/reward_fns.py` score them. Used for B3 (W6)
      Rank-GRPO main loop.

Validation: each output parquet is verified to have the required column
schema before write. If columns are missing/wrong-typed, the script raises
with a clear message rather than silently producing an unusable artifact.

Usage:
    # First build the upstream parquet if it doesn't exist:
    python scripts/build_reward_dataset.py --n-sessions 15000

    # Then convert to TRL formats:
    python scripts/build_trl_datasets.py
    python scripts/build_trl_datasets.py --in data/reward_train.parquet --out-dir data/trl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "reward_train.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "trl"


# ---------------------------------------------------------------------------
# TRL schema contracts (copy of the documented requirements; assert these on
# every output to prevent silent format drift breaking GPU runs).
# ---------------------------------------------------------------------------

KTO_REQUIRED = {"prompt": str, "completion": str, "label": bool}
DPO_REQUIRED = {"prompt": str, "chosen": str, "rejected": str}
GRPO_REQUIRED = {"prompt": str}


def _validate_schema(df: pd.DataFrame, required: dict, name: str) -> None:
    """Assert df has exactly the required columns + types. Raise on mismatch."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} parquet missing required TRL columns: {missing}. "
            f"Got columns: {sorted(df.columns)}. "
            f"Required: {sorted(required)}."
        )
    if df.empty:
        raise ValueError(f"{name} parquet is empty (0 rows). Refusing to write.")
    # Type spot-checks — pandas keeps dtypes via the underlying numpy array.
    for col, expected in required.items():
        sample = df[col].iloc[0]
        if expected is bool:
            if not isinstance(sample, (bool, np.bool_)):
                raise ValueError(
                    f"{name}.{col} dtype expected bool, got {type(sample).__name__} "
                    f"(value: {sample!r}). Cast via .astype(bool) before write."
                )
        elif expected is str:
            if not isinstance(sample, str):
                raise ValueError(
                    f"{name}.{col} dtype expected str, got {type(sample).__name__} "
                    f"(value: {sample!r})."
                )


# ---------------------------------------------------------------------------
# Format builders
# ---------------------------------------------------------------------------

def build_kto(df: pd.DataFrame) -> pd.DataFrame:
    """One-to-one map → (prompt, completion, label: bool).

    Preserves the train/val `split` column from the input so downstream can
    use it for eval.
    """
    if "text_a" not in df.columns or "text_b" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"KTO input must have columns (text_a, text_b, label). "
            f"Got: {sorted(df.columns)}."
        )
    out = pd.DataFrame({
        "prompt": df["text_a"].astype(str).values,
        "completion": df["text_b"].astype(str).values,
        "label": (df["label"].astype(int) == 1).astype(bool).values,
    })
    # Carry split through if present (train/val).
    if "split" in df.columns:
        out["split"] = df["split"].astype(str).values
    _validate_schema(out, KTO_REQUIRED, "kto")
    return out


def _pair_pos_neg(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Pair POS↔NEG within a single homogeneous slice → (prompt, chosen, rejected).

    Caller is responsible for slicing by split before calling — this function
    samples only from the rows it receives, which is what guarantees
    no-cross-split leakage in the parent `build_dpo_random`. Both the initial
    sampling AND the degenerate-pair recovery draw from the same `neg` pool.
    """
    pos = df[df["label"].astype(int) == 1]
    neg = df[df["label"].astype(int) == 0]
    if pos.empty or neg.empty:
        raise ValueError(
            f"Slice has no POS or NEG rows: pos={len(pos)} neg={len(neg)}. "
            f"DPO needs both within each split."
        )
    n = min(len(pos), len(neg))
    pos_sample = pos.sample(n=n, random_state=seed).reset_index(drop=True)
    neg_sample = neg.sample(n=n, random_state=seed + 1).reset_index(drop=True)
    out = pd.DataFrame({
        "prompt": pos_sample["text_a"].astype(str).values,
        "chosen": pos_sample["text_b"].astype(str).values,
        "rejected": neg_sample["text_b"].astype(str).values,
    })
    n_dup = int((out["chosen"] == out["rejected"]).sum())
    if n_dup > 0:
        rng = np.random.default_rng(seed + 7)
        for idx in out.index[out["chosen"] == out["rejected"]]:
            for _ in range(20):
                replacement = neg.sample(
                    n=1, random_state=int(rng.integers(0, 1_000_000))
                ).iloc[0]
                if replacement["text_b"] != out.loc[idx, "chosen"]:
                    out.loc[idx, "rejected"] = str(replacement["text_b"])
                    break
    return out


def build_dpo_random(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Random POS↔NEG pairing → (prompt, chosen, rejected).

    Each pair: prompt = POS row's text_a, chosen = POS text_b, rejected = NEG text_b.
    The chosen row's prompt is used (NEG row's prompt is NOT used as the shared
    prompt — DPO contract says one prompt with chosen+rejected for it).

    No-leakage discipline: when the input has a `split` column, pairing happens
    INDEPENDENTLY within each split. A train-tagged pair's `rejected` text is
    sourced only from train NEG rows (never val). Without this, a val response
    string would appear in the train set's rejected position — silent leakage
    that breaks the train-only / val-only invariant the upstream
    `build_reward_dataset.py --split-val` was designed to enforce.

    Splits with no POS or no NEG rows are skipped with a warning rather than
    aborting the whole build (small val partitions can legitimately end up
    one-sided after the by-session_id split).

    Note: this is *vanilla DPO* (one rejected per pair). S-DPO with N hard
    negatives (W5) requires a different builder — TODO for plan §6.3 B2.
    """
    if "label" not in df.columns:
        raise ValueError(
            f"DPO input must have label column. Got: {sorted(df.columns)}."
        )

    if "split" in df.columns:
        parts = []
        for split_val, sub in df.groupby("split", sort=True):
            try:
                paired = _pair_pos_neg(sub, seed)
            except ValueError as e:
                print(
                    f"[trl-data] WARNING: skipping split={split_val!r} for DPO ({e})",
                    file=sys.stderr,
                )
                continue
            paired["split"] = str(split_val)
            parts.append(paired)
        if not parts:
            raise ValueError(
                "DPO input had a `split` column but no split produced pairs "
                "(every split was missing POS or NEG)."
            )
        out = pd.concat(parts, ignore_index=True)
    else:
        out = _pair_pos_neg(df, seed)

    _validate_schema(out, DPO_REQUIRED, "dpo")
    n_dup_final = int((out["chosen"] == out["rejected"]).sum())
    if n_dup_final > 0:
        raise ValueError(f"{n_dup_final} DPO pairs have chosen == rejected after recovery.")
    return out


def build_grpo_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Prompt-only → (prompt). Deduplicated.

    GRPO is online RL: the policy generates completions and the reward function
    scores them. We just need a corpus of prompts to roll out from.

    No-leakage discipline: when the input has a `split` column, dedup happens
    per-split AND any prompt that appears in train is removed from val. This
    prevents a prompt the model trained on from being treated as a held-out
    val rollout source. Train-precedence (rather than val-precedence) keeps
    the train slice complete; val ends up holding only strictly-unseen prompts.
    """
    if "text_a" not in df.columns:
        raise ValueError(f"GRPO input must have column 'text_a'. Got: {sorted(df.columns)}.")

    if "split" in df.columns:
        train_prompts = set(
            df.loc[df["split"] == "train", "text_a"].astype(str)
        )
        train_uniq = (
            df[df["split"] == "train"]
            .drop_duplicates(subset=["text_a"])[["text_a", "split"]]
            .reset_index(drop=True)
        )
        # All non-train rows: dedup within their own split and drop any prompt
        # that already appears in train. Generalizes to "val" plus any other
        # split label a future caller might introduce.
        non_train = df[df["split"] != "train"].copy()
        non_train = non_train[~non_train["text_a"].astype(str).isin(train_prompts)]
        non_train_uniq = (
            non_train.drop_duplicates(subset=["text_a"])[["text_a", "split"]]
            .reset_index(drop=True)
        )
        combined = pd.concat([train_uniq, non_train_uniq], ignore_index=True)
        out = pd.DataFrame({
            "prompt": combined["text_a"].astype(str).values,
            "split": combined["split"].astype(str).values,
        })
    else:
        unique = (
            df["text_a"]
            .astype(str)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        out = pd.DataFrame({"prompt": unique.values})

    _validate_schema(out, GRPO_REQUIRED, "grpo")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in", dest="in_path", default=str(DEFAULT_IN),
        help=f"Input parquet (default: {DEFAULT_IN})",
    )
    p.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help=f"Output directory for KTO/DPO/GRPO parquets (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--formats", default="kto,dpo,grpo",
        help="Comma-separated subset of formats to build (default: kto,dpo,grpo)",
    )
    args = p.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(
            f"ERROR: input parquet not found at {in_path}.\n"
            f"Run `python scripts/build_reward_dataset.py` first to produce it.",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[trl-data] loading {in_path}")
    df = pd.read_parquet(in_path)
    n_pos = int((df["label"].astype(int) == 1).sum()) if "label" in df.columns else 0
    n_neg = int((df["label"].astype(int) == 0).sum()) if "label" in df.columns else 0
    print(f"[trl-data] input rows: {len(df):,}  (POS={n_pos:,}  NEG={n_neg:,})")

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    written = []

    if "kto" in formats:
        kto = build_kto(df)
        out = out_dir / "kto.parquet"
        kto.to_parquet(out, index=False)
        size_mb = os.path.getsize(out) / 1e6
        n_t = int(kto["label"].sum())
        n_f = int((~kto["label"]).sum())
        print(f"[trl-data] wrote KTO   → {out} ({len(kto):,} rows, "
              f"True={n_t:,} False={n_f:,}, {size_mb:.1f} MB)")
        written.append(("kto", out))

    if "dpo" in formats:
        dpo = build_dpo_random(df, seed=args.seed)
        out = out_dir / "dpo.parquet"
        dpo.to_parquet(out, index=False)
        size_mb = os.path.getsize(out) / 1e6
        print(f"[trl-data] wrote DPO   → {out} ({len(dpo):,} pairs, {size_mb:.1f} MB)")
        written.append(("dpo", out))

    if "grpo" in formats:
        grpo = build_grpo_prompts(df)
        out = out_dir / "grpo_prompts.parquet"
        grpo.to_parquet(out, index=False)
        size_mb = os.path.getsize(out) / 1e6
        print(f"[trl-data] wrote GRPO  → {out} ({len(grpo):,} prompts, {size_mb:.1f} MB)")
        written.append(("grpo", out))

    print(f"[trl-data] {len(written)} format(s) written.")
    print("[trl-data] Next step: validate with the dataset inspector before training:")
    print("           hf_jobs(\"uv\", {")
    print("             \"script\": \"https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py\",")
    print("             \"script_args\": [\"--dataset\", \"<your_hub_dataset_after_upload>\", \"--split\", \"train\"]")
    print("           })")
    return 0


if __name__ == "__main__":
    sys.exit(main())
