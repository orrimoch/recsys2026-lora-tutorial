"""Train a distilled cross-encoder judge that approximates the Gemini-judge
signal on (context, response) text pairs.

Why this exists (W6 review Option B refactor): `r_judge_stub` in
`reward_fns.py` returns 0.0 always, so the W6 GRPO reward effectively
collapses to R_rule (0.15) + R_format (0.05) = 0.20 of total weight, of
which R_rule has AUC 0.51 vs Gemini (random). This script trains a small
cross-encoder on `data/reward_calibration_anchors.parquet` (210k labeled
rows: 105k POS source A + 105k synthetic-perturbation source D, anchors
∈ {1.0, 5.0}) so R_judge can carry actual Gemini-aligned gradient at
training time.

Pipeline (two modes):
  --mode prepare : pure-pandas data prep. Reads anchor parquet, drops
                   source-B (NaN anchor) rows, builds (context, response,
                   label∈[0,1]) JSONL pairs, session-level train/val split.
                   ~1 min on CPU.
  --mode train   : GPU step. Loads the JSONL, fine-tunes a small
                   cross-encoder (default `cross-encoder/ms-marco-MiniLM-L-6-v2`)
                   with MSE loss against the [0,1]-normalized label.
                   ~30 min on A100 / ~2 hr on T4. Pushes to Hub.

Usage:
    # Step 1 — local prep (offline, no GPU):
    python scripts/train_distilled_judge.py --mode prepare \\
        --anchors data/reward_calibration_anchors.parquet \\
        --out-dir data/distilled_judge

    # Step 2 — Colab training:
    python scripts/train_distilled_judge.py --mode train \\
        --data-dir data/distilled_judge \\
        --hub-repo orrimoch/recsys2026-distilled-judge

Output: a Hub repo with the fine-tuned cross-encoder. `reward_fns.py`
loads it via `DistilledJudge(checkpoint=hub_repo)`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHORS = REPO_ROOT / "data" / "reward_calibration_anchors.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "distilled_judge"

# Pair-building template. Includes the Personalization signal (country/age/
# gender) — Gemini's two judging axes are Personalization + Explanation
# Quality, so the cross-encoder must SEE the profile data to learn
# Personalization. Plain-text format keeps the cross-encoder agnostic to
# tokenizer; the model learns whatever encoding works best.
CONTEXT_TEMPLATE = (
    "User query: {user_query}\n"
    "Listener goal: {goal_listener}\n"
    "Recommended track: {track_name} by {artist_name}\n"
    "User profile: country={country_name}, age={age_group}, gender={gender}\n"
    "Prior dialog: {history_text}"
)

# Token caps (cross-encoders are usually 512-token max-budgeted).
MAX_CONTEXT_CHARS = 1200
MAX_RESPONSE_CHARS = 600


# ---------------------------------------------------------------------------
# Anchor normalization + filtering
# ---------------------------------------------------------------------------

def normalize_anchor(a: float) -> float:
    """Map raw judge_anchor ∈ {1.0, 5.0} (today) or [1.0, 5.0] (future
    Gemini-API anchors) into a uniform [0, 1] reward scale.

    The cross-encoder regresses against the normalized label so its
    output (sigmoid'd) stays interpretable as "Gemini-likes-this score".
    """
    return (float(a) - 1.0) / 4.0


def filter_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no judge_anchor (source B distractors).

    Returns a fresh copy so callers can safely mutate without touching
    the parquet roundtrip.
    """
    return df.dropna(subset=["judge_anchor"]).reset_index(drop=True).copy()


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def _safe_str(value, default: str = "") -> str:
    """Render None / NaN / numpy null as `default`, else str(value)."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


def build_pair(row: "pd.Series | dict") -> tuple[str, str]:
    """Render (context, response) as a cross-encoder text pair.

    `row` is a single anchor row (from filter_anchors output). All fields
    that can be None/NaN in real data fall back to "" — the cross-encoder
    handles the empty placeholder fine and we never crash on partial rows.
    """
    if isinstance(row, dict):
        get = lambda k: row.get(k)
    else:
        get = lambda k: row[k] if k in row else None

    ctx = CONTEXT_TEMPLATE.format(
        user_query=_safe_str(get("user_query"))[:MAX_CONTEXT_CHARS],
        goal_listener=_safe_str(get("goal_listener"))[:MAX_CONTEXT_CHARS],
        track_name=_safe_str(get("track_name")),
        artist_name=_safe_str(get("artist_name")),
        country_name=_safe_str(get("country_name")),
        age_group=_safe_str(get("age_group")),
        gender=_safe_str(get("gender")),
        history_text=_safe_str(get("history_text"))[:MAX_CONTEXT_CHARS],
    )
    resp = _safe_str(get("predicted_response"))[:MAX_RESPONSE_CHARS]
    return ctx, resp


# ---------------------------------------------------------------------------
# Session-level train/val split
# ---------------------------------------------------------------------------

def split_train_val(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by session_id so val sessions never appear in train.

    The judge will be evaluated on held-out sessions; row-level shuffling
    would let the model memorize a session's patterns and inflate val
    accuracy (feedback_no_data_leakage applied to judge calibration).
    """
    rng = np.random.default_rng(seed)
    sessions = sorted(df["session_id"].unique())
    rng.shuffle(sessions)
    n_val = int(len(sessions) * val_frac)
    val_sess = set(sessions[:n_val])
    train_df = df[~df["session_id"].isin(val_sess)].reset_index(drop=True)
    val_df = df[df["session_id"].isin(val_sess)].reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# JSONL serialization (Colab-friendly streaming format)
# ---------------------------------------------------------------------------

def write_jsonl(df: pd.DataFrame, path: "Path | str") -> int:
    """Render each row as {context, response, label, session_id}.

    Returns the number of rows written. session_id is kept so the GPU
    training step can re-stratify by session for monitoring.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with p.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            ctx, resp = build_pair(row)
            label = normalize_anchor(row["judge_anchor"])
            f.write(json.dumps({
                "context": ctx,
                "response": resp,
                "label": float(label),
                "session_id": row["session_id"],
            }, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Mode 1 — prepare (CPU)
# ---------------------------------------------------------------------------

def prepare_mode(args) -> int:
    anchor_path = Path(args.anchors)
    if not anchor_path.exists():
        print(f"ERROR: anchor parquet not found: {anchor_path}", file=sys.stderr)
        return 1

    print(f"[judge-prep] loading {anchor_path}")
    df = pd.read_parquet(anchor_path)
    print(f"[judge-prep] input: {len(df):,} rows")

    df = filter_anchors(df)
    print(f"[judge-prep] after dropping NaN anchors: {len(df):,} rows")
    print(f"[judge-prep] label distribution: {df['judge_anchor'].value_counts().to_dict()}")

    train_df, val_df = split_train_val(df, val_frac=args.val_frac, seed=args.seed)
    print(f"[judge-prep] train: {len(train_df):,} rows  val: {len(val_df):,} rows")

    out_dir = Path(args.out_dir)
    n_train = write_jsonl(train_df, out_dir / "train.jsonl")
    n_val = write_jsonl(val_df, out_dir / "val.jsonl")
    print(f"[judge-prep] wrote {n_train:,} → {out_dir/'train.jsonl'}")
    print(f"[judge-prep] wrote {n_val:,} → {out_dir/'val.jsonl'}")
    return 0


# ---------------------------------------------------------------------------
# Mode 2 — train (GPU; lazy-imported so the prepare path doesn't touch torch)
# ---------------------------------------------------------------------------

def train_mode(args) -> int:
    """Fine-tune a cross-encoder regression head on the JSONL pairs.

    GPU step. Prefers `cross-encoder/ms-marco-MiniLM-L-6-v2` as the base —
    pre-trained on query-doc relevance, ~80 MB, ~30 min on A100.
    """
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        TrainingArguments, Trainer,
    )

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        print(f"ERROR: missing JSONL files in {data_dir}. Run --mode prepare first.",
              file=sys.stderr)
        return 1

    print(f"[judge-train] loading dataset from {data_dir}")
    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "val": str(val_path)},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"[judge-train] base model: {args.base_model}")

    def tokenize(batch):
        return tokenizer(
            batch["context"], batch["response"],
            truncation=True, padding="max_length", max_length=args.max_length,
        )
    ds = ds.map(tokenize, batched=True, remove_columns=["context", "response", "session_id"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, problem_type="regression",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        push_to_hub=bool(args.hub_repo),
        hub_model_id=args.hub_repo,
        hub_private_repo=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        tokenizer=tokenizer,
    )
    trainer.train()
    if args.hub_repo:
        trainer.push_to_hub()
        print(f"[judge-train] pushed → https://huggingface.co/{args.hub_repo}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["prepare", "train"], required=True)

    # Prepare args
    p.add_argument("--anchors", default=str(DEFAULT_ANCHORS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # Train args
    p.add_argument("--data-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--output-dir", default="./distilled_judge_run")
    p.add_argument("--hub-repo", default=None,
                   help="HF Hub repo to push the trained judge (e.g. orrimoch/recsys2026-distilled-judge).")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)

    args = p.parse_args(argv)

    if args.mode == "prepare":
        return prepare_mode(args)
    return train_mode(args)


if __name__ == "__main__":
    sys.exit(main())
