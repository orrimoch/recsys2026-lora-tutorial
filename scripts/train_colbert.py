"""Fine-tune `music-colbert-v1` from off-the-shelf colbertv2.0 (W2.b).

Warm-starts `colbert-ir/colbertv2.0` and fine-tunes with PyLate's contrastive loss
on the JSONL triples from `scripts/build_colbert_train_data.py` (query / positive /
hard-negative doc texts). Hyperparams follow the plan §6/§10 (ColBERTv2 / PyLate
defaults): lr 1e-5, AdamW wd 0.01, warmup 10%, batch 32, 1-3 epochs, bf16,
grad-clip 1.0, seed 42, q_len 32 / d_len 96, `[MASK]` query augmentation on.

Leak discipline: trains on the TRAIN-split triples only.

Usage (Colab GPU):
  python scripts/train_colbert.py \
    --train-jsonl experiments/cache/retrieval_v2/colbert_train.jsonl \
    --out-dir experiments/cache/retrieval_v2/colbert/music-colbert-v1 \
    --epochs 1 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Pure data-shaping (unit-tested).
# --------------------------------------------------------------------------- #
def triples_to_contrastive_rows(triples: list[dict]) -> list[dict[str, str]]:
    """Explode each builder triple {query, positive, negatives[]} into flat
    (query, positive, negative) rows — one per negative — for PyLate's
    Contrastive loss. Triples with no negatives are dropped (no pair to form)."""
    rows: list[dict[str, str]] = []
    for t in triples:
        for neg in (t.get("negatives") or []):
            rows.append({
                "query": t["query"],
                "positive": t["positive"],
                "negative": neg,
            })
    return rows


def load_triples(path: str) -> list[dict[str, Any]]:  # pragma: no cover
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Trainer wiring (GPU integration; run in the notebook, not unit-tested).
# --------------------------------------------------------------------------- #
def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default="colbert-ir/colbertv2.0")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--q-len", type=int, default=32)
    ap.add_argument("--d-len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from pylate import losses, models, utils

    rows = triples_to_contrastive_rows(load_triples(args.train_jsonl))
    print(f"[train-colbert] {len(rows)} contrastive rows", file=sys.stderr)
    train_dataset = Dataset.from_list(rows)

    model = models.ColBERT(
        model_name_or_path=args.base_model,
        query_length=args.q_len,
        document_length=args.d_len,
    )
    train_loss = losses.Contrastive(model=model)

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        bf16=True,
        seed=args.seed,
        logging_steps=50,
        save_strategy="no",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    trainer.train()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    print(f"[train-colbert] saved -> {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
