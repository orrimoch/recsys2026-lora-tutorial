"""Fine-tune `music-colbert-v1` from off-the-shelf colbertv2.0 (W2.b).

Warm-starts `colbert-ir/colbertv2.0` and fine-tunes with PyLate's contrastive loss
on the JSONL triples from `scripts/build_colbert_train_data.py` (query / positive /
hard-negative doc texts). Hyperparams follow the plan §6/§10: lr 1e-5, AdamW wd
0.01, warmup 10%, batch 32, 1-3 epochs, bf16, grad-clip 1.0, seed 42, q32/d96.

MODEL SELECTION — validate on the DEV gate, never on a train-internal loss. This
campaign has shown every train-side metric to be anti-correlated with dev, so the
trainer evaluates the live model every `--eval-steps` on the dev turn-1 pool
(the SAME turn-1 recall@20 the #82-reprobe gate measures) and saves+overrides the
single best checkpoint by that metric. Pass `--dev-eval-pack` (built by the nb82
`#82-dev-eval-pack` cell). Without it, falls back to a plain final-only save.

Leak discipline: trains on TRAIN-split triples; validates on TEST-split (dev) turn-1.

Usage (Colab GPU):
  python scripts/train_colbert.py \
    --train-jsonl experiments/cache/retrieval_v2/colbert_train.jsonl \
    --dev-eval-pack experiments/cache/retrieval_v2/colbert_dev_eval.pkl \
    --out-dir experiments/cache/retrieval_v2/colbert/music-colbert-v1 \
    --epochs 2 --batch-size 32 --eval-steps 500 --dev-subset 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))


# --------------------------------------------------------------------------- #
# Pure functions (unit-tested).
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


def wall_recall_at_k(
    ranked_lists: list[list[str]], golds: list[str], wall: list[bool], k: int
) -> float:
    """recall@k over the wall subset only (rows where wall[i] is True). Returns
    0.0 when there are no wall rows (a safe sentinel so best-model max-tracking
    never sees NaN). At turn-1 every row is wall, so this == plain turn-1 recall@k.
    """
    idx = [i for i, w in enumerate(wall) if w]
    if not idx:
        return 0.0
    hits = sum(1.0 for i in idx if golds[i] in ranked_lists[i][:k])
    return hits / len(idx)


def load_triples(path: str) -> list[dict[str, Any]]:  # pragma: no cover
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Dev-eval callback (GPU integration; run in the notebook, not unit-tested).
# --------------------------------------------------------------------------- #
def _make_dev_eval_callback(pack: dict, eval_steps: int, dev_subset: int,
                            k: int, out_dir: str, seed: int):  # pragma: no cover
    """Build a TrainerCallback that, every `eval_steps`, reranks a dev turn-1
    subset with the LIVE model (same MaxSim re-probe as the gate), and
    saves+overrides the single best checkpoint by dev recall@k."""
    import random
    import numpy as np
    import torch
    from transformers import TrainerCallback
    from mcrs.retrieval_modules.colbert_late import ColbertRetriever, pylate_encoders

    # Deterministic dev subset (cap eval cost; the full re-probe is the final read).
    n = len(pack["queries"])
    order = list(range(n))
    random.Random(seed).shuffle(order)
    sub = sorted(order[: min(dev_subset, n)])
    queries = [pack["queries"][i] for i in sub]
    pools = [pack["pools"][i] for i in sub]
    golds = [pack["golds"][i] for i in sub]
    wall = [pack["wall"][i] for i in sub]
    need_tids = sorted({t for pool in pools for t in pool})
    need_texts = [pack["tid_to_text"][t] for t in need_tids]
    print(f"[dev-eval] subset={len(queries)} turn-1 queries, {len(need_tids)} pool docs, "
          f"every {eval_steps} steps -> recall@{k} (best saved to {out_dir})", file=sys.stderr)

    state = {"best": -1.0}

    def _evaluate(model) -> float:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            q_enc, d_enc = pylate_encoders(model)
            retr = ColbertRetriever(doc_embs={}, query_encoder=q_enc, doc_encoder=d_enc)
            retr.encode_docs(need_tids, need_texts)
            ranked = retr.batch_rerank_pool(queries, pools, topk=100)
        if was_training:
            model.train()
        return wall_recall_at_k(ranked, golds, wall, k)

    class DevRecallCallback(TrainerCallback):
        def _eval_and_save(self, model, step, tag):
            score = _evaluate(model)
            best = score > state["best"]
            if best:
                state["best"] = score
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(out_dir)
            print(f"[dev-eval] step {step} ({tag}) recall@{k}={score:.4f}"
                  f"{'  <- NEW BEST, saved (override)' if best else ''}", file=sys.stderr)

        def on_step_end(self, args, st, control, **kwargs):
            if st.global_step > 0 and st.global_step % eval_steps == 0:
                self._eval_and_save(kwargs["model"], st.global_step, "periodic")

        def on_train_end(self, args, st, control, **kwargs):
            # Final eval so the last steps get a chance to be the best.
            self._eval_and_save(kwargs["model"], st.global_step, "final")
            print(f"[dev-eval] DONE best recall@{k}={state['best']:.4f} -> {out_dir}",
                  file=sys.stderr)

    return DevRecallCallback()


def existing_artifact_blocks(path: str, force: bool) -> bool:
    """True if `path` already exists and --force was NOT passed, i.e. the run
    should SKIP to avoid clobbering a live artifact in place. The 2026-06-15
    accident re-ran finetune and overwrote the 0.50 ColBERT model in place
    (save_pretrained), making config-205/209 non-reproducible. Pure -> unit-tested."""
    import os
    return bool(path) and os.path.exists(path) and not force


# --------------------------------------------------------------------------- #
# Trainer wiring (GPU integration; run in the notebook, not unit-tested).
# --------------------------------------------------------------------------- #
def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out-dir if it already exists. WITHOUT this, an "
                         "existing model dir SKIPS training (guards against the accidental "
                         "in-place retrain that clobbered the 0.50 ColBERT).")
    ap.add_argument("--dev-eval-pack", default=None,
                    help="Pickle from nb82 #82-dev-eval-pack; enables dev model selection")
    ap.add_argument("--base-model", default="colbert-ir/colbertv2.0")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--eval-steps", type=int, default=500,
                    help="Dev re-probe cadence (not too often / not too rare)")
    ap.add_argument("--dev-subset", type=int, default=300,
                    help="Turn-1 queries used for in-loop dev eval (caps cost)")
    ap.add_argument("--q-len", type=int, default=96,
                    help="query token budget; 96 keeps the trailing 'goal:' facet "
                         "(was 32, which right-truncated the goal off ~95% of queries)")
    ap.add_argument("--d-len", type=int, default=96)
    ap.add_argument("--k", type=int, default=20, help="recall@k for dev selection")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if existing_artifact_blocks(args.out_dir, args.force):
        print(f"[train-colbert] {args.out_dir} already exists — SKIPPING to avoid "
              f"overwriting it in place. Pass --force to retrain.", file=sys.stderr)
        return

    import pickle
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

    callbacks = []
    if args.dev_eval_pack:
        with open(args.dev_eval_pack, "rb") as f:
            pack = pickle.load(f)
        callbacks.append(_make_dev_eval_callback(
            pack, args.eval_steps, args.dev_subset, args.k, args.out_dir, args.seed))
    else:
        print("[train-colbert] no --dev-eval-pack: final-only save, NO dev selection",
              file=sys.stderr)

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
        save_strategy="no",  # the dev-eval callback owns saving the best model
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
        callbacks=callbacks,
    )
    trainer.train()

    # Fallback: if no dev pack drove a save, persist the final model.
    if not args.dev_eval_pack:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out_dir)
    print(f"[train-colbert] best/final model -> {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
