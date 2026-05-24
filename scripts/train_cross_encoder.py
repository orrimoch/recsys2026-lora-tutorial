"""Stage B: fine-tune BAAI/bge-reranker-v2-m3 via sentence-transformers' CrossEncoder API.

Hyperparams (spec §7):
  - Full FT (no LoRA), lr=2e-5, bs=16, epochs=3, max_length=512, bf16
  - Cross-entropy on pairwise (pos, neg) — one example per (query, pos, neg) pair.

Usage:
  python scripts/train_cross_encoder.py \
    --triples experiments/cache/retrieval_v2/triples_reranker.jsonl \
    --output-dir /content/bge_reranker_finetune \
    --hub-repo OrRim123/recsys2026-bge-reranker-music-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _expand_triples_to_pairs(path: str) -> tuple[list, list]:
    """Each (q, pos, [negs]) row → 1 positive pair + len(negs) negative pairs."""
    pos_pairs = []
    neg_pairs = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            q = obj["query"]
            for p in obj.get("pos", []):
                pos_pairs.append([q, p])
            for n in obj.get("neg", []):
                neg_pairs.append([q, n])
    return pos_pairs, neg_pairs


def pairwise_bce_loss(pos_logits, neg_logits, neg_weights=None):
    """Pairwise binary cross-entropy for the Stage B reranker.

    ``loss = BCE(score(q, pos), 1) + BCE(score(q, neg), 0)`` — the relevance
    objective from the plan (Phase 6 §6.5). ``pos_logits`` / ``neg_logits`` are
    raw scoring-head outputs (B,) / (N,); targets are applied internally so the
    caller never materializes 1/0 tensors.

    ``neg_weights`` (optional, shape matching ``neg_logits``): per-negative
    weights for the negative term — used for v2 rank-weighting of harder Stage A
    negatives. v1 passes None (unweighted), since teacher scores are dormant.
    """
    import torch
    import torch.nn.functional as F

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits)
    )
    neg_targets = torch.zeros_like(neg_logits)
    if neg_weights is not None:
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, neg_targets, weight=neg_weights
        )
    else:
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets)
    return pos_loss + neg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hub-repo", required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    import torch

    pos_pairs, neg_pairs = _expand_triples_to_pairs(args.triples)
    print(f"[train-ce] {len(pos_pairs)} pos, {len(neg_pairs)} neg pairs", file=sys.stderr)

    examples = [InputExample(texts=p, label=1.0) for p in pos_pairs] + \
               [InputExample(texts=n, label=0.0) for n in neg_pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = CrossEncoder(
        args.base_model, num_labels=1, max_length=args.max_length,
        automodel_args={"torch_dtype": torch.bfloat16},
    )
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(loader) * args.epochs),
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
        use_amp=True,
    )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    pushed = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    tok = AutoTokenizer.from_pretrained(args.output_dir)
    print(f"[train-ce] pushing → {args.hub_repo}", file=sys.stderr)
    pushed.push_to_hub(args.hub_repo, private=False)
    tok.push_to_hub(args.hub_repo, private=False)


if __name__ == "__main__":
    main()
