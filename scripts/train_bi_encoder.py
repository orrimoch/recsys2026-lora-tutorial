"""Stage A: custom PEFT-LoRA fine-tune of BAAI/bge-m3 on conversation→track triples.

Why a custom loop (not FlagEmbedding's CLI):
  FlagEmbedding's master `unified_finetune` does NOT expose --use_lora /
  --lora_rank / --lora_alpha. A `torchrun -m FlagEmbedding...` invocation
  with those flags crashes at startup. This script uses sentence-transformers'
  underlying AutoModel + peft.LoraConfig + a small MultipleNegativesRanking
  loss to get equivalent training behavior with the exact LoRA settings the
  plan calls for (r=32 / alpha=64 over attention+FFN projections).

Hyperparameters (spec §6):
  - lr 5e-6, per-device bs 2, train_group_size 8 (1 pos + 7 in-batch negs),
    n_negatives_per_query 15 (sampled from the 15 mined negs per row),
    temperature 0.05, epochs 2.
  - LoRA r=32 alpha=64 over query/key/value/dense projections.
  - bf16 mixed precision via torch.cuda.amp.
  - After training: merge LoRA via peft_model.merge_and_unload(), push merged
    model to Hub.

Usage:
  python scripts/train_bi_encoder.py \
    --triples experiments/cache/retrieval_v2/triples_bge_m3.jsonl \
    --output-dir /content/bge_m3_finetune \
    --hub-repo OrRim123/recsys2026-bge-m3-music-v1 \
    --results-dir /content/drive/MyDrive/recsys2026_retrieval_v2_cache/results/bge_m3 \
    --merge --cleanup-after-push
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional


# LoRA target modules for BGE-M3 (XLM-RoBERTa under the hood).
_BGE_M3_LORA_TARGETS = ["query", "key", "value", "dense"]


class TripleJsonlDataset:
    """Loads JSONL triples produced by scripts/build_bi_encoder_training_data.py.

    Each row: {"query": str, "pos": [str], "neg": [str, ...]}.
    On __getitem__, returns {"query": str, "positive": str, "negatives": [str]*n_negatives}.
    Short neg-lists are upsampled by repeated random sampling from the same row.
    """

    def __init__(self, path: str, n_negatives: int = 15, seed: int = 42):
        import json as _json
        self.rows = []
        with open(path) as f:
            for line in f:
                obj = _json.loads(line)
                if not obj.get("pos") or not obj.get("neg"):
                    continue
                self.rows.append(obj)
        self.n_negatives = int(n_negatives)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        negs = list(row["neg"])
        if len(negs) >= self.n_negatives:
            negs = negs[: self.n_negatives]
        else:
            pad_pool = list(negs) if negs else [""]
            while len(negs) < self.n_negatives:
                negs.append(self.rng.choice(pad_pool))
        return {
            "query": row["query"],
            "positive": row["pos"][0],
            "negatives": negs,
        }


def _collate_batch(batch: list[dict], tokenizer, max_q_len: int, max_p_len: int):
    """Tokenize a list of {query, positive, negatives} rows into tensors."""
    import torch

    queries = [b["query"] for b in batch]
    # positives + negatives per row → (B * (1 + n_negs)) docs.
    docs: list[str] = []
    n_per = 1 + len(batch[0]["negatives"])
    for b in batch:
        docs.append(b["positive"])
        docs.extend(b["negatives"])

    q_enc = tokenizer(queries, max_length=max_q_len, padding=True, truncation=True, return_tensors="pt")
    d_enc = tokenizer(docs, max_length=max_p_len, padding=True, truncation=True, return_tensors="pt")
    return q_enc, d_enc, n_per


def _mean_pool(last_hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """L2-normalized mean-pool over non-padding tokens. Matches BGE-M3 dense head."""
    import torch
    import torch.nn.functional as F

    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return F.normalize(pooled, p=2, dim=1)


def _info_nce_loss(q_emb: "torch.Tensor", d_emb: "torch.Tensor", n_per: int, temperature: float) -> "torch.Tensor":
    """InfoNCE: each query has 1 positive + (n_per - 1) negatives, contiguous in d_emb."""
    import torch
    import torch.nn.functional as F

    B = q_emb.size(0)
    d_emb = d_emb.view(B, n_per, -1)             # (B, n_per, D)
    scores = torch.einsum("bd,bnd->bn", q_emb, d_emb) / temperature  # (B, n_per)
    labels = torch.zeros(B, dtype=torch.long, device=scores.device)  # positive is index 0
    return F.cross_entropy(scores, labels)


def _train(args):
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train-bi-encoder] device={device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=_BGE_M3_LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(device)
    model.print_trainable_parameters()

    ds = TripleJsonlDataset(args.triples, n_negatives=args.n_negatives)
    print(f"[train-bi-encoder] {len(ds)} training triples", file=sys.stderr)
    loader = DataLoader(
        ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: _collate_batch(b, tokenizer, args.query_max_len, args.passage_max_len),
    )

    total_steps = len(loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    step = 0
    for epoch in range(args.epochs):
        for q_enc, d_enc, n_per in loader:
            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            d_enc = {k: v.to(device) for k, v in d_enc.items()}
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                q_out = model(**q_enc)
                d_out = model(**d_enc)
                q_emb = _mean_pool(q_out.last_hidden_state, q_enc["attention_mask"])
                d_emb = _mean_pool(d_out.last_hidden_state, d_enc["attention_mask"])
                loss = _info_nce_loss(q_emb, d_emb, n_per, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.logging_steps == 0:
                writer.add_scalar("train/loss", float(loss.item()), step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                print(f"[train-bi-encoder] step={step}/{total_steps} loss={float(loss.item()):.4f}", file=sys.stderr)

    writer.close()
    # Save the adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train-bi-encoder] adapter saved → {output_dir}", file=sys.stderr)


def _merge_and_push(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    print("[train-bi-encoder] merging LoRA → base", file=sys.stderr)
    base = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, args.output_dir)
    merged = peft_model.merge_and_unload()
    merged_dir = Path(args.output_dir) / "merged"
    merged.save_pretrained(str(merged_dir))
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(str(merged_dir))
    print(f"[train-bi-encoder] merged → {merged_dir}", file=sys.stderr)

    hub_target = f"{args.hub_repo}-merged"
    print(f"[train-bi-encoder] pushing to {hub_target}", file=sys.stderr)
    merged.push_to_hub(hub_target, private=False)
    tok.push_to_hub(hub_target, private=False)
    return merged_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triples", required=True)
    p.add_argument("--base-model", default="BAAI/bge-m3")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hub-repo", required=True,
                   help="HF Hub repo prefix (the merged model is pushed to <hub_repo>-merged).")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--cleanup-after-push", action="store_true")
    p.add_argument("--results-dir", default=None)
    # Hyperparameters (spec §6)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--n-negatives", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--query-max-len", type=int, default=512)
    p.add_argument("--passage-max-len", type=int, default=256)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--logging-steps", type=int, default=50)
    args = p.parse_args()

    _train(args)

    if args.merge:
        _merge_and_push(args)

    if args.results_dir is not None:
        rd = Path(args.results_dir)
        rd.mkdir(parents=True, exist_ok=True)
        runs_src = Path(args.output_dir) / "runs"
        if runs_src.exists():
            shutil.copytree(runs_src, rd / "runs", dirs_exist_ok=True)

    if args.cleanup_after_push and args.merge:
        merged_dir = Path(args.output_dir) / "merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)


if __name__ == "__main__":
    main()
