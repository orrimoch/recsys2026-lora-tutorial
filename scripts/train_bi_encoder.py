"""Stage A: custom PEFT-LoRA fine-tune of BAAI/bge-m3 on conversation→track triples.

Why a custom loop (not FlagEmbedding's CLI):
  FlagEmbedding's master `unified_finetune` does NOT expose --use_lora /
  --lora_rank / --lora_alpha. A `torchrun -m FlagEmbedding...` invocation
  with those flags crashes at startup. This script uses sentence-transformers'
  underlying AutoModel + peft.LoraConfig + a small MultipleNegativesRanking
  loss to get equivalent training behavior with the exact LoRA settings the
  plan calls for (r=32 / alpha=64 over attention+FFN projections).

Hyperparameters (spec §6):
  - lr 5e-6, per-device bs 2, train_group_size 8 (1 pos + 7 negs),
    n_negatives_per_query 15 (all mined negs used; tiny per-row denominator hurts
    contrastive signal — ML reviewer I1, restored from spec §6),
    temperature 0.05, epochs 2, warmup_ratio 0.1.
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
        if len(negs) > self.n_negatives:
            negs = self.rng.sample(negs, self.n_negatives)
        elif len(negs) < self.n_negatives:
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


def _cls_pool(last_hidden: "torch.Tensor") -> "torch.Tensor":
    """L2-normalized CLS token. Matches BGE-M3 inference (CLS token, NOT mean).

    BGEM3FlagModel.encode() reads last_hidden_state[:, 0] at inference time, so
    training must pool the same way or LoRA-adapted weights won't be optimized
    for what production reads.
    """
    import torch.nn.functional as F

    cls = last_hidden[:, 0]
    return F.normalize(cls, p=2, dim=1)


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
    # I3: gradient checkpointing — must be enabled BEFORE get_peft_model,
    # and enable_input_require_grads is required for PEFT compatibility.
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    # Training does CLS-pool on last_hidden_state directly (see _cls_pool), so
    # XLM-RoBERTa's `pooler` submodule is never on the gradient path. Earlier
    # versions added `modules_to_save=["pooler"]` per spec line 140's wording,
    # but that wrapped a module the loss never touches AND collided with the
    # `dense` substring in `target_modules` (peft would also LoRA-wrap pooler.dense).
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

    # Optimizer steps after grad-accum: total_micro_batches / accum_steps.
    accum = max(1, int(args.gradient_accumulation_steps))
    total_micro = len(loader) * args.epochs
    total_steps = max(1, total_micro // accum)
    # I4: only optimize trainable params (saves ~9GB of AdamW state on a 567M model).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    # I2: warmup_ratio 0.1 → 10% linear warmup, then linear decay.
    # N5: end_factor=0.1 (not 0.0) so the final ~10% of steps still updates.
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                     total_iters=max(1, total_steps - warmup_steps)),
        ],
        milestones=[warmup_steps],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Gradient-accumulation training loop. Loss is divided by `accum` so the
    # accumulated gradient matches what a single bs=(per_device*accum) step
    # would produce; optimizer steps only every `accum` micro-batches.
    micro_step = 0
    opt_step = 0
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for q_enc, d_enc, n_per in loader:
            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            d_enc = {k: v.to(device) for k, v in d_enc.items()}
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                q_out = model(**q_enc)
                d_out = model(**d_enc)
                q_emb = _cls_pool(q_out.last_hidden_state)
                d_emb = _cls_pool(d_out.last_hidden_state)
                loss = _info_nce_loss(q_emb, d_emb, n_per, args.temperature) / accum
            loss.backward()
            micro_step += 1
            if micro_step % accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1
                if opt_step % args.logging_steps == 0:
                    # `loss.item()` is the per-micro-batch contribution; multiply
                    # back by accum to get the effective-batch loss for logging.
                    writer.add_scalar("train/loss", float(loss.item()) * accum, opt_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], opt_step)
                    print(f"[train-bi-encoder] opt_step={opt_step}/{total_steps} "
                          f"loss={float(loss.item()) * accum:.4f}", file=sys.stderr)
    # Flush any partial accumulation at end of training.
    if micro_step % accum != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    writer.close()
    # Save the adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train-bi-encoder] adapter saved → {output_dir}", file=sys.stderr)


def _merge_and_push(args):
    """Merge LoRA → base, copy sentence-transformers scaffolding from BAAI/bge-m3
    so the merged repo deploys with CLS-pool + L2-normalize, then upload the whole
    folder (not just the HF model files) to the Hub.

    Why the scaffolding copy: training pools `last_hidden_state[:, 0]` (CLS),
    matching BGE-M3's native head. But `merged.push_to_hub` uploads only the
    HF AutoModel artifacts. Without `modules.json` + `1_Pooling/` + `2_Normalize/`
    in the repo, `SentenceTransformer(<hub_repo>)` silently falls back to mean-pool
    + no-normalize defaults → inference uses mean-pool while training used CLS,
    destroying the fine-tune signal.
    """
    import shutil

    import torch
    from huggingface_hub import HfApi, snapshot_download
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

    # Pull only the sentence-transformers scaffolding files from the base repo
    # (model weights/tokenizer already saved by save_pretrained above).
    print(f"[train-bi-encoder] copying ST scaffolding from {args.base_model}",
          file=sys.stderr)
    st_src = snapshot_download(
        args.base_model,
        allow_patterns=[
            "modules.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
            "1_Pooling/*",
            "2_Normalize/*",
        ],
    )
    for fname in ("modules.json", "sentence_bert_config.json",
                  "config_sentence_transformers.json"):
        src = Path(st_src) / fname
        if src.exists():
            shutil.copy(src, merged_dir / fname)
    for subdir in ("1_Pooling", "2_Normalize"):
        src = Path(st_src) / subdir
        if src.is_dir():
            shutil.copytree(src, merged_dir / subdir, dirs_exist_ok=True)
    print("[train-bi-encoder] ST scaffolding present:",
          sorted(p.name for p in merged_dir.iterdir() if p.name.startswith(("1_", "2_", "modules", "sentence", "config_sentence"))),
          file=sys.stderr)

    # ML-reviewer I3: pin the pool mode in case the upstream BGE-M3 config
    # ever changes. Training optimizes CLS; if the scaffolding ever ships with
    # mean-pool, the merged model deploys wrong.
    import json as _json
    pool_cfg_path = merged_dir / "1_Pooling" / "config.json"
    if pool_cfg_path.exists():
        pool_cfg = _json.loads(pool_cfg_path.read_text())
        assert pool_cfg.get("pooling_mode_cls_token") is True, (
            f"BAAI/bge-m3 ST scaffolding shipped with non-CLS pooling: "
            f"{pool_cfg}. Training used CLS; deploy would mismatch."
        )
        print(f"[train-bi-encoder] pooling pinned: CLS=True ({pool_cfg_path})",
              file=sys.stderr)

    hub_target = f"{args.hub_repo}-merged"
    print(f"[train-bi-encoder] uploading merged folder → {hub_target}", file=sys.stderr)
    api = HfApi()
    api.create_repo(repo_id=hub_target, exist_ok=True, private=False)
    api.upload_folder(folder_path=str(merged_dir), repo_id=hub_target, repo_type="model")
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
    p.add_argument("--gradient-accumulation-steps", type=int, default=16,
                   help="Micro-batches accumulated per optimizer step. With "
                        "per-device-batch-size=2, default 16 → effective batch 32. "
                        "ML reviewer I2: bs=2 + no in-batch negs → very noisy "
                        "InfoNCE signal; accumulating recovers a usable denominator.")
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
