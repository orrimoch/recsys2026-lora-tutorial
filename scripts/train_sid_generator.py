"""W3: fine-tune Qwen2.5-1.5B-Instruct + LoRA as a SID generator.

Pipeline:
  1. Load extended tokenizer + Qwen-1.5B base model.
  2. Untie tied embeddings + add 768 SID tokens + resize embedding matrix.
  3. Wrap with LoRA (r=32, alpha=64, modules_to_save=['embed_tokens','lm_head']).
  4. Read W2 train/val parquets, format examples, run HF Trainer for N epochs.
  5. Push LoRA adapter to Hub. If --merge, also merge + push the full ~3GB model.

Designed to run on Colab L4 (24GB) or Blackwell (95GB). Smoke mode keeps step
budget small for plumbing-validation runs (~10 min).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
)

from mcrs.sid.training_format import collate_training_batch, format_example_for_training
from mcrs.sid.vocab import (
    add_sid_tokens_to_tokenizer, build_sid_to_token_id_lookup,
    extend_model_vocab, untie_embeddings_if_tied,
)


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_LEVELS = 3
CODEBOOK_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_training/train.parquet")
    p.add_argument("--val-parquet", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_training/val.parquet")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_generator")
    p.add_argument("--hub-repo", type=str,
                   default="OrRim123/recsys2026-sid-generator-qwen15b-v1")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run for plumbing test (200 steps, 1k train rows)")
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--merge", action="store_true",
                   help="After training, merge LoRA into base + push merged model (~3GB)")
    return p.parse_args()


def load_and_extend_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok, n_added = add_sid_tokens_to_tokenizer(tok, NUM_LEVELS, CODEBOOK_SIZE)
    print(f"[tokenizer] base vocab + {n_added} SID tokens = {len(tok)} total", file=sys.stderr)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_and_extend_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"[model] tied={model.config.tie_word_embeddings}", file=sys.stderr)
    untie_embeddings_if_tied(model)
    extend_model_vocab(model, len(tokenizer))
    print(
        f"[model] vocab now {model.get_input_embeddings().weight.shape[0]} "
        f"(tied={model.config.tie_word_embeddings})",
        file=sys.stderr,
    )
    return model


def wrap_lora(model, args):
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens", "lm_head"],   # CRITICAL — without this the new SID embeddings never train
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def build_hf_dataset(parquet_path, tokenizer, sid_lookup, max_prompt_len, sample_n=None):
    df = pd.read_parquet(parquet_path)
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"[data] {parquet_path.name}: {len(df)} rows", file=sys.stderr)

    def _gen():
        for row in df.itertuples(index=False):
            yield format_example_for_training(
                query=row.query,
                code_1=int(row.code_1), code_2=int(row.code_2), code_3=int(row.code_3),
                tokenizer=tokenizer, sid_lookup=sid_lookup, max_prompt_len=max_prompt_len,
            )

    return Dataset.from_generator(_gen)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_and_extend_tokenizer()
    sid_lookup = build_sid_to_token_id_lookup(tokenizer, NUM_LEVELS, CODEBOOK_SIZE)

    model = load_and_extend_model(tokenizer)
    model = wrap_lora(model, args)

    sample_n = 1000 if args.smoke else None
    train_ds = build_hf_dataset(args.train_parquet, tokenizer, sid_lookup, args.max_prompt_len, sample_n=sample_n)
    val_ds = build_hf_dataset(args.val_parquet, tokenizer, sid_lookup, args.max_prompt_len, sample_n=200 if args.smoke else None)

    max_steps = 200 if args.smoke else -1
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if not args.smoke else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    def collator(examples):
        return collate_training_batch(examples, pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()

    # Save LoRA adapter locally + push.
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] LoRA adapter at {adapter_dir}", file=sys.stderr)

    hub_lora_repo = f"{args.hub_repo}-lora"
    model.push_to_hub(hub_lora_repo, private=False)
    tokenizer.push_to_hub(hub_lora_repo, private=False)
    print(f"[push] {hub_lora_repo}", file=sys.stderr)

    if args.merge:
        merged = model.merge_and_unload()
        merged_dir = args.output_dir / "merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        merged_repo = f"{args.hub_repo}-merged"
        merged.push_to_hub(merged_repo, private=False, safe_serialization=True)
        tokenizer.push_to_hub(merged_repo, private=False)
        print(f"[push] merged → {merged_repo}", file=sys.stderr)

    # Persist a small summary so notebooks can pick it up.
    summary = {
        "base_model": BASE_MODEL,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "epochs": args.epochs, "lr": args.lr,
        "smoke": args.smoke,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "hub_lora_repo": hub_lora_repo,
        "hub_merged_repo": f"{args.hub_repo}-merged" if args.merge else None,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
