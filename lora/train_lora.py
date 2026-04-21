"""LoRA fine-tune Qwen2.5-3B-Instruct on the Blind-A-aligned SFT dataset.

Implements completion-only loss: only the assistant response tokens contribute
to the gradient. The system prompt + user query are masked with -100 so the
model isn't burning capacity memorizing the prompt template.

Also enables `model.enable_input_require_grads()` so gradients flow through
the embedding layer when gradient_checkpointing + PEFT are combined.

Runs on:
  - Colab A100 (~20-30 min for 1 epoch on ~2k filtered rows, bf16)
  - Local M4 MPS (much slower, prefer Colab for full runs)

CLI:
    python train_lora.py \
        --dataset_path ./data/train_sft \
        --output_dir ./lora_adapters/qwen3b_blinda_v1 \
        --base_model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def detect_device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # bf16 on MPS is flaky for training
    return "cpu", torch.float32


def make_completion_masked_tokenizer(tokenizer, max_length: int):
    """Tokenize each row as (system + user + assistant) with `labels` set to
    -100 on every token BEFORE the assistant turn. Only the assistant response
    contributes to cross-entropy loss.

    Returns a function suitable for `Dataset.map(..., remove_columns=...)`.
    """
    def _tok(row: dict) -> dict:
        full_msgs = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_query"]},
            {"role": "assistant", "content": row["assistant_response"]},
        ]
        prefix_msgs = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_query"]},
        ]
        # Use add_special_tokens=False because the chat template already
        # emits BOS/EOS markers via Qwen's <|im_start|>/<|im_end|>.
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False)
        prefix_text = tokenizer.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True
        )
        full_ids = tokenizer(
            full_text, add_special_tokens=False, truncation=True, max_length=max_length
        ).input_ids
        prefix_ids = tokenizer(
            prefix_text, add_special_tokens=False, truncation=True, max_length=max_length
        ).input_ids

        # If truncation cut off the assistant turn entirely, mask everything;
        # the row contributes nothing to loss (and TRL will skip it if labels
        # are all -100).
        prefix_len = min(len(prefix_ids), len(full_ids))
        labels = [-100] * prefix_len + full_ids[prefix_len:]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }
    return _tok


class PadCollator:
    """Right-pads input_ids/attention_mask/labels to the longest seq in batch.
    Pads input_ids with pad_token_id, attention_mask with 0, labels with -100
    (so padding doesn't contribute to loss)."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = max_len - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_token_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Conservative default — Qwen2.5-3B is well-trained, "
                             "high LR risks catastrophically forgetting v10 style.")
    parser.add_argument("--per_device_batch", type=int, default=1,
                        help="M4-friendly default. Bump to 2-4 on Colab A100.")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Effective batch ~16 with default per_device=1.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=1536,
                        help="M4-friendly default. Bump to 3072 on Colab A100.")
    parser.add_argument("--eval_split_frac", type=float, default=0.02,
                        help="Hold out this fraction for eval (0 to disable).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_modules", type=str, default="attn",
                        choices=["attn", "attn_mlp"],
                        help="`attn` = q,k,v,o (smaller, faster). "
                             "`attn_mlp` = + gate,up,down (more capacity).")
    args = parser.parse_args()

    device, dtype = detect_device_dtype()
    print(f"[train] device={device} dtype={dtype}", file=sys.stderr)

    print(f"[train] loading tokenizer {args.base_model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[train] loading base model {args.base_model} ({dtype})", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    # Gradient checkpointing + disabling cache for training.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # Required for gradient_checkpointing + PEFT: makes embedding outputs
    # require grad so the LoRA layers downstream get backward signal.
    model.enable_input_require_grads()

    print(f"[train] loading SFT dataset from {args.dataset_path}", file=sys.stderr)
    ds = load_from_disk(args.dataset_path)
    print(f"[train] dataset rows: {len(ds)}", file=sys.stderr)

    # Pre-tokenize with completion-only mask. This bypasses TRL's automatic
    # tokenization (SFTTrainer detects 'input_ids' and skips its own pipeline).
    print(f"[train] tokenizing with completion-only mask (max_length={args.max_length})",
          file=sys.stderr)
    tokenize_fn = make_completion_masked_tokenizer(tok, args.max_length)
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, num_proc=1)
    # Drop rows where mask consumed the entire sequence (assistant truncated).
    n_before = len(ds)
    ds = ds.filter(lambda r: any(t != -100 for t in r["labels"]))
    n_after = len(ds)
    if n_after < n_before:
        print(f"[train] dropped {n_before - n_after} rows with no completion tokens",
              file=sys.stderr)

    if args.eval_split_frac and args.eval_split_frac > 0 and len(ds) >= 50:
        split = ds.train_test_split(test_size=args.eval_split_frac, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
        print(f"[train] train={len(train_ds)} eval={len(eval_ds)}", file=sys.stderr)
    else:
        train_ds = ds
        eval_ds = None

    target_mods = (
        ["q_proj", "k_proj", "v_proj", "o_proj"]
        if args.target_modules == "attn"
        else ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_mods,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=100 if eval_ds is not None else None,
        report_to=[],  # silence wandb/etc by default
        seed=args.seed,
        max_seq_length=args.max_length,
        gradient_checkpointing=True,
        # Disable TRL's own dataset preprocessing — our pre-tokenized rows have
        # input_ids/labels/attention_mask already.
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tok,
        data_collator=PadCollator(pad_token_id=tok.pad_token_id),
    )
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_adapter")
    # Use the PEFT model's save_pretrained directly to ensure adapter-only save.
    trainer.model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print(f"[train] saved adapter to {final_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
