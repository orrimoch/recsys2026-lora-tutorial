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


def _free_gpu(label: str = "") -> None:
    """Release Python references + run gc + clear CUDA cache.

    Called at strategic points: after trainer.train() (frees Adam optimizer state ~3.8GB),
    after merge (frees the pre-merge LoRA-wrapped model), after push (frees merged ~3GB).
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1e9
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 1e9
        print(f"[gpu] {label}: {before:.2f} GB -> {after:.2f} GB allocated", file=sys.stderr)


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
                   help="Tiny run for plumbing test (200 steps, 1k train rows, ~10 min)")
    p.add_argument("--tiny", action="store_true",
                   help="Even tinier e2e plumbing test (30 steps, 100 train rows, ~3 min)")
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
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in --output-dir if any exists. "
                        "Picks up model + optimizer + scheduler + RNG state. Use after a "
                        "Colab disconnect or OOM.")
    p.add_argument("--cleanup-after-push", action="store_true",
                   help="After pushing adapter (and merged) to Hub, delete the local "
                        "adapter_dir, merged_dir, and trainer checkpoints. Use when the Hub "
                        "is the canonical store and local disk is at a premium.")
    p.add_argument("--report-to", default="tensorboard",
                   choices=["tensorboard", "wandb", "trackio", "none"],
                   help="Logging backend for live training metrics graphs (default: tensorboard). "
                        "TensorBoard writes ~5-20 MB of logs under output_dir/runs/<timestamp>/; "
                        "view via %%tensorboard --logdir <path> in Colab.")
    p.add_argument("--save-best", action="store_true",
                   help="Disk-friendly mode: keep at most 2 checkpoints (best + latest), "
                        "save model weights ONLY (no optimizer state — drops checkpoint size "
                        "from ~4.8GB to ~975MB), evaluate eval_loss every save_steps, and "
                        "load_best_model_at_end so the pushed model is the best by val loss. "
                        "Trade-off: ~5x less disk, but cannot resume mid-training (no optimizer).")
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Disable gradient checkpointing. Trades memory for ~30%% speedup. "
                        "Use on big-VRAM GPUs (Blackwell 95GB, A100 80GB). On L4 24GB this "
                        "may OOM at batch>=16.")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="If set, copy per-experiment artifacts to <results-dir>/<run-id>/ "
                        "BEFORE --cleanup-after-push fires. Persists: tensorboard runs/, "
                        "training_summary.json, results.txt (hub repos + final metrics). "
                        "Use a Drive path so artifacts survive Colab runtime death.")
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

    # --tiny is the smallest plumbing-test mode (~3 min); --smoke is the medium
    # plumbing-test mode (~10 min); no flag = full training.
    if args.tiny:
        train_sample_n, val_sample_n, max_steps = 100, 30, 30
        save_steps_eff = 30
    elif args.smoke:
        train_sample_n, val_sample_n, max_steps = 1000, 200, 200
        save_steps_eff = 500
    else:
        train_sample_n, val_sample_n, max_steps = None, None, -1
        save_steps_eff = 500

    train_ds = build_hf_dataset(args.train_parquet, tokenizer, sid_lookup, args.max_prompt_len, sample_n=train_sample_n)
    val_ds = build_hf_dataset(args.val_parquet, tokenizer, sid_lookup, args.max_prompt_len, sample_n=val_sample_n)

    # --save-best (Option B): disk-friendly + best-by-val-loss model selection.
    # save_only_model=True drops optimizer state from each checkpoint
    # (~4.8GB → ~975MB), losing resume capability but cutting peak disk 5x.
    # Requires eval to determine "best", so eval_strategy is forced to steps.
    save_best_kwargs: dict = {}
    if args.save_best:
        save_best_kwargs = dict(
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    eval_strategy = "no" if (args.smoke or args.tiny) else "steps"
    if args.save_best:
        eval_strategy = "steps"  # required for load_best_model_at_end
    eval_steps = save_steps_eff if args.save_best else 200

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
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps_eff,
        save_total_limit=2,
        report_to=args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        **save_best_kwargs,
    )

    def collator(examples):
        return collate_training_batch(examples, pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    # Resume from last checkpoint if requested AND one exists. HF picks the
    # latest checkpoint in output_dir automatically when passed True.
    resume_arg = args.resume if args.resume else None
    if args.resume:
        ckpts = sorted(args.output_dir.glob("checkpoint-*"))
        if ckpts:
            print(f"[resume] from latest checkpoint: {ckpts[-1].name}", file=sys.stderr)
        else:
            print(f"[resume] no checkpoints in {args.output_dir}; starting fresh", file=sys.stderr)
            resume_arg = None
    trainer.train(resume_from_checkpoint=resume_arg)

    # Capture final-step metrics for the results.txt summary later.
    final_log = trainer.state.log_history[-1] if trainer.state.log_history else {}

    # Free Adam optimizer state (~3.8GB on GPU for 476M trainable params) — done
    # with training, no longer needed for the save/merge/push steps.
    trainer.optimizer = None
    trainer.lr_scheduler = None
    _free_gpu("after trainer.train")

    # Save LoRA adapter locally + push.
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] LoRA adapter at {adapter_dir}", file=sys.stderr)

    hub_lora_repo = f"{args.hub_repo}-lora"
    model.push_to_hub(hub_lora_repo, private=False)
    tokenizer.push_to_hub(hub_lora_repo, private=False)
    print(f"[push] LoRA adapter repo: https://huggingface.co/{hub_lora_repo}", file=sys.stderr)

    hub_merged_repo = None
    if args.merge:
        merged = model.merge_and_unload()
        # del + null out trainer's reference to actually free the pre-merge
        # LoRA-wrapped model from VRAM. Without nulling trainer.model and
        # trainer.model_wrapped, the only Python reference is gone but trainer
        # still pins the tensors → torch.cuda.empty_cache reclaims nothing.
        del model
        trainer.model = None
        trainer.model_wrapped = None
        _free_gpu("after merge_and_unload")

        merged_dir = args.output_dir / "merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        hub_merged_repo = f"{args.hub_repo}-merged"
        # safetensors is the default in recent transformers; push_to_hub no longer
        # accepts safe_serialization (only save_pretrained does, on line above).
        merged.push_to_hub(hub_merged_repo, private=False)
        tokenizer.push_to_hub(hub_merged_repo, private=False)
        print(f"[push] merged model repo: https://huggingface.co/{hub_merged_repo}", file=sys.stderr)
        del merged
        _free_gpu("after merged push")

    # Persist a small summary so notebooks can pick it up.
    summary = {
        "base_model": BASE_MODEL,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "epochs": args.epochs, "lr": args.lr,
        "smoke": args.smoke, "tiny": args.tiny,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "hub_lora_repo": hub_lora_repo,
        "hub_merged_repo": hub_merged_repo,
        "final_metrics": final_log,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Persist per-experiment artifacts to a Drive-friendly location BEFORE cleanup.
    # Survives Colab runtime death and --cleanup-after-push.
    if args.results_dir is not None:
        import shutil
        from datetime import datetime
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.hub_repo.split('/')[-1]}"
        per_exp_dir = args.results_dir / run_id
        per_exp_dir.mkdir(parents=True, exist_ok=True)
        # Copy training_summary.json
        shutil.copy2(args.output_dir / "training_summary.json", per_exp_dir / "training_summary.json")
        # Copy tensorboard runs/ if they exist (for post-mortem viewing)
        runs_src = args.output_dir / "runs"
        if runs_src.exists():
            shutil.copytree(runs_src, per_exp_dir / "runs", dirs_exist_ok=True)
        # Write a human-readable results.txt with key info
        results_lines = [
            f"# W3 SID Generator Training — {run_id}",
            "",
            f"Base model:        {BASE_MODEL}",
            f"Mode:              {'tiny' if args.tiny else 'smoke' if args.smoke else 'full'}",
            f"Epochs:            {args.epochs}",
            f"LR:                {args.lr}",
            f"LoRA r/alpha:      {args.lora_r} / {args.lora_alpha}",
            f"n_train / n_val:   {len(train_ds)} / {len(val_ds)}",
            "",
            f"Hub repos:",
            f"  LoRA adapter:    https://huggingface.co/{hub_lora_repo}",
        ]
        if hub_merged_repo:
            results_lines.append(f"  Merged model:    https://huggingface.co/{hub_merged_repo}")
        results_lines += [
            "",
            f"Final-step metrics:",
        ]
        for k, v in final_log.items():
            results_lines.append(f"  {k:24s} {v}")
        results_lines += [
            "",
            f"TensorBoard logs:  {per_exp_dir / 'runs'}",
            f"  view via:        %tensorboard --logdir {per_exp_dir / 'runs'}",
        ]
        (per_exp_dir / "results.txt").write_text("\n".join(results_lines) + "\n")
        print(f"[results] persisted to {per_exp_dir}", file=sys.stderr)

    # Optional: cleanup local copies once Hub push has succeeded. Saves ~5 GB
    # (checkpoints) + ~975 MB (adapter) + ~3 GB (merged) on whatever disk
    # output_dir lives on. Hub remains the canonical store.
    # NOTE: this does NOT touch output_dir/runs/ (TensorBoard logs) — copy them
    # to --results-dir if you want them to survive cleanup.
    if args.cleanup_after_push:
        import shutil
        for ckpt in args.output_dir.glob("checkpoint-*"):
            print(f"[cleanup] rm {ckpt}", file=sys.stderr)
            shutil.rmtree(ckpt, ignore_errors=True)
        adapter_dir = args.output_dir / "adapter"
        if adapter_dir.exists():
            print(f"[cleanup] rm {adapter_dir}", file=sys.stderr)
            shutil.rmtree(adapter_dir, ignore_errors=True)
        if args.merge:
            merged_dir = args.output_dir / "merged"
            if merged_dir.exists():
                print(f"[cleanup] rm {merged_dir}", file=sys.stderr)
                shutil.rmtree(merged_dir, ignore_errors=True)
        _free_gpu("after cleanup")


if __name__ == "__main__":
    main()
