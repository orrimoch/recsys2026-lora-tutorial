"""Precompute StateTracker user_state for every train turn, cached for parity.

The bge_m3_ft v2 query carries a serve-safe [STATE] block (the StateTracker
LM-extracted user_state: mood/intent/energy/sonic_pref/era_pref/familiarity).
For train/serve parity the SAME extractor must produce the state at TRAINING
time. This batch-extracts user_state over the conversation split and caches it
to {cache_dir}/state/{session_id}__{turn_number}.json (the StateTracker cache
format), so build_bi_encoder_training_data can load it and emit an identical
[STATE] block. The raw `thought` field is NEVER used (leakage-unsafe — empty in
Blind-A); user_state is computed at both train and serve, so no schema mismatch.

GPU/Colab job (loads a small instruct LM). Idempotent — re-runs hit the cache.

Usage:
  python scripts/precompute_train_user_state.py \
    --lm-type Qwen/Qwen2.5-1.5B-Instruct \
    --cache-dir music-crs-baselines/experiments/cache \
    --conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINES = REPO / "music-crs-baselines"
sys.path.insert(0, str(BASELINES))


def main():
    ap = argparse.ArgumentParser(description="Precompute StateTracker user_state over a conversation split.")
    ap.add_argument("--lm-type", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="StateTracker extractor LM (the documented default).")
    ap.add_argument("--cache-dir", default=str(BASELINES / "experiments" / "cache"),
                    help="States cached to {cache_dir}/state/. Point at a Drive-backed dir on Colab.")
    ap.add_argument("--conv-hf", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    ap.add_argument("--split", default="train", help="'train' (+ 'test' for the final model).")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-sessions", type=int, default=0, help="0 = all; >0 = smoke cap.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--use-vllm", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from datasets import load_dataset
    from mcrs.lm_modules import load_lm_module
    from mcrs.query_rewriters.state_tracker import StateTracker

    lm = load_lm_module(args.lm_type, device=args.device, attn_implementation=args.attn,
                        dtype=torch.bfloat16, use_vllm=args.use_vllm)
    prompt_path = BASELINES / "mcrs" / "system_prompts" / "state_extraction.txt"
    st = StateTracker(lm, prompt_path=str(prompt_path), cache_dir=args.cache_dir,
                      max_new_tokens=args.max_new_tokens)

    ds = load_dataset(args.conv_hf, split=args.split)
    if args.max_sessions:
        ds = ds.select(range(min(args.max_sessions, len(ds))))

    n, fallback = 0, 0
    for sess in ds:
        sid = sess["session_id"]
        df = pd.DataFrame(sess["conversations"])
        for _, music in df[df["role"] == "music"].iterrows():
            tn = int(music["turn_number"])
            urow = df[(df["turn_number"] == tn) & (df["role"] == "user")]
            if urow.empty:
                continue
            user_query = str(urow.iloc[0]["content"])
            prior = df[df["turn_number"] < tn]
            history_text = "\n".join(f"{t['role']}: {t['content']}" for _, t in prior.iterrows())
            state = st.extract(sid, tn, user_query, history_text)   # caches as a side-effect
            n += 1
            if StateTracker.was_fallback(state):
                fallback += 1
            if n % 2000 == 0:
                print(f"[precompute-state] {n} turns extracted ({fallback} fallbacks)", flush=True)

    print(f"[precompute-state] DONE: {n} turn states cached to {args.cache_dir}/state/ "
          f"({fallback} parse-fallbacks)")


if __name__ == "__main__":
    main()
