"""Smoke-test the StateTracker on a few train sessions (Task #5).

Runs end-to-end: load Qwen-1.5B → for each (session, turn), call
StateTracker.extract → check parse-validity, latency, sample outputs.

The plan's A1 spec calls for ≥99% parse-validity. On MPS this script processes
5 sessions × 8 turns = 40 generations in ~2 min as a smoke. The full 100-session
test (~13 min on MPS / <1 min on Colab A100) should be re-run on Colab once
this smoke passes.

Usage:
    python scripts/smoke_state_tracker.py                 # 5 sessions, default
    python scripts/smoke_state_tracker.py --n-sessions 25
    python scripts/smoke_state_tracker.py --model Qwen/Qwen2.5-0.5B-Instruct  # faster MPS
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mcrs.lm_modules.llama import LLAMA_MODEL  # noqa: E402
from mcrs.query_rewriters.state_tracker import StateTracker  # noqa: E402
from reward_fns import dump_json  # noqa: E402

PROMPT_PATH = REPO_ROOT / "music-crs-baselines" / "mcrs" / "system_prompts" / "state_extraction.txt"
# Use experiments/cache/ to match the cf-bpr convention and avoid the
# `rm -rf cache` purge in run_inference_devset.py:71 (Gap 3). The smoke
# script doesn't go through that purge in practice, but keeping caches
# in one place makes the convention consistent.
CACHE_DIR = REPO_ROOT / "music-crs-baselines" / "experiments" / "cache" / "state_smoke"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_history(convos: pd.DataFrame, turn_n: int) -> str:
    """Concatenate prior turns up to turn_n. Music turns expand to 'track by artist'."""
    prior = convos[convos["turn_number"] < turn_n]
    if prior.empty:
        return ""
    lines = []
    for _, r in prior.tail(8).iterrows():  # cap to 8 latest entries
        role = "assistant" if r["role"] == "music" else r["role"]
        content = str(r["content"])[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-sessions", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto if omitted)")
    p.add_argument("--max-new-tokens", type=int, default=96)
    args = p.parse_args(argv)

    device = args.device or _pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"[smoke] device={device} dtype={dtype} model={args.model}")

    print(f"[smoke] loading model (this is the slow part)…")
    t0 = time.time()
    lm = LLAMA_MODEL(
        model_name=args.model,
        device=device,
        attn_implementation="eager",
        dtype=dtype,
    )
    print(f"[smoke] model loaded in {time.time() - t0:.1f}s")

    tracker = StateTracker(
        lm=lm,
        prompt_path=PROMPT_PATH,
        cache_dir=str(CACHE_DIR),
        max_new_tokens=args.max_new_tokens,
        debug_failures=True,
    )

    # Load train sessions
    print(f"[smoke] loading {args.n_sessions} train sessions")
    from datasets import load_dataset
    import random
    tr = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(tr)), args.n_sessions) if args.n_sessions < len(tr) else range(len(tr))
    sessions = tr.select(idx).to_list()

    # Walk turns
    timings = []
    samples = []
    print(f"[smoke] extracting states…")
    t_total = time.time()
    for sess in sessions:
        sid = sess["session_id"]
        convos = pd.DataFrame(sess.get("conversations") or [])
        if convos.empty:
            continue
        for turn_n in sorted(convos["turn_number"].unique()):
            tn = int(turn_n)
            tdf = convos[convos["turn_number"] == tn]
            user_row = tdf[tdf["role"] == "user"]
            if user_row.empty:
                continue
            user_query = str(user_row.iloc[0]["content"])
            history = _build_history(convos, tn)

            t0 = time.time()
            state = tracker.extract(sid, tn, user_query, history)
            dt = time.time() - t0
            timings.append(dt)

            if len(samples) < 6:
                samples.append({
                    "session_id": sid[:8],
                    "turn": tn,
                    "user_query": user_query[:80],
                    "state": state,
                    "ms": int(dt * 1000),
                })

    elapsed = time.time() - t_total
    rep = tracker.report()
    print()
    print("=" * 70)
    print("STATE TRACKER SMOKE — RESULTS")
    print("=" * 70)
    print(f"  sessions:            {len(sessions)}")
    print(f"  total calls:         {rep['calls']}")
    print(f"  cache hits:          {rep['cache_hits']}")
    print(f"  ok first try:        {rep['ok_first_try']}")
    print(f"  ok after retry:      {rep['ok_after_retry']}")
    print(f"  fallback to prior:   {rep['fallback_to_prior']}")
    print(f"  drop:                {rep['drop']}")
    print(f"  parse validity (excl cache): {rep['parse_validity_excl_cache']:.3f}")
    print(f"  total time:          {elapsed:.1f}s")
    if timings:
        print(f"  per-call mean: {sum(timings)/len(timings)*1000:.0f}ms  "
              f"max: {max(timings)*1000:.0f}ms")

    print("\nSample states:")
    for s in samples:
        print(f"\n  session={s['session_id']}  turn={s['turn']}  ms={s['ms']}")
        print(f"  query: {s['user_query']!r}")
        print(f"  state: {s['state']}")

    if tracker.failures:
        print(f"\n  --- {len(tracker.failures)} parse failures (first 3 raw outputs) ---")
        for f in tracker.failures[:3]:
            print(f"\n  session={f['session_id'][:8]}  turn={f['turn_number']}")
            print(f"  query:    {f['user_query']!r}")
            print(f"  attempt1: {f['raw_output_attempt1']!r}")
            print(f"  attempt2: {f['raw_output_attempt2']!r}")

    # Persist a JSON report so it can be cited in experiments_log.md.
    out = REPO_ROOT / "data" / "smoke_state_tracker_report.json"
    dump_json({
        "model": args.model,
        "device": device,
        "n_sessions": args.n_sessions,
        "stats": rep,
        "elapsed_seconds": round(elapsed, 1),
        "per_call_mean_ms": round(sum(timings)/len(timings)*1000, 1) if timings else None,
        "samples": samples,
    }, out)
    print(f"\n[smoke] report → {out}")

    # Gate per plan A1: parse_validity_excl_cache ≥ 0.99 (relax to 0.95 for smoke
    # since N is small).
    pv = rep["parse_validity_excl_cache"]
    if pv >= 0.95:
        print(f"\n  PASS  parse validity {pv:.3f} ≥ 0.95 (smoke threshold)")
        return 0
    else:
        print(f"\n  FAIL  parse validity {pv:.3f} < 0.95 — investigate model/prompt")
        return 2


if __name__ == "__main__":
    sys.exit(main())
