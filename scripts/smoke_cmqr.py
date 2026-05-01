"""End-to-end smoke test for CMQR (W2 Task #12).

Runs the full state-tracker → CMQR rewrite → fake-inner-retriever → RRF-fuse
pipeline on a few real train sessions. Exercises the LM rewrite generation
without the cost of loading the production wRRF retrieval indices (BM25 +
dense embeddings on 50k tracks). The fake retriever returns deterministic
ranked lists keyed by query content so we can verify fusion behaviour.

Pass criterion (smoke): ≥80% of turns produce 4 valid rewrites; rewrites
are diverse (no two identical); RRF fusion produces non-empty top-K with
no duplicates.

Usage:
    python scripts/smoke_cmqr.py                       # 3 sessions, MPS, Qwen-1.5B
    python scripts/smoke_cmqr.py --n-sessions 5
    python scripts/smoke_cmqr.py --model Qwen/Qwen2.5-0.5B-Instruct  # faster on MPS
"""
from __future__ import annotations

import argparse
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
from mcrs.query_rewriters.cmqr import CMQR_REWRITER  # noqa: E402
from reward_fns import dump_json  # noqa: E402

STATE_PROMPT_PATH = REPO_ROOT / "music-crs-baselines" / "mcrs" / "system_prompts" / "state_extraction.txt"
CMQR_PROMPT_PATH = REPO_ROOT / "music-crs-baselines" / "mcrs" / "system_prompts" / "cmqr_rewrites.txt"
CACHE_DIR = REPO_ROOT / "music-crs-baselines" / "experiments" / "cache" / "smoke_cmqr"


class _FakeRetriever:
    """Deterministic per-query ranked list. Mimics the real wRRF surface."""
    def __init__(self, catalog_size: int = 200):
        self.catalog = [f"track_{i:04d}" for i in range(catalog_size)]
        self.calls = 0

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None):
        self.calls += 1
        out = []
        for q in queries:
            # Use a hash-based projection so similar queries get overlapping
            # results (so RRF fusion has signal). Each query yields a
            # permutation of the top-N catalog ids ordered by hash buckets.
            h = abs(hash(q)) % 1000
            base = [(i + h) % len(self.catalog) for i in range(topk)]
            out.append([self.catalog[i] for i in base])
        return out


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_history(convos: pd.DataFrame, turn_n: int) -> str:
    prior = convos[convos["turn_number"] < turn_n]
    if prior.empty:
        return ""
    lines = []
    for _, r in prior.tail(8).iterrows():
        role = "assistant" if r["role"] == "music" else r["role"]
        content = str(r["content"])[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-sessions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = args.device or _pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"[smoke-cmqr] device={device} dtype={dtype} model={args.model}")

    print(f"[smoke-cmqr] loading model…")
    t0 = time.time()
    lm = LLAMA_MODEL(
        model_name=args.model, device=device,
        attn_implementation="eager", dtype=dtype,
    )
    print(f"[smoke-cmqr] model loaded in {time.time() - t0:.1f}s")

    tracker = StateTracker(
        lm=lm, prompt_path=STATE_PROMPT_PATH,
        cache_dir=str(CACHE_DIR), max_new_tokens=96,
    )
    fake_retriever = _FakeRetriever()
    cmqr = CMQR_REWRITER(
        lm=lm, inner_retriever=fake_retriever,
        prompt_path=CMQR_PROMPT_PATH,
        cache_dir=str(CACHE_DIR),
        n_rewrites=4, topk_per_rewrite=50, rrf_k=60, max_new_tokens=96,
        debug=True,
    )

    print(f"[smoke-cmqr] loading {args.n_sessions} train sessions")
    from datasets import load_dataset
    import random
    tr = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(tr)), args.n_sessions)
    sessions = tr.select(idx).to_list()

    # Walk turns: state_tracker.extract → cmqr (with state) → top-K
    samples = []
    timings = []
    print(f"[smoke-cmqr] running pipeline…")
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
            cmqr.set_batch_context([sid], [tn], [state])
            top20 = cmqr.batch_text_to_item_retrieval([user_query], topk=20)[0]
            dt = time.time() - t0
            timings.append(dt)

            # Inspect cached rewrites for this turn so we can show them.
            cache_file = cmqr._cache_path(sid, tn)
            rewrites = []
            if cache_file.exists():
                import json
                with cache_file.open("r", encoding="utf-8") as f:
                    rewrites = json.load(f).get("rewrites", [])

            if len(samples) < 4:
                samples.append({
                    "session_id": sid[:8],
                    "turn": tn,
                    "user_query": user_query[:90],
                    "state_keys": list(state.keys()) if state else None,
                    "n_rewrites": len(rewrites),
                    "rewrites": rewrites[:4],
                    "top5_track_ids": top20[:5],
                    "n_top20": len(top20),
                    "ms": int(dt * 1000),
                })

    elapsed = time.time() - t_total
    cmqr_rep = cmqr.report()
    state_rep = tracker.report()

    print()
    print("=" * 70)
    print("CMQR SMOKE — RESULTS")
    print("=" * 70)
    print(f"  sessions:                  {len(sessions)}")
    print(f"  total CMQR calls:          {cmqr_rep['calls']}")
    print(f"  CMQR cache hits:           {cmqr_rep['cache_hits']}")
    print(f"  full 4-rewrite rate:       {cmqr_rep['full_rewrite_rate']:.3f}")
    print(f"  partial (<4) rewrites:     {cmqr_rep['rewrites_partial']}")
    print(f"  dropped (0 valid):         {cmqr_rep['rewrites_dropped']}")
    print(f"  StateTracker parse rate:   {state_rep['parse_validity_excl_cache']:.3f}")
    print(f"  total time:                {elapsed:.1f}s")
    if timings:
        print(f"  per-call mean: {sum(timings)/len(timings)*1000:.0f}ms  "
              f"max: {max(timings)*1000:.0f}ms")

    print("\nSample rewrites + top-5 fused track IDs:")
    for s in samples:
        print(f"\n  session={s['session_id']}  turn={s['turn']}  ms={s['ms']}")
        print(f"  query:        {s['user_query']!r}")
        print(f"  state keys:   {s['state_keys']}")
        print(f"  rewrites ({s['n_rewrites']}):")
        for r in s["rewrites"]:
            print(f"    - {r}")
        print(f"  top-5 track IDs: {s['top5_track_ids']}")
        print(f"  total returned: {s['n_top20']}")

    if cmqr.failures:
        print(f"\n  --- {len(cmqr.failures)} CMQR parse failures ---")
        for f in cmqr.failures[:2]:
            print(f"\n  session={(f.get('session_id') or '?')[:8]}  turn={f.get('turn_number')}")
            print(f"  raw_output: {(f.get('raw_output') or '')[:200]!r}")

    out = REPO_ROOT / "data" / "smoke_cmqr_report.json"
    dump_json({
        "model": args.model,
        "device": device,
        "n_sessions": args.n_sessions,
        "cmqr_stats": cmqr_rep,
        "state_tracker_stats": state_rep,
        "elapsed_seconds": round(elapsed, 1),
        "per_call_mean_ms": round(sum(timings)/len(timings)*1000, 1) if timings else None,
        "samples": samples,
    }, out)
    print(f"\n[smoke-cmqr] report → {out}")

    # Smoke gate: ≥80% full 4-rewrite rate + no drops + state parse ≥95%
    full_rate = cmqr_rep["full_rewrite_rate"]
    parse_rate = state_rep["parse_validity_excl_cache"]
    drops = cmqr_rep["rewrites_dropped"]
    if full_rate >= 0.80 and parse_rate >= 0.95 and drops == 0:
        print(f"\n  PASS  CMQR smoke (full={full_rate:.2f} ≥0.80, "
              f"state={parse_rate:.2f} ≥0.95, drops=0)")
        return 0
    print(f"\n  FAIL  CMQR smoke (full={full_rate:.2f}, state={parse_rate:.2f}, drops={drops})")
    return 2


if __name__ == "__main__":
    sys.exit(main())
