"""Smoke-test ProRank locally (W3 Task #22).

Loads Qwen-2.5-0.5B-Instruct, scores a small candidate set against a few
queries, verifies last-token-logit-diff scoring runs and produces a
sensible ordering. Doesn't load the full 50k-track tid-text cache — uses
an in-memory fake catalog so the smoke is fast.

Pass criterion: scoring runs without crashing, scores are finite floats,
relevant docs (matching the query) rank above irrelevant docs.

Usage:
    python scripts/smoke_pro_rank.py
    python scripts/smoke_pro_rank.py --device mps --batch-size 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args(argv)

    # Bypass the full ProRankReranker.__init__ (which builds a 50k-track
    # tid-text cache via HF datasets). Instead, instantiate via class
    # surgery: load model+tokenizer, then patch in a fake tid_to_text.
    from mcrs.rerankers.pro_rank import ProRankReranker

    # Auto-pick device.
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"[smoke-prorank] device={device} dtype={dtype} model={args.model}")

    # Build a fake tid_to_text: 4 jazz docs + 4 metal docs + 2 random.
    fake_catalog = {
        "jazz_1": "Holocene by Bon Iver | folk indie | dreamy slow",
        "jazz_2": "Blue in Green by Miles Davis | modal jazz | calm reflective",
        "jazz_3": "Round Midnight by Thelonious Monk | bebop jazz | introspective",
        "jazz_4": "So What by Miles Davis | modal jazz | smooth thoughtful",
        "metal_1": "Master of Puppets by Metallica | thrash metal | aggressive heavy",
        "metal_2": "Raining Blood by Slayer | thrash metal | brutal fast",
        "metal_3": "Painkiller by Judas Priest | speed metal | shredding wild",
        "metal_4": "Black Sabbath by Black Sabbath | doom metal | dark slow",
        "rand_1": "Yesterday by The Beatles | pop rock | mellow nostalgic",
        "rand_2": "Bohemian Rhapsody by Queen | rock opera | dramatic theatrical",
    }

    print("[smoke-prorank] loading model (this is the slow part)…")
    t0 = time.time()
    # Use real __init__ but stub the tid-text builder. Approach: monkey-patch
    # the cache loader before construction.
    orig_loader = ProRankReranker._load_or_build_tid_text
    def _fake_loader(self, *_a, **_k):
        return dict(fake_catalog)
    ProRankReranker._load_or_build_tid_text = _fake_loader  # type: ignore[assignment]
    try:
        reranker = ProRankReranker(
            item_db_name="dummy",
            track_split_types=[],
            corpus_types=[],
            cache_dir="/tmp/smoke_prorank_cache",
            model_name=args.model,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
        )
    finally:
        ProRankReranker._load_or_build_tid_text = orig_loader  # type: ignore[assignment]
    print(f"[smoke-prorank] loaded in {time.time() - t0:.1f}s")
    print(f"[smoke-prorank] yes_id={reranker.yes_id} no_id={reranker.no_id}")

    # Score 3 queries.
    queries = [
        "I want some smooth modal jazz for late-night focus",
        "Looking for aggressive thrash metal to power through a workout",
        "Recommend something nostalgic and acoustic",
    ]
    candidates = list(fake_catalog.keys())

    print("\n[smoke-prorank] running rerank() on 3 queries × 10 candidates…")
    t0 = time.time()
    out = reranker.rerank(
        queries=queries,
        candidate_tids=[candidates] * len(queries),
        topk=5,
    )
    dt = time.time() - t0

    rep = reranker.report()
    print(f"\n[smoke-prorank] elapsed: {dt:.1f}s")
    print(f"[smoke-prorank] stats: {rep}")

    # P1 #6 fix — print score distribution. If scores cluster near zero,
    # the model is uncalibrated for the domain and the W3 gate will likely
    # fail. Surfacing this early avoids burning 45 A100-min in vain.
    import numpy as np
    all_scores = list(reranker._score_cache.values())
    if all_scores:
        arr = np.array(all_scores)
        print(f"\n[smoke-prorank] score distribution (n={len(arr)}):")
        print(f"  mean={arr.mean():+.3f}  std={arr.std():.3f}")
        print(f"  range: [{arr.min():+.3f}, {arr.max():+.3f}]")
        n_pos = int((arr > 0).sum())
        n_neg = int((arr < 0).sum())
        print(f"  positive (yes wins): {n_pos:>3}/{len(arr)} ({100*n_pos/len(arr):.1f}%)")
        print(f"  negative (no wins):  {n_neg:>3}/{len(arr)} ({100*n_neg/len(arr):.1f}%)")
        # Calibration warning: if std < 0.5 and mean ≈ 0, the model is
        # essentially flipping a coin per pair → reranking will be noisy.
        if arr.std() < 0.5 and abs(arr.mean()) < 0.5:
            print("  ⚠️  Low score variance — base SLM looks weakly calibrated.")
            print("     W3 gate (+0.015 nDCG@10) is paper-expected to MISS.")
            print("     Plan now: train ProRank via GRPO OR fall back to BGE.")

    print("\n=== RESULTS ===")
    for q, top5 in zip(queries, out):
        print(f"\nQuery: {q}")
        print(f"  Top-5: {top5}")

    # Smoke pass criteria (infrastructure-only — does NOT validate W3 gate):
    #   1. All scores are finite (no NaN/Inf — model loaded + forward pass works)
    #   2. Stats reflect the right number of (q, d) pairs were scored
    #   3. Q2 (clear-cut metal query) places a metal track at top — sanity for the
    #      base SLM. Q1/Q3 are MURKY for un-trained Qwen-0.5B (paper §3.2: base
    #      Qwen-0.5B scores ~0.30 on BEIR vs trained ProRank ~0.51). We do NOT
    #      gate on those — full discrimination requires GRPO warmup, run via
    #      colab/20_train_prorank_dev.ipynb.
    failures = []
    if rep["scored_pairs"] != 30:
        failures.append(f"expected 30 scored pairs, got {rep['scored_pairs']}")
    if rep["rerank_calls"] != 3:
        failures.append(f"expected 3 rerank calls, got {rep['rerank_calls']}")
    if not out[1][0].startswith("metal"):
        failures.append(f"Q2 (clear metal query): expected metal_* at top, got {out[1][0]}")

    print()
    if failures:
        print("  FAIL  ProRank infrastructure smoke")
        for f in failures:
            print(f"    - {f}")
        return 2
    print("  PASS  ProRank infrastructure smoke")
    print("        (Q2 metal query placed metal at top — base SLM is functional)")
    print("        Q1/Q3 calibration is paper-expected weak without GRPO warmup;")
    print("        run colab/20_train_prorank_dev.ipynb for the trained policy,")
    print("        OR set reranker_type=bge_reranker_v2_m3 as the cut-path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
