"""Offline retrieval-only evaluation on a held-out slice of train conversations.

Tier-1 purpose (this file): pre-test retrieval changes BEFORE burning a Blind-A
submission slot. Blind-A has only 80 rows and ±0.05 nDCG@20 noise per
`project_fresh_model_state.md`; this harness runs ~800+ (query, gold) pairs
from held-out train sessions, giving a far tighter signal.

What it does:
    1. Deterministic stratified sample of N train sessions (seed-fixed).
    2. For each turn in each session with role=='music', extract
       (user_query, chat_history, gold_track_id). All music turns contribute
       one (query, gold) pair.
    3. Load retrieval module from the yaml config — NO LM loaded (Tier 1).
    4. Run batch retrieval, score nDCG@{1,10,20} + catalog diversity
       against gold tids (binary relevance per row).
    5. Append a row to documents/offline_eval_log.md.

Tier-2 (not in this file): full two-step pipeline on 80-row holdout with
response-quality heuristics. Runs on Colab or local smoke. Build later.

Usage:
    python scripts/offline_eval.py --tid 021-two-step-wrrf-lyrics-qwen15b-blindsetA
    python scripts/offline_eval.py --tid 020-two-step-wrrf-lyrics-qwen15b-devset --holdout-size 300

NOTE: runs the retrieval stack only. If the yaml references Qwen2.5-3B or a
reranker, those get skipped. This is deliberate — Tier 1 is for fast retrieval
signal, not generation/response evaluation.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
EVALUATOR_DIR = REPO_ROOT / "music-crs-evaluator"
sys.path.insert(0, str(BASELINES_DIR))
sys.path.insert(0, str(EVALUATOR_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from datasets import load_dataset  # noqa: E402

from mcrs.db_item import MusicCatalogDB  # noqa: E402
from mcrs.retrieval_modules import load_retrieval_module  # noqa: E402
from metrics.metrics_recsys import get_ndcg  # noqa: E402
from metrics.metrics_diversity import compute_catalog_diversity  # noqa: E402


def load_train_holdout(size: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic stratified holdout by `conversation_goal.category`.

    Returns a list of session dicts (up to `size`). Stratification keeps
    the same category mix as full train so retrieval metrics aren't skewed
    by one goal type dominating.
    """
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    # Group session indices by conversation_goal.category.
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, sess in enumerate(ds):
        cat = (sess.get("conversation_goal") or {}).get("category") or "unknown"
        by_cat[cat].append(i)

    rng = random.Random(seed)
    total = sum(len(v) for v in by_cat.values())
    picked: list[int] = []
    for cat, indices in by_cat.items():
        n_cat = max(1, round(len(indices) / total * size))
        rng.shuffle(indices)
        picked.extend(indices[:n_cat])
    rng.shuffle(picked)
    picked = picked[:size]  # truncate if rounding overshot
    return ds.select(picked).to_list()


def build_query_gold_pairs(
    holdout: list[dict], item_db: MusicCatalogDB,
) -> list[dict[str, Any]]:
    """For each music turn in each held-out session, build one
    (user_query, chat_history, gold_tid) eval pair.

    Chat-history formatting matches inference pipeline
    (CRS_BASELINE.batch_chat line 169): newline-joined "role: content" where
    music turns have their track_id expanded to metadata via item_db.
    """
    pairs: list[dict[str, Any]] = []
    for sess in holdout:
        convos = sess.get("conversations") or []
        df = pd.DataFrame(convos)
        if df.empty:
            continue
        # Find music turns — one eval pair per music turn.
        music_rows = df[df["role"] == "music"]
        for _, music_row in music_rows.iterrows():
            turn_n = music_row["turn_number"]
            gold_tid = music_row["content"]
            # Chat history = all turns STRICTLY BEFORE this music turn at
            # this turn_number. That includes the user turn at turn_n (the
            # query that prompted the recommendation) but NOT the music turn
            # itself (which is the gold we're predicting).
            prior_idx = []
            for i, row in df.iterrows():
                if row["turn_number"] < turn_n:
                    prior_idx.append(i)
                elif row["turn_number"] == turn_n and row["role"] == "user":
                    prior_idx.append(i)
            if not prior_idx:
                continue
            prior = df.loc[prior_idx]
            # Format same as inference
            history_lines = []
            user_query = None
            for _, t in prior.iterrows():
                role = "assistant" if t["role"] == "music" else t["role"]
                content = t["content"]
                if t["role"] == "music":
                    try:
                        content = item_db.id_to_metadata(content)
                    except Exception:
                        content = str(content)
                history_lines.append(f"{role}: {content}")
                if t["role"] == "user" and t["turn_number"] == turn_n:
                    user_query = t["content"]
            retrieval_input = "\n".join(history_lines)
            pairs.append({
                "session_id": sess["session_id"],
                "turn_number": int(turn_n),
                "user_query": user_query,
                "retrieval_input": retrieval_input,
                "gold_tid": gold_tid,
            })
    return pairs


def run_tier1(
    tid: str, holdout_size: int = 200, seed: int = 42, topk_batch: int = 20,
) -> dict[str, float]:
    """Run retrieval-only offline eval for the given config."""
    cfg = OmegaConf.load(BASELINES_DIR / "config" / f"{tid}.yaml")
    print(f"[offline-eval] config: {tid}")
    print(f"[offline-eval] retrieval_type={cfg.retrieval_type}  corpus_types={list(cfg.corpus_types)}")

    # Paths referenced in the yaml resolve against music-crs-baselines/ (same
    # as during actual inference). Temporarily chdir there so cache_dir like
    # '../experiments/cache' points to the same place.
    origin_cwd = os.getcwd()
    os.chdir(BASELINES_DIR)
    try:
        print(f"[offline-eval] loading holdout: size={holdout_size} seed={seed}")
        holdout = load_train_holdout(holdout_size, seed)
        print(f"[offline-eval] loading item_db: {cfg.item_db_name}")
        item_db = MusicCatalogDB(
            cfg.item_db_name, list(cfg.track_split_types), list(cfg.corpus_types),
        )
        print(f"[offline-eval] building eval pairs from {len(holdout)} sessions")
        pairs = build_query_gold_pairs(holdout, item_db)
        print(f"[offline-eval] built {len(pairs)} (query, gold_tid) pairs")

        print(f"[offline-eval] loading retrieval: {cfg.retrieval_type}")
        retrieval = load_retrieval_module(
            cfg.retrieval_type,
            cfg.item_db_name,
            list(cfg.track_split_types),
            list(cfg.corpus_types),
            cfg.cache_dir,
        )
        retrieval_inputs = [p["retrieval_input"] for p in pairs]
        print(f"[offline-eval] running batch retrieval (topk={topk_batch})")
        preds_per_row = retrieval.batch_text_to_item_retrieval(
            retrieval_inputs, topk=topk_batch,
        )
    finally:
        os.chdir(origin_cwd)

    # Score each (gold, preds) pair
    ndcg1 = np.mean([get_ndcg([p["gold_tid"]], preds, 1) for p, preds in zip(pairs, preds_per_row)])
    ndcg10 = np.mean([get_ndcg([p["gold_tid"]], preds, 10) for p, preds in zip(pairs, preds_per_row)])
    ndcg20 = np.mean([get_ndcg([p["gold_tid"]], preds, 20) for p, preds in zip(pairs, preds_per_row)])
    # Catalog diversity — flatten all preds across all queries.
    catalog_size = len(item_db.metadata_dict) if hasattr(item_db, "metadata_dict") else 47071
    flat_preds = [tid for preds in preds_per_row for tid in preds]
    catdiv = compute_catalog_diversity(flat_preds, catalog_size)
    # Retrieval-side composite per plan §2.5.1 (LexDiv=0 because no LM).
    composite_retrieval = 0.50 * ndcg20 + 0.10 * catdiv

    scores = {
        "ndcg@1": round(float(ndcg1), 4),
        "ndcg@10": round(float(ndcg10), 4),
        "ndcg@20": round(float(ndcg20), 4),
        "catalog_diversity": round(float(catdiv), 4),
        "composite_retrieval": round(float(composite_retrieval), 4),
        "n_pairs": len(pairs),
        "holdout_size": holdout_size,
        "seed": seed,
    }

    print("\n=== Tier-1 offline eval result ===")
    for k, v in scores.items():
        print(f"  {k:22}: {v}")
    return scores


def append_log_row(tid: str, scores: dict[str, Any]) -> None:
    """Append a single row to documents/offline_eval_log.md (create if missing)."""
    log_path = REPO_ROOT / "documents" / "offline_eval_log.md"
    header_lines = [
        "# Offline eval log — Tier-1 retrieval-only",
        "",
        "Append-only. Populated by `scripts/offline_eval.py`. Tier-1 is fast,",
        "deterministic-seeded, retrieval-only; use this as the Blind-A pre-filter.",
        "",
        "Rule: a candidate config ships to Blind-A ONLY if its offline",
        "composite_retrieval + 2*sigma ≥ prior champion's offline composite",
        "AND nDCG@20 ≥ wRRF baseline. Sigma estimated from repeated-seed runs.",
        "",
        "## Table",
        "",
        "| tid | seed | holdout_size | n_pairs | nDCG@1 | nDCG@10 | nDCG@20 | CatDiv | composite_retrieval |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(header_lines) + "\n")
    row = (
        f"| {tid} | {scores['seed']} | {scores['holdout_size']} | {scores['n_pairs']} |"
        f" {scores['ndcg@1']} | {scores['ndcg@10']} | {scores['ndcg@20']} |"
        f" {scores['catalog_diversity']} | {scores['composite_retrieval']} |"
    )
    with log_path.open("a") as f:
        f.write(row + "\n")
    print(f"[offline-eval] row appended to {log_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Tier-1 offline retrieval eval.")
    p.add_argument("--tid", required=True,
                   help="Config filename (no .yaml) under music-crs-baselines/config/")
    p.add_argument("--holdout-size", type=int, default=200,
                   help="Number of train sessions to hold out (default 200).")
    p.add_argument("--seed", type=int, default=42,
                   help="Deterministic holdout sample seed.")
    p.add_argument("--no-log", action="store_true",
                   help="Skip appending to offline_eval_log.md")
    args = p.parse_args()

    scores = run_tier1(args.tid, holdout_size=args.holdout_size, seed=args.seed)
    if not args.no_log:
        append_log_row(args.tid, scores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
