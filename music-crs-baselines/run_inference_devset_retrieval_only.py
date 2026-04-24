"""Retrieval-only inference entrypoint (plan §5).

Sibling of `run_inference_devset.py`. Loads only the retriever + item_db
(no LM), which drops a full 1000-session devset run from ~45 min to ~3 min
on a Mac. Activated when the config has `lm_type: null` (or the field is
absent).

Schema: writes the same JSON shape as the stock entrypoint, with
`predicted_response: "ok"` as a schema-valid stub (the evaluator only
uses `predicted_response` for lexical_diversity, not nDCG).

Contract (plan §5, Fixes F5/F6, verification §13.4/§13.5):
- Exactly one record per (session, turn ∈ 1..8) → 8000 records on the full
  devset; 800 records on `--subset 100`.
- `predicted_track_ids` = 20 **distinct** track_ids, always. We request
  topk=40, dedup, truncate to 20, and backfill from a popularity pool if
  fewer than 20 survive dedup.

CLI:
    python run_inference_devset_retrieval_only.py --tid <tid> [--subset N]

Reads config from `config/<tid>.yaml`; writes predictions to
`exp/inference/devset/<tid>.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

import pandas as pd
from datasets import load_dataset, concatenate_datasets
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs.db_item import MusicCatalogDB
from mcrs.retrieval_modules import load_retrieval_module


RETRIEVAL_TOPK = 40   # plan Fix F6 — request extra to survive dedup.
FINAL_K = 20          # challenge schema.


def chat_history_parser(conversations, item_db, target_turn_number):
    """Mirror of `run_inference_devset.chat_history_parser`, but takes the item_db
    directly instead of a music_crs wrapper. Logic is copied, not edited
    (plan §5 item 2: "copied, not edited")."""
    df_conversation = pd.DataFrame(conversations)
    df_history = df_conversation[df_conversation["turn_number"] < target_turn_number]
    chat_history: list[dict[str, str]] = []
    for turn_data in df_history.to_dict(orient="records"):
        current_role = turn_data["role"]
        current_content = turn_data["content"]
        if turn_data["role"] == "music":
            current_role = "assistant"
            current_content = item_db.id_to_metadata(turn_data["content"])
        chat_history.append({"role": current_role, "content": current_content})
    df_current_turn = df_conversation[df_conversation["turn_number"] == target_turn_number]
    user_query = df_current_turn.iloc[0]["content"]
    return chat_history, user_query


def build_popularity_pool(config) -> list[str]:
    """Ordered pool of candidate track_ids for dedup backfill.

    Preference order:
      1. `popularity` column on the metadata dataset, descending (if present).
      2. track frequency in the `train` split's `conversations` (music-role
         turns), descending.
      3. Catalog insertion order (every track appears exactly once).

    Only called on the rare path where fewer than 20 distinct track_ids
    survive dedup — the list is built lazily on first need.
    """
    metadata = load_dataset(config.item_db_name)
    concat = concatenate_datasets([metadata[s] for s in config.track_split_types])

    # Option 1: explicit popularity column.
    cols = set(concat.column_names)
    if "popularity" in cols:
        rows = [(r["track_id"], r["popularity"]) for r in concat]
        rows.sort(key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
        return [tid for tid, _ in rows]

    # Option 2: train-split conversational popularity.
    try:
        train = load_dataset(config.test_dataset_name, split="train")
        counts: Counter[str] = Counter()
        for item in train:
            for turn in item["conversations"]:
                if turn["role"] == "music":
                    counts[turn["content"]] += 1
        ranked = [tid for tid, _ in counts.most_common()]
        seen = set(ranked)
        tail = [tid for tid in (r["track_id"] for r in concat) if tid not in seen]
        return ranked + tail
    except Exception:
        # Option 3: catalog order.
        return [r["track_id"] for r in concat]


def dedup_and_pad(candidates: list[str], pool: list[str] | None) -> list[str]:
    """Return exactly FINAL_K distinct track_ids (plan Fix F6)."""
    unique = list(dict.fromkeys(candidates))[:FINAL_K]
    if len(unique) == FINAL_K:
        return unique
    if pool is None:
        # Caller should backfill with a pool if this happens — return what we have;
        # main() will call again with the lazy pool.
        return unique
    seen = set(unique)
    for tid in pool:
        if tid not in seen:
            unique.append(tid)
            seen.add(tid)
            if len(unique) == FINAL_K:
                break
    return unique


def main(args):
    # Do NOT `rm -rf cache` here (the stock script does that; retrieval-only
    # iteration pins `cache_dir: ../../experiments/cache` as a shared index
    # to avoid the 2–3 min rebuild on every run, plan Fix P3).
    config = OmegaConf.load(f"config/{args.tid}.yaml")

    retrieval = load_retrieval_module(
        retrieval_type=config.retrieval_type,
        dataset_name=config.item_db_name,
        track_split_types=config.track_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
    )
    item_db = MusicCatalogDB(config.item_db_name, config.track_split_types, config.corpus_types)

    db = load_dataset(config.test_dataset_name, split="test")
    if args.subset is not None:
        db = db.select(range(min(args.subset, len(db))))

    batch_data: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for item in db:
        user_id = item["user_id"]
        session_id = item["session_id"]
        for target_turn_number in range(1, 9):
            chat_history, user_query = chat_history_parser(item["conversations"], item_db, target_turn_number)
            retrieval_input = "\n".join(
                f"{t['role']}: {t['content']}" for t in (chat_history + [{"role": "user", "content": user_query}])
            )
            batch_data.append(retrieval_input)
            metadata.append({"session_id": session_id, "user_id": user_id, "turn_number": target_turn_number})

    inference_results: list[dict[str, Any]] = []
    pool: list[str] | None = None
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch retrieval"):
        batch = batch_data[i:i + args.batch_size]
        batch_meta = metadata[i:i + args.batch_size]
        batch_hits = retrieval.batch_text_to_item_retrieval(batch, topk=RETRIEVAL_TOPK)
        for j, hits in enumerate(batch_hits):
            picked = dedup_and_pad(hits, pool=None)
            if len(picked) < FINAL_K:
                if pool is None:
                    pool = build_popularity_pool(config)
                picked = dedup_and_pad(hits, pool=pool)
            assert len(picked) == FINAL_K, f"dedup/backfill failed: {len(picked)}"
            assert len(set(picked)) == FINAL_K, f"duplicates after dedup: {picked}"
            inference_results.append({
                "session_id": batch_meta[j]["session_id"],
                "user_id": batch_meta[j]["user_id"],
                "turn_number": batch_meta[j]["turn_number"],
                "predicted_track_ids": picked,
                "predicted_response": "ok",  # schema-valid stub; evaluator uses only for lex_div.
            })

    os.makedirs("exp/inference/devset", exist_ok=True)
    out_path = f"exp/inference/devset/{args.tid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)
    print(f"wrote {len(inference_results)} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval-only batch inference for Music CRS (no LM).")
    parser.add_argument("--tid", type=str, required=True, help="Config identifier (config/<tid>.yaml).")
    parser.add_argument("--batch_size", type=int, default=64, help="Retrieval batch size (CPU-bound).")
    parser.add_argument("--subset", type=int, default=None,
                        help="Truncate test set to first N sessions (default: full 1000).")
    args = parser.parse_args()
    main(args)
