"""Stage A training data builder.

Reads existing W2 train.parquet (raw subset) → mines hard negatives via
zero-shot BGE-M3 → writes JSONL triples consumable by FlagEmbedding's
unified_finetune.

Usage:
  python scripts/build_bi_encoder_training_data.py \
    --train-parquet experiments/cache/sid_training/train.parquet \
    --track-meta-hf talkpl-ai/TalkPlayData-Challenge-Track-Metadata \
    --bge-m3-model BAAI/bge-m3 \
    --output experiments/cache/retrieval_v2/triples_bge_m3.jsonl \
    --percpos-threshold 0.80 --k-negs 15 --pool-size 200 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from mcrs.retrieval_modules.bge_m3_format import format_query_text, format_track_text
from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query


def build_triples_for_row(
    row: dict,
    gold_track_id: str,
    neg_track_ids: list[str],
    track_text_map: dict[str, str],
) -> dict:
    """Build one JSONL triple for a (query, gold, negs) tuple.

    Negatives not present in track_text_map are silently dropped (catalog drift guard).
    """
    query = format_query_text(
        chat_history=row.get("chat_history") or [],
        current_user_query=row.get("current_user_query", ""),
        user_profile=row.get("user_profile_raw"),
        conversation_goal=row.get("conversation_goal"),
        mode="raw",  # match production's default in build_retrieval_query
    )
    return {
        "query": query,
        "pos": [track_text_map[gold_track_id]],
        "neg": [track_text_map[tid] for tid in neg_track_ids if tid in track_text_map],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--bge-m3-model", default="BAAI/bge-m3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--percpos-threshold", type=float, default=0.80)
    parser.add_argument("--k-negs", type=int, default=15)
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke cap; 0 = all")
    args = parser.parse_args()

    # 1. Load train data (raw subset only — metadata-source rows aren't conversation→track pairs)
    train = pd.read_parquet(args.train_parquet)
    train = train[train["source"] == "raw"].reset_index(drop=True)
    if args.max_rows > 0:
        train = train.head(args.max_rows)
    print(f"[hn-miner] {len(train)} raw conversation pairs", file=sys.stderr)

    # 2. Load track metadata + build text map
    from datasets import load_dataset
    track_meta = load_dataset(args.track_meta_hf, split="all_tracks")
    track_ids: list[str] = []
    track_texts: list[str] = []
    for trow in tqdm(track_meta, desc="format tracks"):
        tid = trow["track_id"]
        text = format_track_text(
            track_name=trow.get("track_name", "unknown"),
            artist_name=trow.get("artist_name"),
            album_name=trow.get("album_name"),
            release_date=trow.get("release_date"),
            tag_list=trow.get("tag_list"),
        )
        track_ids.append(tid)
        track_texts.append(text)
    track_text_map = dict(zip(track_ids, track_texts))

    # 3. Encode all tracks with zero-shot BGE-M3
    import torch
    from FlagEmbedding import BGEM3FlagModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(args.bge_m3_model, use_fp16=True, device=device)
    print("[hn-miner] encoding catalog tracks...", file=sys.stderr)
    track_embs = model.encode(
        track_texts, batch_size=args.batch_size, max_length=256,
    )["dense_vecs"]
    track_embs = np.asarray(track_embs, dtype=np.float32)
    # Ensure unit-norm (BGE-M3 should return L2-normalized; guard against version differences)
    norms = np.linalg.norm(track_embs, axis=1, keepdims=True)
    track_embs = track_embs / np.clip(norms, 1e-9, None)
    np.save(str(Path(args.output).with_suffix(".track_embs.npy")), track_embs)

    # 4. Mine negatives per query, write JSONL
    print(f"[hn-miner] mining negatives per query → {args.output}", file=sys.stderr)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with open(args.output, "w") as f_out:
        for i in tqdm(range(0, len(train), args.batch_size), desc="mine"):
            batch_rows = train.iloc[i:i+args.batch_size].to_dict("records")
            batch_queries = [
                format_query_text(
                    chat_history=r.get("chat_history") or [],
                    current_user_query=r.get("current_user_query", ""),
                    user_profile=r.get("user_profile_raw"),
                    conversation_goal=r.get("conversation_goal"),
                    mode="raw",
                )
                for r in batch_rows
            ]
            batch_embs = model.encode(
                batch_queries, batch_size=args.batch_size, max_length=512,
            )["dense_vecs"]
            batch_embs = np.asarray(batch_embs, dtype=np.float32)
            # Normalize queries (defensive — same guard as track_embs)
            qnorms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
            batch_embs = batch_embs / np.clip(qnorms, 1e-9, None)
            for j, row in enumerate(batch_rows):
                gold_tid = row["track_id"]
                if gold_tid not in track_text_map:
                    n_skipped += 1
                    continue
                try:
                    negs = mine_negatives_for_query(
                        query_emb=batch_embs[j],
                        track_embs=track_embs,
                        track_ids=track_ids,
                        gold_track_id=gold_tid,
                        percpos_threshold=args.percpos_threshold,
                        k_negs=args.k_negs,
                        pool_size=args.pool_size,
                        seed=42 + i + j,
                    )
                except ValueError as e:
                    n_skipped += 1
                    continue
                if len(negs) < 2:
                    n_skipped += 1
                    continue
                triple = build_triples_for_row(row, gold_tid, negs, track_text_map)
                f_out.write(json.dumps(triple) + "\n")
                n_written += 1

    print(f"[hn-miner] DONE → {args.output} (wrote {n_written}, skipped {n_skipped})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
