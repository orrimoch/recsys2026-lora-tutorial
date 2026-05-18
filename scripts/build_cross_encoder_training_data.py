"""Stage B training data builder.

Mines HNs via the FINE-TUNED BGE-M3 (Stage A output) so the cross-encoder
sees in-distribution negatives per spec §7.

Walks the HF conversation dataset (NOT train.parquet — schema lacks
chat_history / current_user_query / user_profile_raw / conversation_goal).

Usage:
  python scripts/build_cross_encoder_training_data.py \
    --train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset \
    --bge-m3-ft-hub OrRim123/recsys2026-bge-m3-music-v1-merged \
    --output experiments/cache/retrieval_v2/triples_reranker.jsonl \
    --percpos-threshold 0.80 --k-negs 7
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from tqdm import tqdm

from mcrs.retrieval_modules.bge_m3_format import format_query_text, format_track_text
from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
from build_bi_encoder_training_data import _iter_conversation_turns


def build_ce_triple(query: str, gold_track_text: str, neg_track_texts: list[str]) -> dict:
    """One JSONL row for the cross-encoder trainer."""
    return {"query": query, "pos": [gold_track_text], "neg": list(neg_track_texts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-conv-hf", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--bge-m3-ft-hub", required=True)
    parser.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--percpos-threshold", type=float, default=0.80)
    parser.add_argument("--k-negs", type=int, default=7)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"[ce-build] walking {args.train_conv_hf} (train split)", file=sys.stderr)
    conv_ds = load_dataset(args.train_conv_hf, split="train")
    train_rows = _iter_conversation_turns(conv_ds)
    if args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]
    print(f"[ce-build] {len(train_rows)} per-music-turn rows", file=sys.stderr)

    # Track text map
    track_meta = load_dataset(args.track_meta_hf, split="all_tracks")
    track_ids: list[str] = []
    track_texts: list[str] = []
    for trow in tqdm(track_meta, desc="format tracks"):
        track_ids.append(trow["track_id"])
        track_texts.append(format_track_text(
            track_name=trow.get("track_name", "unknown"),
            artist_name=trow.get("artist_name"),
            album_name=trow.get("album_name"),
            release_date=trow.get("release_date"),
            tag_list=trow.get("tag_list"),
        ))
    track_text_map = dict(zip(track_ids, track_texts))
    if len(set(track_ids)) != len(track_ids):
        dupes = [tid for tid, c in Counter(track_ids).items() if c > 1]
        raise RuntimeError(f"track catalog has duplicates (first 5: {dupes[:5]})")

    # Encode catalog with the fine-tuned model
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.bge_m3_ft_hub, device=device)
    print("[ce-build] encoding catalog with fine-tuned BGE-M3", file=sys.stderr)
    track_embs = model.encode(track_texts, batch_size=args.batch_size, normalize_embeddings=True,
                              show_progress_bar=True)
    track_embs = np.asarray(track_embs, dtype=np.float32)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with open(args.output, "w") as f_out:
        for i in tqdm(range(0, len(train_rows), args.batch_size), desc="mine"):
            batch = train_rows[i:i + args.batch_size]
            batch_queries = [format_query_text(
                chat_history=r.get("chat_history") or [],
                current_user_query=r.get("current_user_query", ""),
                user_profile=r.get("user_profile_raw"),
                conversation_goal=r.get("conversation_goal"),
                mode="raw",
            ) for r in batch]
            batch_embs = model.encode(batch_queries, batch_size=args.batch_size,
                                       normalize_embeddings=True, show_progress_bar=False)
            batch_embs = np.asarray(batch_embs, dtype=np.float32)
            for j, row in enumerate(batch):
                gold = row["track_id"]
                if gold not in track_text_map:
                    n_skipped += 1; continue
                try:
                    negs = mine_negatives_for_query(
                        query_emb=batch_embs[j], track_embs=track_embs, track_ids=track_ids,
                        gold_track_id=gold, percpos_threshold=args.percpos_threshold,
                        k_negs=args.k_negs, pool_size=args.pool_size, seed=42 + i + j,
                    )
                except ValueError:
                    n_skipped += 1; continue
                if len(negs) < 2:
                    n_skipped += 1; continue
                triple = build_ce_triple(
                    query=batch_queries[j],
                    gold_track_text=track_text_map[gold],
                    neg_track_texts=[track_text_map[n] for n in negs if n in track_text_map],
                )
                f_out.write(json.dumps(triple) + "\n")
                n_written += 1

    print(f"[ce-build] DONE → {args.output} (wrote {n_written}, skipped {n_skipped})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
