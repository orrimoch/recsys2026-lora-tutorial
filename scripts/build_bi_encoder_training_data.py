"""Stage A training data builder.

Walks the HF conversation dataset (talkpl-ai/TalkPlayData-Challenge-Dataset,
train split) → assembles per-music-turn tuples → mines hard negatives via
zero-shot BGE-M3 → writes JSONL triples consumable by FlagEmbedding's
unified_finetune.

Why walk the raw HF dataset (NOT W2's train.parquet)? W2's train.parquet
schema is (source, session_id, track_id, query, code_1, code_2, code_3) —
it does NOT carry chat_history, current_user_query, user_profile_raw, or
conversation_goal. Reading those from a row dict returns the defaults and
silently produces useless queries (`"user: "` for every example).

The walker mirrors `mcrs.sid.training_data.build_raw_conversation_pairs`'s
iteration shape so the (history, query, profile, goal, gold_tid) tuples we
build for bi-encoder fine-tuning are structurally identical to the ones
SID training already uses.

Usage:
  python scripts/build_bi_encoder_training_data.py \
    --train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset \
    --track-meta-hf talkpl-ai/TalkPlayData-Challenge-Track-Metadata \
    --bge-m3-model BAAI/bge-m3 \
    --output experiments/cache/retrieval_v2/triples_bge_m3.jsonl \
    --percpos-threshold 0.80 --k-negs 15 --pool-size 200 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))

import numpy as np
from tqdm import tqdm

from mcrs.retrieval_modules.bge_m3_format import format_query_text, format_track_text
from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query


def build_triples_for_row(
    row: dict,
    gold_track_id: str,
    neg_track_ids: list[str],
    track_text_map: dict[str, str],
    query_mode: str = "bge_m3_structured",
) -> dict:
    """Build one JSONL triple for a (query, gold, negs) tuple.

    `query_mode` MUST match the deployment YAML's query_preprocessing_mode or
    the fine-tune is optimized for a distribution the runtime never sees.

    Negatives not present in track_text_map are silently dropped (catalog drift guard).
    """
    query = format_query_text(
        chat_history=row.get("chat_history") or [],
        current_user_query=row.get("current_user_query", ""),
        user_profile=row.get("user_profile_raw"),
        conversation_goal=row.get("conversation_goal"),
        mode=query_mode,
    )
    return {
        "query": query,
        "pos": [track_text_map[gold_track_id]],
        "neg": [track_text_map[tid] for tid in neg_track_ids if tid in track_text_map],
        # ML-reviewer I-3: carry the gold track_id so training-time full-catalog
        # nDCG eval (in train_bi_encoder.py) can score against the actual ~50k
        # catalog instead of just the 16 mined candidates.
        "pos_tid": gold_track_id,
        # Session-level data leak fix: emit session_id so TripleJsonlDataset can
        # split val SESSION-disjoint from train. Without this, row-shuffle val
        # splits create 95%+ session overlap with train -> model memorizes
        # session-level patterns -> in-training val metric is inflated relative
        # to true dev generalization (observed: val=0.24 but dev=0.11).
        "session_id": row.get("session_id"),
    }


def _iter_conversation_turns(sessions) -> list[dict[str, Any]]:
    """Walk the HF conversation dataset and emit one row per music-recommendation turn.

    Mirrors `mcrs.sid.training_data.build_raw_conversation_pairs`'s iteration
    logic exactly — at each music turn, emit a tuple carrying chat_history
    (turns BEFORE this music turn), current_user_query (the most recent user
    utterance), and gold_track_id (the recommended track at this turn).

    Yields dicts with keys: chat_history, current_user_query, user_profile_raw,
    conversation_goal, track_id, session_id.
    """
    rows: list[dict[str, Any]] = []
    for sess_idx, session in enumerate(sessions):
        convs = session.get("conversations", [])
        user_profile = session.get("user_profile")
        conversation_goal = session.get("conversation_goal")
        session_id = (
            session.get("session_id")
            or session.get("id")
            or session.get("conversation_id")
            or f"session_{sess_idx}"
        )
        chat_history: list[dict[str, str]] = []
        pending_user_query: Optional[str] = None
        for turn in convs:
            role = turn.get("role")
            content = turn.get("content") or ""
            if role == "user":
                pending_user_query = content
            elif role == "music":
                # ML-reviewer I-2: truthy check (NOT `is not None`) so empty-string
                # user content doesn't produce a degenerate row mapping a blank
                # [QUERY]:  prefix to a specific track. `"" is not None` was True,
                # silently injecting noise rows.
                if pending_user_query and content:
                    rows.append({
                        "session_id": str(session_id),
                        "chat_history": list(chat_history),
                        "current_user_query": pending_user_query,
                        "user_profile_raw": user_profile,
                        "conversation_goal": conversation_goal,
                        "track_id": content,
                    })
                if pending_user_query:
                    chat_history.append({"role": "user", "content": pending_user_query})
                chat_history.append({"role": "assistant", "content": content})
                pending_user_query = None
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-conv-hf",
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="HF conversation dataset to walk (train split).",
    )
    parser.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--bge-m3-model", default="BAAI/bge-m3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--percpos-threshold", type=float, default=0.80,
                        help="Only used when --mining-strategy=percpos.")
    parser.add_argument("--k-negs", type=int, default=15)
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke cap; 0 = all")
    parser.add_argument("--mining-strategy", type=str, default="percpos",
                        choices=["percpos", "simans"],
                        help="percpos: NV-Retriever filter then uniform sample "
                             "(skips queries with insufficient surviving negatives). "
                             "simans: Gaussian-weighted sample without filter "
                             "(no skip; targets moderate-difficulty negatives via "
                             "exp(-(s_i - s_pos + a)^2/b); see Zhou et al. EMNLP 2022).")
    parser.add_argument("--simans-a", type=float, default=0.1,
                        help="SimANS target offset: peak weight at s_pos - a. "
                             "Only used when --mining-strategy=simans.")
    parser.add_argument("--simans-b", type=float, default=0.05,
                        help="SimANS spread (smaller = narrower peak). "
                             "Only used when --mining-strategy=simans.")
    parser.add_argument("--query-mode", type=str, default="bge_m3_structured",
                        choices=["raw", "last_user", "last_user_with_goal", "bge_m3_structured"],
                        help="Query format for HN-mining queries AND emitted training "
                             "queries. MUST match query_preprocessing_mode in the "
                             "deployment YAML or the fine-tune is optimized for a "
                             "distribution the runtime never sees. Default "
                             "bge_m3_structured = matches config 180.")
    args = parser.parse_args()

    # Lazy imports — FlagEmbedding has a heavy CUDA-touching init; keeps unit tests fast.
    from datasets import load_dataset

    # 1. Load train conversations and assemble per-music-turn tuples.
    print(
        f"[hn-miner] loading conversations from {args.train_conv_hf} (train split)...",
        file=sys.stderr,
    )
    conv_ds = load_dataset(args.train_conv_hf, split="train")
    train_rows = _iter_conversation_turns(conv_ds)
    if args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]
    print(f"[hn-miner] {len(train_rows)} raw conversation→track pairs", file=sys.stderr)

    # 2. Load track metadata + build text map
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

    # Validate catalog uniqueness ONCE up-front. `mine_negatives_for_query`
    # raises on duplicates per query; doing it here turns N silent skips into
    # one loud fail-fast at startup.
    if len(set(track_ids)) != len(track_ids):
        dupes = [tid for tid, c in Counter(track_ids).items() if c > 1]
        raise RuntimeError(
            f"track catalog has {len(dupes)} duplicate track_ids — first 5: {dupes[:5]}. "
            "mine_negatives_for_query requires unique IDs."
        )

    # 3. Encode all tracks with zero-shot BGE-M3
    # Lazy imports — FlagEmbedding has a heavy CUDA-touching init; keeps unit tests fast.
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

    # 4. Mine negatives per query, write JSONL
    print(f"[hn-miner] mining negatives per query → {args.output}", file=sys.stderr)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_no_gold = 0
    n_skipped_miner_error = 0
    n_skipped_too_few_negs = 0
    first_error_logged = False
    with open(args.output, "w") as f_out:
        for i in tqdm(range(0, len(train_rows), args.batch_size), desc="mine"):
            batch_rows = train_rows[i:i + args.batch_size]
            batch_queries = [
                format_query_text(
                    chat_history=r.get("chat_history") or [],
                    current_user_query=r.get("current_user_query", ""),
                    user_profile=r.get("user_profile_raw"),
                    conversation_goal=r.get("conversation_goal"),
                    mode=args.query_mode,
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
                    n_skipped_no_gold += 1
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
                        strategy=args.mining_strategy,
                        simans_a=args.simans_a,
                        simans_b=args.simans_b,
                    )
                except ValueError as e:
                    n_skipped_miner_error += 1
                    if not first_error_logged:
                        print(f"[hn-miner] first miner ValueError: {e}", file=sys.stderr)
                        first_error_logged = True
                    continue
                if len(negs) < 2:
                    n_skipped_too_few_negs += 1
                    continue
                triple = build_triples_for_row(row, gold_tid, negs, track_text_map,
                                                query_mode=args.query_mode)
                f_out.write(json.dumps(triple) + "\n")
                n_written += 1

    print(
        f"[hn-miner] DONE → {args.output} (wrote {n_written}; "
        f"skipped_no_gold={n_skipped_no_gold}, "
        f"skipped_miner_error={n_skipped_miner_error}, "
        f"skipped_too_few_negs={n_skipped_too_few_negs})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
