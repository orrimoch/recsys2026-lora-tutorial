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
    # Filter neg track_ids once so `neg` and `neg_tids` stay index-aligned.
    kept_neg_tids = [tid for tid in neg_track_ids if tid in track_text_map]
    return {
        "query": query,
        "pos": [track_text_map[gold_track_id]],
        "neg": [track_text_map[tid] for tid in kept_neg_tids],
        # ML-reviewer I-3: gold track_id for full-catalog val nDCG.
        "pos_tid": gold_track_id,
        # Δ3 issue C extension (RocketQAv2 / BGE-M3 §3.3): per-neg track_ids
        # parallel to `neg`. Used by `_info_nce_loss_in_batch_masked` to mask
        # the cross-pos-into-neg-slot collision (this query's gold appearing
        # as a hard negative for another query in the batch).
        "neg_tids": kept_neg_tids,
        # Δ2 (§6.5): user_id is the train/val split key — every session of a
        # given user lives in exactly one partition. session_id is retained
        # as the legacy split key and for batch-sampling diagnostics.
        "user_id": row.get("user_id"),
        "session_id": row.get("session_id"),
    }


def _format_history_music_turn(
    track_id: str,
    metadata_dict: dict,
    corpus_types: list,
) -> str:
    """Mirror `MusicCatalogDB.id_to_metadata` byte-for-byte so the
    [HISTORY]: music-turn text matches what Blind-A / devset inference
    feeds the encoder via `chat_history_parser` -> `id_to_metadata`.

    Format (matches mcrs/db_item/music_catalog.py:id_to_metadata):
        'track_id: <id>, <ct1>: <vals>, <ct2>: <vals>, ...'
    where each `<vals>` is `", ".join(metadata[ct]).lower()`.

    Falls back to the raw track_id when metadata_dict doesn't have the
    track (catalog drift). Same fallback shape as `id_to_metadata` would
    behave under a missing key, except we return the bare ID instead of
    raising — the train builder shouldn't crash on one stale conv row.
    """
    if track_id not in metadata_dict:
        return track_id
    md = metadata_dict[track_id]
    parts = [f"track_id: {track_id}"]
    for ct in corpus_types:
        val = md.get(ct)
        if val is None:
            joined = ""
        elif isinstance(val, list):
            joined = ", ".join(str(v) for v in val)
        else:
            joined = str(val)
        parts.append(f"{ct}: {joined.lower()}")
    return ", ".join(parts)


def _iter_conversation_turns(
    sessions,
    metadata_dict: Optional[dict] = None,
    corpus_types: Optional[list] = None,
) -> list[dict[str, Any]]:
    """Walk the HF conversation dataset and emit one row per music-recommendation turn.

    At each music turn: emit a row carrying chat_history (turns BEFORE this
    music turn), current_user_query, and gold_track_id.

    `metadata_dict` + `corpus_types` (optional): when provided, music turns
    appended to chat_history are EXPANDED via `_format_history_music_turn`
    (mirrors `id_to_metadata`). This matches production inference exactly,
    closing the train/eval feature-parity gap surfaced by the chat_history
    raw-ID-vs-metadata-text discussion.

    Back-compat: when `metadata_dict` is None, music turns are appended as
    raw track_ids (legacy behavior; preserves existing dev-eval callers).

    Yields dicts with keys: user_id, session_id, chat_history,
    current_user_query, user_profile_raw, conversation_goal, track_id.
    """
    expand_history = metadata_dict is not None and corpus_types is not None
    rows: list[dict[str, Any]] = []
    for sess_idx, session in enumerate(sessions):
        convs = session.get("conversations", [])
        user_profile = session.get("user_profile")
        conversation_goal = session.get("conversation_goal")
        user_id = session.get("user_id")
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
                # [QUERY]:  prefix to a specific track.
                if pending_user_query and content:
                    rows.append({
                        "user_id": str(user_id) if user_id is not None else None,
                        "session_id": str(session_id),
                        "chat_history": list(chat_history),
                        "current_user_query": pending_user_query,
                        "user_profile_raw": user_profile,
                        "conversation_goal": conversation_goal,
                        "track_id": content,
                    })
                if pending_user_query:
                    chat_history.append({"role": "user", "content": pending_user_query})
                # Train/inference parity: expand the music-turn track_id into
                # the same id_to_metadata format that production's
                # chat_history_parser feeds the encoder. Falls back to raw
                # track_id when metadata is unavailable (legacy callers).
                if expand_history:
                    music_text = _format_history_music_turn(
                        content, metadata_dict, corpus_types,
                    )
                else:
                    music_text = content
                chat_history.append({"role": "assistant", "content": music_text})
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
    parser.add_argument("--history-corpus-types", type=str,
                        default="track_name,artist_name,album_name",
                        help="Comma-separated catalog fields used to expand "
                             "[HISTORY]: music-turn IDs into id_to_metadata "
                             "format at MINING time. Default matches production "
                             "config 021 (corpus_types: [track_name, artist_name, "
                             "album_name]). Closes the train/eval feature-parity "
                             "gap in the [HISTORY] block.")
    args = parser.parse_args()

    # Lazy imports — FlagEmbedding has a heavy CUDA-touching init; keeps unit tests fast.
    from datasets import load_dataset

    # 1. Load track metadata FIRST (we need metadata_dict for history
    #    expansion in step 2). Also build the format_track_text text map
    #    used as pos/neg payloads.
    print(f"[hn-miner] loading track metadata from {args.track_meta_hf}", file=sys.stderr)
    track_meta = load_dataset(args.track_meta_hf, split="all_tracks")
    metadata_dict: dict = {}
    track_ids: list[str] = []
    track_texts: list[str] = []
    history_corpus_types = [
        ct.strip() for ct in args.history_corpus_types.split(",") if ct.strip()
    ]
    for trow in tqdm(track_meta, desc="format tracks"):
        tid = trow["track_id"]
        metadata_dict[tid] = dict(trow)  # raw row preserved for id_to_metadata mirror
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

    # 2. Load train conversations and assemble per-music-turn tuples.
    #    metadata_dict + history_corpus_types are threaded in so music turns
    #    in chat_history get expanded to id_to_metadata format (matches
    #    Blind-A / devset inference exactly).
    print(
        f"[hn-miner] loading conversations from {args.train_conv_hf} (train split); "
        f"history expansion via id_to_metadata format with corpus_types="
        f"{history_corpus_types}",
        file=sys.stderr,
    )
    conv_ds = load_dataset(args.train_conv_hf, split="train")
    train_rows = _iter_conversation_turns(
        conv_ds, metadata_dict=metadata_dict, corpus_types=history_corpus_types,
    )
    if args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]
    print(f"[hn-miner] {len(train_rows)} raw conversation→track pairs", file=sys.stderr)

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
