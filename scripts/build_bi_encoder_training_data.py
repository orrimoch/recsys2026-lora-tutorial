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

The `--bge-m3-model` CLI flag is misleadingly named (historical); it accepts any
sentence-transformers-compatible model. Tested with:
  - BAAI/bge-m3 (default; 567M params, 1024-dim, multilingual)
  - BAAI/bge-base-en-v1.5 (110M params, 768-dim, English-only; 5× smaller — allows
    larger batch sizes during training/mining)
  - BAAI/bge-large-en-v1.5 (335M params, 1024-dim, English-only)

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
from mcrs.retrieval_modules.hn_miner import batch_mine_negatives, mine_negatives_for_query


def build_triples_for_row(
    row: dict,
    gold_track_id: str,
    neg_track_ids: list[str],
    track_text_map: dict[str, str],
    query_mode: str = "bge_m3_structured",
    metadata_dict: Optional[dict] = None,
    tag_vocab: Optional[dict[str, int]] = None,
    release_year_lookup: Optional[dict[str, int]] = None,
    max_tags: int = 20,
) -> dict:
    """Build one JSONL triple for a (query, gold, negs) tuple.

    `query_mode` MUST match the deployment YAML's query_preprocessing_mode or
    the fine-tune is optimized for a distribution the runtime never sees.

    Negatives not present in track_text_map are silently dropped (catalog drift guard).

    Multi-modal fields (Phase 1, fresh-model branch):
      When ``tag_vocab`` and/or ``release_year_lookup`` are provided, the
      output triple additionally carries:
        - ``tag_ids_pos`` / ``tag_ids_neg``: list[list[int]] of tag-vocab IDs
          (truncated to ``max_tags``; pad-id 0 reserved).
        - ``release_year_pos`` / ``release_year_neg``: list[int] (-1 = unknown).
      These extra fields drive the multi-modal model's tag-embedding +
      release-year tokens (see ``MultiModalBiEncoder`` in
      ``mcrs/training/multimodal_bi_encoder.py``).

    DESIGN NOTE: teacher-score-driven multi-positive promotion is NOT done
    here. The triples file stays single-positive (canonical) and the
    training dataloader joins ``teacher_scores.parquet`` at __getitem__
    time to promote alternates dynamically. This keeps the builder a
    single, fast pass (no reranker dependency) AND lets multi-positive
    threshold be tuned without rebuilding triples.
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
    out = {
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
    if tag_vocab is not None and metadata_dict is not None:
        out["tag_ids_pos"] = [_track_to_tag_ids(gold_track_id, metadata_dict, tag_vocab, max_tags)]
        out["tag_ids_neg"] = [
            _track_to_tag_ids(tid, metadata_dict, tag_vocab, max_tags) for tid in kept_neg_tids
        ]
    if release_year_lookup is not None:
        out["release_year_pos"] = [int(release_year_lookup.get(gold_track_id, -1))]
        out["release_year_neg"] = [int(release_year_lookup.get(tid, -1)) for tid in kept_neg_tids]
    return out


def _track_to_tag_ids(
    track_id: str,
    metadata_dict: dict,
    tag_vocab: dict[str, int],
    max_tags: int = 20,
) -> list[int]:
    """Map a track's ``tag_list`` (lowercased, stripped) to tag-vocab IDs.

    Matches ``scripts/precompute_multimodal_artifacts.py:_build_metadata_artifacts``
    normalization (lowercase + strip) so the same tag string always maps to
    the same vocab id at train and inference time. Unknown tags are skipped.
    """
    md = metadata_dict.get(track_id) if metadata_dict else None
    if not md:
        return []
    tags = md.get("tag_list") or []
    if isinstance(tags, str):
        tags = [tags]
    ids: list[int] = []
    for t in tags[:max_tags]:
        norm = str(t).strip().lower()
        tid = tag_vocab.get(norm)
        if tid is not None:
            ids.append(tid)
    return ids


def _load_multimodal_artifacts(cache_dir: str) -> tuple[Optional[dict], Optional[dict]]:
    """Load tag_vocab.json + release_year_lookup.json from a multi-modal
    artifact cache dir (built by ``scripts/precompute_multimodal_artifacts.py``).

    Returns ``(tag_vocab, release_year_lookup)``. Either can be None if its
    artifact file is missing — caller decides how strict to be.
    """
    import os as _os
    tag_path = _os.path.join(cache_dir, "tag_vocab.json")
    year_path = _os.path.join(cache_dir, "release_year_lookup.json")
    tag_vocab = None
    year_lookup = None
    if _os.path.isfile(tag_path):
        with open(tag_path, "r") as f:
            tag_vocab = json.load(f)
        print(f"[mm-artifacts] loaded tag_vocab: {len(tag_vocab)} entries from {tag_path}",
              file=sys.stderr)
    else:
        print(f"[mm-artifacts] WARN: {tag_path} missing — tag_ids_* will not be emitted",
              file=sys.stderr)
    if _os.path.isfile(year_path):
        with open(year_path, "r") as f:
            year_lookup = json.load(f)
        print(f"[mm-artifacts] loaded release_year_lookup: {len(year_lookup)} tracks from {year_path}",
              file=sys.stderr)
    else:
        print(f"[mm-artifacts] WARN: {year_path} missing — release_year_* will not be emitted",
              file=sys.stderr)
    return tag_vocab, year_lookup


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
    # Phase 1 (fresh-model branch): multi-modal training-data fields.
    # When ``--multimodal-artifacts`` is set, the builder loads
    # ``tag_vocab.json`` + ``release_year_lookup.json`` from that dir and
    # emits ``tag_ids_pos`` / ``tag_ids_neg`` / ``release_year_pos`` /
    # ``release_year_neg`` in every triple. Backwards-compatible:
    # without the flag the JSONL stays in the original schema.
    parser.add_argument(
        "--multimodal-artifacts", type=str, default="",
        help="Optional path to the cache dir built by "
             "scripts/precompute_multimodal_artifacts.py. When set, the "
             "builder loads tag_vocab.json + release_year_lookup.json from "
             "this dir and adds tag_ids_{pos,neg} + release_year_{pos,neg} "
             "fields to every triple. Leave empty to keep the legacy text-only "
             "schema (current default).",
    )
    parser.add_argument(
        "--max-tags", type=int, default=20,
        help="Truncate per-track tag_ids to this length (matches the model's "
             "tag-token capacity). Only used when --multimodal-artifacts is set.",
    )
    args = parser.parse_args()

    # Phase 1: optionally load multi-modal artifacts (tag_vocab + release_year_lookup).
    tag_vocab: Optional[dict] = None
    release_year_lookup: Optional[dict] = None
    if args.multimodal_artifacts:
        tag_vocab, release_year_lookup = _load_multimodal_artifacts(args.multimodal_artifacts)

    # Lazy imports — FlagEmbedding has a heavy CUDA-touching init; keeps unit tests fast.
    from datasets import load_dataset

    # 1. Load track metadata FIRST. Build the id_to_metadata-aligned text map
    #    used for BOTH (a) pos/neg payloads in training triples AND (b)
    #    [HISTORY] music-turn expansion via _format_history_music_turn AND
    #    (c) the catalog re-embed in nb 70 cell 6.
    #
    # PATCH 4 (2026-05-22, train/inference format alignment): pos/neg now use
    # the SAME `id_to_metadata` format that production's chat_history_parser
    # produces at inference time. Previously pos/neg used format_track_text
    # (5 fields, pipe-separated, original case) while [HISTORY] mentions of
    # the same tracks used id_to_metadata (4 fields, comma, lowercased) —
    # forcing the encoder to learn TWO representations of every track.
    # Aligning both sides removes that asymmetry; the catalog re-embed in
    # nb 70 cell 6 must also use this format to stay consistent.
    print(f"[hn-miner] loading track metadata from {args.track_meta_hf}", file=sys.stderr)
    track_meta = load_dataset(args.track_meta_hf, split="all_tracks")
    metadata_dict: dict = {}
    track_ids: list[str] = []
    history_corpus_types = [
        ct.strip() for ct in args.history_corpus_types.split(",") if ct.strip()
    ]
    for trow in tqdm(track_meta, desc="index tracks"):
        tid = trow["track_id"]
        metadata_dict[tid] = dict(trow)
        track_ids.append(tid)
    # Build pos/neg text via the SAME formatter as [HISTORY] music turns +
    # catalog vectors (id_to_metadata mirror). Closes the train/inference
    # format gap surfaced by the BlindA nDCG regression diagnosis.
    track_text_map = {
        tid: _format_history_music_turn(tid, metadata_dict, history_corpus_types)
        for tid in track_ids
    }
    if track_ids:
        sample_text = track_text_map[track_ids[0]]
        print(f"[hn-miner] track_text_map format (sample): {sample_text[:200]}", file=sys.stderr)

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

    # 3. Encode all tracks with the zero-shot encoder for HN mining.
    # Materialize track_texts from track_text_map in track_ids order so the
    # encoder output rows align with track_ids[i].
    track_texts = [track_text_map[tid] for tid in track_ids]
    # Lazy imports — heavy CUDA init; keeps unit tests fast.
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Encoder loaded via sentence-transformers (works for BGE-M3, bge-base-en-v1.5,
    # bge-large-en-v1.5, etc.). Production's DENSE_LOCAL also uses ST, so the
    # mining-time encoding contract matches the inference-time contract exactly.
    # FP16 on GPU for speed (matches the old BGEM3FlagModel use_fp16=True behavior).
    print(f"[hn-miner] loading zero-shot encoder: {args.bge_m3_model}", file=sys.stderr)
    model = SentenceTransformer(args.bge_m3_model, device=device)
    # max_seq_length used for BOTH catalog tracks and queries (sentence-transformers
    # doesn't take per-call max_length). 512 covers our query format (~300 tokens) +
    # leaves headroom for tracks (~50-80 tokens); also matches bge-base-en-v1.5's
    # native 512 cap.
    model.max_seq_length = 512
    if device == "cuda":
        model = model.half()  # FP16 inference; ~2× faster forward, ~50% less VRAM
    print("[hn-miner] encoding catalog tracks...", file=sys.stderr)
    track_embs = model.encode(
        track_texts, batch_size=args.batch_size,
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    )
    track_embs = np.asarray(track_embs, dtype=np.float32)
    # Defensive re-normalize (ST normalize_embeddings=True already does this;
    # the redundant pass is cheap and guards against version differences).
    norms = np.linalg.norm(track_embs, axis=1, keepdims=True)
    track_embs = track_embs / np.clip(norms, 1e-9, None)

    # 4. Mine negatives per query, write JSONL.
    #
    # Vectorization: track_embs is uploaded to GPU ONCE (saves ~380GB of
    # transfers vs uploading per batch). Each batch then issues a single
    # GPU matmul + topk via batch_mine_negatives, instead of per-query
    # CPU argsort. ~5-6× faster wallclock on Blackwell at typical batch
    # sizes — the previous per-query CPU loop dominated mining time.
    print(f"[hn-miner] mining negatives per query → {args.output}", file=sys.stderr)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_no_gold = 0
    n_skipped_miner_error = 0
    n_skipped_too_few_negs = 0
    first_error_logged = False

    # Pin catalog to GPU once for the whole mining run.
    track_embs_dev = torch.from_numpy(track_embs)
    if device == "cuda":
        track_embs_dev = track_embs_dev.to("cuda")
        print(f"[hn-miner] pinned catalog to GPU: {track_embs_dev.shape} dtype={track_embs_dev.dtype}",
              file=sys.stderr)

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
                batch_queries, batch_size=args.batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
            batch_embs = np.asarray(batch_embs, dtype=np.float32)
            qnorms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
            batch_embs = batch_embs / np.clip(qnorms, 1e-9, None)

            # Vectorized mining: one GPU matmul + one topk for the whole batch.
            # batch_mine_negatives returns one entry per query (None when gold
            # isn't in the catalog). Seeds match the legacy per-query loop's
            # 42 + global_idx convention (test_..._matches_build_script_..._convention).
            gold_tids = [r["track_id"] for r in batch_rows]
            try:
                batch_negs = batch_mine_negatives(
                    query_embs=batch_embs,
                    track_embs=track_embs_dev,
                    track_ids=track_ids,
                    gold_track_ids=gold_tids,
                    percpos_threshold=args.percpos_threshold,
                    k_negs=args.k_negs,
                    pool_size=args.pool_size,
                    seed=42 + i,
                    strategy=args.mining_strategy,
                    simans_a=args.simans_a,
                    simans_b=args.simans_b,
                )
            except ValueError as e:
                # Whole-batch failure (shape/validation). Falls back to per-row
                # accounting: count each row as a miner error.
                n_skipped_miner_error += len(batch_rows)
                if not first_error_logged:
                    print(f"[hn-miner] first miner ValueError: {e}", file=sys.stderr)
                    first_error_logged = True
                continue

            for j, row in enumerate(batch_rows):
                gold_tid = gold_tids[j]
                negs = batch_negs[j]
                if negs is None:
                    # gold_tid not in catalog
                    n_skipped_no_gold += 1
                    continue
                if len(negs) < 2:
                    n_skipped_too_few_negs += 1
                    continue
                triple = build_triples_for_row(
                    row, gold_tid, negs, track_text_map,
                    query_mode=args.query_mode,
                    metadata_dict=metadata_dict,
                    tag_vocab=tag_vocab,
                    release_year_lookup=release_year_lookup,
                    max_tags=args.max_tags,
                )
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
