"""Stage B training data builder (multi-modal, multi-positive).

Phase 6a of the Stage B cross-encoder plan. Candidate generation now uses the
TRAINED multi-modal Stage A retriever (``DENSE_MULTIMODAL_LOCAL``), NOT the old
text-only ``bge-m3`` SentenceTransformer. For each training query we take the
Stage A top-``pool_size`` (default 100) candidates; positives = gold + multi-
positive promotions from the Phase 0 ``teacher_scores.parquet`` (any candidate
with ``teacher_score >= multipositive_threshold * gold_score``); negatives =
the remaining Stage A candidates (the hardest, distribution-matched — far harder
than zero-shot SimANS-mined negatives).

Output schema mirrors ``scripts/build_bi_encoder_training_data.py`` BYTE-FOR-BYTE
(same field names) so the analogous Stage B dataloader can join to the same
memmapped CLAP / CF arrays. Track-IDs + modality IDs only — CLAP/CF tensors are
NOT inlined (the dataloader joins memmapped arrays; inlining would inflate the
JSONL by ~12 GB). On top of the Stage A fields, the Stage B builder bakes the
multi-positive promotion INTO the row (``pos`` / ``pos_tids`` become lists; the
Phase 6a plan: "Positives = gold + multi-positive promotions") and carries the
teacher scores (``pos_teacher_score`` / ``pos_teacher_scores`` /
``neg_teacher_scores``) so the pairwise-BCE trainer can rank-weight harder negs.

v1 NOTE (2026-05-24, decision: ship single-positive Stage B): multi-positive
promotion is DORMANT with the current ``teacher_scores.parquet``. That parquet
is keyed by ``(pos_tid, sorted(mined_neg_tids))`` built from the bi-encoder
triples (~7-15 SimANS negs/row), but candidate generation here uses the Stage A
top-``pool_size`` (~100) pool — so the composite key never matches and every
row falls back to single-positive (gold) + Stage A hard negatives. That IS the
intended v1 data (cross-attention over distribution-matched hard negs is the
core Stage B lift); ``pos_teacher_score`` / ``*_teacher_scores`` come out absent
and the trainer uses plain (unweighted) pairwise BCE. True multi-positive needs
a teacher RE-SCORE over the Stage A top-100 pool (v2) — the promotion logic in
``select_positives_and_negatives`` is kept ready for that.

Walks the HF conversation dataset (NOT train.parquet — that schema lacks
chat_history / current_user_query / user_profile_raw / conversation_goal),
mirroring the Stage A builder's walker.

Usage:
  python scripts/build_cross_encoder_training_data.py \
    --train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset \
    --stage-a-hub-repo OrRim123/recsys2026-bge-base-en-music-mm-v1-merged \
    --multimodal-artifacts experiments/cache/multimodal \
    --teacher-scores-path experiments/cache/multimodal/teacher_scores.parquet \
    --multipositive-threshold 0.85 \
    --output experiments/cache/retrieval_v2/triples_reranker_mm.jsonl \
    --pool-size 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from tqdm import tqdm

from mcrs.retrieval_modules.bge_m3_format import format_query_text
from build_bi_encoder_training_data import (
    _format_history_music_turn,
    _iter_conversation_turns,
    _load_multimodal_artifacts,
    _track_to_tag_ids,
)


# Back-compat: the original text-only single-positive triple builder. Kept so
# existing callers / tests of the legacy shape continue to work.
def build_ce_triple(query: str, gold_track_text: str, neg_track_texts: list[str]) -> dict:
    """One legacy single-positive JSONL row for the cross-encoder trainer."""
    return {"query": query, "pos": [gold_track_text], "neg": list(neg_track_texts)}


@dataclass
class CandidateSelection:
    """Result of splitting a Stage A candidate pool into positives + negatives.

    Positives = gold first, then any multi-positive promotions (teacher score
    >= threshold * gold_score), in descending teacher-score order. Negatives =
    the remaining candidates, EXCLUDING the gold and every promoted positive.
    Teacher-score parallel lists are None when no teacher entry is available.
    """

    pos_tids: list[str]
    neg_tids: list[str]
    pos_teacher_scores: Optional[list[float]]
    neg_teacher_scores: Optional[list[float]]


def select_positives_and_negatives(
    gold_tid: str,
    candidate_tids: list[str],
    teacher: Optional[dict],
    multipositive_threshold: float = 0.85,
) -> CandidateSelection:
    """Split a Stage A top-K candidate pool into positives + negatives.

    Args:
      gold_tid: the labeled gold track for this query.
      candidate_tids: Stage A top-K retrieved track_ids (MAY contain the gold
        when Stage A ranked it highly — it is filtered out of the negatives).
      teacher: optional ``{"pos_score": float, "neg_tid_to_score": {tid: score}}``
        entry for this row (from ``teacher_scores.parquet``). When None,
        single-positive (gold only) with all candidates as negatives.
      multipositive_threshold: candidate -> positive iff
        ``teacher_score >= multipositive_threshold * gold_score`` (RocketQAv2
        §3.3 recipe; default 0.85 — mirrors the Stage A builder + dataloader).

    Returns a ``CandidateSelection``. Negatives are guaranteed disjoint from
    positives (the gold + any promotion is removed from the candidate pool).
    """
    pos_tids: list[str] = [gold_tid]
    pos_teacher_scores: Optional[list[float]]
    neg_teacher_scores: Optional[list[float]]

    if teacher is not None and float(teacher.get("pos_score", 0.0)) > 0:
        gold_score = float(teacher["pos_score"])
        cutoff = float(multipositive_threshold) * gold_score
        neg_tid_to_score = teacher.get("neg_tid_to_score") or {}
        # Promote any candidate the teacher rated within threshold of the gold.
        # Keep promotions in descending teacher-score order for determinism;
        # ties broken by first appearance in candidate_tids.
        promotions = [
            tid for tid in candidate_tids
            if tid != gold_tid and float(neg_tid_to_score.get(tid, float("-inf"))) >= cutoff
        ]
        promotions.sort(key=lambda t: -float(neg_tid_to_score[t]))
        pos_tids.extend(promotions)
        pos_set = set(pos_tids)
        neg_tids = [tid for tid in candidate_tids if tid not in pos_set]
        pos_teacher_scores = [gold_score] + [float(neg_tid_to_score[t]) for t in promotions]
        neg_teacher_scores = [float(neg_tid_to_score.get(t, 0.0)) for t in neg_tids]
    else:
        pos_set = set(pos_tids)
        neg_tids = [tid for tid in candidate_tids if tid not in pos_set]
        pos_teacher_scores = None
        neg_teacher_scores = None

    return CandidateSelection(
        pos_tids=pos_tids,
        neg_tids=neg_tids,
        pos_teacher_scores=pos_teacher_scores,
        neg_teacher_scores=neg_teacher_scores,
    )


def build_mm_ce_triple(
    query: str,
    selection: CandidateSelection,
    track_text_map: dict[str, str],
    row: dict,
    metadata_dict: Optional[dict] = None,
    tag_vocab: Optional[dict[str, int]] = None,
    release_year_lookup: Optional[dict[str, int]] = None,
    max_tags: int = 20,
) -> dict:
    """Build one multi-modal, multi-positive JSONL triple for Stage B.

    Output schema mirrors ``build_bi_encoder_training_data.build_triples_for_row``
    field-for-field (``query`` / ``pos`` / ``neg`` / ``pos_tid`` / ``neg_tids`` /
    ``user_id`` / ``session_id`` / ``tag_ids_pos`` / ``tag_ids_neg`` /
    ``release_year_pos`` / ``release_year_neg``), with two Stage-B-specific
    extensions:

      - ``pos`` / ``pos_tids`` / ``tag_ids_pos`` / ``release_year_pos`` are
        multi-positive lists (gold first, then promotions). Stage A keeps these
        single-element and promotes at dataloader time; Stage B bakes the
        promotion in per the Phase 6a plan.
      - ``pos_teacher_score`` (gold scalar) / ``pos_teacher_scores`` (per-positive)
        / ``neg_teacher_scores`` (per-negative) carry the Phase 0 teacher scores
        so the pairwise-BCE trainer can rank-weight harder negatives.

    ``pos_tid`` stays the GOLD track_id (single string) so it remains a valid
    key into ``teacher_scores.parquet`` (which is keyed by (pos_tid, sorted
    neg_tids)) — exactly as the Stage A dataloader expects.

    Track-IDs + modality IDs only; no CLAP/CF tensors inlined (the dataloader
    joins memmapped arrays). Candidate tids absent from ``track_text_map``
    (catalog drift) are dropped from BOTH ``neg`` and the parallel modality /
    teacher-score lists so all per-negative lists stay index-aligned.
    """
    gold_tid = selection.pos_tids[0]

    # Filter positives + negatives to those present in the text map once, so
    # every parallel list (text, tids, tag_ids, years, teacher scores) stays
    # index-aligned (mirrors Stage A's single-pass neg filter).
    kept_pos_idx = [i for i, t in enumerate(selection.pos_tids) if t in track_text_map]
    kept_neg_idx = [i for i, t in enumerate(selection.neg_tids) if t in track_text_map]
    pos_tids = [selection.pos_tids[i] for i in kept_pos_idx]
    neg_tids = [selection.neg_tids[i] for i in kept_neg_idx]

    out: dict[str, Any] = {
        "query": query,
        "pos": [track_text_map[t] for t in pos_tids],
        "neg": [track_text_map[t] for t in neg_tids],
        # Gold-only string: the teacher_scores.parquet key + full-catalog val.
        "pos_tid": gold_tid,
        # Full multi-positive set (gold first, then promotions).
        "pos_tids": pos_tids,
        # Per-neg track_ids parallel to `neg` (in-batch mask + sampler).
        "neg_tids": neg_tids,
        # Split keys (mirror Stage A): user_id is the train/val partition key.
        "user_id": row.get("user_id"),
        "session_id": row.get("session_id"),
    }

    if tag_vocab is not None and metadata_dict is not None:
        out["tag_ids_pos"] = [
            _track_to_tag_ids(t, metadata_dict, tag_vocab, max_tags) for t in pos_tids
        ]
        out["tag_ids_neg"] = [
            _track_to_tag_ids(t, metadata_dict, tag_vocab, max_tags) for t in neg_tids
        ]
    if release_year_lookup is not None:
        out["release_year_pos"] = [int(release_year_lookup.get(t, -1)) for t in pos_tids]
        out["release_year_neg"] = [int(release_year_lookup.get(t, -1)) for t in neg_tids]

    if selection.pos_teacher_scores is not None:
        pos_scores = [selection.pos_teacher_scores[i] for i in kept_pos_idx]
        out["pos_teacher_score"] = float(pos_scores[0]) if pos_scores else 0.0
        out["pos_teacher_scores"] = [float(s) for s in pos_scores]
    if selection.neg_teacher_scores is not None:
        out["neg_teacher_scores"] = [
            float(selection.neg_teacher_scores[i]) for i in kept_neg_idx
        ]

    return out


def _load_teacher_scores(parquet_path: str) -> dict:
    """Load teacher_scores.parquet → dict keyed by (pos_tid, sorted_neg_tids).

    Byte-for-byte mirror of ``scripts/train_bi_encoder.py:_load_teacher_scores``
    so the Stage B builder, the Stage A trainer, and the precompute script all
    key the parquet identically (no index drift across scripts).

    Value is ``{pos_score: float, neg_tid_to_score: dict[str, float]}``.
    """
    import pandas as _pd

    df = _pd.read_parquet(parquet_path)
    out: dict = {}
    for _, prow in df.iterrows():
        pos_tid = str(prow["pos_tid"])
        neg_tids = [str(t) for t in list(prow["neg_tids"])]
        neg_scores = [float(s) for s in list(prow["neg_scores"])]
        key = (pos_tid, tuple(sorted(neg_tids)))
        out[key] = {
            "pos_score": float(prow["pos_score"]),
            "neg_tid_to_score": dict(zip(neg_tids, neg_scores)),
        }
    print(f"[ce-build] loaded {len(out)} teacher rows from {parquet_path}", file=sys.stderr)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-conv-hf", default="talkpl-ai/TalkPlayData-Challenge-Dataset",
                        help="HF conversation dataset to walk (train split).")
    parser.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--output", required=True)
    # --- Stage A multi-modal retriever (candidate generation) ---
    parser.add_argument("--stage-a-hub-repo", required=True,
                        help="Path OR Hub repo of the trained multi-modal Stage A "
                             "MultiModalBiEncoder (DENSE_MULTIMODAL_LOCAL model_dir). "
                             "Consistent with nb 70/71 CONFIG STAGE_A_HUB_REPO.")
    parser.add_argument("--multimodal-artifacts", required=True,
                        help="Path to the precompute cache dir (built by "
                             "scripts/precompute_multimodal_artifacts.py): tag_vocab.json, "
                             "release_year_lookup.json, user_cf*, track_clap/cf*. Used "
                             "for (a) DENSE_MULTIMODAL_LOCAL query-side user injection AND "
                             "(b) tag_ids_* / release_year_* emission. Mirrors nb 70/71 "
                             "MULTIMODAL_ARTIFACTS.")
    parser.add_argument("--backbone-override", default="",
                        help="Optional base-model override for the multi-modal encoder "
                             "(passed to MultiModalBiEncoder.from_pretrained).")
    parser.add_argument("--embed-label", default="default",
                        help="Subdir under dense_local/<safe>/ holding the catalog "
                             "embeddings (matches embed_catalog_multimodal.py).")
    parser.add_argument("--cache-dir", default="./cache",
                        help="Parent of dense_local/ where DENSE_MULTIMODAL_LOCAL reads "
                             "the precomputed catalog pickle.")
    parser.add_argument("--query-max-len", type=int, default=384)
    parser.add_argument("--pool-size", type=int, default=100,
                        help="Stage A top-K candidate pool per query (the plan feeds "
                             "Stage B top-100). Positives are promoted out of this pool; "
                             "the rest are negatives.")
    # --- Multi-positive promotion (Phase 0 teacher scores) ---
    parser.add_argument("--teacher-scores-path", default="",
                        help="Path to teacher_scores.parquet (Phase 0 / "
                             "scripts/precompute_reranker_scores.py). When set, candidates "
                             "with teacher_score >= multipositive_threshold * gold_score "
                             "are promoted to positives. Mirrors nb 70/71 TEACHER_SCORES_PATH.")
    parser.add_argument("--multipositive-threshold", type=float, default=0.85,
                        help="Candidate -> positive iff teacher_score >= this * gold_score. "
                             "Default 0.85 (RocketQAv2 §3.3); mirrors the Stage A builder.")
    # --- Multi-modal emission ---
    parser.add_argument("--max-tags", type=int, default=20,
                        help="Truncate per-track tag_ids to this length (matches the "
                             "model's tag-token capacity).")
    parser.add_argument("--history-corpus-types", type=str,
                        default="track_name,artist_name,album_name",
                        help="Catalog fields used to format pos/neg track text + expand "
                             "[HISTORY] music turns (id_to_metadata mirror). Default matches "
                             "production config 021 + the Stage A builder.")
    parser.add_argument("--query-mode", type=str, default="bge_m3_structured",
                        choices=["raw", "last_user", "last_user_with_goal", "bge_m3_structured"],
                        help="Query format for retrieval AND emitted training queries. "
                             "MUST match the deployment YAML query_preprocessing_mode.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke cap; 0 = all")
    args = parser.parse_args()

    from datasets import load_dataset

    # --- 1. Multi-modal emission artifacts (tag_vocab + release_year_lookup) ---
    tag_vocab, release_year_lookup = _load_multimodal_artifacts(args.multimodal_artifacts)

    # --- 2. Track metadata → id_to_metadata-aligned text map (Stage A parity) ---
    print(f"[ce-build] loading track metadata from {args.track_meta_hf}", file=sys.stderr)
    track_meta = load_dataset(args.track_meta_hf, split="all_tracks")
    metadata_dict: dict = {}
    track_ids: list[str] = []
    history_corpus_types = [c.strip() for c in args.history_corpus_types.split(",") if c.strip()]
    for trow in tqdm(track_meta, desc="index tracks"):
        tid = trow["track_id"]
        metadata_dict[tid] = dict(trow)
        track_ids.append(tid)
    if len(set(track_ids)) != len(track_ids):
        dupes = [tid for tid, c in Counter(track_ids).items() if c > 1]
        raise RuntimeError(f"track catalog has duplicates (first 5: {dupes[:5]})")
    track_text_map = {
        tid: _format_history_music_turn(tid, metadata_dict, history_corpus_types)
        for tid in track_ids
    }

    # --- 3. Train conversations → per-music-turn rows (history expanded) ---
    print(f"[ce-build] walking {args.train_conv_hf} (train split)", file=sys.stderr)
    conv_ds = load_dataset(args.train_conv_hf, split="train")
    train_rows = _iter_conversation_turns(
        conv_ds, metadata_dict=metadata_dict, corpus_types=history_corpus_types,
    )
    if args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]
    print(f"[ce-build] {len(train_rows)} per-music-turn rows", file=sys.stderr)

    # --- 4. Optional teacher scores for multi-positive promotion ---
    teacher_scores: Optional[dict] = None
    if args.teacher_scores_path:
        teacher_scores = _load_teacher_scores(args.teacher_scores_path)

    # --- 5. Trained multi-modal Stage A retriever (candidate generation) ---
    # DENSE_MULTIMODAL_LOCAL loads the fused 768-d catalog pickle + the
    # multi-modal model, and injects per-user CF into the query side. This is
    # the SAME retriever production wRRF uses — Stage B's candidate pool is
    # therefore distribution-matched to deployment, not a text-only proxy.
    from mcrs.retrieval_modules.dense_multimodal_local import DENSE_MULTIMODAL_LOCAL

    print(f"[ce-build] loading Stage A retriever {args.stage_a_hub_repo}", file=sys.stderr)
    retriever = DENSE_MULTIMODAL_LOCAL(
        dataset_name=args.track_meta_hf,
        split_types=[args.history_corpus_types],
        corpus_types=history_corpus_types,
        cache_dir=args.cache_dir,
        model_dir=args.stage_a_hub_repo,
        embed_label=args.embed_label,
        backbone_override=args.backbone_override or None,
        multimodal_artifacts=args.multimodal_artifacts,
        query_max_len=args.query_max_len,
    )

    # --- 6. Retrieve top-K per query, split pos/neg, write JSONL ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_no_gold = 0
    n_skipped_too_few_negs = 0
    with open(args.output, "w") as f_out:
        for i in tqdm(range(0, len(train_rows), args.batch_size), desc="retrieve"):
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
            batch_user_ids = [r.get("user_id") for r in batch_rows]
            # Stage A top-K candidates per (query, user). User-aware query side.
            batch_cands = retriever.batch_text_to_item_retrieval(
                batch_queries, topk=args.pool_size, user_ids=batch_user_ids,
            )
            for j, row in enumerate(batch_rows):
                gold = row["track_id"]
                if gold not in track_text_map:
                    n_skipped_no_gold += 1
                    continue
                cand_tids = list(batch_cands[j])
                teacher = None
                if teacher_scores is not None:
                    # v1: this key (gold + Stage A top-K negs) will not match the
                    # parquet (keyed on the bi-encoder's mined negs), so teacher
                    # stays None and rows fall back to single-positive. See the
                    # module-level v1 NOTE. Wired for the v2 re-score regardless.
                    key = (str(gold), tuple(sorted(str(t) for t in cand_tids if t != gold)))
                    teacher = teacher_scores.get(key)
                selection = select_positives_and_negatives(
                    gold_tid=gold,
                    candidate_tids=cand_tids,
                    teacher=teacher,
                    multipositive_threshold=args.multipositive_threshold,
                )
                triple = build_mm_ce_triple(
                    query=batch_queries[j],
                    selection=selection,
                    track_text_map=track_text_map,
                    row=row,
                    metadata_dict=metadata_dict,
                    tag_vocab=tag_vocab,
                    release_year_lookup=release_year_lookup,
                    max_tags=args.max_tags,
                )
                if len(triple["neg"]) < 2:
                    n_skipped_too_few_negs += 1
                    continue
                f_out.write(json.dumps(triple) + "\n")
                n_written += 1

    print(
        f"[ce-build] DONE → {args.output} (wrote {n_written}; "
        f"skipped_no_gold={n_skipped_no_gold}, "
        f"skipped_too_few_negs={n_skipped_too_few_negs})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
