"""Build a PyLate PLAID index over the full 47k catalog with music-colbert-v1
(Stage B / W2.c of the ColBERT plan).

This is the artifact that tests the plan's RECALL thesis. The pool-reranker
(ColbertRetriever) can only reorder the union's own top-100 — it can never surface
a gold the union missed. A full-catalog PLAID index CAN: it retrieves over all 47k
tracks, so `ColbertIndexRetriever` can add new-artist/wall golds the union's
BM25+dense+session channels never reached. Doc text is UUID-stripped (RCA #4),
matching the training docs.

Usage (Colab GPU):
  python scripts/build_colbert_index.py \
    --model-dir experiments/cache/retrieval_v2/colbert/music-colbert-v1 \
    --index-folder experiments/cache/retrieval_v2/colbert/plaid \
    --index-name colbert-music-v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))


def existing_artifact_blocks(path: str, force: bool) -> bool:
    """True if `path` already exists and --force was NOT passed, i.e. SKIP the
    build to avoid clobbering a live index in place. PLAID's override=True silently
    overwrote the 0.50 index during the 2026-06-15 retrain accident. Pure -> unit-tested."""
    import os
    return bool(path) and os.path.exists(path) and not force


def main():  # pragma: no cover
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="Fine-tuned ColBERT dir (music-colbert-v1) or a hub name")
    ap.add_argument("--index-folder", required=True)
    ap.add_argument("--index-name", default="colbert-music-v1")
    ap.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--d-len", type=int, default=96,
                    help="document token budget. 96 for raw metadata; raise (e.g. 128) for "
                         "--enrich-tags so curated tags survive. MUST equal train_colbert --d-len.")
    ap.add_argument("--enrich-tags", action="store_true",
                    help="EXP-217: append curated genre/mood tags (catalog-freq filtered, "
                         "top-k) to each doc. MUST match build_colbert_train_data --enrich-tags "
                         "+ same --tag-min-freq/--tag-top-k (doc-side train/index parity).")
    ap.add_argument("--tag-min-freq", type=int, default=50)
    ap.add_argument("--tag-top-k", type=int, default=15)
    ap.add_argument("--force", action="store_true",
                    help="overwrite the index if it already exists. WITHOUT this, an "
                         "existing index SKIPS the build (guards against the accidental "
                         "in-place rebuild that clobbered the 0.50 PLAID index).")
    args = ap.parse_args()

    index_path = os.path.join(args.index_folder, args.index_name)
    if existing_artifact_blocks(index_path, args.force):
        print(f"[colbert-index] {index_path} already exists — SKIPPING to avoid "
              f"overwriting it in place. Pass --force to rebuild.", file=sys.stderr)
        return

    from pylate import indexes, models
    from mcrs.db_item.music_catalog import MusicCatalogDB
    from mcrs.retrieval_modules.colbert_late import (
        DEFAULT_Q_LEN, strip_track_id_prefix, build_tag_vocab, colbert_doc_text,
    )

    corpus = ["track_name", "artist_name", "album_name"]
    item_db = MusicCatalogDB(args.track_meta_hf, ["all_tracks"], corpus)
    tids = list(item_db.metadata_dict.keys())
    if args.enrich_tags:
        # EXP-217: vocab built over the WHOLE catalog (stable freq; identical to the
        # train builder when --tag-min-freq matches) -> byte-identical enriched docs.
        vocab = build_tag_vocab(
            ((item_db.metadata_dict.get(t) or {}).get("tag_list") for t in tids),
            min_freq=args.tag_min_freq)
        docs = [colbert_doc_text(t, item_db.id_to_metadata, item_db.metadata_dict,
                                 vocab, args.tag_top_k) for t in tids]
        print(f"[colbert-index] {len(tids)} docs TAG-ENRICHED "
              f"(vocab={len(vocab)} @min_freq={args.tag_min_freq}, top_k={args.tag_top_k})",
              file=sys.stderr)
    else:
        docs = [strip_track_id_prefix(item_db.id_to_metadata(t)) for t in tids]
        print(f"[colbert-index] {len(tids)} catalog docs (UUID-stripped)", file=sys.stderr)

    model = models.ColBERT(
        model_name_or_path=args.model_dir,
        query_length=DEFAULT_Q_LEN,
        document_length=args.d_len,
    )
    doc_embeddings = model.encode(
        docs, batch_size=args.batch_size, is_query=False, show_progress_bar=True
    )

    index = indexes.PLAID(
        # override=args.force (defense-in-depth): the skip-guard above already
        # returns unless --force, so reaching here means new index OR --force.
        # Tying override to --force ensures even a path-layout mismatch in the
        # guard can't silently clobber a live index in place.
        index_folder=args.index_folder, index_name=args.index_name, override=args.force
    )
    index.add_documents(documents_ids=tids, documents_embeddings=doc_embeddings)
    print(f"[colbert-index] DONE -> {args.index_folder}/{args.index_name}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
