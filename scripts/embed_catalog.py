"""Compute dense embeddings for the 47K-track catalog with any HF embedding model.

Used for Phase 1+ retriever swaps (BGE-M3, Qwen3-Embedding-4B, ...) where the
challenge-provided embeddings aren't available for the chosen model. One-time
cost (~30–60 min on Blackwell-95GB for 47K tracks); subsequent runs reuse the
cached pickle.

Usage:
    python scripts/embed_catalog.py \\
        --model BAAI/bge-m3 \\
        --label bge-m3-metadata \\
        --fields track_name artist_name album_name tag_list release_date \\
        --batch-size 64

Output:
    music-crs-baselines/experiments/cache/dense_local/<safe_model>/<label>/track_embeddings.pkl
    Format: {"track_ids": list[str], "track_mat": np.ndarray (N, dim, float32, L2-normalized)}
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"


def build_doc_text(metadata: dict, fields: list[str]) -> str:
    """Build a single doc-text string from a catalog row by concatenating named fields.

    Each field renders as 'field: value' (joined by ', ' if value is a list).
    Empty lists and absent fields are silently skipped to avoid 'field: ' artefacts.
    """
    parts: list[str] = []
    for field in fields:
        if field not in metadata:
            continue
        val = metadata[field]
        if isinstance(val, list):
            if not val:
                continue
            rendered = ", ".join(str(v) for v in val)
        else:
            if val is None or val == "":
                continue
            rendered = str(val)
        parts.append(f"{field}: {rendered}")
    return " | ".join(parts)


def main():
    import argparse
    import os
    import pickle
    import sys

    import numpy as np

    parser = argparse.ArgumentParser(description="Compute catalog embeddings with a HF model.")
    parser.add_argument("--model", required=True, help="HF model name (e.g., BAAI/bge-m3, Qwen/Qwen3-Embedding-4B).")
    parser.add_argument("--label", required=True, help="Short label identifying the doc-text recipe (e.g., bge-m3-metadata).")
    parser.add_argument(
        "--fields", nargs="+",
        default=["track_name", "artist_name", "album_name", "tag_list", "release_date"],
        help="Catalog metadata fields to concatenate into doc text.",
    )
    parser.add_argument("--catalog-dataset", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--catalog-split", default="all_tracks")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-root", default=str(BASELINES_DIR / "experiments" / "cache" / "dense_local"))
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Smoke-test cap; encode only the first N tracks (default: full catalog).")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"[embed_catalog] loading {args.catalog_dataset}[{args.catalog_split}]", file=sys.stderr)
    ds = load_dataset(args.catalog_dataset, split=args.catalog_split)
    if args.max_tracks is not None:
        ds = ds.select(range(min(args.max_tracks, len(ds))))

    track_ids: list[str] = []
    doc_texts: list[str] = []
    for row in ds:
        track_ids.append(row["track_id"])
        doc_texts.append(build_doc_text(row, args.fields))
    print(f"[embed_catalog] built {len(doc_texts)} doc texts", file=sys.stderr)
    print(f"[embed_catalog] sample doc[0]: {doc_texts[0][:200]}", file=sys.stderr)

    print(f"[embed_catalog] loading sentence-transformers model: {args.model}", file=sys.stderr)
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = SentenceTransformer(args.model, device=device)
    print(f"[embed_catalog] device={device}", file=sys.stderr)

    embeddings = model.encode(
        doc_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"[embed_catalog] embeddings shape={embeddings.shape} dtype={embeddings.dtype}", file=sys.stderr)

    safe_model = args.model.replace("/", "_")
    out_dir = Path(args.cache_root) / safe_model / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "track_embeddings.pkl"
    with out_path.open("wb") as f:
        pickle.dump({"track_ids": track_ids, "track_mat": embeddings}, f)
    print(f"[embed_catalog] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
