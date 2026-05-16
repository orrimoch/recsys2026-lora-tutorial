"""Build the SID training data parquet by combining raw conversations,
metadata-as-query, and (optional) doc2query sources.

Reads:
  - W1 SID lookup: experiments/cache/sid/track_to_sid.parquet
  - HF: talkpl-ai/TalkPlayData-Challenge-Dataset (split=train)
  - HF: talkpl-ai/TalkPlayData-Challenge-Track-Metadata (split=all_tracks)
  - (Optional) experiments/cache/doc2query/<safe_model>/queries.parquet

Writes:
  - experiments/cache/sid_training/train.parquet
  - experiments/cache/sid_training/val.parquet
  - experiments/cache/sid_training/summary.json

Usage:
    python scripts/build_sid_training_data.py
    python scripts/build_sid_training_data.py --no-doc2query   # skip doc2query
    python scripts/build_sid_training_data.py --max-sessions 100   # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
from datasets import load_dataset

from mcrs.sid.training_data import (
    build_doc2query_pairs,
    build_metadata_as_query_pairs,
    build_raw_conversation_pairs,
    stratified_split,
)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build SID training data.")
    parser.add_argument("--no-doc2query", action="store_true",
                        help="Skip doc2query source even if its parquet exists.")
    parser.add_argument("--n-turns-window", type=int, default=3,
                        help="Chat history window for raw conversation pairs.")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Smoke: cap on train sessions read.")
    parser.add_argument(
        "--doc2query-model", default="Qwen_Qwen2.5-1.5B-Instruct",
        help="Subdir under experiments/cache/doc2query/ to load synthetic queries from.",
    )
    args = parser.parse_args()

    # 1. Load W1 SID lookup
    sid_path = REPO_ROOT / "experiments" / "cache" / "sid" / "track_to_sid.parquet"
    if not sid_path.exists():
        raise SystemExit(f"Missing {sid_path}. Run W1 (notebook 60) first.")
    sid_df = pd.read_parquet(sid_path)
    track_to_sid: dict[str, tuple[int, int, int]] = {
        row.track_id: (int(row.code_1), int(row.code_2), int(row.code_3))
        for row in sid_df.itertuples()
    }
    sid_hash = sha256_of_file(sid_path)
    print(f"[build_sid_training_data] loaded W1 SIDs: {len(track_to_sid)} tracks, "
          f"sha256={sid_hash[:12]}...", file=sys.stderr)

    # 2. Build raw conversation pairs
    print("[build_sid_training_data] loading TalkPlayData-Challenge-Dataset[train]...",
          file=sys.stderr)
    conv_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset", split="train",
    )
    if args.max_sessions is not None:
        conv_ds = conv_ds.select(range(min(args.max_sessions, len(conv_ds))))
    sessions = [dict(row) for row in conv_ds]
    raw_pairs = build_raw_conversation_pairs(
        sessions, track_to_sid, n_turns_window=args.n_turns_window,
    )
    print(f"[build_sid_training_data] raw conversation pairs: {len(raw_pairs)}",
          file=sys.stderr)

    # 3. Build metadata-as-query pairs
    print("[build_sid_training_data] loading Track-Metadata[all_tracks]...",
          file=sys.stderr)
    meta_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks",
    )
    track_metadata = [dict(row) for row in meta_ds]
    meta_pairs = build_metadata_as_query_pairs(track_metadata, track_to_sid)
    print(f"[build_sid_training_data] metadata-as-query pairs: {len(meta_pairs)}",
          file=sys.stderr)

    # 4. (Optional) Doc2query pairs
    doc2query_pairs: list = []
    doc2query_path = (
        REPO_ROOT / "experiments" / "cache" / "doc2query"
        / args.doc2query_model / "queries.parquet"
    )
    if args.no_doc2query:
        print("[build_sid_training_data] --no-doc2query: skipping doc2query source",
              file=sys.stderr)
    elif doc2query_path.exists():
        print(f"[build_sid_training_data] loading doc2query from {doc2query_path}",
              file=sys.stderr)
        d2q_df = pd.read_parquet(doc2query_path)
        d2q_rows = [dict(row) for row in d2q_df.to_dict(orient="records")]
        doc2query_pairs = build_doc2query_pairs(d2q_rows, track_to_sid)
        print(f"[build_sid_training_data] doc2query pairs: {len(doc2query_pairs)}",
              file=sys.stderr)
    else:
        print(f"[build_sid_training_data] WARNING: doc2query parquet not found at "
              f"{doc2query_path}. Shipping without doc2query (run notebook 54 to add).",
              file=sys.stderr)

    # 5. Combine + stratified split
    all_pairs = raw_pairs + meta_pairs + doc2query_pairs
    df = pd.DataFrame(all_pairs)
    print(f"[build_sid_training_data] total pairs: {len(df)} "
          f"(raw={len(raw_pairs)}, metadata={len(meta_pairs)}, doc2query={len(doc2query_pairs)})",
          file=sys.stderr)

    # Group-level split for the 'raw' source so all turns from one session land in
    # the same partition (avoids train↔val leakage via overlapping chat history).
    # 'metadata' and 'doc2query' rows are independent → row-level split is fine.
    train, val = stratified_split(
        df, val_frac=args.val_frac, seed=args.seed,
        group_by={"raw": "session_id"},
    )
    n_val_sessions = val[val["source"] == "raw"]["session_id"].nunique() if "session_id" in val.columns else 0
    n_train_sessions = train[train["source"] == "raw"]["session_id"].nunique() if "session_id" in train.columns else 0
    print(f"[build_sid_training_data] raw split is session-level: "
          f"{n_train_sessions} train sessions / {n_val_sessions} val sessions "
          f"(disjoint by construction)", file=sys.stderr)

    # 6. Write outputs
    out_dir = REPO_ROOT / "experiments" / "cache" / "sid_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    print(f"[build_sid_training_data] wrote {train_path} ({len(train)} rows)",
          file=sys.stderr)
    print(f"[build_sid_training_data] wrote {val_path} ({len(val)} rows)",
          file=sys.stderr)

    # 7. Summary
    summary = {
        "w1_sid_sha256": sid_hash,
        "n_tracks_with_sid": len(track_to_sid),
        "sources": {
            "raw": len(raw_pairs),
            "metadata": len(meta_pairs),
            "doc2query": len(doc2query_pairs),
        },
        "total": len(df),
        "train_rows": len(train),
        "val_rows": len(val),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_turns_window": args.n_turns_window,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[build_sid_training_data] wrote {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
