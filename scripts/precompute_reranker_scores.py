"""Precompute teacher reranker scores for every (query, doc) pair in a
bi-encoder triples JSONL.

Used by:
  - Multi-positive label promotion (any candidate with
    ``score >= multipos_threshold * gold_score`` becomes an additional
    positive).
  - MarginMSE distillation loss
    (``MSE(student_margin, teacher_margin)`` where margin = ``s_pos - s_neg``).

Input: a triples JSONL file produced by
``scripts/build_bi_encoder_training_data.py``. Each row has ``pos_tid`` +
``neg_tids`` (+ ``query`` text).

Output: a Parquet file with one row per input row. Schema:

    row_idx     : int64               (line index in input JSONL)
    pos_tid     : string
    pos_score   : float32             (teacher's score for the gold)
    neg_tids    : list<string>
    neg_scores  : list<float32>

Companion JSONL (``{output}.partial``) is written line-by-line during the
run for resume support. The final parquet is built once all rows are
scored. Re-running on the same output: skips rows already in the partial,
then resumes.

Default teacher: ``BAAI/bge-reranker-v2-m3`` (~568M, cross-encoder).
Loads with bf16 on CUDA/MPS, fp32 on CPU. Same loader pattern as
``mcrs/rerankers/bge_reranker.py:31-54``.

Usage:
  python scripts/precompute_reranker_scores.py \\
    --triples experiments/cache/retrieval_v2/triples_bge_base_en_music_v1.jsonl \\
    --output experiments/cache/multimodal/teacher_scores.parquet

Estimated wall-clock: ~2-3 hr on Blackwell bf16 for 115K rows × 16 pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_LEN = 256  # matches mcrs/rerankers/bge_reranker.py:109


def _load_reranker(model_name: str):
    """Mirror mcrs/rerankers/bge_reranker.py:31-54 loader pattern."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    print(f"[teacher] loaded {model_name} on {device} dtype={dtype}", flush=True)
    return tokenizer, model, device


def _iter_triples(path: Path) -> Iterator[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f)


def _score_pairs(
    tokenizer, model, device, pairs: list[tuple[str, str]], max_len: int
) -> np.ndarray:
    """Mirror mcrs/rerankers/bge_reranker.py:_score_batch.

    pairs: list of (query, doc) tuples. Returns 1-D float32 array of
    relevance scores, one per pair.
    """
    import torch

    queries = [p[0] for p in pairs]
    docs = [p[1] for p in pairs]
    inputs = tokenizer(
        queries, docs,
        padding=True, truncation=True, max_length=max_len, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.view(-1)
    return logits.float().cpu().numpy()


def _build_pairs_for_row(row: dict) -> tuple[str, str, list[str], list[str], list[str]]:
    """Returns (query, pos_text, pos_tid, neg_texts, neg_tids)."""
    query = row["query"]
    pos_text = row["pos"][0]
    pos_tid = row["pos_tid"]
    neg_texts = list(row["neg"])
    neg_tids = list(row["neg_tids"])
    if len(neg_texts) != len(neg_tids):
        raise ValueError(
            f"row has mismatched neg/neg_tids lengths: "
            f"{len(neg_texts)} vs {len(neg_tids)}"
        )
    return query, pos_text, pos_tid, neg_texts, neg_tids


def _finalize_parquet(partial_path: Path, output_path: Path) -> None:
    """Convert the .partial JSONL into the final Parquet output."""
    import pandas as pd

    print(f"[teacher] finalizing {partial_path} -> {output_path}", flush=True)
    rows = []
    with open(partial_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("partial file is empty — nothing to finalize")

    df = pd.DataFrame(rows)
    # Cast scores to float32 for compactness.
    df["pos_score"] = df["pos_score"].astype("float32")
    df["neg_scores"] = df["neg_scores"].apply(lambda xs: np.asarray(xs, dtype=np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="zstd")
    print(
        f"[teacher] wrote {output_path} rows={len(df)} "
        f"size_mb={output_path.stat().st_size / 1e6:.1f}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--triples", required=True, type=Path,
        help="Input triples JSONL (output of build_bi_encoder_training_data.py).",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output parquet path. A {output}.partial JSONL is written incrementally for resume.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Teacher reranker (default {DEFAULT_MODEL}).")
    p.add_argument(
        "--rows-per-batch", type=int, default=4,
        help="How many input rows to batch into one model call. Each row contributes "
             "(1 pos + N negs) pairs to the batch.",
    )
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--max-rows", type=int, default=0, help="0 = no cap; useful for smoke tests.")
    p.add_argument(
        "--finalize-only", action="store_true",
        help="Skip scoring; just convert the existing .partial JSONL into the final parquet.",
    )
    args = p.parse_args()

    if not args.triples.exists():
        raise FileNotFoundError(f"triples not found: {args.triples}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")

    if args.finalize_only:
        _finalize_parquet(partial_path, args.output)
        return

    # Resume: skip rows already in the partial.
    completed = _count_lines(partial_path)
    total = _count_lines(args.triples)
    if args.max_rows > 0:
        total = min(total, args.max_rows)
    if completed >= total:
        print(f"[teacher] partial already has {completed} rows; finalizing only", flush=True)
        _finalize_parquet(partial_path, args.output)
        return
    print(
        f"[teacher] resume — {completed}/{total} rows already scored. "
        f"Loading model...",
        flush=True,
    )

    tokenizer, model, device = _load_reranker(args.model)

    # Stream input, skip completed, score remaining, append to partial.
    from tqdm import tqdm

    pbar = tqdm(total=total, initial=completed, desc="[teacher]")
    with open(partial_path, "a") as out_f:
        batch_rows: list[dict] = []

        def _flush(rows_chunk: list[dict]) -> None:
            if not rows_chunk:
                return
            # Build a flat pair list, remember per-row offsets so we can
            # split scores back per-row.
            all_pairs: list[tuple[str, str]] = []
            row_pos_offsets: list[int] = []  # start index of each row's pos in the flat list
            row_neg_ranges: list[tuple[int, int]] = []  # (start, end_exclusive) of each row's negs
            for r in rows_chunk:
                q, pos_text, _, neg_texts, _ = _build_pairs_for_row(r)
                pos_idx = len(all_pairs)
                all_pairs.append((q, pos_text))
                row_pos_offsets.append(pos_idx)
                neg_start = len(all_pairs)
                for nt in neg_texts:
                    all_pairs.append((q, nt))
                row_neg_ranges.append((neg_start, len(all_pairs)))

            scores = _score_pairs(tokenizer, model, device, all_pairs, args.max_len)

            for row_idx_in_chunk, r in enumerate(rows_chunk):
                _, _, pos_tid, _, neg_tids = _build_pairs_for_row(r)
                pos_score = float(scores[row_pos_offsets[row_idx_in_chunk]])
                ns, ne = row_neg_ranges[row_idx_in_chunk]
                neg_scores = scores[ns:ne].tolist()
                out_record = {
                    "row_idx": r["__row_idx__"],
                    "pos_tid": pos_tid,
                    "pos_score": pos_score,
                    "neg_tids": neg_tids,
                    "neg_scores": neg_scores,
                }
                out_f.write(json.dumps(out_record) + "\n")
            out_f.flush()
            pbar.update(len(rows_chunk))

        for i, row in enumerate(_iter_triples(args.triples)):
            if i < completed:
                continue
            if args.max_rows > 0 and i >= args.max_rows:
                break
            row["__row_idx__"] = i
            batch_rows.append(row)
            if len(batch_rows) >= args.rows_per_batch:
                _flush(batch_rows)
                batch_rows = []

        # Final flush
        _flush(batch_rows)
    pbar.close()

    _finalize_parquet(partial_path, args.output)


if __name__ == "__main__":
    main()
