"""Generic offline eval harness for the v2 retrieval pipeline.

Computes nDCG@K, recall@K, MRR per turn against a JSONL of predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def compute_ndcg_at_k(retrieved: list[str], gold: str, k: int) -> float:
    """nDCG@k for a single-relevant-item case: 1/log2(rank+1) if gold in top-k, else 0."""
    top_k = retrieved[:k]
    if gold not in top_k:
        return 0.0
    rank = top_k.index(gold) + 1  # 1-indexed
    return 1.0 / math.log2(rank + 1)


def compute_recall_at_k(retrieved: list[str], gold: str, k: int) -> float:
    """Recall@K for single-relevant-item: 1 if gold in top-k else 0."""
    return 1.0 if gold in retrieved[:k] else 0.0


def compute_mrr(retrieved: list[str], gold: str) -> float:
    """MRR for one query: 1/rank or 0."""
    try:
        rank = retrieved.index(gold) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-jsonl", required=True,
                        help="JSONL with rows {query_id, gold, retrieved: [...]}.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.predictions_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))

    ndcgs = [compute_ndcg_at_k(r["retrieved"], r["gold"], 20) for r in rows]
    recalls_20 = [compute_recall_at_k(r["retrieved"], r["gold"], 20) for r in rows]
    recalls_100 = [compute_recall_at_k(r["retrieved"], r["gold"], 100) for r in rows]
    mrrs = [compute_mrr(r["retrieved"], r["gold"]) for r in rows]

    metrics = {
        "n_queries": len(rows),
        "mean_ndcg_at_20": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "mean_recall_at_20": sum(recalls_20) / len(recalls_20) if recalls_20 else 0.0,
        "mean_recall_at_100": sum(recalls_100) / len(recalls_100) if recalls_100 else 0.0,
        "mean_mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
    }
    Path(args.output_json).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
