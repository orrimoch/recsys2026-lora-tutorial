"""Compare two blindset prediction.json runs by per-turn nDCG@20 + paired-bootstrap CI.

Bridges the gap between run_inference_blindset.py's prediction.json output and
the paired_bootstrap_ci utility (which lives in compare_diagnostic_runs.py).

Usage:
    python scripts/compare_blind_predictions.py \\
        --pred_a music-crs-baselines/exp/inference/dev/170-...json \\
        --pred_b music-crs-baselines/exp/inference/dev/132-...json \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --split test \\
        --label_a 'wRRF+SID (170)' \\
        --label_b 'wRRF current (132)' \\
        --output experiments/diagnostic_runs/w4_gate.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_diagnostic_runs import paired_bootstrap_ci


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_a", type=Path, required=True,
                   help="prediction.json from run A (e.g. config 170)")
    p.add_argument("--pred_b", type=Path, required=True,
                   help="prediction.json from run B (e.g. config 132)")
    p.add_argument("--dataset", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Dataset",
                   help="HF dataset for gold lookups")
    p.add_argument("--split", type=str, default="test",
                   help="Dataset split holding gold tracks (dev=test split)")
    p.add_argument("--label_a", type=str, default="A")
    p.add_argument("--label_b", type=str, default="B")
    p.add_argument("--k", type=int, default=20, help="nDCG@k")
    p.add_argument("--n_resamples", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the JSON comparison")
    return p.parse_args()


def ndcg_at_k(retrieved: list[str], gold: str, k: int) -> float:
    """nDCG@k with binary relevance (single gold)."""
    for rank, tid in enumerate(retrieved[:k]):
        if tid == gold:
            return 1.0 / math.log2(rank + 2)
    return 0.0


def _load_pred(path: Path) -> dict[str, list[str]]:
    """Return {session_id_or_index: predicted_track_list} from a blindset prediction.json.

    Tolerant of two shapes seen in practice:
      list[dict]: each dict has 'session_id' or 'id' + 'predicted_items' or 'tracks'
      dict[str, dict]: keyed by session id, value has 'predicted_items' or 'tracks'
    """
    raw = json.loads(path.read_text())
    out: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for sid, rec in raw.items():
            tracks = rec.get("predicted_items") or rec.get("tracks") or rec.get("track_ids") or []
            out[str(sid)] = list(tracks)
    elif isinstance(raw, list):
        for i, rec in enumerate(raw):
            sid = rec.get("session_id") or rec.get("id") or str(i)
            tracks = rec.get("predicted_items") or rec.get("tracks") or rec.get("track_ids") or []
            out[str(sid)] = list(tracks)
    else:
        raise ValueError(f"Unexpected prediction.json shape: {type(raw)}")
    return out


def _load_gold(dataset: str, split: str) -> dict[str, str]:
    """Return {session_id: gold_track_id} from the dataset's held-out split.

    Falls back to indexing the rows if session_id absent.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    out: dict[str, str] = {}
    for i, row in enumerate(ds):
        sid = row.get("session_id") or row.get("id") or str(i)
        # Gold track is whatever the dataset's evaluation schema designates.
        # Try common field names; the first non-None wins.
        gold = (
            row.get("target_track_id")
            or row.get("gold_track_id")
            or row.get("target")
            or row.get("track_id")
        )
        if gold:
            out[str(sid)] = str(gold)
    return out


def main():
    args = parse_args()
    preds_a = _load_pred(args.pred_a)
    preds_b = _load_pred(args.pred_b)
    gold = _load_gold(args.dataset, args.split)

    # Compute per-session nDCG for both; pair by session id.
    common = sorted(set(preds_a) & set(preds_b) & set(gold))
    if not common:
        raise SystemExit(
            f"No overlapping session ids between pred_a ({len(preds_a)}), "
            f"pred_b ({len(preds_b)}), and gold ({len(gold)}). "
            "Check that --dataset / --split point at the same data."
        )

    diffs: list[float] = []
    per_session = []
    for sid in common:
        a = ndcg_at_k(preds_a[sid], gold[sid], args.k)
        b = ndcg_at_k(preds_b[sid], gold[sid], args.k)
        diffs.append(a - b)
        per_session.append({"session_id": sid, "ndcg_a": a, "ndcg_b": b, "delta": a - b})

    mean_a = sum(p["ndcg_a"] for p in per_session) / len(per_session)
    mean_b = sum(p["ndcg_b"] for p in per_session) / len(per_session)
    mean_delta, ci_lo, ci_hi = paired_bootstrap_ci(
        diffs, n_resamples=args.n_resamples, alpha=args.alpha, seed=42,
    )

    out = {
        "n_paired": len(common),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "mean_ndcg_a": mean_a,
        "mean_ndcg_b": mean_b,
        "delta_mean": mean_delta,
        "paired_bootstrap_ci": {"lo": ci_lo, "hi": ci_hi, "alpha": args.alpha},
        "per_session": per_session[:50],  # truncate for readability
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
