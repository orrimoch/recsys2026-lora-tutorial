"""Compare two blindset prediction.json runs by per-turn nDCG@20 + paired-bootstrap CI.

Bridges the gap between run_inference_blindset.py's prediction.json output and
the paired_bootstrap_ci utility (which lives in compare_diagnostic_runs.py).

Usage:
    python scripts/compare_blind_predictions.py \\
        --pred_a music-crs-baselines/exp/inference/dev/170-...json \\
        --pred_b music-crs-baselines/exp/inference/dev/132-...json \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --gold_split test \\
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
    p.add_argument("--gold_split", type=str, default="test",
                   help="Dataset split holding gold tracks. For dev evaluation use 'test' "
                        "(the TalkPlayData dev set is the 'test' split per HF convention).")
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
    """Return {composite_key: predicted_track_list} from a prediction.json.

    Composite key = f"{session_id}__{turn_number}" so multi-turn sessions don't
    collapse. run_inference_blindset.py writes one record per (session, turn)
    with keys: session_id, user_id, turn_number, predicted_track_ids.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list of records, got {type(raw).__name__}")
    out: dict[str, list[str]] = {}
    for rec in raw:
        sid = rec.get("session_id")
        tn = rec.get("turn_number")
        if sid is None or tn is None:
            continue
        tracks = (
            rec.get("predicted_track_ids")
            or rec.get("predicted_items")
            or rec.get("tracks")
            or rec.get("track_ids")
            or []
        )
        out[f"{sid}__{tn}"] = list(tracks)
    return out


def _load_gold(dataset: str, split: str) -> dict[str, str]:
    """Return {composite_key: gold_track_id} from the dataset.

    Composite key = f"{session_id}__{turn_number}". Mirrors the canonical
    extraction at music-crs-evaluator/make_ground_truth.py:parsing_groundtruth:
    for each turn in 1..8, the gold track is the 2nd content row of that turn's
    conversations.
    """
    import pandas as pd
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    out: dict[str, str] = {}
    for row in ds:
        sid = row.get("session_id")
        if sid is None:
            continue
        convs = row.get("conversations")
        if not convs:
            continue
        df = pd.DataFrame(convs)
        if "turn_number" not in df.columns:
            continue
        for tn in range(1, 9):
            sub = df[df["turn_number"] == tn]
            if len(sub) < 2:
                continue
            gold = sub.iloc[1]["content"]
            if gold:
                out[f"{sid}__{tn}"] = str(gold)
    return out


def main():
    args = parse_args()
    preds_a = _load_pred(args.pred_a)
    preds_b = _load_pred(args.pred_b)
    gold = _load_gold(args.dataset, args.gold_split)

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
