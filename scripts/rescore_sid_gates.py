"""Re-evaluate validation gates on EXISTING quantizer checkpoints + assignments,
without retraining. Use when gate logic changes (e.g. gate 3 v1 -> v2 dominance metric).

Reads each `quantizer_seed{S}_assignments.parquet` + previous gates JSON,
recomputes gate 3 (and gate 2) using current `mcrs.sid.validation`
implementations, overwrites the gates JSON.

Usage:
    python scripts/rescore_sid_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
from datasets import load_dataset

from mcrs.sid.validation import validate_codebook_utilization, validate_cluster_purity


# Match the production thresholds in build_sid_quantizer.py.
LEVEL_THRESHOLDS = [0.50, 0.25, 0.15]
GATE_3_THRESHOLD = 0.20
SEEDS = [42, 123, 7]


def main():
    cache_root = REPO_ROOT / "experiments" / "cache" / "sid"

    print("Loading TalkPlayData-Challenge-Track-Metadata for tag_lookup...", file=sys.stderr)
    meta_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    tag_lookup = {row["track_id"]: row.get("tag_list") or [] for row in meta_ds}

    for s in SEEDS:
        gates_path = cache_root / f"quantizer_seed{s}_gates.json"
        assign_path = cache_root / f"quantizer_seed{s}_assignments.parquet"
        if not gates_path.exists() or not assign_path.exists():
            print(f"seed {s}: missing artifacts, skipping", file=sys.stderr)
            continue

        prev = json.loads(gates_path.read_text())
        df = pd.read_parquet(assign_path)

        # Re-evaluate gate 2 with current per-level thresholds.
        g2_results = []
        for level in range(3):
            assignments = df[f"code_{level + 1}"].tolist()
            passed, util = validate_codebook_utilization(
                assignments, codebook_size=256, threshold=LEVEL_THRESHOLDS[level],
            )
            g2_results.append({
                "level": level + 1, "passed": passed,
                "utilization": util, "threshold": LEVEL_THRESHOLDS[level],
            })
        g2_pass = all(r["passed"] for r in g2_results)

        # Re-evaluate gate 3 with current dominance metric.
        level1_buckets: dict[str, list[str]] = {}
        for tid, c1 in zip(df["track_id"].tolist(), df["code_1"].tolist()):
            level1_buckets.setdefault(str(int(c1)), []).append(tid)
        g3_pass, g3_purity = validate_cluster_purity(
            level1_buckets, tag_lookup,
            n_samples=100, threshold=GATE_3_THRESHOLD, seed=s,
        )

        new_gates = dict(prev)
        new_gates["gate_2_codebook_utilization"] = {"passed": g2_pass, "per_level": g2_results}
        new_gates["gate_3_cluster_purity"] = {"passed": g3_pass, "purity": g3_purity}
        new_gates["all_gates_passed"] = (
            prev["gate_1_relative_mse"]["passed"] and g2_pass and g3_pass
        )
        gates_path.write_text(json.dumps(new_gates, indent=2))
        print(
            f"seed {s}: g3_purity={g3_purity:.3f} g3_pass={g3_pass} "
            f"all_passed={new_gates['all_gates_passed']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
