"""Compare two Phase-0-style retrieval diagnostic runs.

Usage:
    python scripts/compare_diagnostic_runs.py \\
        --baseline data/phase0_diagnostic_records.jsonl \\
        --experiment data/phase1_bge_m3_records.jsonl \\
        --out-md documents/phase1_bge_m3_vs_baseline.md \\
        --label-baseline "v1 (BM25 + Qwen-0.6B + wRRF)" \\
        --label-experiment "BGE-M3"

The records files are produced by `scripts/phase0_retrieval_diagnostic.py
--out-records ...`. Two records are paired by (session_id, turn_number); only
turns present in both runs are compared.

Reports:
- Per-component (bm25 / dense / fused) deltas in recall@{5,20,50,100} and
  nDCG@20 with paired-bootstrap CIs and a sig-level verdict.
- Failure-mode migration matrix: how many baseline misses became experiment
  hits (and vice versa) per category.
- Per-slice deltas in recall@20 (query length / depth / artist mention).

Why this matters: Blind-A is capped at ~3 submissions/week with ±0.05 noise
on 80 rows. The dev split has ~8000 turns — paired bootstrap on it gives a
much tighter signal so we don't burn submissions on uncertain bets.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = REPO_ROOT / "music-crs-evaluator"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

# Reuse the categorization + I/O from the diagnostic itself.
from scripts.phase0_retrieval_diagnostic import (  # noqa: E402
    bucket_depth,
    bucket_query_length,
    categorize_miss,
    has_artist_mention,
    read_records_jsonl,
)
from metrics.metrics_recsys import get_ndcg, get_recall  # noqa: E402


_COMPONENT_FIELD = {
    "bm25": "bm25_top_100",
    "dense": "dense_top_100",
    "fused": "fused_top_k",
}
_K_VALUES = (5, 20, 50, 100)


def paired_bootstrap_ci(
    paired_diffs: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) of the per-turn paired difference.

    The reported `mean` is the arithmetic mean of `paired_diffs` (not a
    bootstrap statistic). `ci_low` and `ci_high` are the alpha/2 and
    1-alpha/2 quantiles of the bootstrap distribution of resampled means.
    """
    import numpy as np

    if not paired_diffs:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(paired_diffs, dtype=np.float64)
    mean = float(arr.mean())
    if np.all(arr == 0):
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return (mean, lo, hi)


def compute_migration_matrix(
    baseline_cats: list[str], experiment_cats: list[str]
) -> dict[str, dict[str, int]]:
    """Return {baseline_cat: {experiment_cat: count}} from index-aligned lists."""
    if len(baseline_cats) != len(experiment_cats):
        raise ValueError(
            f"baseline ({len(baseline_cats)}) and experiment ({len(experiment_cats)}) "
            f"category lists must be the same length"
        )
    matrix: dict[str, dict[str, int]] = {}
    for b, e in zip(baseline_cats, experiment_cats):
        matrix.setdefault(b, {}).setdefault(e, 0)
        matrix[b][e] += 1
    return matrix


def _per_turn_recall(rec: dict, component: str, k: int) -> float:
    return get_recall(gold=[rec["gold_id"]], preds=rec[_COMPONENT_FIELD[component]], k=k)


def _per_turn_ndcg(rec: dict, component: str, k: int = 20) -> float:
    return get_ndcg(gold=[rec["gold_id"]], preds=rec[_COMPONENT_FIELD[component]], k=k)


def _verdict(mean: float, lo: float, hi: float) -> str:
    if lo > 0:
        return "improved (CI excludes 0)"
    if hi < 0:
        return "regressed (CI excludes 0)"
    return "no significant change"


def compare_runs(
    baseline_records: list[dict],
    experiment_records: list[dict],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Inner-join records by (session_id, turn_number) and compute deltas."""
    base_idx = {(r["session_id"], r["turn_number"]): r for r in baseline_records}
    exp_idx = {(r["session_id"], r["turn_number"]): r for r in experiment_records}
    common_keys = sorted(set(base_idx) & set(exp_idx))

    paired = [(base_idx[k], exp_idx[k]) for k in common_keys]
    n = len(paired)

    # Per-component metric deltas with paired-bootstrap CIs.
    per_component: dict[str, dict[str, dict]] = {}
    for comp in _COMPONENT_FIELD:
        comp_block: dict[str, dict] = {}
        for k in _K_VALUES:
            diffs = [
                _per_turn_recall(e, comp, k) - _per_turn_recall(b, comp, k)
                for b, e in paired
            ]
            mean, lo, hi = paired_bootstrap_ci(diffs, n_resamples, alpha, seed)
            comp_block[f"recall@{k}"] = {
                "mean_delta": mean,
                "ci_low": lo,
                "ci_high": hi,
                "verdict": _verdict(mean, lo, hi),
            }
        diffs = [
            _per_turn_ndcg(e, comp) - _per_turn_ndcg(b, comp) for b, e in paired
        ]
        mean, lo, hi = paired_bootstrap_ci(diffs, n_resamples, alpha, seed)
        comp_block["ndcg@20"] = {
            "mean_delta": mean,
            "ci_low": lo,
            "ci_high": hi,
            "verdict": _verdict(mean, lo, hi),
        }
        per_component[comp] = comp_block

    # Failure-mode migration matrix on the fused output.
    base_cats = [
        categorize_miss(
            gold_id=b["gold_id"],
            bm25_top_k=b["bm25_top_100"],
            dense_top_k=b["dense_top_100"],
            fused_top_k=b["fused_top_k"],
        )
        for b, _ in paired
    ]
    exp_cats = [
        categorize_miss(
            gold_id=e["gold_id"],
            bm25_top_k=e["bm25_top_100"],
            dense_top_k=e["dense_top_100"],
            fused_top_k=e["fused_top_k"],
        )
        for _, e in paired
    ]
    migration = compute_migration_matrix(base_cats, exp_cats)

    # Per-slice deltas for recall@20 on fused.
    slice_deltas = {}
    slice_keys = {
        "query_length": lambda r: bucket_query_length(r["user_query"]),
        "depth": lambda r: bucket_depth(r["turn_number"]),
        "artist_mention": lambda r: "yes" if has_artist_mention(r["user_query"]) else "no",
    }
    for slice_name, key_fn in slice_keys.items():
        bucket_diffs: dict[str, list[float]] = {}
        for b, e in paired:
            key = key_fn(b)  # base + exp slice keys must match (same dataset)
            bucket_diffs.setdefault(key, []).append(
                _per_turn_recall(e, "fused", 20) - _per_turn_recall(b, "fused", 20)
            )
        slice_deltas[slice_name] = {}
        for key, diffs in bucket_diffs.items():
            mean, lo, hi = paired_bootstrap_ci(diffs, n_resamples, alpha, seed)
            slice_deltas[slice_name][key] = {
                "n": len(diffs),
                "mean_delta": mean,
                "ci_low": lo,
                "ci_high": hi,
                "verdict": _verdict(mean, lo, hi),
            }

    return {
        "n_paired": n,
        "n_baseline_only": len(set(base_idx) - set(exp_idx)),
        "n_experiment_only": len(set(exp_idx) - set(base_idx)),
        "per_component_deltas": per_component,
        "failure_migration": migration,
        "slice_deltas": slice_deltas,
    }


def render_comparison_md(
    summary: dict,
    label_baseline: str,
    label_experiment: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Comparison: **{label_experiment}** vs **{label_baseline}**\n")
    lines.append(
        f"- paired turns: {summary['n_paired']} "
        f"(baseline-only={summary['n_baseline_only']}, experiment-only={summary['n_experiment_only']})"
    )
    lines.append("- CIs are 95% paired-bootstrap (n_resamples=1000) on per-turn deltas\n")

    lines.append("## Per-component deltas (experiment − baseline)\n")
    for comp, block in summary["per_component_deltas"].items():
        lines.append(f"### {comp}\n")
        lines.append("| Metric | Δ mean | 95% CI | Verdict |")
        lines.append("|---|---|---|---|")
        for metric in ("recall@5", "recall@20", "recall@50", "recall@100", "ndcg@20"):
            d = block[metric]
            lines.append(
                f"| {metric} | {d['mean_delta']:+.4f} | "
                f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | {d['verdict']} |"
            )
        lines.append("")

    lines.append("## Failure-mode migration (rows=baseline, cols=experiment)\n")
    all_cats = sorted({k for k in summary["failure_migration"]} |
                      {c for row in summary["failure_migration"].values() for c in row})
    header = "| baseline ↓ \\ experiment → | " + " | ".join(all_cats) + " | row total |"
    lines.append(header)
    lines.append("|" + "---|" * (len(all_cats) + 2))
    for b in all_cats:
        row = summary["failure_migration"].get(b, {})
        cells = [str(row.get(c, 0)) for c in all_cats]
        total = sum(row.values())
        lines.append(f"| {b} | " + " | ".join(cells) + f" | {total} |")
    lines.append("")
    lines.append("**Read the table:** the `not_in_either → hit_in_top_k` cell is the "
                 "key win — it's the count of previously-unreachable golds that the "
                 "new pipeline now retrieves. The diagonal is unchanged turns. "
                 "Cells above the diagonal (baseline_hit → experiment_miss) are regressions.\n")

    lines.append("## Per-slice Δ recall@20 (fused)\n")
    for slice_name, buckets in summary["slice_deltas"].items():
        lines.append(f"### {slice_name}\n")
        lines.append("| Bucket | n | Δ mean | 95% CI | Verdict |")
        lines.append("|---|---|---|---|---|")
        for key in sorted(buckets):
            d = buckets[key]
            lines.append(
                f"| {key} | {d['n']} | {d['mean_delta']:+.4f} | "
                f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | {d['verdict']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compare two Phase 0 diagnostic runs.")
    parser.add_argument("--baseline", required=True, help="Path to baseline records JSONL.")
    parser.add_argument("--experiment", required=True, help="Path to experiment records JSONL.")
    parser.add_argument("--out-json", default=None, help="Optional path for raw comparison JSON.")
    parser.add_argument("--out-md", required=True, help="Path for the markdown comparison report.")
    parser.add_argument("--label-baseline", default="baseline", help="Label for baseline column.")
    parser.add_argument("--label-experiment", default="experiment", help="Label for experiment column.")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_records = list(read_records_jsonl(args.baseline))
    experiment_records = list(read_records_jsonl(args.experiment))

    summary = compare_runs(
        baseline_records=baseline_records,
        experiment_records=experiment_records,
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        seed=args.seed,
    )
    md = render_comparison_md(summary, args.label_baseline, args.label_experiment)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out_md} and {out_json}", file=sys.stderr)
    else:
        print(f"wrote {out_md}", file=sys.stderr)

    headline = summary["per_component_deltas"]["fused"]["recall@20"]
    print(
        f"FUSED Δ recall@20 = {headline['mean_delta']:+.4f} "
        f"[{headline['ci_low']:+.4f}, {headline['ci_high']:+.4f}] — {headline['verdict']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
