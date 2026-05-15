"""Phase 0 retrieval diagnostic — measure where the current pipeline fails on dev.

Reports per-component recall@{5,20,50,100}, MRR, nDCG@20, and a failure-mode
breakdown for missed gold tracks (BM25-only / dense-only / both-low-rank / neither).
Slices results by query length, conversation depth, and artist-mention presence.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = REPO_ROOT / "music-crs-evaluator"
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from metrics.metrics_recsys import (  # noqa: E402
    get_ndcg,
    get_recall,
    get_reciprocal_rank,
)

# "by Capitalized" attribution pattern OR a quoted song-title-like string
_ARTIST_MENTION_RE = re.compile(r'\bby\s+[A-Z][a-zA-Z]+|"[^"]+"|“[^”]+”')

_COMPONENT_FIELD = {
    "bm25": "bm25_top_100",
    "dense": "dense_top_100",
    "fused": "fused_top_k",
}
_K_VALUES = (5, 20, 50, 100)
_FAILURE_KEYS = (
    "hit_in_top_k",
    "in_both_low_rank",
    "not_in_bm25_only",
    "not_in_dense_only",
    "not_in_either",
)


def categorize_miss(
    *,
    gold_id: str,
    bm25_top_k: list[str],
    dense_top_k: list[str],
    fused_top_k: list[str],
) -> str:
    if gold_id in fused_top_k:
        return "hit_in_top_k"
    in_bm25 = gold_id in bm25_top_k
    in_dense = gold_id in dense_top_k
    if in_bm25 and in_dense:
        return "in_both_low_rank"
    if in_dense:
        return "not_in_bm25_only"
    if in_bm25:
        return "not_in_dense_only"
    return "not_in_either"


def bucket_query_length(query: str) -> str:
    n = len(query.split())
    if n < 10:
        return "short"
    if n <= 30:
        return "medium"
    return "long"


def bucket_depth(turn_number: int) -> str:
    if turn_number <= 1:
        return "1"
    if turn_number <= 4:
        return "2-4"
    return "5+"


def has_artist_mention(query: str) -> bool:
    return _ARTIST_MENTION_RE.search(query) is not None


def write_records_jsonl(records: list[dict], path) -> None:
    """Write per-turn diagnostic records as JSONL. Consumed by compare_diagnostic_runs."""
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_records_jsonl(path):
    """Lazily yield per-turn records from a JSONL file."""
    import json
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _slice_recall_at_20(records: list[dict], key_fn) -> dict[str, dict]:
    buckets: dict[str, list[float]] = {}
    for rec in records:
        k = key_fn(rec)
        buckets.setdefault(k, [0.0, 0])
        gold = rec["gold_id"]
        preds = rec["fused_top_k"]
        buckets[k][0] += get_recall(gold=[gold], preds=preds, k=20)
        buckets[k][1] += 1
    return {
        key: {"recall@20": (hits / n if n else 0.0), "n": n}
        for key, (hits, n) in buckets.items()
    }


def summarize_diagnostic(per_turn_records: list[dict]) -> dict:
    """Aggregate per-turn diagnostic records into a final summary report."""
    n = len(per_turn_records)

    per_component_metrics: dict[str, dict[str, float]] = {}
    for comp, field in _COMPONENT_FIELD.items():
        recall_sums = {k: 0.0 for k in _K_VALUES}
        mrr_sum = 0.0
        ndcg20_sum = 0.0
        for rec in per_turn_records:
            preds = rec[field]
            gold = rec["gold_id"]
            for k in _K_VALUES:
                recall_sums[k] += get_recall(gold=[gold], preds=preds, k=k)
            mrr_sum += get_reciprocal_rank(gold=gold, preds=preds)
            ndcg20_sum += get_ndcg(gold=[gold], preds=preds, k=20)
        per_component_metrics[comp] = {
            **{f"recall@{k}": (recall_sums[k] / n if n else 0.0) for k in _K_VALUES},
            "mrr": mrr_sum / n if n else 0.0,
            "ndcg@20": ndcg20_sum / n if n else 0.0,
        }

    breakdown = Counter({k: 0 for k in _FAILURE_KEYS})
    for rec in per_turn_records:
        cat = categorize_miss(
            gold_id=rec["gold_id"],
            bm25_top_k=rec["bm25_top_100"],
            dense_top_k=rec["dense_top_100"],
            fused_top_k=rec["fused_top_k"],
        )
        breakdown[cat] += 1

    slice_breakdown = {
        "query_length": _slice_recall_at_20(
            per_turn_records, lambda r: bucket_query_length(r["user_query"])
        ),
        "depth": _slice_recall_at_20(
            per_turn_records, lambda r: bucket_depth(r["turn_number"])
        ),
        "artist_mention": _slice_recall_at_20(
            per_turn_records,
            lambda r: "yes" if has_artist_mention(r["user_query"]) else "no",
        ),
    }

    return {
        "n_turns": n,
        "per_component_metrics": per_component_metrics,
        "failure_breakdown": dict(breakdown),
        "slice_breakdown": slice_breakdown,
    }


# ----------------------------------------------------------------------------
# Orchestration: load dev split, run BM25/dense/fused, collect per-turn records
# ----------------------------------------------------------------------------
# Imports are deferred (under main) to keep the unit-test path fast — model
# instantiation pulls in transformers + datasets which take ~3s to import.

BASELINES_DIR = REPO_ROOT / "music-crs-baselines"

# The bare retriever specs used by `wrrf_bm25_dense_lyrics_v1` (the v1 fused
# pipeline). For Phase 0 we instantiate them standalone so we can capture
# top-100 per component, separately from any CMQR/ProRank overlay.
_BM25_CORPUS = ["track_name", "artist_name", "album_name", "release_date", "tag_list"]
_DENSE_EMBED_COL = "metadata-qwen3_embedding_0.6b"


def _build_chat_history_input(conversations, item_db, target_turn_number):
    """Reproduce run_inference_devset_retrieval_only.chat_history_parser shape:
    flatten chat history + current user query into a single retrieval input string.
    """
    import pandas as pd

    df_conversation = pd.DataFrame(conversations)
    df_history = df_conversation[df_conversation["turn_number"] < target_turn_number]
    chat_history: list[dict[str, str]] = []
    gold_id = None
    for turn_data in df_history.to_dict(orient="records"):
        role = turn_data["role"]
        content = turn_data["content"]
        if role == "music":
            role = "assistant"
            content = item_db.id_to_metadata(turn_data["content"])
        chat_history.append({"role": role, "content": content})
    df_current = df_conversation[df_conversation["turn_number"] == target_turn_number]
    user_row = df_current[df_current["role"] == "user"]
    music_row = df_current[df_current["role"] == "music"]
    if user_row.empty or music_row.empty:
        return None, None, None
    user_query = user_row.iloc[0]["content"]
    gold_id = music_row.iloc[0]["content"]
    retrieval_input = "\n".join(
        f"{t['role']}: {t['content']}"
        for t in (chat_history + [{"role": "user", "content": user_query}])
    )
    return retrieval_input, user_query, gold_id


def collect_per_turn_records(
    config_path: str,
    sample: int | None = None,
    n_top: int = 100,
    fused_k: int = 20,
    batch_size: int = 64,
    n_turns_per_session: int = 8,
) -> list[dict]:
    """Run BM25, dense, and fused (wRRF) retrievers on dev; return diagnostic records."""
    if str(BASELINES_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINES_DIR))

    import os
    from datasets import load_dataset
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from mcrs.db_item import MusicCatalogDB
    from mcrs.retrieval_modules import load_retrieval_module

    config = OmegaConf.load(config_path)
    cwd = os.getcwd()
    os.chdir(str(BASELINES_DIR))  # cache_dir paths in configs are relative to baselines/
    try:
        item_db = MusicCatalogDB(
            config.item_db_name, config.track_split_types, config.corpus_types
        )
        bm25 = load_retrieval_module(
            retrieval_type="bm25",
            dataset_name=config.item_db_name,
            track_split_types=config.track_split_types,
            corpus_types=_BM25_CORPUS,
            cache_dir=config.cache_dir,
        )
        dense = load_retrieval_module(
            retrieval_type="dense_metadata_qwen3_instruct",
            dataset_name=config.item_db_name,
            track_split_types=config.track_split_types,
            corpus_types=config.corpus_types,
            cache_dir=config.cache_dir,
        )
        fused = load_retrieval_module(
            retrieval_type=config.retrieval_type,
            dataset_name=config.item_db_name,
            track_split_types=config.track_split_types,
            corpus_types=config.corpus_types,
            cache_dir=config.cache_dir,
        )

        db = load_dataset(config.test_dataset_name, split="test")
        if sample is not None:
            db = db.select(range(min(sample, len(db))))

        inputs: list[str] = []
        meta: list[dict] = []
        for item in db:
            session_id = item["session_id"]
            for tn in range(1, n_turns_per_session + 1):
                ri, uq, gold = _build_chat_history_input(item["conversations"], item_db, tn)
                if ri is None:
                    continue
                inputs.append(ri)
                meta.append({
                    "session_id": session_id,
                    "turn_number": tn,
                    "user_query": uq,
                    "gold_id": gold,
                })

        records: list[dict] = []
        for i in tqdm(range(0, len(inputs), batch_size), desc="Phase0 retrieval"):
            batch = inputs[i:i + batch_size]
            batch_meta = meta[i:i + batch_size]
            bm25_hits = bm25.batch_text_to_item_retrieval(batch, topk=n_top)
            dense_hits = dense.batch_text_to_item_retrieval(batch, topk=n_top)
            fused_hits = fused.batch_text_to_item_retrieval(batch, topk=fused_k)
            for j, m in enumerate(batch_meta):
                records.append({
                    **m,
                    "bm25_top_100": list(bm25_hits[j]),
                    "dense_top_100": list(dense_hits[j]),
                    "fused_top_k": list(fused_hits[j]),
                })
        return records
    finally:
        os.chdir(cwd)


def render_markdown_report(summary: dict, config_path: str, sample: int | None) -> str:
    pcm = summary["per_component_metrics"]
    fb = summary["failure_breakdown"]
    sb = summary["slice_breakdown"]
    n = summary["n_turns"]
    scope = f"sample={sample}" if sample else "full dev"

    lines: list[str] = []
    lines.append(f"# Phase 0 retrieval-diagnostic baseline\n")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Scope: {scope} ({n} turns)\n")

    lines.append("## Per-component metrics\n")
    lines.append("| Component | recall@5 | recall@20 | recall@50 | recall@100 | MRR | nDCG@20 |")
    lines.append("|---|---|---|---|---|---|---|")
    for comp in ("bm25", "dense", "fused"):
        m = pcm[comp]
        lines.append(
            f"| {comp} | {m['recall@5']:.3f} | {m['recall@20']:.3f} | "
            f"{m['recall@50']:.3f} | {m['recall@100']:.3f} | {m['mrr']:.3f} | {m['ndcg@20']:.3f} |"
        )

    lines.append("\n## Failure-mode breakdown (per turn)\n")
    lines.append("| Category | Count | Share |")
    lines.append("|---|---|---|")
    for k in _FAILURE_KEYS:
        c = fb.get(k, 0)
        lines.append(f"| {k} | {c} | {(c / n if n else 0.0):.1%} |")

    lines.append("\n## Recall@20 by slice\n")
    for dim, buckets in sb.items():
        lines.append(f"### {dim}\n")
        lines.append("| Bucket | n | recall@20 |")
        lines.append("|---|---|---|")
        for key, vals in sorted(buckets.items()):
            lines.append(f"| {key} | {vals['n']} | {vals['recall@20']:.3f} |")
        lines.append("")

    lines.append("\n## Decision gate (per Phase 0 plan)\n")
    miss_counts = {k: fb.get(k, 0) for k in _FAILURE_KEYS if k != "hit_in_top_k"}
    total_misses = sum(miss_counts.values())
    if total_misses == 0:
        lines.append("- No misses → pipeline is at ceiling. Investigate evaluator/dataset.")
    else:
        biggest_miss = max(miss_counts, key=miss_counts.get)
        lines.append(f"- Biggest miss category: **{biggest_miss}** ({miss_counts[biggest_miss]}/{total_misses} of misses).")
        lever_map = {
            "not_in_either": "Embedder upgrade and/or document enrichment (BGE-M3, doc2query, tag_list).",
            "not_in_bm25_only": "BM25 preprocessing — multi-field, synonym/tag enrichment, query expansion.",
            "not_in_dense_only": "Dense embedder upgrade or fine-tuning (BGE-M3 / Qwen3-Embedding-4B / contrastive FT).",
            "in_both_low_rank": "Fusion + reranker — RRF weights, cross-encoder reranker, fine-tuned ProRank.",
        }
        lines.append(f"- Suggested lever: {lever_map.get(biggest_miss, 'investigate further.')}")
    return "\n".join(lines) + "\n"


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Phase 0 retrieval diagnostic.")
    parser.add_argument(
        "--config",
        default=str(BASELINES_DIR / "config" / "110-prorank-rerank-devset.yaml"),
        help="Path to pipeline config YAML (default: 110-prorank-rerank-devset.yaml).",
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N sessions for fast dry-run (default: full dev).")
    parser.add_argument("--n-top", type=int, default=100, help="Per-component top-N (default 100).")
    parser.add_argument("--fused-k", type=int, default=20, help="Fused/final top-K (default 20).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-json", default=str(REPO_ROOT / "data" / "phase0_diagnostic.json"))
    parser.add_argument(
        "--out-md",
        default=str(REPO_ROOT / "documents" / "phase0_baseline_report.md"),
    )
    parser.add_argument(
        "--out-records",
        default=str(REPO_ROOT / "data" / "phase0_diagnostic_records.jsonl"),
        help="Per-turn records JSONL (consumed by scripts/compare_diagnostic_runs.py).",
    )
    args = parser.parse_args()

    records = collect_per_turn_records(
        config_path=args.config,
        sample=args.sample,
        n_top=args.n_top,
        fused_k=args.fused_k,
        batch_size=args.batch_size,
    )
    summary = summarize_diagnostic(records)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_records = Path(args.out_records)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown_report(summary, args.config, args.sample), encoding="utf-8")
    write_records_jsonl(records, out_records)
    print(f"wrote {out_json}, {out_md}, {out_records}", file=sys.stderr)
    print(f"n_turns={summary['n_turns']}, fused recall@20={summary['per_component_metrics']['fused']['recall@20']:.3f}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
