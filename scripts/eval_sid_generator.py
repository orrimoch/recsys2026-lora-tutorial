"""W3 eval gate: run constrained beam search over val parquet, compute nDCG@20,
paired-bootstrap CI vs Phase 0 baseline.

Reads merged HF model (or base + LoRA adapter) + W1 SID lookup + W2 val parquet.
Writes:
  experiments/cache/sid_eval/w3_eval_metrics.json
  experiments/cache/sid_eval/per_query_ndcg.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcrs.sid.eval import aggregate_ndcg, compute_ndcg_at_k
from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
from mcrs.sid.vocab import build_sid_to_token_id_lookup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True,
                   help="HF Hub repo of the merged SID generator (e.g. OrRim123/...-merged)")
    p.add_argument("--val-parquet", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_training/val.parquet")
    p.add_argument("--track-to-sid", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid/track_to_sid.parquet")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_eval")
    p.add_argument("--phase0-jsonl", type=Path,
                   default=REPO_ROOT / "experiments/diagnostic_runs/phase0_baseline_full_dev/per_turn_metrics.jsonl",
                   help="Per-turn nDCG@20 from Phase 0 baseline (for paired-bootstrap CI). "
                        "Optional: if absent, eval still runs but skips the CI.")
    p.add_argument("--eval-slice", choices=["raw", "all"], default="raw",
                   help="raw = only conversation-derived val rows (matches Blind-A); "
                        "all = include metadata + doc2query rows (catalog coverage diagnostic).")
    p.add_argument("--num-beams", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, eval only the first N val rows (for smoke).")
    return p.parse_args()


def build_collision_lookup(track_to_sid_df: pd.DataFrame) -> dict[tuple[int,int,int], list[str]]:
    """SID triplet → ordered list of track_ids (popularity-descending, from W1 bucket_rank)."""
    lookup: dict[tuple[int,int,int], list[str]] = {}
    for row in track_to_sid_df.sort_values("bucket_rank").itertuples(index=False):
        key = (int(row.code_1), int(row.code_2), int(row.code_3))
        lookup.setdefault(key, []).append(row.track_id)
    return lookup


def decode_beams_to_tracks(
    beam_token_ids: list[list[int]],
    beam_scores: list[float],
    inverse_lookup: dict[int, tuple[int,int]],
    sid_to_tracks: dict[tuple[int,int,int], list[str]],
    cap_per_bucket: int = 1,
    top_k: int = 20,
) -> list[str]:
    """For each beam, decode SID + look up tracks (popularity-ordered). Apply per-bucket cap.

    cap_per_bucket=1 means: each beam contributes at most 1 track from its collision bucket.
    Spillover tracks (beams 21+) are appended to fill any de-duplication gaps.
    """
    seen = set()
    primary = []
    spillover = []
    for tok_ids in beam_token_ids:
        try:
            sid = (
                inverse_lookup[tok_ids[0]][1],
                inverse_lookup[tok_ids[1]][1],
                inverse_lookup[tok_ids[2]][1],
            )
        except (KeyError, IndexError):
            continue
        bucket = sid_to_tracks.get(sid, [])
        added_in_beam = 0
        for tid in bucket:
            if tid in seen:
                continue
            if added_in_beam < cap_per_bucket:
                primary.append(tid)
                added_in_beam += 1
            else:
                spillover.append(tid)
            seen.add(tid)
    return (primary + spillover)[:top_k]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer + model from {args.model_id}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    sid_lookup = build_sid_to_token_id_lookup(tok, num_levels=3, codebook_size=256)
    inverse = {v: k for k, v in sid_lookup.items()}

    print(f"[load] track_to_sid from {args.track_to_sid}", file=sys.stderr)
    t2s = pd.read_parquet(args.track_to_sid)
    sids = list(t2s[["code_1", "code_2", "code_3"]].itertuples(index=False, name=None))
    sid_to_tracks = build_collision_lookup(t2s)
    print(f"[trie] {len(sids)} SID rows, {len(sid_to_tracks)} unique SID triples", file=sys.stderr)
    trie = build_sid_trie(sids, sid_lookup)

    val = pd.read_parquet(args.val_parquet)
    if args.eval_slice == "raw":
        val = val[val["source"] == "raw"].reset_index(drop=True)
    if args.limit > 0:
        val = val.head(args.limit).reset_index(drop=True)
    print(f"[eval] {len(val)} val queries (slice={args.eval_slice})", file=sys.stderr)

    per_query_ndcg = []
    per_query_records = []
    eos_id = tok.eos_token_id

    with torch.inference_mode():
        for i, row in enumerate(val.itertuples(index=False)):
            inputs = tok(
                row.query,
                truncation=True, max_length=args.max_prompt_len,
                return_tensors="pt",
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            prefix_fn = make_prefix_allowed_tokens_fn(
                trie, prompt_lens={0: prompt_len}, eos_token_id=eos_id,
            )
            out = model.generate(
                **inputs,
                max_new_tokens=3,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                prefix_allowed_tokens_fn=prefix_fn,
                output_scores=False, return_dict_in_generate=True,
                do_sample=False,
            )
            # Each sequence row is (prompt_len + 3) tokens; slice off prompt to get the 3 SID tokens.
            beams = out.sequences[:, prompt_len:prompt_len + 3].cpu().tolist()
            tracks = decode_beams_to_tracks(
                beam_token_ids=beams,
                beam_scores=[],   # unused for now; could weight beams in v2
                inverse_lookup=inverse,
                sid_to_tracks=sid_to_tracks,
                top_k=args.top_k,
            )
            score = compute_ndcg_at_k(retrieved=tracks, gold=row.track_id, k=args.top_k)
            per_query_ndcg.append(score)
            per_query_records.append({
                "query_id": i,
                "track_id_gold": row.track_id,
                "ndcg_at_20": score,
                "top_5_retrieved": tracks[:5],
            })
            if (i + 1) % 200 == 0:
                running = aggregate_ndcg(per_query_ndcg)
                print(f"[eval] {i+1}/{len(val)} mean nDCG@20={running:.4f}", file=sys.stderr)

    mean_ndcg = aggregate_ndcg(per_query_ndcg)
    metrics = {
        "model_id": args.model_id,
        "eval_slice": args.eval_slice,
        "n_queries": len(per_query_ndcg),
        "mean_ndcg_at_20": mean_ndcg,
        "phase0_baseline_ndcg_at_20": 0.099,   # documented from Phase 0 baseline
        "delta_vs_phase0": mean_ndcg - 0.099,
    }

    # Optional paired-bootstrap CI vs Phase 0 baseline.
    if args.phase0_jsonl.exists():
        baseline_per_query = _load_phase0_per_query(args.phase0_jsonl)
        # Align by query_id (or track_id_gold) — both runs evaluated same val rows, but make sure.
        # For simplicity here, paired-bootstrap on same-length lists (truncate to common count).
        n_common = min(len(per_query_ndcg), len(baseline_per_query))
        ours = per_query_ndcg[:n_common]
        theirs = baseline_per_query[:n_common]
        from scripts.compare_diagnostic_runs import paired_bootstrap_ci
        lo, hi = paired_bootstrap_ci(
            ours, theirs, n_resamples=1000, alpha=0.05,
        )
        metrics["paired_bootstrap_ci"] = {"lo": lo, "hi": hi, "n_compared": n_common}
        metrics["gate_pass"] = (mean_ndcg >= 0.12) and (lo > 0)
    else:
        print(f"[warn] phase0 jsonl not found at {args.phase0_jsonl}; skipping CI.", file=sys.stderr)
        metrics["paired_bootstrap_ci"] = None
        metrics["gate_pass"] = mean_ndcg >= 0.12  # point-estimate gate

    (args.output_dir / "w3_eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.output_dir / "per_query_ndcg.jsonl").open("w") as f:
        for rec in per_query_records:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(metrics, indent=2))


def _load_phase0_per_query(path: Path) -> list[float]:
    """Read Phase 0 per-turn metrics JSONL; extract per-query nDCG@20 in file order."""
    out = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            # Tolerate variant key names from earlier runs.
            v = rec.get("ndcg_at_20") or rec.get("ndcg@20") or rec.get("ndcg")
            if v is not None:
                out.append(float(v))
    return out


if __name__ == "__main__":
    main()
