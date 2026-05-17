"""W3 v1 post-mortem: three diagnostics that disambiguate WHY a clean SFT run
(eval loss 1.48) failed the nDCG@20 gate (0.0204).

Q1 — Per-position teacher-forced top-k accuracy.
     Single forward pass per row with the gold SID appended; record argmax at each
     of the 3 SID positions, restricted to that level's token vocab AND unrestricted.

Q2 — Generated-SID popularity collapse.
     Constrained greedy generation (same trie as eval), tally the generated SIDs.
     Compare to the gold-SID distribution on the same val rows. If a handful of SIDs
     dominate the generated output, the model has collapsed.

Q3 — Unconstrained-generation validity.
     Greedy decoding WITHOUT the prefix trie. Check (a) per-position level validity,
     (b) whether the triplet maps to a real track. Tells us if the model learned the
     SID space on its own or merely rides the trie.

Loads val.parquet + track_to_sid.parquet (W2 + W1 outputs) and the merged W3 model
from HF Hub. All three checks run in a single loop so model load + tokenizer setup
happen once.

Usage:
    python scripts/diagnose_sid_generator.py \\
        --model-id OrRim123/recsys2026-sid-generator-qwen15b-v1-merged \\
        --val-parquet experiments/cache/sid_training/val.parquet \\
        --track-to-sid experiments/cache/sid/track_to_sid.parquet \\
        --output-dir experiments/cache/sid_eval/diagnostic_v1 \\
        --limit 500              # smoke; remove for full eval

Writes:
    <output-dir>/sid_diagnostic_metrics.json   — aggregate Q1+Q2+Q3 numbers
    <output-dir>/per_query.jsonl               — per-row breakdown for drill-down
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcrs.sid.diagnostics import (
    aggregate_validity,
    build_level_token_ids,
    is_triplet_valid,
    popularity_stats,
    topk_accuracy,
)
from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
from mcrs.sid.vocab import build_sid_to_token_id_lookup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True,
                   help="HF Hub repo of the merged SID generator")
    p.add_argument("--val-parquet", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_training/val.parquet")
    p.add_argument("--track-to-sid", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid/track_to_sid.parquet")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_eval/diagnostic_v1")
    p.add_argument("--eval-slice", choices=["raw", "all"], default="raw",
                   help="raw = only conversation-derived rows (matches Blind-A semantics)")
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, run on the first N rows only (for smoke)")
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--checks", default="q1,q2,q3",
                   help="Comma-separated subset of {q1,q2,q3}")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_collision_lookup(t2s: pd.DataFrame) -> dict[tuple[int, int, int], list[str]]:
    """SID triplet → popularity-ordered track ids; mirrors eval_sid_generator.py."""
    lookup: dict[tuple[int, int, int], list[str]] = {}
    for row in t2s.sort_values("bucket_rank").itertuples(index=False):
        key = (int(row.code_1), int(row.code_2), int(row.code_3))
        lookup.setdefault(key, []).append(row.track_id)
    return lookup


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = set(c.strip() for c in args.checks.split(","))

    device = pick_device()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    if device == "mps" and dtype == torch.bfloat16:
        # bf16 on MPS is flaky pre-torch 2.5; fp16 is the safe default for Mac.
        print("[warn] downgrading bf16 -> fp16 on MPS for stability", file=sys.stderr)
        dtype = torch.float16

    print(f"[load] tokenizer + model from {args.model_id} (device={device}, dtype={dtype})",
          file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    sid_lookup = build_sid_to_token_id_lookup(tok, num_levels=3, codebook_size=256)
    inverse = {v: k for k, v in sid_lookup.items()}
    level_token_ids = build_level_token_ids(sid_lookup)
    level_token_tensors = {
        lvl: torch.tensor(level_token_ids[lvl], dtype=torch.long, device=device)
        for lvl in (0, 1, 2)
    }

    print(f"[load] track_to_sid from {args.track_to_sid}", file=sys.stderr)
    t2s = pd.read_parquet(args.track_to_sid)
    sid_to_tracks = build_collision_lookup(t2s)
    sids = list({(int(r.code_1), int(r.code_2), int(r.code_3))
                 for r in t2s.itertuples(index=False)})
    print(f"[trie] {len(sid_to_tracks)} unique SID triples", file=sys.stderr)
    trie = build_sid_trie(sids, sid_lookup) if "q2" in checks else None

    val = pd.read_parquet(args.val_parquet)
    if args.eval_slice == "raw":
        val = val[val["source"] == "raw"].reset_index(drop=True)
    if args.limit > 0:
        val = val.head(args.limit).reset_index(drop=True)
    print(f"[eval] {len(val)} val queries (slice={args.eval_slice}, checks={sorted(checks)})",
          file=sys.stderr)

    # Per-row records for the JSONL drill-down.
    per_row = []

    # Q1 accumulators: rankings + golds per level, both restricted to level vocab and unrestricted.
    q1_rankings_lvl: dict[int, list[list[int]]] = {0: [], 1: [], 2: []}
    q1_golds_lvl: dict[int, list[int]] = {0: [], 1: [], 2: []}
    q1_unrestricted_top1_in_level: dict[int, int] = {0: 0, 1: 0, 2: 0}

    # Q2 accumulators: counter of generated SIDs + gold SIDs.
    q2_generated = Counter()
    q2_gold = Counter()

    # Q3 accumulators: per-row validity records.
    q3_records: list[dict] = []

    eos_id = tok.eos_token_id

    with torch.inference_mode():
        for i, row in enumerate(val.itertuples(index=False)):
            gold = (int(row.code_1), int(row.code_2), int(row.code_3))
            q2_gold[gold] += 1
            gold_token_ids = [sid_lookup[(lvl, gold[lvl])] for lvl in (0, 1, 2)]

            prompt_enc = tok(
                row.query,
                truncation=True, max_length=args.max_prompt_len,
                return_tensors="pt", add_special_tokens=False,
            ).to(device)
            prompt_len = prompt_enc["input_ids"].shape[1]

            row_record = {"row_idx": i, "gold_sid": gold}

            # ----- Q1 — teacher-forced per-position accuracy -----
            if "q1" in checks:
                gold_t = torch.tensor([gold_token_ids], dtype=torch.long, device=device)
                input_ids = torch.cat([prompt_enc["input_ids"], gold_t], dim=1)
                attn = torch.cat([prompt_enc["attention_mask"],
                                  torch.ones_like(gold_t)], dim=1)
                logits = model(input_ids=input_ids, attention_mask=attn).logits[0]
                # logits at position `prompt_len - 1` predicts the token at `prompt_len`
                # (which is sid_token_0), and so on.
                q1_pred_per_pos = []
                for lvl in (0, 1, 2):
                    pos = prompt_len - 1 + lvl
                    step_logits = logits[pos]
                    # Restricted ranking: argsort over level-lvl SID tokens only.
                    lvl_ids = level_token_tensors[lvl]
                    lvl_logits = step_logits[lvl_ids]
                    sorted_idx = torch.argsort(lvl_logits, descending=True)
                    restricted_ranking = lvl_ids[sorted_idx].tolist()
                    q1_rankings_lvl[lvl].append(restricted_ranking)
                    q1_golds_lvl[lvl].append(gold_token_ids[lvl])
                    # Unrestricted top-1: does the whole-vocab argmax even land in level-lvl?
                    unrestricted_top1 = int(step_logits.argmax().item())
                    if inverse.get(unrestricted_top1, (-1, -1))[0] == lvl:
                        q1_unrestricted_top1_in_level[lvl] += 1
                    q1_pred_per_pos.append({
                        "level": lvl,
                        "gold_token_id": gold_token_ids[lvl],
                        "restricted_top1_token_id": restricted_ranking[0],
                        "restricted_top1_correct": restricted_ranking[0] == gold_token_ids[lvl],
                        "unrestricted_top1_token_id": unrestricted_top1,
                        "unrestricted_top1_in_level": (
                            inverse.get(unrestricted_top1, (-1, -1))[0] == lvl
                        ),
                    })
                row_record["q1"] = q1_pred_per_pos

            # ----- Q2 — constrained greedy generation -----
            if "q2" in checks:
                prefix_fn = make_prefix_allowed_tokens_fn(
                    trie, prompt_lens={0: prompt_len}, eos_token_id=eos_id,
                )
                out = model.generate(
                    **prompt_enc,
                    max_new_tokens=3,
                    num_beams=1,
                    prefix_allowed_tokens_fn=prefix_fn,
                    do_sample=False,
                    output_scores=False, return_dict_in_generate=True,
                )
                emitted = out.sequences[0, prompt_len:prompt_len + 3].tolist()
                try:
                    gen_sid = (
                        inverse[emitted[0]][1],
                        inverse[emitted[1]][1],
                        inverse[emitted[2]][1],
                    )
                    q2_generated[gen_sid] += 1
                    row_record["q2_generated_sid"] = gen_sid
                except (KeyError, IndexError):
                    # Should not happen with the trie; record explicitly if it does.
                    row_record["q2_generated_sid"] = None
                    row_record["q2_generation_anomaly"] = emitted

            # ----- Q3 — unconstrained greedy generation -----
            if "q3" in checks:
                out = model.generate(
                    **prompt_enc,
                    max_new_tokens=3,
                    num_beams=1,
                    do_sample=False,
                    output_scores=False, return_dict_in_generate=True,
                )
                emitted = out.sequences[0, prompt_len:prompt_len + 3].tolist()
                if len(emitted) < 3:
                    emitted = emitted + [-1] * (3 - len(emitted))
                validity = is_triplet_valid(emitted, inverse, sid_to_tracks)
                q3_records.append(validity)
                row_record["q3"] = {
                    "emitted_token_ids": emitted,
                    **validity,
                }

            per_row.append(row_record)
            if (i + 1) % 100 == 0:
                print(f"[diag] {i+1}/{len(val)}", file=sys.stderr)

    # ----- Aggregate -----
    metrics: dict = {"model_id": args.model_id, "n_queries": len(val),
                     "eval_slice": args.eval_slice, "checks": sorted(checks)}

    if "q1" in checks:
        q1 = {}
        n = len(q1_rankings_lvl[0])
        for lvl in (0, 1, 2):
            q1[f"level_{lvl}"] = {
                "top_1_restricted_acc": topk_accuracy(q1_rankings_lvl[lvl], q1_golds_lvl[lvl], k=1),
                "top_5_restricted_acc": topk_accuracy(q1_rankings_lvl[lvl], q1_golds_lvl[lvl], k=5),
                "unrestricted_top1_in_level_rate": (
                    q1_unrestricted_top1_in_level[lvl] / n if n else 0.0
                ),
            }
        metrics["q1_per_position_accuracy"] = q1

    if "q2" in checks:
        gen_stats = popularity_stats(q2_generated, top_ks=[1, 10, 50, 100])
        gold_stats = popularity_stats(q2_gold, top_ks=[1, 10, 50, 100])
        metrics["q2_popularity_collapse"] = {
            "generated": _serialize_popularity(gen_stats),
            "gold": _serialize_popularity(gold_stats),
            "top_5_generated_sids": [
                {"sid": list(k), "count": v}
                for k, v in q2_generated.most_common(5)
            ],
            "top_5_gold_sids": [
                {"sid": list(k), "count": v}
                for k, v in q2_gold.most_common(5)
            ],
        }

    if "q3" in checks:
        metrics["q3_unconstrained_validity"] = aggregate_validity(q3_records)

    metrics_path = args.output_dir / "sid_diagnostic_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default))
    jsonl_path = args.output_dir / "per_query.jsonl"
    with jsonl_path.open("w") as f:
        for r in per_row:
            f.write(json.dumps(r, default=_json_default) + "\n")
    print(f"[done] wrote {metrics_path}", file=sys.stderr)
    print(json.dumps(metrics, indent=2, default=_json_default))


def _serialize_popularity(stats: dict) -> dict:
    """JSON-safe: tuple keys -> lists."""
    out = dict(stats)
    if isinstance(out.get("top_1_key"), tuple):
        out["top_1_key"] = list(out["top_1_key"])
    return out


def _json_default(o):
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


if __name__ == "__main__":
    main()
