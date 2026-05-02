"""Unified composite eval — closes plan §10 verification step.

Reads a `prediction.json` (output of run_inference_devset.py /
run_inference_blindset.py) plus gold-truth + per-row context, computes
per-row reward components via `compose_r_turn`, and writes a results JSON
with both per-row breakdown and aggregate summary.

Replaces the ad-hoc cell-14 logic in `colab/32_train_responder_grpo.ipynb`
and similar inline computations in `colab/33`/`40`/`41`. One canonical
codepath, one JSON output schema, easy to diff between runs.

Two ways to provide gold + context:

  1. **HF dataset mode** (production use). Pass `--hf-dataset` (e.g.
     `talkpl-ai/TalkPlayData-Challenge-Dataset`) + `--hf-split` (`test` or
     `train`) and the script materialises gold+context tables internally
     by walking the session-nested HF rows. This is what the colab
     notebooks would call.

  2. **Offline JSON mode** (testing + post-hoc analysis). Pass
     `--gold-json` (list of {session_id, turn_number, gold_track_id})
     and `--context-json` (list of {session_id, turn_number, track_name,
     artist_name, history_text, user_profile}). Lets us compute composite
     scores from cached files without re-loading the HF dataset every
     time.

Optional:
  - `--judge-checkpoint <hub_repo>` loads a `DistilledJudge` and uses it
    for R_judge (default: stub returns 0).

Usage examples:

    # post-W6 dev-eval composite (production):
    python scripts/responder_eval.py \\
        --predictions exp/inference/devset/220-...json \\
        --hf-dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --hf-split test \\
        --judge-checkpoint orrimoch/recsys2026-distilled-judge-2026-05-09 \\
        --out exp/eval/220-...composite.json

    # offline testing:
    python scripts/responder_eval.py \\
        --predictions /tmp/pred.json --gold-json /tmp/gold.json \\
        --context-json /tmp/ctx.json --out /tmp/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
from reward_fns import compose_r_turn, DistilledJudge  # noqa: E402


# ---------------------------------------------------------------------------
# Schema validation (input prediction.json)
# ---------------------------------------------------------------------------

PREDICTION_REQUIRED = {
    "session_id", "user_id", "turn_number",
    "predicted_track_ids", "predicted_response",
}


def validate_prediction_schema(predictions) -> None:
    """Raise ValueError if prediction.json shape is wrong.

    Mirrors the validator in scripts/validate_prediction.py but checks
    only the per-row contract, not split-specific row counts (those live
    in the original validator).
    """
    if not isinstance(predictions, list):
        raise ValueError(
            f"predictions must be a list, got {type(predictions).__name__}"
        )
    for i, row in enumerate(predictions):
        if not isinstance(row, dict):
            raise ValueError(f"row[{i}] is not a dict")
        missing = PREDICTION_REQUIRED - set(row.keys())
        if missing:
            raise ValueError(
                f"row[{i}] missing required field(s): {sorted(missing)}"
            )


# ---------------------------------------------------------------------------
# Per-row composite
# ---------------------------------------------------------------------------

def compose_row(
    pred_row: dict,
    gold_lookup: "dict[tuple, str]",
    context_lookup: "dict[tuple, dict]",
    judge: "DistilledJudge | None" = None,
) -> dict:
    """Score one prediction row.

    Args:
        pred_row: one entry from prediction.json. Required keys:
            session_id, turn_number, predicted_track_ids, predicted_response.
        gold_lookup: (sid, tn) → gold_track_id. Missing entries → r_retr=0.
        context_lookup: (sid, tn) → dict with optional keys:
            track_name, artist_name, history_text, user_profile.
        judge: optional DistilledJudge instance. None → R_judge contributes 0.

    Returns:
        dict with r_turn + all sub-scores.
    """
    sid = pred_row["session_id"]
    tn = int(pred_row["turn_number"])
    key = (sid, tn)

    gold = gold_lookup.get(key, "")
    ctx = context_lookup.get(key, {})

    top1_meta = None
    if ctx.get("track_name") or ctx.get("artist_name"):
        top1_meta = {
            "track_name": str(ctx.get("track_name") or ""),
            "artist_name": str(ctx.get("artist_name") or ""),
        }

    # Judge score: when no judge, default 0 (back-compat with stub).
    judge_score = 0.0
    if judge is not None:
        # Use the predicted_response as both context-summary and response.
        # The crs_baseline-style "User query: ... | Recommended track: ..."
        # context isn't reconstructable from prediction.json alone; the
        # judge's training signal will degrade gracefully on a thin context.
        ctx_str = " ".join([
            f"User query: {ctx.get('user_query', '')}",
            f"Recommended track: {(ctx.get('track_name') or '')} by {(ctx.get('artist_name') or '')}",
            f"Prior dialog: {ctx.get('history_text', '')}",
        ])
        judge_score = judge.score(ctx_str, str(pred_row["predicted_response"]))

    comps = compose_r_turn(
        predicted_track_ids=list(pred_row["predicted_track_ids"]),
        gold_track_id=gold,
        response_text=str(pred_row["predicted_response"]),
        valid_catalog=None,
        top1_meta=top1_meta,
        user_state=None,  # not reconstructable from prediction.json
        user_profile=ctx.get("user_profile"),
        history_text=str(ctx.get("history_text", "")),
        judge_score=judge_score,
    )
    # Carry session/turn for downstream merge.
    comps = dict(comps)
    comps["session_id"] = sid
    comps["turn_number"] = tn
    return comps


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_AGG_KEYS = ("r_turn", "r_retr", "r_rule", "r_format", "r_judge", "r_user_prof")


def aggregate(per_row: "list[dict]") -> dict:
    """Mean across rows for each reward component, plus row count.

    Empty input → n=0 + all means 0.0 (no divide-by-zero).
    """
    n = len(per_row)
    summary: dict = {"n": n}
    if n == 0:
        for k in _AGG_KEYS:
            summary[f"mean_{k}"] = 0.0
        return summary
    for k in _AGG_KEYS:
        summary[f"mean_{k}"] = sum(float(r.get(k, 0.0)) for r in per_row) / n
    return summary


def recall_at_k(predictions: "list[dict]", gold_lookup: "dict[tuple, str]", k: int) -> float:
    """Fraction of prediction rows where gold is in top-k of predicted_track_ids.

    Rows with no known gold in `gold_lookup` are excluded from both
    numerator and denominator (no gold → no positive to find).
    """
    n_with_gold = 0
    n_hit = 0
    for row in predictions:
        sid = row["session_id"]
        tn = int(row["turn_number"])
        gold = gold_lookup.get((sid, tn))
        if not gold:
            continue
        n_with_gold += 1
        preds = list(row["predicted_track_ids"])[:k]
        if gold in preds:
            n_hit += 1
    if n_with_gold == 0:
        return 0.0
    return n_hit / n_with_gold


# ---------------------------------------------------------------------------
# HF-dataset mode helpers (production)
# ---------------------------------------------------------------------------

def _build_lookups_from_hf(dataset_name: str, hf_split: str):
    """Materialize gold + context lookups from the HF session-nested dataset.

    Used by `--hf-dataset` mode. Walks each session's `conversations` array
    once, extracts `music`-role rows for gold tracks, and reconstructs the
    user_query + track_name + history_text per turn.

    Returns (gold_lookup, context_lookup).
    """
    from datasets import load_dataset
    print(f"[responder-eval] loading {dataset_name}[{hf_split}]")
    ds = load_dataset(dataset_name, split=hf_split)

    gold: "dict[tuple, str]" = {}
    ctx: "dict[tuple, dict]" = {}
    # Track name lookup (stub — for full track_name population we'd need
    # the Track-Metadata dataset; the eval still works without it because
    # `r_rule` skips its mention sub-checks when top1_meta is None).
    for sess in ds:
        sid = sess["session_id"]
        msgs = sorted(
            sess.get("conversations") or [],
            key=lambda m: (int(m["turn_number"]), m["role"]),
        )
        for msg in msgs:
            if msg["role"] == "music":
                gold[(sid, int(msg["turn_number"]))] = str(msg["content"])
        for tn in range(1, 9):
            user = next(
                (m for m in msgs if int(m["turn_number"]) == tn and m["role"] == "user"),
                None,
            )
            if user is None:
                continue
            prior = [m for m in msgs if int(m["turn_number"]) < tn]
            history = "\n".join(f"{m['role']}: {m['content']}" for m in prior)
            ctx[(sid, tn)] = {
                "user_query": str(user["content"]),
                "history_text": history,
                # track_name / artist_name require Track-Metadata; left None.
                # user_profile requires User-Metadata; left None.
            }
    return gold, ctx


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True,
                   help="Input prediction.json from run_inference_*.py.")
    p.add_argument("--out", required=True,
                   help="Output composite results JSON.")
    # Gold + context: pick one mode.
    p.add_argument("--hf-dataset", default=None,
                   help="HF dataset name (e.g. talkpl-ai/TalkPlayData-Challenge-Dataset).")
    p.add_argument("--hf-split", default="test",
                   help="HF split — defaults 'test' (the dev set).")
    p.add_argument("--gold-json", default=None,
                   help="Offline mode: list of {session_id, turn_number, gold_track_id}.")
    p.add_argument("--context-json", default=None,
                   help="Offline mode: list of per-row context dicts.")
    p.add_argument("--judge-checkpoint", default=None,
                   help="HF Hub repo of the trained DistilledJudge. Optional.")
    args = p.parse_args(argv)

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: predictions not found: {pred_path}", file=sys.stderr)
        return 1
    with pred_path.open(encoding="utf-8") as f:
        predictions = json.load(f)
    try:
        validate_prediction_schema(predictions)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[responder-eval] loaded {len(predictions):,} prediction rows")

    # Resolve gold + context lookups.
    if args.gold_json:
        with open(args.gold_json, encoding="utf-8") as f:
            gold_rows = json.load(f)
        gold_lookup = {
            (r["session_id"], int(r["turn_number"])): str(r["gold_track_id"])
            for r in gold_rows
        }
        if args.context_json:
            with open(args.context_json, encoding="utf-8") as f:
                ctx_rows = json.load(f)
            context_lookup = {
                (r["session_id"], int(r["turn_number"])): {
                    k: r.get(k) for k in (
                        "user_query", "track_name", "artist_name",
                        "history_text", "user_profile",
                    )
                }
                for r in ctx_rows
            }
        else:
            context_lookup = {}
    elif args.hf_dataset:
        gold_lookup, context_lookup = _build_lookups_from_hf(args.hf_dataset, args.hf_split)
    else:
        print(
            "ERROR: must pass either --hf-dataset or --gold-json (and optionally --context-json).",
            file=sys.stderr,
        )
        return 1

    # Optional judge.
    judge = None
    if args.judge_checkpoint:
        print(f"[responder-eval] loading judge {args.judge_checkpoint}")
        judge = DistilledJudge(checkpoint=args.judge_checkpoint)
        judge.warmup()

    print("[responder-eval] scoring rows…")
    per_row = [compose_row(r, gold_lookup, context_lookup, judge=judge) for r in predictions]
    summary = aggregate(per_row)
    summary["recall_at_1"] = recall_at_k(predictions, gold_lookup, 1)
    summary["recall_at_10"] = recall_at_k(predictions, gold_lookup, 10)
    summary["recall_at_20"] = recall_at_k(predictions, gold_lookup, 20)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_row": per_row}, f, ensure_ascii=False, indent=2)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n[responder-eval] wrote {out_path} ({size_kb:.1f} KB)")
    print("[responder-eval] summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
