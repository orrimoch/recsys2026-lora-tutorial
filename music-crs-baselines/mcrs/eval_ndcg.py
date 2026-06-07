"""nDCG@k with turn-stratified reporting for the Music CRS dev harness.

Tier-0 #1: Blind-A is single-turn / zero-history, so the turn-1 stratum is the
only honest proxy for Blind. The flat dev mean over all 8 turns over-credits the
played-track-seeded channels (same_artist / SASRec sequence / related_artist)
that are structurally dead on Blind. Use ndcg_by_turn() and gate ship-to-Blind
decisions on the turn-1 number, not the overall average.

ndcg_at_k matches the official evaluator (music-crs-evaluator/metrics
get_ndcg) for the single-binary-gold case used by this challenge:
    DCG  = sum over top-k of rel / log2(rank_1indexed + 1)   (rel in {0,1})
    IDCG = 1                                                  (exactly one gold)
=> score = 1/log2(rank_1indexed + 1) if the gold is in the top-k, else 0.
"""
from __future__ import annotations

import math
from collections import defaultdict


def ndcg_at_k(gold, ranked, k: int = 20) -> float:
    """Single-gold nDCG@k. `gold` is one track_id; `ranked` is the predicted
    list (best first). Returns 1/log2(rank+1) if gold is in the top-k, else 0."""
    for pos, tid in enumerate(ranked[:k]):
        if tid == gold:
            return 1.0 / math.log2(pos + 2)  # pos is 0-indexed -> rank = pos+1
    return 0.0


def ndcg_by_turn(ranked_lists, golds, turn_numbers, k: int = 20) -> dict:
    """Per-turn and overall nDCG@k.

    Args:
        ranked_lists: list of predicted track_id lists (one per dev row).
        golds: list of gold track_ids, aligned with ranked_lists.
        turn_numbers: list of turn indices (1-based), aligned with ranked_lists.
        k: cutoff (default 20).

    Returns a dict:
        {
          "overall": flat mean nDCG@k over all rows,
          "macro":   mean over turns of each turn's mean (matches the official
                     macro-average when every session has all turns),
          "turn1":   the turn-1 (cold / Blind proxy) mean, or None if absent,
          "per_turn": {turn_number: {"ndcg": mean, "n": count}},
        }
    """
    if not (len(ranked_lists) == len(golds) == len(turn_numbers)):
        raise ValueError("ranked_lists, golds, turn_numbers must be the same length")

    per_turn_scores: dict = defaultdict(list)
    all_scores: list[float] = []
    for ranked, gold, tn in zip(ranked_lists, golds, turn_numbers):
        s = ndcg_at_k(gold, ranked, k=k)
        all_scores.append(s)
        per_turn_scores[int(tn)].append(s)

    per_turn = {
        tn: {"ndcg": sum(v) / len(v), "n": len(v)}
        for tn, v in sorted(per_turn_scores.items())
    }
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    macro = (sum(d["ndcg"] for d in per_turn.values()) / len(per_turn)
             if per_turn else 0.0)
    turn1 = per_turn.get(1, {}).get("ndcg")
    return {"overall": overall, "macro": macro, "turn1": turn1, "per_turn": per_turn}


def recall_by_turn(ranked_lists, golds, turn_numbers, k: int = 100) -> dict:
    """Per-turn and overall recall@k (1 if the gold is in the top-k, else 0).

    For validating a new recall channel before any reranker retrain: the turn-1
    stratum is the cold / Blind proxy where session channels are dead, so a new
    content channel must lift turn-1 recall to be worth shipping. Same return
    shape as ndcg_by_turn: {overall, macro, turn1, per_turn:{tn:{recall,n}}}.
    """
    if not (len(ranked_lists) == len(golds) == len(turn_numbers)):
        raise ValueError("ranked_lists, golds, turn_numbers must be the same length")

    per_turn_scores: dict = defaultdict(list)
    all_scores: list[float] = []
    for ranked, gold, tn in zip(ranked_lists, golds, turn_numbers):
        hit = 1.0 if gold in ranked[:k] else 0.0
        all_scores.append(hit)
        per_turn_scores[int(tn)].append(hit)

    per_turn = {
        tn: {"recall": sum(v) / len(v), "n": len(v)}
        for tn, v in sorted(per_turn_scores.items())
    }
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    macro = (sum(d["recall"] for d in per_turn.values()) / len(per_turn)
             if per_turn else 0.0)
    turn1 = per_turn.get(1, {}).get("recall")
    return {"overall": overall, "macro": macro, "turn1": turn1, "per_turn": per_turn}


def format_by_turn(report: dict, label: str = "") -> str:
    """Human-readable one-block summary for notebook/CLI logs. Works for both
    ndcg_by_turn (per-turn key 'ndcg') and recall_by_turn (key 'recall')."""
    # Detect the metric key from a per-turn entry; fall back to 'ndcg'.
    any_entry = next(iter(report["per_turn"].values()), {})
    metric = "recall" if "recall" in any_entry else "ndcg"
    tag = f"{metric}{'@100' if metric == 'recall' else '@20'}"
    lines = []
    t1 = report.get("turn1")
    head = f"{label + ' ' if label else ''}{tag} overall(flat)={report['overall']:.4f}  "
    lines.append(head + f"macro={report['macro']:.4f}  "
                 f"turn1(cold/Blind proxy)={'n/a' if t1 is None else f'{t1:.4f}'}")
    for tn, d in report["per_turn"].items():
        lines.append(f"  turn {tn:>2}: {tag}={d[metric]:.4f}  (n={d['n']})")
    return "\n".join(lines)
