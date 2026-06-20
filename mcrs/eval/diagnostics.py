"""F3 — diagnostic metrics for module development (NOT the official selection score).

recall@K (official has recall disabled, so this is ours), nDCG@K (official get_ndcg, parity by
construction), mean hit-rank, with overall / by-segment / by-turn breakdowns. Clearly namespaced
from the official path; selection only ever reads score_official.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from mcrs.contracts import SubmissionRow
from mcrs.eval.harness import GoldRow
from mcrs.eval.official import official_ndcg


def recall_at_k(retrieved: Sequence[str], gold: str, k: int) -> float:
    return 1.0 if gold in retrieved[:k] else 0.0


def hit_rank(retrieved: Sequence[str], gold: str) -> Optional[int]:
    try:
        return list(retrieved).index(gold) + 1
    except ValueError:
        return None


def gold_rank_distribution(
    ranked_ids: Sequence[Sequence[str]],
    golds: Sequence[str],
    *,
    buckets: Sequence[int] = (20, 50, 100, 200, 500),
) -> dict:
    """Bucket each gold's 1-based hit rank in its reranked list into half-open bands [1..b0], (b0..b1]…
    plus a `not_in_pool` band for golds absent (or ranked beyond the last bucket).

    Answers "how far down does a fix have to reach?" for the in-pool-but-not-top-k golds: if they
    cluster in 21-50 the reranker is close (cheap feature/retrain win); if they spread into 101-500 it
    is badly mis-ordering them (low-EV for reranker tuning). Returns {n, labels (ordered), buckets}."""
    if len(ranked_ids) != len(golds):
        raise ValueError(f"ranked_ids ({len(ranked_ids)}) and golds ({len(golds)}) must be 1:1 aligned.")
    edges = list(buckets)
    labels, prev = [], 0
    for b in edges:
        labels.append(f"{prev + 1}-{b}")
        prev = b
    labels.append("not_in_pool")
    counts = {l: 0 for l in labels}
    for ids, g in zip(ranked_ids, golds):
        r = hit_rank(ids, g)                       # 1-based rank, or None if absent
        if r is None or r > edges[-1]:
            counts["not_in_pool"] += 1
            continue
        for b, lab in zip(edges, labels):
            if r <= b:
                counts[lab] += 1
                break
    return {"n": len(golds), "labels": labels, "buckets": counts}


def feature_rescue_analysis(
    base_ranked: Sequence[Sequence[str]],
    feat_ranked: Sequence[Sequence[str]],
    golds: Sequence[str],
    *,
    top_k: int = 20,
    band: tuple[int, int] = (20, 50),
) -> dict:
    """Would a candidate feature's OWN ranking rescue the base reranker's just-missed golds?

    `base_ranked[i]` (e.g. K2's order) and `feat_ranked[i]` (e.g. a fine-tuned cross-encoder's order)
    are full ordered id lists over the SAME pool for turn i; `golds[i]` is the gold. Per gold:
      converted    base_rank <= top_k                                   (base already converts it)
      rescuable    band[0] < base_rank <= band[1]  AND feat_rank <= top_k  (feature could promote it)
      mid_missed   band[0] < base_rank <= band[1]  AND feat_rank  > top_k  (feature doesn't help here)
      deep         base_rank > band[1]                                  (base ranks it far down)
      not_in_pool  base_rank is None

    `rescue_rate` = rescuable / (rescuable + mid_missed): of the base's just-below-cutoff golds, the
    fraction the feature ranks into its top_k — the 2a go/no-go for stacking the feature.
    `demotion_risk` = (base-top_k golds the feature ranks > top_k) / converted: how much the feature
    would fight the base on golds it already had right (a signal-noise check). Pure, no model/GPU."""
    if not (len(base_ranked) == len(feat_ranked) == len(golds)):
        raise ValueError(f"base_ranked ({len(base_ranked)}), feat_ranked ({len(feat_ranked)}) and "
                         f"golds ({len(golds)}) must be 1:1 aligned.")
    lo, hi = band
    counts = {"converted": 0, "rescuable": 0, "mid_missed": 0, "deep": 0, "not_in_pool": 0}
    demoted = 0
    for b_ids, f_ids, g in zip(base_ranked, feat_ranked, golds):
        br = hit_rank(b_ids, g)
        fr = hit_rank(f_ids, g)
        if br is None:
            counts["not_in_pool"] += 1
        elif br <= top_k:
            counts["converted"] += 1
            if fr is None or fr > top_k:
                demoted += 1
        elif br <= hi:
            counts["rescuable" if (fr is not None and fr <= top_k) else "mid_missed"] += 1
        else:
            counts["deep"] += 1
    band_total = counts["rescuable"] + counts["mid_missed"]
    return {
        "n": len(golds),
        "counts": counts,
        "rescue_rate": (counts["rescuable"] / band_total) if band_total else 0.0,
        "demotion_risk": (demoted / counts["converted"]) if counts["converted"] else 0.0,
    }


def _conversion_block(ranked_ids, golds, idxs, pool_k: int, top_k: int) -> dict[str, float]:
    if not idxs:
        return {"n": 0, "recall_at_pool": 0.0, "recall_at_top": 0.0,
                "conversion": 0.0, "recall_loss": 0.0, "ranking_loss": 0.0}
    n = len(idxs)
    in_pool = sum(recall_at_k(ranked_ids[i], golds[i], pool_k) for i in idxs) / n
    in_top = sum(recall_at_k(ranked_ids[i], golds[i], top_k) for i in idxs) / n
    return {
        "n": n,
        "recall_at_pool": in_pool,                              # gold reachable at all (the ceiling)
        "recall_at_top": in_top,                                # gold ranked into the scored top-k
        "conversion": (in_top / in_pool) if in_pool else 0.0,   # of reachable golds, fraction converted
        "recall_loss": 1.0 - in_pool,                           # never reached the pool -> recall's job
        "ranking_loss": in_pool - in_top,                       # in pool, mis-ranked -> reranker's job
    }


def conversion_diagnosis(
    ranked_ids: Sequence[Sequence[str]],
    golds: Sequence[str],
    *,
    pool_k: int = 500,
    top_k: int = 20,
    segments: Optional[Sequence[str]] = None,
) -> dict:
    """Decompose the nDCG@top_k gap into recall-bound vs ranking-bound headroom (Option-0).

    `ranked_ids[i]` is the reranker's FULL ordered candidate ids for turn i (e.g. K2's reranked pool);
    `golds[i]` is that turn's gold tid. Splits the miss budget into:
      - recall_loss   = 1 - recall@pool_k         (gold never reached the pool — ColBERT/recall lever)
      - ranking_loss  = recall@pool_k - recall@top_k  (gold in the pool but not in the top-k — K2/CE lever)
    and reports `conversion` = recall@top_k / recall@pool_k (of the reachable golds, the fraction the
    reranker lands in the scored top-k). `verdict` names the larger headroom so effort goes where the
    gap actually is. Pure over precomputed ranked lists — the GPU run that produces them is the caller's.
    """
    if top_k > pool_k:
        raise ValueError(f"top_k ({top_k}) must be <= pool_k ({pool_k}): the scored top-k is a prefix "
                         "of the pool, else ranking_loss is negative/meaningless.")
    if len(ranked_ids) != len(golds):
        raise ValueError(f"ranked_ids ({len(ranked_ids)}) and golds ({len(golds)}) must be 1:1 aligned.")
    if segments is not None and len(segments) != len(golds):
        raise ValueError(f"segments ({len(segments)}) must be 1:1 with golds ({len(golds)}).")
    ranked_ids = [list(r) for r in ranked_ids]
    overall = _conversion_block(ranked_ids, golds, list(range(len(golds))), pool_k, top_k)
    if overall["ranking_loss"] > overall["recall_loss"]:
        overall["verdict"] = "ranking-bound"
    elif overall["recall_loss"] > overall["ranking_loss"]:
        overall["verdict"] = "recall-bound"
    else:
        overall["verdict"] = "balanced"
    if segments is not None:
        overall["by_segment"] = {
            s: _conversion_block(ranked_ids, golds, [i for i in range(len(golds)) if segments[i] == s],
                                 pool_k, top_k)
            for s in sorted(set(segments))
        }
    return overall


@dataclass(frozen=True)
class DiagnosticReport:
    overall: dict[str, float]
    by_segment: dict[str, dict[str, float]]
    by_turn: dict[int, dict[str, float]]


def _aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    metric_keys = [k for k in rows[0] if k.startswith(("recall@", "ndcg@"))]
    out = {k: sum(r[k] for r in rows) / len(rows) for k in metric_keys}
    hits = [r["_hit_rank"] for r in rows if r["_hit_rank"] is not None]
    out["hit_rate"] = len(hits) / len(rows)
    out["mean_hit_rank"] = (sum(hits) / len(hits)) if hits else float("nan")
    return out


def score_diagnostic(
    predictions: Sequence[SubmissionRow],
    ground_truth: Sequence[GoldRow],
    catalog_size: int,
    k_recall: tuple[int, ...] = (1, 10, 20, 50, 100, 200, 500),
    k_ndcg: tuple[int, ...] = (1, 10, 20),
    segment_of: Optional[dict[str, str]] = None,
) -> DiagnosticReport:
    pred_by = {(p.session_id, p.turn_number): p for p in predictions}
    rows: list[dict] = []
    for g in ground_truth:
        p = pred_by.get((g.session_id, g.turn_number))
        if p is None:
            raise KeyError(f"no prediction row for (session, turn)=({g.session_id}, {g.turn_number})")
        ids, gold = p.predicted_track_ids, g.ground_truth_track_id
        row = {
            "session_id": g.session_id, "turn_number": g.turn_number,
            "_hit_rank": hit_rank(ids, gold),
            "_seg": (segment_of or {}).get(g.session_id),
        }
        row.update({f"recall@{k}": recall_at_k(ids, gold, k) for k in k_recall})
        row.update({f"ndcg@{k}": official_ndcg([gold], ids, k) for k in k_ndcg})
        rows.append(row)

    overall = _aggregate(rows)
    by_turn = {t: _aggregate([r for r in rows if r["turn_number"] == t])
               for t in sorted({r["turn_number"] for r in rows})}
    by_segment: dict[str, dict[str, float]] = {}
    if segment_of:
        for seg in sorted({r["_seg"] for r in rows if r["_seg"] is not None}):
            by_segment[seg] = _aggregate([r for r in rows if r["_seg"] == seg])
    return DiagnosticReport(overall=overall, by_segment=by_segment, by_turn=by_turn)
