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
