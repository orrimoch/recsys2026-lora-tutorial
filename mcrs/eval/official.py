"""F3 — official-parity scoring. Replicates evaluate_devset.py EXACTLY, calling the official
metric functions (never re-deriving the math). This is the only selection number.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from mcrs.contracts import SubmissionRow
from mcrs.eval import ensure_evaluator_on_path
from mcrs.eval.harness import GoldRow


def _official_funcs():
    ensure_evaluator_on_path()
    from metrics.metrics_diversity import compute_catalog_diversity, compute_lexical_diversity
    from metrics.metrics_recsys import compute_recsys_metrics
    return compute_recsys_metrics, compute_catalog_diversity, compute_lexical_diversity


def official_ndcg(gold: list[str], preds: list[str], k: int) -> float:
    ensure_evaluator_on_path()
    from metrics.metrics_recsys import get_ndcg
    return get_ndcg(gold, preds, k)


def score_official(
    predictions: Sequence[SubmissionRow],
    ground_truth: Sequence[GoldRow],
    catalog_size: int,
    k_values: tuple[int, ...] = (1, 10, 20),
) -> dict[str, float]:
    compute_recsys_metrics, compute_catalog_diversity, compute_lexical_diversity = _official_funcs()
    pred_by = {(p.session_id, p.turn_number): p for p in predictions}

    results, recommended, responses = [], [], []
    for g in ground_truth:
        key = (g.session_id, g.turn_number)
        if key not in pred_by:
            raise KeyError(f"no prediction row for (session, turn)={key}")
        p = pred_by[key]
        # official compute_recsys_metrics raises ValueError on duplicate preds/gold
        m = compute_recsys_metrics(p.predicted_track_ids, [g.ground_truth_track_id], list(k_values))
        recommended.extend(p.predicted_track_ids)
        responses.append(p.predicted_response)
        results.append({"session_id": g.session_id, "turn_number": g.turn_number, **m})

    # Aggregation EXACTLY per evaluate_devset.py: turn-bucket mean -> mean over turns (no session avg)
    df = pd.DataFrame(results)
    turn_wise = df.drop(columns=["session_id"]).groupby("turn_number").agg("mean")
    macro = {k: float(v) for k, v in turn_wise.mean(axis=0).to_dict().items()}

    macro["catalog_diversity"] = compute_catalog_diversity(recommended, catalog_size)
    macro["lexical_diversity"] = compute_lexical_diversity(responses)
    macro["total_catalog_size"] = catalog_size
    return macro
