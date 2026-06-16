"""K2 training driver — turn the fused retrieval pool over a set of turns into LGBMReranker groups.

Runs the SAME query construction + fusion as serve (so train==serve), then attaches each turn's
gold. K2.build_training_data skips groups whose gold isn't in the pool.
"""
from __future__ import annotations

from typing import Callable, Optional

from mcrs.contracts import TurnContext


def build_rerank_groups(
    query_builder, fusion, turns: list[TurnContext],
    gold_fn: Callable[[TurnContext], Optional[str]],
    topk: int, topk_internal: Optional[int] = None,
) -> list[tuple]:
    queries = [query_builder.build(t).text for t in turns]
    bc = [{"history_tids": t.history_tids, "user_id": t.user_id} for t in turns]
    uids = [t.user_id for t in turns]
    pools = fusion.fuse(queries, topk, topk_internal=topk_internal, batch_context=bc, user_ids=uids)
    return [(t, pool, gold_fn(t)) for t, pool in zip(turns, pools)]
