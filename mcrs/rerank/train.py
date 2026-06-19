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
    per_channel_query_builders: Optional[dict] = None,
) -> list[tuple]:
    queries = [query_builder.build(t).text for t in turns]
    bc = [{"history_tids": t.history_tids, "user_id": t.user_id} for t in turns]
    uids = [t.user_id for t in turns]
    # Fail loud if a routing key matches no channel's query_key — otherwise that channel silently
    # falls back to the full query and the focused-query routing is a no-op, mistraining K2 on a
    # full-vs-focused-skewed pool (mirrors the same guard in InferenceHarness.__init__).
    channels = getattr(fusion, "channels", None)
    if per_channel_query_builders and channels is not None:
        channel_keys = {getattr(ch, "query_key", None) for ch in channels}
        unmatched = [k for k in per_channel_query_builders if k not in channel_keys]
        if unmatched:
            raise ValueError(
                f"per_channel_query_builders keys {unmatched} match no channel query_key "
                f"{sorted(k for k in channel_keys if k)} — routing would silently fall back to the "
                f"full query (train/serve skew)")
    # Route per-channel queries (e.g. ColBERT's focused query) EXACTLY like InferenceHarness.run, so the
    # K2 training pool == the serve pool for every channel (no train/serve skew on a routed channel).
    pcq = {key: [qb.build(t).text for t in turns]
           for key, qb in (per_channel_query_builders or {}).items()} or None
    pools = fusion.fuse(queries, topk, topk_internal=topk_internal, batch_context=bc, user_ids=uids,
                        per_channel_queries=pcq)
    return [(t, pool, gold_fn(t)) for t, pool in zip(turns, pools)]
