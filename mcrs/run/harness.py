"""D1 — inference & submission harness.

Orchestrates the causal spine per turn: R1 query → R7 fuse → (optional K2/K3 rerank) → L1 filter
→ (optional S1 responder) → SubmissionRow. Plus strict JSON writing and a precheck validator.
Reranker/responder are optional so a BM25→RRF→filter→trivial-responder first submission works.
See `.claude/documents/features/80_D1_inference_submission_harness.md`.
"""
from __future__ import annotations

import json
from typing import Iterable, Optional, Sequence

from mcrs.contracts import RankedList, SubmissionRow, TurnContext


class InferenceHarness:
    def __init__(self, query_builder, fusion, filt, responder=None, reranker=None,
                 topk: int = 300, topk_internal: Optional[int] = None,
                 per_channel_query_builders: Optional[dict] = None) -> None:
        self.qb = query_builder
        self.fusion = fusion
        self.filt = filt
        self.responder = responder
        self.reranker = reranker
        self.topk = topk
        self.topk_internal = topk_internal
        # {query_key: QueryBuilder} — a channel whose `query_key` matches gets this builder's query
        # (e.g. ColBERT gets the focused recency_window=1 query) while others keep the full query.
        self.per_channel_query_builders = per_channel_query_builders or {}
        # Fail loud if a routing key matches no channel's query_key — otherwise that channel silently
        # falls back to the full query and the focused-query routing is a no-op (e.g. key "colbert_ft"
        # vs the config default query_key "colbert").
        channels = getattr(self.fusion, "channels", None)
        if self.per_channel_query_builders and channels is not None:
            channel_keys = {getattr(ch, "query_key", None) for ch in channels}
            unmatched = [k for k in self.per_channel_query_builders if k not in channel_keys]
            if unmatched:
                raise ValueError(
                    f"per_channel_query_builders keys {unmatched} match no channel query_key "
                    f"{sorted(k for k in channel_keys if k)} — routing would silently fall back to the "
                    f"full query")

    def run(self, turns: Sequence[TurnContext], show_progress: bool = False) -> list[SubmissionRow]:
        turns = list(turns)
        queries = [self.qb.build(t).text for t in turns]
        bc = [{"history_tids": t.history_tids, "user_id": t.user_id} for t in turns]
        uids = [t.user_id for t in turns]
        # build per-channel query lists (1:1 with turns) for any configured query_key
        per_channel_queries = {key: [qb.build(t).text for t in turns]
                               for key, qb in self.per_channel_query_builders.items()} or None
        pools = self.fusion.fuse(queries, self.topk, topk_internal=self.topk_internal,
                                 batch_context=bc, user_ids=uids,
                                 per_channel_queries=per_channel_queries)
        rows: list[SubmissionRow] = []
        pairs = zip(turns, pools)
        if show_progress:                       # per-turn bar: the rerank loop is the slow part (CE forwards)
            try:
                from tqdm.auto import tqdm
                pairs = tqdm(list(pairs), total=len(turns), desc="rerank turns", unit="turn")
            except ImportError:
                pass
        for t, pool in pairs:
            ranked = self.reranker.rerank(t, pool) if self.reranker else RankedList(turn=t, items=pool)
            ids = self.filt.apply(ranked)
            resp = self.responder.respond(t, [{"track_id": i} for i in ids]) if self.responder else ""
            rows.append(SubmissionRow(t.session_id, t.user_id, t.turn_number, ids, resp))
        return rows


def write_submission(rows: Sequence[SubmissionRow], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in rows], f, ensure_ascii=False, indent=2)


def validate_submission(
    rows: Sequence[SubmissionRow], catalog=None,
    expected_keys: Optional[Iterable[tuple[str, int]]] = None, top_k: int = 20,
) -> None:
    """Raise ValueError on any strict-schema violation (rule #1 + submission spec)."""
    seen_rows: set[tuple[str, int]] = set()
    for r in rows:
        key = (r.session_id, r.turn_number)
        if key in seen_rows:
            raise ValueError(f"duplicate submission row for {key}")
        seen_rows.add(key)
        ids = r.predicted_track_ids
        if len(ids) > top_k:
            raise ValueError(f"{key}: {len(ids)} ids > top_k={top_k}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{key}: duplicate predicted_track_ids")
        if catalog is not None:
            bad = [i for i in ids if i not in catalog]
            if bad:
                raise ValueError(f"{key}: non-catalog ids {bad[:3]}")
    if expected_keys is not None:
        missing = set(expected_keys) - seen_rows
        if missing:
            raise ValueError(f"missing {len(missing)} required (session, turn) rows, e.g. {list(missing)[:3]}")
