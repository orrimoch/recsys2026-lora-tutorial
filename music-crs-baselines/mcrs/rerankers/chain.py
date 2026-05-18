"""Chain reranker: runs a list of rerankers in sequence.

Each stage's `topk` shrinks the candidate set; the final stage's output
respects the caller's `topk` (truncated if necessary).
"""
from __future__ import annotations

from typing import Any, Optional


class CHAIN_RERANKER:
    def __init__(self, stages: list[tuple[str, int, Any]]):
        """Args:
            stages: list of (name, topk, reranker_instance) tuples. Each is
                applied in order; reranker outputs are fed to the next stage.
        """
        if not stages:
            raise ValueError("CHAIN_RERANKER requires at least one stage")
        self.stages = stages

    def rerank(
        self,
        queries: list[str],
        candidate_tids: list[list[str]],
        topk: int,
        user_ids: Optional[list[Optional[str]]] = None,
        goal_categories: Optional[list[Optional[str]]] = None,
        goal_specificities: Optional[list[Optional[str]]] = None,
        user_profiles_raw: Optional[list[Any]] = None,
    ) -> list[list[str]]:
        side_channels = {
            "user_ids": user_ids,
            "goal_categories": goal_categories,
            "goal_specificities": goal_specificities,
            "user_profiles_raw": user_profiles_raw,
        }
        current = candidate_tids
        for stage_idx, (name, stage_topk, reranker) in enumerate(self.stages):
            # Final stage respects the caller's topk; earlier stages use their own.
            is_last = stage_idx == len(self.stages) - 1
            this_topk = topk if is_last else stage_topk
            try:
                current = reranker.rerank(queries, current, topk=this_topk, **side_channels)
            except TypeError:
                # Back-compat: reranker predates side-channel kwargs.
                current = reranker.rerank(queries, current, topk=this_topk)
        return current
