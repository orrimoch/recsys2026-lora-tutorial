"""R1 — causal query construction.

Builds the F2 Query from a TurnContext using only turns 1..t: recency-weighted concat of the
user utterances + the goal, with a token-budget cap that drops OLDEST first and never truncates
the latest utterance or the goal (§8 alignment; the cap value is decided by P0). Pure, no deps.
"""
from __future__ import annotations

from dataclasses import dataclass

from mcrs.contracts import Query, TurnContext


def _ntok(s: str) -> int:
    return len(s.split())


@dataclass
class QueryBuilder:
    context_cap: int = 0      # max whitespace tokens (0 = unbounded)
    recency_window: int = 0   # keep only the last N utterances (0 = all)

    def build(self, ctx: TurnContext) -> Query:
        utts = list(ctx.utterances)
        if self.recency_window and len(utts) > self.recency_window:
            utts = utts[-self.recency_window:]
        goal = ctx.goal or ""
        kept = self._apply_cap(utts, goal)
        parts = kept + ([goal] if goal else [])
        return Query(text=" ".join(p for p in parts if p))

    def _apply_cap(self, utts: list[str], goal: str) -> list[str]:
        if not self.context_cap or not utts:
            return utts
        latest = utts[-1]
        budget = self.context_cap - _ntok(goal) - _ntok(latest)
        kept_head: list[str] = []
        for u in reversed(utts[:-1]):           # newest-of-the-older first
            if budget - _ntok(u) < 0:
                break
            budget -= _ntok(u)
            kept_head.insert(0, u)
        return kept_head + [latest]
