"""R1 — causal query construction.

Builds the F2 Query from a TurnContext using only turns 1..t: recency-weighted concat of the
user utterances + the goal, with a token-budget cap that drops OLDEST first and never truncates
the latest utterance or the goal (§8 alignment; the cap value is decided by P0). Pure, no deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from mcrs.contracts import Query, TurnContext


def _ntok(s: str) -> int:
    return len(s.split())


@dataclass
class QueryBuilder:
    context_cap: int = 0      # max whitespace tokens (0 = unbounded)
    recency_window: int = 0   # keep only the last N utterances (0 = all)
    markers: bool = False                                   # enriched template (K3b §4.1)
    taste_items: int = 0                                    # max history tracks in the taste: clause
    track_label_fn: Optional[Callable[[str], Optional[str]]] = field(default=None, repr=False)  # tid -> "artist – title" | None

    def build(self, ctx: TurnContext) -> Query:
        if not self.markers:
            return self._build_plain(ctx)
        return self._build_enriched(ctx)

    def _build_plain(self, ctx: TurnContext) -> Query:
        utts = list(ctx.utterances)
        if self.recency_window and len(utts) > self.recency_window:
            utts = utts[-self.recency_window:]
        goal = ctx.goal or ""
        kept = self._apply_cap(utts, goal)
        parts = kept + ([goal] if goal else [])
        return Query(text=" ".join(p for p in parts if p))

    def _build_enriched(self, ctx: TurnContext) -> Query:
        utts = list(ctx.utterances)
        latest = utts[-1] if utts else ""
        older = utts[:-1]
        if self.recency_window and len(older) > self.recency_window:
            older = older[-self.recency_window:]
        lines: list[str] = []
        if latest:
            lines.append(f"request: {latest}")
        if older:
            lines.append("context: " + " ".join(older))
        if ctx.goal:
            lines.append(f"goal: {ctx.goal}")
        if self.taste_items and ctx.history_tids and self.track_label_fn:
            labels: list[str] = []
            for tid in reversed(ctx.history_tids):          # chronological -> newest-first
                lab = self.track_label_fn(tid)
                if lab:
                    labels.append(lab)
                if len(labels) >= self.taste_items:
                    break
            if labels:
                lines.append("taste: " + "; ".join(labels))
        return Query(text="\n".join(lines))

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
