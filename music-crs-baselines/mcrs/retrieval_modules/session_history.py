"""Recover the list of track_ids already played in a session, for the
session-aware recall channels (same-artist, session-CF).

Prefers an explicit batch_context['history_tids'] (raw played track_ids,
populated by crs_baseline + the eval harness). Falls back to scanning
chat_history for music-turn contents that are valid catalog track_ids
(works only when music turns carry raw ids, not expanded text)."""
from __future__ import annotations

from typing import Optional


def played_tids_from_context(ctx: Optional[dict], catalog_tids: set) -> list[str]:
    if not ctx:
        return []
    explicit = ctx.get("history_tids")
    if explicit:
        return [str(t) for t in explicit if str(t) in catalog_tids]
    out: list[str] = []
    for turn in ctx.get("chat_history", []) or []:
        if turn.get("role") in ("music", "assistant"):
            c = str(turn.get("content", ""))
            if c in catalog_tids:
                out.append(c)
    return out
