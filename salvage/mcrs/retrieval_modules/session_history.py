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
            # Prefer an explicit raw track_id (the inference parser expands
            # music-turn content to metadata TEXT, so content is no longer a
            # catalog id; track_id carries the original id). Fall back to
            # content for turns that still hold a raw id (eval harness / tests).
            tid = turn.get("track_id")
            if tid is not None and str(tid) in catalog_tids:
                out.append(str(tid))
                continue
            c = str(turn.get("content", ""))
            if c in catalog_tids:
                out.append(c)
    return out


def session_match_features(cand_meta: dict, played_meta: list[dict]) -> dict:
    """Structural session-continuity features for one candidate.

    SHARED between training (scripts/build_lgbm_features.py) and inference
    (mcrs/rerankers/lgbm_rerank.py) to guarantee identical train/serve features.

    Keys: same_artist, same_album, artist_in_session_count, session_tag_overlap.
    All comparisons are case-insensitive and whitespace-trimmed.
    """
    c_artist = str(cand_meta.get("artist_name") or "").strip().lower()
    c_album = str(cand_meta.get("album_name") or "").strip().lower()
    artists = [str(m.get("artist_name") or "").strip().lower() for m in played_meta]
    albums = [str(m.get("album_name") or "").strip().lower() for m in played_meta]
    c_tags = {str(t).strip().lower() for t in (cand_meta.get("tag_list") or []) if t}
    session_tags: set[str] = set()
    for m in played_meta:
        for t in (m.get("tag_list") or []):
            if t:
                session_tags.add(str(t).strip().lower())
    return {
        "same_artist": int(bool(c_artist) and c_artist in artists),
        "same_album": int(bool(c_album) and c_album in albums),
        "artist_in_session_count": sum(1 for a in artists if a and a == c_artist),
        "session_tag_overlap": len(c_tags & session_tags),
    }
