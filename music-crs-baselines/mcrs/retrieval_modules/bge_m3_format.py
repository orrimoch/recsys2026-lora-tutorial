"""Query + track text formatters for BGE-M3 fine-tune + inference.

CRITICAL: format_query_text() must produce the EXACT same string as
crs_baseline.batch_chat builds at inference. This is the train/eval parity
contract — fine-tuning with one format and inferring with another is the
canonical generative-retrieval failure mode.
"""
from __future__ import annotations

from typing import Optional


def format_track_text(
    track_name: str,
    artist_name: Optional[str] = None,
    album_name: Optional[str] = None,
    release_date: Optional[str] = None,
    tag_list: Optional[list[str]] = None,
) -> str:
    """5-field track text matching BM25 corpus.

    Format: 'track_name: X | artist_name: Y | album_name: Z | release_date: W | tag_list: a, b, c'
    Missing fields render as the field name followed by 'unknown'.
    """
    parts = [f"track_name: {track_name}"]
    parts.append(f"artist_name: {artist_name or 'unknown'}")
    parts.append(f"album_name: {album_name or 'unknown'}")
    parts.append(f"release_date: {release_date or 'unknown'}")
    if tag_list:
        parts.append(f"tag_list: {', '.join(tag_list)}")
    else:
        parts.append("tag_list: ")
    return " | ".join(parts)


def format_query_text(
    chat_history: list[dict[str, str]],
    current_user_query: str,
    user_profile: Optional[dict] = None,
    conversation_goal: Optional[dict] = None,
    max_history_turns: int = 6,
) -> str:
    """Query text matching crs_baseline.batch_chat's runtime construction.

    Layout:
        [USER]: age=X, country=Y, prefers=Z
        [GOAL]: listener_goal_text
        [HISTORY]: U: prev_user_msg / A: prev_assistant_msg / ...
        [QUERY]: current_user_query
    """
    parts: list[str] = []
    if user_profile is not None:
        parts.append(
            f"[USER]: age={user_profile.get('age', '?')}, "
            f"country={user_profile.get('country_code', '?')}, "
            f"prefers={user_profile.get('preferred_musical_culture', '?')}"
        )
    if conversation_goal is not None:
        goal = conversation_goal.get("listener_goal", "")
        if goal:
            parts.append(f"[GOAL]: {goal}")
    if chat_history:
        msgs = chat_history[-max_history_turns:]
        hist_parts = []
        for m in msgs:
            role = "U" if m.get("role") == "user" else "A"
            hist_parts.append(f"{role}: {m.get('content', '')}")
        parts.append("[HISTORY]: " + " / ".join(hist_parts))
    parts.append(f"[QUERY]: {current_user_query}")
    return "\n".join(parts)
