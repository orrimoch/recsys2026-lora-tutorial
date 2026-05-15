"""SID training-data builders + the cross-cutting query formatter.

The query formatter `format_query_for_sid_input` is used at BOTH training
(W2) and inference (W4) to build the SID generator's prompt string. Training
and inference MUST produce identical strings for identical inputs — keep all
formatting decisions in this single function.
"""
from __future__ import annotations

from typing import Any, Optional


_MAX_HISTORY_MSG_CHARS = 200


def format_query_for_sid_input(
    chat_history: list[dict[str, str]],
    current_user_query: str,
    user_profile: Optional[dict[str, Any]] = None,
    conversation_goal: Optional[dict[str, Any]] = None,
) -> str:
    """Build the SID generator's input prompt from session components.

    Format (line-delimited blocks; sections omitted if their input is None/empty):
      [USER]: age=<int>, country=<2-letter>, prefers=<musical culture>
      [GOAL]: <listener_goal text>
      [HISTORY]:
        user: <truncated msg>
        assistant: <truncated msg>
        ...
      [QUERY]: <current user message>
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
        parts.append("[HISTORY]:")
        for msg in chat_history:
            role = "user" if msg.get("role") == "user" else "assistant"
            content = msg.get("content", "") or ""
            if len(content) > _MAX_HISTORY_MSG_CHARS:
                content = content[:_MAX_HISTORY_MSG_CHARS]
            parts.append(f"  {role}: {content}")
    parts.append(f"[QUERY]: {current_user_query}")
    return "\n".join(parts)


def windowed_chat_history(
    chat_history: list[dict[str, str]],
    n_turns: Optional[int] = None,
) -> list[dict[str, str]]:
    """Keep last N turn-pairs (= 2N messages) of chat history; no-op if n_turns is None."""
    if n_turns is None or n_turns <= 0:
        return chat_history
    max_msgs = 2 * n_turns
    if len(chat_history) <= max_msgs:
        return chat_history
    return chat_history[-max_msgs:]


def build_metadata_as_query_pairs(
    track_metadata: list[dict[str, Any]],
    track_to_sid: dict[str, tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """For each track in metadata that has a SID assignment, build one
    (query, SID) training pair. Query = concatenated metadata fields.

    Skips tracks not in track_to_sid (orphaned by W1 quantization).
    """
    pairs: list[dict[str, Any]] = []
    for row in track_metadata:
        tid = row["track_id"]
        if tid not in track_to_sid:
            continue
        # Build query from available metadata fields
        parts = []
        for field in ("track_name", "artist_name", "album_name", "tag_list"):
            val = row.get(field)
            if isinstance(val, list) and val:
                rendered = ", ".join(str(v) for v in val if v)
                if rendered:
                    parts.append(f"{field}: {rendered}")
        release = row.get("release_date") or ""
        if release:
            parts.append(f"release_date: {release}")
        query = " | ".join(parts)
        c1, c2, c3 = track_to_sid[tid]
        pairs.append({
            "source": "metadata",
            "track_id": tid,
            "query": query,
            "code_1": c1, "code_2": c2, "code_3": c3,
        })
    return pairs
