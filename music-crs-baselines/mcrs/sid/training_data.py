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


def build_raw_conversation_pairs(
    sessions: list[dict[str, Any]],
    track_to_sid: dict[str, tuple[int, int, int]],
    n_turns_window: int = 3,
) -> list[dict[str, Any]]:
    """For each music-role turn in train conversations, emit one (query, SID) pair.

    The query string is built by `format_query_for_sid_input(chat_history_so_far,
    user_query_at_this_turn, user_profile, conversation_goal)`. Chat history is
    windowed to the last n_turns_window turn-pairs.

    Skips turns where the gold track is not in track_to_sid (e.g. test-set tracks).
    """
    pairs: list[dict[str, Any]] = []
    for session in sessions:
        convs = session.get("conversations", [])
        user_profile = session.get("user_profile")
        conversation_goal = session.get("conversation_goal")

        # Walk the conversation in order. At each music-role turn, emit a pair.
        # The chat history at that point is everything BEFORE the current music turn.
        chat_history: list[dict[str, str]] = []
        pending_user_query: Optional[str] = None
        for turn in convs:
            role = turn.get("role")
            content = turn.get("content") or ""
            if role == "user":
                pending_user_query = content
                # Note: don't append yet — wait until we see if this turn produces a music response
            elif role == "music":
                # Emit a pair if we have a pending user query AND the track has a SID
                if pending_user_query is not None and content in track_to_sid:
                    windowed = windowed_chat_history(chat_history, n_turns=n_turns_window)
                    query_str = format_query_for_sid_input(
                        chat_history=windowed,
                        current_user_query=pending_user_query,
                        user_profile=user_profile,
                        conversation_goal=conversation_goal,
                    )
                    c1, c2, c3 = track_to_sid[content]
                    pairs.append({
                        "source": "raw",
                        "track_id": content,
                        "query": query_str,
                        "code_1": c1, "code_2": c2, "code_3": c3,
                    })
                # Append the user query and assistant track summary to chat_history
                if pending_user_query is not None:
                    chat_history.append({"role": "user", "content": pending_user_query})
                # The "assistant" message in chat history is the track ID (W4 inference also
                # uses this — id_to_metadata is applied at inference time, but for SID
                # training the bare ID is sufficient signal of "what was just played")
                chat_history.append({"role": "assistant", "content": content})
                pending_user_query = None
    return pairs
