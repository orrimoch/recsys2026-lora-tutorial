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
