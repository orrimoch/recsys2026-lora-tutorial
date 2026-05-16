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
        # Build query from available metadata fields. Handle BOTH list-typed
        # (e.g., track_name=["Yesterday"]) AND scalar-string (e.g., track_name="Yesterday")
        # variants — the dataset has both depending on field/version.
        parts = []
        for field in ("track_name", "artist_name", "album_name", "tag_list"):
            val = row.get(field)
            if isinstance(val, list) and val:
                rendered = ", ".join(str(v) for v in val if v)
            elif isinstance(val, str) and val:
                rendered = val
            else:
                rendered = ""
            if rendered:
                parts.append(f"{field}: {rendered}")
        release = row.get("release_date") or ""
        if release:
            parts.append(f"release_date: {release}")
        query = " | ".join(parts)
        # Skip tracks with completely empty metadata — emitting an empty query
        # would teach the SID generator to map empty input to this SID, which
        # corrupts cold-start behavior at inference. Caller logs the count.
        if not query:
            continue
        c1, c2, c3 = track_to_sid[tid]
        pairs.append({
            "source": "metadata",
            "session_id": None,   # uniform schema; metadata has no session structure
            "track_id": tid,
            "query": query,
            "code_1": c1, "code_2": c2, "code_3": c3,
        })
    return pairs


def _extract_session_id(session: dict[str, Any], fallback_idx: int) -> str:
    """Find a session-level identifier. Tries session_id, id, conversation_id;
    falls back to a synthetic 'session_<idx>' so every session always has one."""
    for key in ("session_id", "id", "conversation_id"):
        val = session.get(key)
        if val:
            return str(val)
    return f"session_{fallback_idx}"


def build_raw_conversation_pairs(
    sessions: list[dict[str, Any]],
    track_to_sid: dict[str, tuple[int, int, int]],
    n_turns_window: int = 3,
) -> list[dict[str, Any]]:
    """For each music-role turn in train conversations, emit one (query, SID) pair.

    The query string is built by `format_query_for_sid_input(chat_history_so_far,
    user_query_at_this_turn, user_profile, conversation_goal)`. Chat history is
    windowed to the last n_turns_window turn-pairs.

    Each pair carries a `session_id` so downstream stratified_split can keep all
    turns from one session in the same partition (avoids train↔val leakage where
    the model sees session-N's earlier turns in train and a later turn in val).

    Skips turns where the gold track is not in track_to_sid (e.g. test-set tracks).
    """
    pairs: list[dict[str, Any]] = []
    for sess_idx, session in enumerate(sessions):
        convs = session.get("conversations", [])
        user_profile = session.get("user_profile")
        conversation_goal = session.get("conversation_goal")
        session_id = _extract_session_id(session, sess_idx)

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
                        "session_id": session_id,
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


def build_doc2query_pairs(
    doc2query_rows: list[dict[str, Any]],
    track_to_sid: dict[str, tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """For each track with synthetic queries (from notebook 54's doc2query
    output), emit one (query, SID) pair per synthetic query.

    Skips tracks not in track_to_sid; skips empty query lists.
    """
    pairs: list[dict[str, Any]] = []
    for row in doc2query_rows:
        tid = row.get("track_id")
        raw_queries = row.get("synthetic_queries")
        # Parquet stores list[str] as a numpy object-array; `array or []` raises
        # "ambiguous truth value". Coerce to a plain Python list before testing.
        if raw_queries is None:
            continue
        if hasattr(raw_queries, "tolist"):
            queries = raw_queries.tolist()
        else:
            queries = list(raw_queries)
        if tid not in track_to_sid or len(queries) == 0:
            continue
        c1, c2, c3 = track_to_sid[tid]
        for q in queries:
            pairs.append({
                "source": "doc2query",
                "session_id": None,   # uniform schema; doc2query has no session structure
                "track_id": tid,
                "query": q,
                "code_1": c1, "code_2": c2, "code_3": c3,
            })
    return pairs


def subsample_one_turn_per_session(
    df: "pd.DataFrame",
    *,
    source: str = "raw",
    session_col: str = "session_id",
    seed: int = 42,
) -> "pd.DataFrame":
    """For rows of `source`, keep exactly ONE row per `session_id` (random, seeded).

    Mirrors Blind-A's structure (80 unique sessions × 1 turn each) so val nDCG@20
    is a structurally honest estimator of Blind-A nDCG@20. Without this, val has
    multiple turns per session and biases the metric (different chat-history-length
    distribution than Blind-A's 1-turn-per-session shape).

    Rows from other sources (metadata, doc2query) are left untouched (they have
    no session structure and Blind-A has no analogue for them anyway).
    """
    import pandas as pd

    if df.empty or session_col not in df.columns:
        return df.reset_index(drop=True)
    src_mask = df["source"] == source
    src_rows = df[src_mask]
    other_rows = df[~src_mask]
    if src_rows.empty:
        return df.reset_index(drop=True)
    # Sample 1 row per non-null session_id deterministically.
    sampled = (
        src_rows.dropna(subset=[session_col])
        .groupby(session_col, group_keys=False, sort=False)
        .apply(lambda g: g.sample(n=1, random_state=seed))
    )
    return pd.concat([sampled, other_rows], ignore_index=True)


def stratified_split(
    df: "pd.DataFrame",
    val_frac: float = 0.05,
    seed: int = 42,
    group_by: Optional[dict[str, str]] = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Stratified split: each `source` group gets val_frac in val.

    By default, splits at row level per source. Pass `group_by={source: column}`
    to split at GROUP level for that source — all rows sharing a group value go
    to the same partition. Use this for the 'raw' source with column='session_id'
    to avoid train↔val leakage (turns from the same session would otherwise be
    spread across both partitions, and the model could memorize via overlapping
    chat history).

    Deterministic given seed. Sources with <= 1/val_frac rows put 0 rows in val
    (avoids tiny/empty val per-source slices).
    """
    import numpy as np
    import pandas as pd

    group_by = group_by or {}
    train_parts: list = []
    val_parts: list = []
    for source, group in df.groupby("source"):
        gcol = group_by.get(source)
        if gcol is None or gcol not in group.columns or group[gcol].isna().all():
            # Row-level split (current behavior; correct when rows are independent).
            n = len(group)
            n_val = int(round(val_frac * n))
            shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            val_parts.append(shuffled.iloc[:n_val])
            train_parts.append(shuffled.iloc[n_val:])
        else:
            # Group-level split: pick whole groups for val. All rows of each group
            # go to the same partition.
            unique_groups = group[gcol].dropna().unique()
            n_val_groups = int(round(val_frac * len(unique_groups)))
            rng = np.random.default_rng(seed)
            shuffled_groups = rng.permutation(unique_groups)
            val_groups = set(shuffled_groups[:n_val_groups].tolist())
            val_mask = group[gcol].isin(val_groups)
            val_parts.append(group[val_mask].reset_index(drop=True))
            train_parts.append(group[~val_mask].reset_index(drop=True))
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    return train, val
