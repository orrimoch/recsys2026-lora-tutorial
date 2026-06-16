"""Query + track text formatters for BGE-M3 fine-tune + inference.

`format_query_text` delegates to `crs_baseline.build_retrieval_query` for
train/eval parity — fine-tuning on a format the runtime never produces is
the canonical generative-retrieval failure mode (see W3 SID v1 gate
failure). Modes supported: 'raw', 'last_user', 'last_user_with_goal'.

`format_track_text` produces a DELIBERATELY enriched 5-field track string
(track_name | artist_name | album_name | release_date | tag_list) — this
intentionally diverges from BM25's 3-field newline-separated corpus and
is the contract for the new BGE-M3 corpus (per Phase 1 §6 of the plan).
"""
from __future__ import annotations

from typing import Optional

from mcrs.crs_baseline import build_retrieval_query


def _unwrap_field(v):
    """The HF TalkPlay catalog returns string fields as single-element lists
    sometimes (e.g. track_name=['A Forbidden Dance'] instead of 'A Forbidden Dance').
    Unwrap single-element lists to their string value so the formatter doesn't
    emit Python list-repr brackets in the track text (which add noise tokens
    the encoder has to learn to ignore)."""
    if isinstance(v, list):
        if len(v) == 0:
            return None
        if len(v) == 1:
            return v[0]
        # Multi-element list: join with comma (rare; e.g. multi-artist tracks).
        return ", ".join(str(x) for x in v)
    return v


def format_track_text(
    track_name,
    artist_name=None,
    album_name=None,
    release_date=None,
    tag_list=None,
) -> str:
    """5-field pipe-separated track text for the BGE-M3 corpus.

    Format:
        'track_name: X | artist_name: Y | album_name: Z | '
        'release_date: W | tag_list: a, b, c'

    DELIBERATELY diverges from BM25's 3-field newline-separated corpus —
    this is the contract for the new BGE-M3 corpus (per Phase 1 §6). The
    enrichment (release_date + tag_list) is expected to help dense
    retrieval pick up temporal/genre signals BM25 misses.

    Missing optional fields render as the field name followed by 'unknown'
    (or empty for `tag_list`) rather than being dropped, so the field
    order and separator count are stable for every row.

    Single-element lists in any field (a known TalkPlay catalog quirk) are
    unwrapped — so `track_name=['A Forbidden Dance']` renders as
    `track_name: A Forbidden Dance`, NOT `track_name: ['A Forbidden Dance']`.
    """
    tn = _unwrap_field(track_name) or "unknown"
    an = _unwrap_field(artist_name) or "unknown"
    al = _unwrap_field(album_name) or "unknown"
    rd = _unwrap_field(release_date) or "unknown"
    parts = [f"track_name: {tn}"]
    parts.append(f"artist_name: {an}")
    parts.append(f"album_name: {al}")
    parts.append(f"release_date: {rd}")
    # tag_list may be a list of strings (preferred) or a single-element list
    # wrapping a list (rare). Unwrap once, then join.
    tags = tag_list
    if isinstance(tags, list) and len(tags) == 1 and isinstance(tags[0], list):
        tags = tags[0]
    if tags:
        parts.append(f"tag_list: {', '.join(str(t) for t in tags)}")
    else:
        parts.append("tag_list: ")
    return " | ".join(parts)


def format_query_text(
    chat_history: list[dict[str, str]],
    current_user_query: str,
    user_profile: Optional[dict] = None,
    conversation_goal: Optional[dict] = None,
    max_history_turns: int = 6,
    mode: str = "raw",
    state: Optional[dict] = None,
) -> str:
    """Build retriever query by delegating to production's `build_retrieval_query`.

    This is a thin wrapper over `mcrs.crs_baseline.build_retrieval_query`
    so fine-tune and inference share a single source of truth for the
    query string. Supported modes (pass-through to production):

        'raw'                — newline-joined "role: content" across all
                               turns (champion default).
        'last_user'          — just the last user-turn content (no role
                               prefix). Reduces noise for dense encoders.
        'last_user_with_goal' — 'last_user' plus ' || goal: <listener_goal>'
                               when `conversation_goal['listener_goal']` is
                               provided.

    Args:
        chat_history: prior turns as `[{"role": ..., "content": ...}, ...]`.
        current_user_query: the current user utterance. Appended as a
            final `{"role": "user", "content": current_user_query}` turn
            before delegating, so it always counts as the last user turn.
        user_profile: CURRENTLY UNUSED. Retained for signature stability
            with existing callers; production `build_retrieval_query`
            does not consume user_profile. May be revisited in Phase 2 as
            a separate context-prefix feature.
        conversation_goal: dict; `listener_goal` field is forwarded as
            `goal_text` to production. Only consulted when
            `mode == "last_user_with_goal"`.
        max_history_turns: truncate `chat_history` to its last N entries
            BEFORE appending the current_user_query and delegating.
        mode: production query mode; see above. Defaults to 'raw'.

    Returns:
        The query string produced by `build_retrieval_query`.
    """
    # user_profile is forwarded only for mode='bge_m3_structured' (the [USER]
    # block needs age/country/gender). Other modes ignore it.
    trimmed_history = chat_history[-max_history_turns:] if chat_history else []
    session_memory = trimmed_history + [
        {"role": "user", "content": current_user_query}
    ]

    goal_text = ""
    if conversation_goal is not None:
        goal_text = conversation_goal.get("listener_goal", "") or ""

    return build_retrieval_query(
        session_memory,
        mode=mode,
        goal_text=goal_text,
        user_profile=user_profile,
        max_history_turns=max_history_turns,
        state=state,
    )
