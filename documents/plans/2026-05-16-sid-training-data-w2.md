# SID Training Data (W2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the (query, SID-sequence) training corpus that the W3 SID generator will fine-tune on. Combines 3 sources (raw conversations + metadata-as-query + optional doc2query) into a single stratified train/val parquet, ready for W3's PyTorch Dataset.

**Architecture:** Pure-function builders (one per source) emit unified `(source, track_id, query, code_1, code_2, code_3)` rows. An orchestration script reads W1's `track_to_sid.parquet` lookup, runs the builders, concatenates, stratified-splits 95/5 per source, writes train + val parquets. A PyTorch `Dataset` class wraps the train parquet for W3's dataloader.

**Tech Stack:** Python 3.10, `pandas`+`pyarrow` (existing), `datasets` (existing), `torch` (existing — only the Dataset class needs it). No new pip deps.

**Spec reference:** `documents/specs/2026-05-15-sid-retrieval-design.md` §3.2 (training data composition), §3.3 (`format_query_for_sid_input`).

**Inputs from prior weeks**:
- W1 artifact: `experiments/cache/sid/track_to_sid.parquet` — columns `track_id, code_1, code_2, code_3, popularity, bucket_rank`. Verified 47,071 rows, SHA256 `3ed8fe9f930a...`.
- (Optional) `experiments/cache/doc2query/Qwen_Qwen2.5-1.5B-Instruct/queries.parquet` from notebook 54 — columns `track_id, synthetic_queries (list[str])`. If absent, W2 ships without doc2query (~55K pairs instead of ~290K) and the user can re-run later.

---

## File structure

| Path | Type | Responsibility |
|---|---|---|
| `mcrs/sid/training_data.py` | new | Pure functions: `format_query_for_sid_input`, `build_raw_conversation_pairs`, `build_metadata_as_query_pairs`, `build_doc2query_pairs`, `stratified_split`, `windowed_chat_history` |
| `mcrs/sid/generator_dataloader.py` | new | `SIDTrainingDataset(torch.utils.data.Dataset)` — wraps train parquet for W3 |
| `scripts/build_sid_training_data.py` | new | Orchestration: load W1 SID lookup → run 3 builders → concat → stratified split → write train + val parquets |
| `tests/test_sid_training_data.py` | new | TDD tests for the 6 pure functions |
| `tests/test_sid_generator_dataloader.py` | new | TDD tests for the Dataset class |
| `colab/61_build_sid_training_data.ipynb` | new | Colab wrapper: clone, deps, run script, inspect output |

**Output artifacts** (under `data/sid_training/`, NOT cached on Drive — committed to git so W3 has reproducible inputs; they're text-mostly and small):
- `train.parquet` — ~95% of pairs, columns `source, track_id, query, code_1, code_2, code_3`
- `val.parquet` — ~5% of pairs, stratified by `source`, same columns
- `summary.json` — counts per source, total, train/val ratios, hash of W1 SID lookup used

---

## Task 1: Setup — verify W1 artifact + scaffold directory

**Files:**
- Verify: `experiments/cache/sid/track_to_sid.parquet` exists (W1 output)
- Create: `data/sid_training/` (empty directory)

- [ ] **Step 1: Verify W1 SID lookup is on disk**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
ls -la experiments/cache/sid/track_to_sid.parquet 2>&1 || echo "MISSING — W1 must complete first"
```

Expected: file exists. If MISSING, abort — W1 must ship its artifact via Colab notebook 60 + pick_best before W2 can run.

- [ ] **Step 2: Read W1 schema sanity check**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('experiments/cache/sid/track_to_sid.parquet')
print(f'rows: {len(df)}')
print(f'columns: {df.columns.tolist()}')
print(f'sample: {df.head(2).to_dict(orient=\"records\")}')
"
```

Expected: 47,071 rows, columns include `track_id, code_1, code_2, code_3, popularity, bucket_rank`.

- [ ] **Step 3: Create the data output directory**

```bash
mkdir -p data/sid_training
```

- [ ] **Step 4: Commit the empty directory marker**

```bash
touch data/sid_training/.gitkeep
git add data/sid_training/.gitkeep
git commit -m "sid w2: scaffold data/sid_training/ output directory"
```

---

## Task 2: TDD `format_query_for_sid_input` (the cross-cutting query formatter)

This is the single function used at BOTH training (W2) and inference (W4) to convert a session's components into a prompt string for the SID generator. Critical that training and inference produce IDENTICAL strings for the same input.

**Files:**
- Create: `music-crs-baselines/mcrs/sid/training_data.py`
- Create: `tests/test_sid_training_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sid_training_data.py`:

```python
"""Tests for SID training data builders + query formatter."""
from typing import Any


def test_format_query_for_sid_input_includes_user_profile_when_present():
    """User profile keys (age, country_code, preferred_musical_culture) appear in output."""
    from mcrs.sid.training_data import format_query_for_sid_input

    out = format_query_for_sid_input(
        chat_history=[],
        current_user_query="play me something dreamy",
        user_profile={"age": 36, "country_code": "MX", "preferred_musical_culture": "Anglo-American Rock"},
        conversation_goal=None,
    )
    assert "age=36" in out
    assert "country=MX" in out
    assert "Anglo-American Rock" in out
    assert "dreamy" in out


def test_format_query_for_sid_input_includes_conversation_goal_listener_goal():
    """conversation_goal.listener_goal text appears in output."""
    from mcrs.sid.training_data import format_query_for_sid_input

    out = format_query_for_sid_input(
        chat_history=[],
        current_user_query="next track",
        user_profile=None,
        conversation_goal={"category": "F", "listener_goal": "find energetic 90s rock"},
    )
    assert "find energetic 90s rock" in out
    assert "next track" in out


def test_format_query_for_sid_input_omits_user_profile_block_when_none():
    """When user_profile is None, no [USER] line in output."""
    from mcrs.sid.training_data import format_query_for_sid_input

    out = format_query_for_sid_input(
        chat_history=[],
        current_user_query="anything",
        user_profile=None,
        conversation_goal=None,
    )
    assert "[USER]" not in out
    assert "anything" in out


def test_format_query_for_sid_input_includes_chat_history_with_role_labels():
    """chat_history messages render with role + content."""
    from mcrs.sid.training_data import format_query_for_sid_input

    chat = [
        {"role": "user", "content": "I want indie rock"},
        {"role": "assistant", "content": "track_name: Mr Brightside, artist_name: The Killers"},
    ]
    out = format_query_for_sid_input(
        chat_history=chat,
        current_user_query="something newer",
        user_profile=None,
        conversation_goal=None,
    )
    assert "user: I want indie rock" in out
    assert "Mr Brightside" in out
    assert "something newer" in out


def test_format_query_for_sid_input_truncates_long_history_messages():
    """Each history message is truncated to 200 chars to keep prompt tractable."""
    from mcrs.sid.training_data import format_query_for_sid_input

    long_msg = "x" * 300
    chat = [{"role": "user", "content": long_msg}]
    out = format_query_for_sid_input(
        chat_history=chat,
        current_user_query="ok",
        user_profile=None,
        conversation_goal=None,
    )
    # The assistant's content is truncated to 200 chars max
    assert "x" * 300 not in out
    assert "x" * 200 in out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest tests/test_sid_training_data.py -q
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'mcrs.sid.training_data'`.

- [ ] **Step 3: Write the implementation**

Create `music-crs-baselines/mcrs/sid/training_data.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/training_data.py tests/test_sid_training_data.py
git commit -m "sid w2: format_query_for_sid_input — cross-cutting query formatter (TDD, 5 tests)"
```

---

## Task 3: TDD `windowed_chat_history` helper

W4 inference and W2 training data builder both window the chat history to the last N turn-pairs (per Phase 0 finding that long queries lose recall).

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/training_data.py`
- Modify: `tests/test_sid_training_data.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/test_sid_training_data.py`:

```python
def test_windowed_chat_history_returns_last_n_turn_pairs():
    """N=3 keeps the last 6 messages (3 user + 3 assistant pairs)."""
    from mcrs.sid.training_data import windowed_chat_history

    chat = []
    for i in range(5):  # 5 user-assistant turn-pairs = 10 messages
        chat.append({"role": "user", "content": f"u{i}"})
        chat.append({"role": "assistant", "content": f"a{i}"})

    out = windowed_chat_history(chat, n_turns=3)
    assert len(out) == 6
    # Should keep the last 6 (i=2 onwards)
    assert out[0]["content"] == "u2"
    assert out[-1]["content"] == "a4"


def test_windowed_chat_history_no_op_when_n_turns_is_none():
    """n_turns=None preserves full history."""
    from mcrs.sid.training_data import windowed_chat_history

    chat = [{"role": "user", "content": str(i)} for i in range(20)]
    out = windowed_chat_history(chat, n_turns=None)
    assert len(out) == 20


def test_windowed_chat_history_no_op_when_history_short():
    """When len(chat) <= 2*n_turns, no truncation needed."""
    from mcrs.sid.training_data import windowed_chat_history

    chat = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    out = windowed_chat_history(chat, n_turns=3)
    assert out == chat


def test_windowed_chat_history_handles_empty_input():
    from mcrs.sid.training_data import windowed_chat_history
    assert windowed_chat_history([], n_turns=3) == []
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_training_data.py::test_windowed_chat_history_returns_last_n_turn_pairs -q
```

Expected: `ImportError: cannot import name 'windowed_chat_history'`.

- [ ] **Step 3: Implement**

Append to `music-crs-baselines/mcrs/sid/training_data.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: `9 passed` (5 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/training_data.py tests/test_sid_training_data.py
git commit -m "sid w2: windowed_chat_history (TDD, 4 tests; matches notebook 56 windowing)"
```

---

## Task 4: TDD `build_metadata_as_query_pairs`

For each track, build a synthetic query from its catalog metadata text. One pair per track → 47K pairs ensures every track has ≥1 training example.

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/training_data.py`
- Modify: `tests/test_sid_training_data.py`

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_sid_training_data.py`:

```python
def test_build_metadata_as_query_pairs_one_pair_per_track():
    """Returns one (query, sid) pair for each track present in the SID lookup."""
    from mcrs.sid.training_data import build_metadata_as_query_pairs

    track_metadata = [
        {"track_id": "t1", "track_name": ["Bohemian Rhapsody"], "artist_name": ["Queen"], "album_name": ["A Night at the Opera"], "tag_list": ["rock"], "release_date": "1975"},
        {"track_id": "t2", "track_name": ["Yesterday"], "artist_name": ["The Beatles"], "album_name": ["Help!"], "tag_list": ["pop"], "release_date": "1965"},
    ]
    track_to_sid = {"t1": (1, 2, 3), "t2": (4, 5, 6)}

    pairs = build_metadata_as_query_pairs(track_metadata, track_to_sid)
    assert len(pairs) == 2
    by_tid = {p["track_id"]: p for p in pairs}
    assert "Bohemian Rhapsody" in by_tid["t1"]["query"]
    assert "Queen" in by_tid["t1"]["query"]
    assert by_tid["t1"]["code_1"] == 1
    assert by_tid["t1"]["code_2"] == 2
    assert by_tid["t1"]["code_3"] == 3
    assert by_tid["t1"]["source"] == "metadata"


def test_build_metadata_as_query_pairs_skips_tracks_not_in_sid_lookup():
    """If a track has no SID assignment, skip it (don't crash, don't synthesize)."""
    from mcrs.sid.training_data import build_metadata_as_query_pairs

    track_metadata = [
        {"track_id": "t1", "track_name": ["A"], "artist_name": ["B"], "album_name": [""], "tag_list": [], "release_date": ""},
        {"track_id": "t_missing", "track_name": ["C"], "artist_name": ["D"], "album_name": [""], "tag_list": [], "release_date": ""},
    ]
    track_to_sid = {"t1": (1, 2, 3)}  # t_missing absent

    pairs = build_metadata_as_query_pairs(track_metadata, track_to_sid)
    assert len(pairs) == 1
    assert pairs[0]["track_id"] == "t1"


def test_build_metadata_as_query_pairs_handles_empty_metadata_fields():
    """Tracks with empty/missing metadata fields still produce a pair (using whatever's available)."""
    from mcrs.sid.training_data import build_metadata_as_query_pairs

    track_metadata = [
        {"track_id": "t1", "track_name": ["Only Title"], "artist_name": [], "album_name": [], "tag_list": [], "release_date": ""},
    ]
    track_to_sid = {"t1": (1, 2, 3)}

    pairs = build_metadata_as_query_pairs(track_metadata, track_to_sid)
    assert len(pairs) == 1
    assert "Only Title" in pairs[0]["query"]
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: 3 new failures with `ImportError: cannot import name 'build_metadata_as_query_pairs'`.

- [ ] **Step 3: Implement**

Append to `music-crs-baselines/mcrs/sid/training_data.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/training_data.py tests/test_sid_training_data.py
git commit -m "sid w2: build_metadata_as_query_pairs (TDD, 3 tests, ensures catalog coverage)"
```

---

## Task 5: TDD `build_raw_conversation_pairs`

For each turn in the train conversations, build a (windowed-history-prefixed query, SID-of-music-turn) pair.

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/training_data.py`
- Modify: `tests/test_sid_training_data.py`

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_sid_training_data.py`:

```python
def test_build_raw_conversation_pairs_one_pair_per_music_turn():
    """For each music-role turn in conversations, emit one (query, SID) pair."""
    from mcrs.sid.training_data import build_raw_conversation_pairs

    sessions = [
        {
            "session_id": "s1", "user_id": "u1",
            "user_profile": {"age": 30, "country_code": "US", "preferred_musical_culture": "Pop"},
            "conversation_goal": {"category": "F", "listener_goal": "find rock"},
            "conversations": [
                {"turn_number": 1, "role": "user", "content": "play rock"},
                {"turn_number": 1, "role": "music", "content": "t1"},
                {"turn_number": 2, "role": "user", "content": "more"},
                {"turn_number": 2, "role": "music", "content": "t2"},
            ],
        },
    ]
    track_to_sid = {"t1": (1, 2, 3), "t2": (4, 5, 6)}

    pairs = build_raw_conversation_pairs(sessions, track_to_sid, n_turns_window=3)
    assert len(pairs) == 2
    # First pair: user_query="play rock", target=t1's SID=(1,2,3)
    assert "play rock" in pairs[0]["query"]
    assert pairs[0]["code_1"] == 1
    assert pairs[0]["track_id"] == "t1"
    assert pairs[0]["source"] == "raw"
    # Second pair: user_query="more", target=t2's SID=(4,5,6) — should include t1 in chat history
    assert pairs[1]["code_1"] == 4
    assert "more" in pairs[1]["query"]


def test_build_raw_conversation_pairs_skips_music_turns_with_no_sid():
    """If a track in conversations isn't in track_to_sid (e.g. test-only track), skip its pair."""
    from mcrs.sid.training_data import build_raw_conversation_pairs

    sessions = [
        {
            "session_id": "s1", "user_id": "u1",
            "user_profile": None, "conversation_goal": None,
            "conversations": [
                {"turn_number": 1, "role": "user", "content": "play rock"},
                {"turn_number": 1, "role": "music", "content": "t_missing"},
            ],
        },
    ]
    track_to_sid: dict = {}  # t_missing not in lookup

    pairs = build_raw_conversation_pairs(sessions, track_to_sid, n_turns_window=3)
    assert pairs == []


def test_build_raw_conversation_pairs_handles_session_without_user_profile_or_goal():
    """Sessions with None profile/goal still produce pairs (just no [USER]/[GOAL] sections)."""
    from mcrs.sid.training_data import build_raw_conversation_pairs

    sessions = [
        {
            "session_id": "s1", "user_id": "u1",
            "user_profile": None, "conversation_goal": None,
            "conversations": [
                {"turn_number": 1, "role": "user", "content": "anything"},
                {"turn_number": 1, "role": "music", "content": "t1"},
            ],
        },
    ]
    track_to_sid = {"t1": (1, 2, 3)}

    pairs = build_raw_conversation_pairs(sessions, track_to_sid, n_turns_window=3)
    assert len(pairs) == 1
    assert "[USER]" not in pairs[0]["query"]
    assert "anything" in pairs[0]["query"]
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: 3 new failures with `ImportError: cannot import name 'build_raw_conversation_pairs'`.

- [ ] **Step 3: Implement**

Append to `music-crs-baselines/mcrs/sid/training_data.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: `15 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/training_data.py tests/test_sid_training_data.py
git commit -m "sid w2: build_raw_conversation_pairs (TDD, 3 tests, windowed history per Phase 0)"
```

---

## Task 6: TDD `build_doc2query_pairs` (optional source — handle absence gracefully)

If notebook 54's doc2query parquet exists, expand each track into ~5 synthetic-query pairs. If the parquet is missing, return an empty list (caller logs a warning).

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/training_data.py`
- Modify: `tests/test_sid_training_data.py`

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_sid_training_data.py`:

```python
def test_build_doc2query_pairs_one_pair_per_synthetic_query():
    """A track with N synthetic queries produces N (query, SID) pairs."""
    from mcrs.sid.training_data import build_doc2query_pairs

    doc2query_rows = [
        {"track_id": "t1", "synthetic_queries": ["something rock", "energetic 80s", "guitar solo banger"]},
        {"track_id": "t2", "synthetic_queries": ["dreamy ambient"]},
    ]
    track_to_sid = {"t1": (1, 2, 3), "t2": (4, 5, 6)}

    pairs = build_doc2query_pairs(doc2query_rows, track_to_sid)
    assert len(pairs) == 4  # 3 from t1 + 1 from t2
    t1_pairs = [p for p in pairs if p["track_id"] == "t1"]
    assert len(t1_pairs) == 3
    queries = [p["query"] for p in t1_pairs]
    assert "something rock" in queries
    assert all(p["code_1"] == 1 for p in t1_pairs)
    assert all(p["source"] == "doc2query" for p in t1_pairs)


def test_build_doc2query_pairs_skips_tracks_not_in_sid_lookup():
    """If a track has synthetic queries but no SID assignment, skip all its pairs."""
    from mcrs.sid.training_data import build_doc2query_pairs

    doc2query_rows = [
        {"track_id": "t1", "synthetic_queries": ["q1"]},
        {"track_id": "t_missing", "synthetic_queries": ["q2", "q3"]},
    ]
    track_to_sid = {"t1": (1, 2, 3)}

    pairs = build_doc2query_pairs(doc2query_rows, track_to_sid)
    assert len(pairs) == 1
    assert pairs[0]["track_id"] == "t1"


def test_build_doc2query_pairs_handles_empty_synthetic_queries_list():
    """A track with empty synthetic_queries list produces no pairs."""
    from mcrs.sid.training_data import build_doc2query_pairs

    doc2query_rows = [{"track_id": "t1", "synthetic_queries": []}]
    track_to_sid = {"t1": (1, 2, 3)}

    pairs = build_doc2query_pairs(doc2query_rows, track_to_sid)
    assert pairs == []
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: 3 new failures with `ImportError`.

- [ ] **Step 3: Implement**

Append to `music-crs-baselines/mcrs/sid/training_data.py`:

```python
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
        queries = row.get("synthetic_queries") or []
        if tid not in track_to_sid or not queries:
            continue
        c1, c2, c3 = track_to_sid[tid]
        for q in queries:
            pairs.append({
                "source": "doc2query",
                "track_id": tid,
                "query": q,
                "code_1": c1, "code_2": c2, "code_3": c3,
            })
    return pairs
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: `18 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/training_data.py tests/test_sid_training_data.py
git commit -m "sid w2: build_doc2query_pairs (TDD, 3 tests, optional source)"
```

---

## Task 7: TDD `stratified_split`

Split combined pairs DataFrame into train/val with deterministic 95/5 ratio per source.

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/training_data.py`
- Modify: `tests/test_sid_training_data.py`

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_sid_training_data.py`:

```python
def test_stratified_split_preserves_source_proportions():
    """Each source's train/val ratio is approximately 95/5."""
    import pandas as pd
    from mcrs.sid.training_data import stratified_split

    df = pd.DataFrame([
        {"source": "raw", "track_id": f"t{i}", "query": f"q{i}", "code_1": 1, "code_2": 2, "code_3": 3}
        for i in range(100)
    ] + [
        {"source": "metadata", "track_id": f"m{i}", "query": f"mq{i}", "code_1": 4, "code_2": 5, "code_3": 6}
        for i in range(200)
    ])

    train, val = stratified_split(df, val_frac=0.05, seed=42)

    raw_train = (train["source"] == "raw").sum()
    raw_val = (val["source"] == "raw").sum()
    assert raw_train + raw_val == 100
    assert 4 <= raw_val <= 6  # ~5% of 100

    meta_train = (train["source"] == "metadata").sum()
    meta_val = (val["source"] == "metadata").sum()
    assert meta_train + meta_val == 200
    assert 9 <= meta_val <= 11  # ~5% of 200


def test_stratified_split_is_deterministic_with_seed():
    """Same seed produces same split."""
    import pandas as pd
    from mcrs.sid.training_data import stratified_split

    df = pd.DataFrame([
        {"source": "raw", "track_id": f"t{i}", "query": f"q{i}", "code_1": 1, "code_2": 2, "code_3": 3}
        for i in range(50)
    ])

    t1, v1 = stratified_split(df, val_frac=0.10, seed=42)
    t2, v2 = stratified_split(df, val_frac=0.10, seed=42)

    pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
    pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))


def test_stratified_split_handles_single_row_per_source():
    """When a source has only 1 row, it goes to train (val gets 0)."""
    import pandas as pd
    from mcrs.sid.training_data import stratified_split

    df = pd.DataFrame([
        {"source": "raw", "track_id": "t1", "query": "q", "code_1": 1, "code_2": 2, "code_3": 3},
    ])
    train, val = stratified_split(df, val_frac=0.05, seed=42)
    assert len(train) == 1
    assert len(val) == 0


def test_stratified_split_total_rows_preserved():
    """train + val == original df (no rows lost)."""
    import pandas as pd
    from mcrs.sid.training_data import stratified_split

    df = pd.DataFrame([
        {"source": "raw" if i % 2 == 0 else "metadata",
         "track_id": f"t{i}", "query": f"q{i}",
         "code_1": 0, "code_2": 0, "code_3": 0}
        for i in range(40)
    ])
    train, val = stratified_split(df, val_frac=0.20, seed=42)
    assert len(train) + len(val) == 40
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: 4 new failures with `ImportError`.

- [ ] **Step 3: Implement**

Append to `music-crs-baselines/mcrs/sid/training_data.py`:

```python
def stratified_split(
    df: "pd.DataFrame",
    val_frac: float = 0.05,
    seed: int = 42,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Stratified split: each `source` group gets val_frac of its rows in val.

    Deterministic given seed. Sources with <= 1/val_frac rows put 0 rows in val
    (avoids tiny/empty val per-source slices).
    """
    import pandas as pd
    train_parts = []
    val_parts = []
    for source, group in df.groupby("source"):
        n = len(group)
        n_val = int(round(val_frac * n))
        # Shuffle deterministically per source
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_parts.append(shuffled.iloc[:n_val])
        train_parts.append(shuffled.iloc[n_val:])
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    return train, val
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sid_training_data.py -q
```

Expected: `22 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/training_data.py tests/test_sid_training_data.py
git commit -m "sid w2: stratified_split (TDD, 4 tests, deterministic per-source 95/5)"
```

---

## Task 8: Build `scripts/build_sid_training_data.py` orchestration

**Files:**
- Create: `scripts/build_sid_training_data.py`

- [ ] **Step 1: Write the script**

Create `scripts/build_sid_training_data.py`:

```python
"""Build the SID training data parquet by combining raw conversations,
metadata-as-query, and (optional) doc2query sources.

Reads:
  - W1 SID lookup: experiments/cache/sid/track_to_sid.parquet
  - HF: talkpl-ai/TalkPlayData-Challenge-Dataset (split=train)
  - HF: talkpl-ai/TalkPlayData-Challenge-Track-Metadata (split=all_tracks)
  - (Optional) experiments/cache/doc2query/<safe_model>/queries.parquet

Writes:
  - data/sid_training/train.parquet
  - data/sid_training/val.parquet
  - data/sid_training/summary.json

Usage:
    python scripts/build_sid_training_data.py
    python scripts/build_sid_training_data.py --no-doc2query   # skip doc2query
    python scripts/build_sid_training_data.py --max-sessions 100   # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
from datasets import load_dataset

from mcrs.sid.training_data import (
    build_doc2query_pairs,
    build_metadata_as_query_pairs,
    build_raw_conversation_pairs,
    stratified_split,
)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build SID training data.")
    parser.add_argument("--no-doc2query", action="store_true",
                        help="Skip doc2query source even if its parquet exists.")
    parser.add_argument("--n-turns-window", type=int, default=3,
                        help="Chat history window for raw conversation pairs.")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Smoke: cap on train sessions read.")
    parser.add_argument(
        "--doc2query-model", default="Qwen_Qwen2.5-1.5B-Instruct",
        help="Subdir under experiments/cache/doc2query/ to load synthetic queries from.",
    )
    args = parser.parse_args()

    # 1. Load W1 SID lookup
    sid_path = REPO_ROOT / "experiments" / "cache" / "sid" / "track_to_sid.parquet"
    if not sid_path.exists():
        raise SystemExit(f"Missing {sid_path}. Run W1 (notebook 60) first.")
    sid_df = pd.read_parquet(sid_path)
    track_to_sid: dict[str, tuple[int, int, int]] = {
        row.track_id: (int(row.code_1), int(row.code_2), int(row.code_3))
        for row in sid_df.itertuples()
    }
    sid_hash = sha256_of_file(sid_path)
    print(f"[build_sid_training_data] loaded W1 SIDs: {len(track_to_sid)} tracks, "
          f"sha256={sid_hash[:12]}...", file=sys.stderr)

    # 2. Build raw conversation pairs
    print("[build_sid_training_data] loading TalkPlayData-Challenge-Dataset[train]...",
          file=sys.stderr)
    conv_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset", split="train",
    )
    if args.max_sessions is not None:
        conv_ds = conv_ds.select(range(min(args.max_sessions, len(conv_ds))))
    sessions = [dict(row) for row in conv_ds]
    raw_pairs = build_raw_conversation_pairs(
        sessions, track_to_sid, n_turns_window=args.n_turns_window,
    )
    print(f"[build_sid_training_data] raw conversation pairs: {len(raw_pairs)}",
          file=sys.stderr)

    # 3. Build metadata-as-query pairs
    print("[build_sid_training_data] loading Track-Metadata[all_tracks]...",
          file=sys.stderr)
    meta_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks",
    )
    track_metadata = [dict(row) for row in meta_ds]
    meta_pairs = build_metadata_as_query_pairs(track_metadata, track_to_sid)
    print(f"[build_sid_training_data] metadata-as-query pairs: {len(meta_pairs)}",
          file=sys.stderr)

    # 4. (Optional) Doc2query pairs
    doc2query_pairs: list = []
    doc2query_path = (
        REPO_ROOT / "experiments" / "cache" / "doc2query"
        / args.doc2query_model / "queries.parquet"
    )
    if args.no_doc2query:
        print("[build_sid_training_data] --no-doc2query: skipping doc2query source",
              file=sys.stderr)
    elif doc2query_path.exists():
        print(f"[build_sid_training_data] loading doc2query from {doc2query_path}",
              file=sys.stderr)
        d2q_df = pd.read_parquet(doc2query_path)
        d2q_rows = [dict(row) for row in d2q_df.to_dict(orient="records")]
        doc2query_pairs = build_doc2query_pairs(d2q_rows, track_to_sid)
        print(f"[build_sid_training_data] doc2query pairs: {len(doc2query_pairs)}",
              file=sys.stderr)
    else:
        print(f"[build_sid_training_data] WARNING: doc2query parquet not found at "
              f"{doc2query_path}. Shipping without doc2query (run notebook 54 to add).",
              file=sys.stderr)

    # 5. Combine + stratified split
    all_pairs = raw_pairs + meta_pairs + doc2query_pairs
    df = pd.DataFrame(all_pairs)
    print(f"[build_sid_training_data] total pairs: {len(df)} "
          f"(raw={len(raw_pairs)}, metadata={len(meta_pairs)}, doc2query={len(doc2query_pairs)})",
          file=sys.stderr)

    train, val = stratified_split(df, val_frac=args.val_frac, seed=args.seed)

    # 6. Write outputs
    out_dir = REPO_ROOT / "data" / "sid_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    print(f"[build_sid_training_data] wrote {train_path} ({len(train)} rows)",
          file=sys.stderr)
    print(f"[build_sid_training_data] wrote {val_path} ({len(val)} rows)",
          file=sys.stderr)

    # 7. Summary
    summary = {
        "w1_sid_sha256": sid_hash,
        "n_tracks_with_sid": len(track_to_sid),
        "sources": {
            "raw": len(raw_pairs),
            "metadata": len(meta_pairs),
            "doc2query": len(doc2query_pairs),
        },
        "total": len(df),
        "train_rows": len(train),
        "val_rows": len(val),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_turns_window": args.n_turns_window,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[build_sid_training_data] wrote {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax + CLI help**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -c "import ast; ast.parse(open('scripts/build_sid_training_data.py').read()); print('AST OK')"
python scripts/build_sid_training_data.py --help 2>&1 | head -15
```

Expected: AST OK + CLI help showing 5 args.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_sid_training_data.py
git commit -m "sid w2: build_sid_training_data.py orchestration (3 sources -> stratified split)"
```

---

## Task 9: TDD `SIDTrainingDataset` PyTorch Dataset class

W3's training loop will iterate over this. Returns `(query_str, target_codes_tensor)` per item.

**Files:**
- Create: `music-crs-baselines/mcrs/sid/generator_dataloader.py`
- Create: `tests/test_sid_generator_dataloader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sid_generator_dataloader.py`:

```python
"""Tests for SIDTrainingDataset (PyTorch Dataset wrapping the train parquet)."""
import pandas as pd
import pytest


@pytest.fixture
def tiny_train_parquet(tmp_path):
    """A 4-row parquet with the exact schema build_sid_training_data.py emits."""
    df = pd.DataFrame([
        {"source": "raw", "track_id": "t1", "query": "play rock",
         "code_1": 5, "code_2": 10, "code_3": 200},
        {"source": "metadata", "track_id": "t2", "query": "track_name: Yesterday | artist: Beatles",
         "code_1": 7, "code_2": 14, "code_3": 99},
        {"source": "doc2query", "track_id": "t3", "query": "dreamy ambient track",
         "code_1": 3, "code_2": 6, "code_3": 12},
        {"source": "raw", "track_id": "t4", "query": "next",
         "code_1": 0, "code_2": 0, "code_3": 1},
    ])
    path = tmp_path / "train.parquet"
    df.to_parquet(path, index=False)
    return path


def test_sid_training_dataset_length_matches_parquet_row_count(tiny_train_parquet):
    from mcrs.sid.generator_dataloader import SIDTrainingDataset
    ds = SIDTrainingDataset(tiny_train_parquet)
    assert len(ds) == 4


def test_sid_training_dataset_returns_query_string_and_target_codes(tiny_train_parquet):
    """__getitem__ returns (query_str, [code_1, code_2, code_3]) tuple."""
    import torch
    from mcrs.sid.generator_dataloader import SIDTrainingDataset
    ds = SIDTrainingDataset(tiny_train_parquet)
    query, codes = ds[0]
    assert query == "play rock"
    assert isinstance(codes, torch.Tensor)
    assert codes.dtype == torch.long
    assert codes.tolist() == [5, 10, 200]


def test_sid_training_dataset_indexing_handles_all_rows(tiny_train_parquet):
    from mcrs.sid.generator_dataloader import SIDTrainingDataset
    ds = SIDTrainingDataset(tiny_train_parquet)
    for i in range(len(ds)):
        q, c = ds[i]
        assert isinstance(q, str)
        assert c.shape == (3,)


def test_sid_training_dataset_filter_by_source_keeps_only_specified(tiny_train_parquet):
    """Optional filter_source arg keeps only rows from that source."""
    from mcrs.sid.generator_dataloader import SIDTrainingDataset
    ds = SIDTrainingDataset(tiny_train_parquet, filter_source="raw")
    assert len(ds) == 2  # 2 raw rows in the fixture
    queries = [ds[i][0] for i in range(len(ds))]
    assert "play rock" in queries
    assert "next" in queries
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_generator_dataloader.py -q
```

Expected: 4 failures with `ModuleNotFoundError: No module named 'mcrs.sid.generator_dataloader'`.

- [ ] **Step 3: Implement**

Create `music-crs-baselines/mcrs/sid/generator_dataloader.py`:

```python
"""PyTorch Dataset wrapping the W2 training parquet for W3 generator fine-tune."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class SIDTrainingDataset(Dataset):
    """Lightweight Dataset over a SID training parquet.

    __getitem__ returns (query_string, target_codes_tensor[3, dtype=long]).
    The W3 trainer is responsible for tokenizing the query string and
    constructing the loss against the 3 target codes (which map to the
    expanded vocabulary of SID tokens).
    """

    def __init__(
        self,
        parquet_path,
        filter_source: Optional[str] = None,
    ) -> None:
        df = pd.read_parquet(parquet_path)
        if filter_source is not None:
            df = df[df["source"] == filter_source].reset_index(drop=True)
        self.queries: list[str] = df["query"].tolist()
        codes = df[["code_1", "code_2", "code_3"]].to_numpy(dtype="int64")
        self.codes = torch.from_numpy(codes)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.queries[idx], self.codes[idx]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sid_generator_dataloader.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/generator_dataloader.py tests/test_sid_generator_dataloader.py
git commit -m "sid w2: SIDTrainingDataset (TDD, 4 tests, PyTorch Dataset wrapping train parquet)"
```

---

## Task 10: Build `colab/61_build_sid_training_data.ipynb`

**Files:**
- Create: `colab/61_build_sid_training_data.ipynb`

- [ ] **Step 1: Write the notebook**

```bash
cat > /Users/orrimoch/PythonProjs/recsys2026/colab/61_build_sid_training_data.ipynb <<'NB_EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 61 — SID Training Data (W2)\n",
    "\n",
    "Combines raw conversations + metadata-as-query + (optional) doc2query into a single stratified train/val parquet for W3 generator fine-tune.\n",
    "\n",
    "**Inputs**:\n",
    "- W1 artifact `experiments/cache/sid/track_to_sid.parquet` (must exist on Drive — run notebook 60 first)\n",
    "- HF `talkpl-ai/TalkPlayData-Challenge-Dataset[train]`\n",
    "- HF `talkpl-ai/TalkPlayData-Challenge-Track-Metadata[all_tracks]`\n",
    "- (Optional) `experiments/cache/doc2query/.../queries.parquet` (run notebook 54 first if you want this source)\n",
    "\n",
    "**Outputs** (committed to git, NOT cached on Drive — text-mostly, small):\n",
    "- `data/sid_training/train.parquet`\n",
    "- `data/sid_training/val.parquet`\n",
    "- `data/sid_training/summary.json`\n",
    "\n",
    "**Wallclock**: ~5–10 min on L4 (no GPU needed; mostly HF dataset download + dataframe building)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1) Clone fresh-model.\n",
    "BRANCH = 'fresh-model'\n",
    "!rm -rf /content/recsys2026\n",
    "!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026\n",
    "%cd /content/recsys2026"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2) HF auth + Drive mount + symlinks for both SID + doc2query caches.\n",
    "import os\n",
    "from google.colab import userdata, drive\n",
    "os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n",
    "drive.mount('/content/drive')\n",
    "os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'\n",
    "\n",
    "for repo_path, drive_subdir in [\n",
    "    ('experiments/cache/sid', 'recsys2026_sid_cache'),\n",
    "    ('experiments/cache/doc2query', 'recsys2026_doc2query_cache'),\n",
    "]:\n",
    "    drive_path = f'/content/drive/MyDrive/{drive_subdir}'\n",
    "    full_repo = f'/content/recsys2026/{repo_path}'\n",
    "    os.makedirs(drive_path, exist_ok=True)\n",
    "    os.makedirs(os.path.dirname(full_repo), exist_ok=True)\n",
    "    if os.path.lexists(full_repo):\n",
    "        !rm -rf {full_repo}\n",
    "    !ln -s {drive_path} {full_repo}\n",
    "    print(f'symlinked {full_repo} -> {drive_path}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3) Install deps (no GPU work — minimal stack).\n",
    "!pip install -q --upgrade datasets 'pandas<3.0' tqdm pyarrow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4) Verify W1 SID lookup is on Drive.\n",
    "import os\n",
    "sid_path = 'experiments/cache/sid/track_to_sid.parquet'\n",
    "if not os.path.exists(sid_path):\n",
    "    raise FileNotFoundError(f'{sid_path} missing — run notebook 60 (W1) first')\n",
    "import pandas as pd\n",
    "sid_df = pd.read_parquet(sid_path)\n",
    "print(f'W1 SID lookup: {len(sid_df)} tracks, columns={sid_df.columns.tolist()}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5) (Optional smoke) Build training data on first 100 train sessions only — fast sanity check.\n",
    "!python scripts/build_sid_training_data.py --max-sessions 100 --no-doc2query"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5b) Inspect the smoke output.\n",
    "import json, pandas as pd\n",
    "summary = json.load(open('data/sid_training/summary.json'))\n",
    "print(json.dumps(summary, indent=2))\n",
    "train = pd.read_parquet('data/sid_training/train.parquet')\n",
    "val = pd.read_parquet('data/sid_training/val.parquet')\n",
    "print(f'\\ntrain shape: {train.shape}')\n",
    "print(f'val shape: {val.shape}')\n",
    "print('\\ntrain head:')\n",
    "print(train.head(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6) Full run — all train sessions + metadata + (doc2query if present).\n",
    "# Wallclock: ~5-10 min mostly HF dataset download (cached after first run).\n",
    "!python scripts/build_sid_training_data.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7) Final inspection.\n",
    "import json, pandas as pd\n",
    "summary = json.load(open('data/sid_training/summary.json'))\n",
    "print(json.dumps(summary, indent=2))\n",
    "print('\\nper-source counts in train:')\n",
    "train = pd.read_parquet('data/sid_training/train.parquet')\n",
    "print(train['source'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8) Commit + push so W3 has reproducible inputs (training data is text, ~10-50 MB).\n",
    "!git config user.email 'orrimoch@gmail.com'\n",
    "!git config user.name 'Or Rimoch (Colab)'\n",
    "!git add data/sid_training/train.parquet data/sid_training/val.parquet data/sid_training/summary.json\n",
    "!git commit -m 'sid w2: training data parquet (raw + metadata + optional doc2query)'\n",
    "# !git push origin {BRANCH}   # uncomment when ready"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
NB_EOF
```

- [ ] **Step 2: Validate notebook JSON**

```bash
python -c "import json; nb=json.load(open('colab/61_build_sid_training_data.ipynb')); print(f'OK, {len(nb[\"cells\"])} cells')"
```

Expected: `OK, 9 cells`.

- [ ] **Step 3: Commit + push**

```bash
git add colab/61_build_sid_training_data.ipynb
git commit -m "sid w2: notebook 61 (smoke + full + commit-back)"
git push origin fresh-model
```

---

## Task 11: Final test sweep + code-reviewer agent

- [ ] **Step 1: Run all SID tests**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest tests/test_sid_preprocessing.py tests/test_sid_validation.py tests/test_sid_quantizer.py tests/test_sid_training_data.py tests/test_sid_generator_dataloader.py -q
```

Expected: 32 (W1) + 22 (W2 training_data) + 4 (W2 dataloader) = **58 SID tests passing**.

- [ ] **Step 2: Run the full repo test sweep**

```bash
python -m pytest tests/ -q --ignore=tests/test_local_eval.py --ignore=tests/test_wave0_integration.py --ignore=tests/test_wave1_integration.py --ignore=tests/test_wave2_integration.py 2>&1 | tail -3
```

Expected: ~420+ tests passing (392 baseline + 32 W1 + 26 W2 = 450 give or take).

- [ ] **Step 3: Verify clean git state + push**

```bash
git status -s   # should be clean
git log --oneline -12   # should show ~10 W2 commits
git push origin fresh-model
```

- [ ] **Step 4: Invoke code-reviewer agent**

Dispatch via Agent tool:

```
Agent({
  description: "Review SID W2 training data build",
  subagent_type: "superpowers:code-reviewer",
  prompt: "Review the W2 implementation of SID training data prep against the design spec and W1's actual artifacts.

  Spec: documents/specs/2026-05-15-sid-retrieval-design.md §3.2-§3.3
  Plan: documents/plans/2026-05-16-sid-training-data-w2.md
  W1 artifact contract: experiments/cache/sid/track_to_sid.parquet has 47K rows, 3017 unique SIDs, 2037 collision buckets per project_w1_sid_quantizer_implementation.md

  Files to review (newly added on fresh-model):
  - music-crs-baselines/mcrs/sid/training_data.py
  - music-crs-baselines/mcrs/sid/generator_dataloader.py
  - scripts/build_sid_training_data.py
  - tests/test_sid_training_data.py
  - tests/test_sid_generator_dataloader.py
  - colab/61_build_sid_training_data.ipynb

  Check:
  1. Does format_query_for_sid_input match the spec §3.3 contract for both training and inference?
  2. Are the 3 builders correctly handling W1's quirk (3017 SIDs / collision buckets averaging 22 tracks)? Specifically: do tracks NOT in track_to_sid get correctly skipped without crashing?
  3. Does stratified_split actually preserve source proportions or could rounding edge cases drop a source from val entirely?
  4. Is the dataloader's tensor return shape (3,) consistent with what W3's generator will expect (3 SID tokens per query)?
  5. The Colab notebook symlinks BOTH sid + doc2query cache dirs to Drive — does the path layout match what build_sid_quantizer expected from W1?
  6. Any silent failure modes when doc2query is absent (warning logged but not blocking)?

  Report: APPROVE / APPROVE-WITH-CHANGES / NEEDS-MAJOR-REVISION + numbered findings + top-5 changes if any."
})
```

- [ ] **Step 5: Address any reviewer findings**

If APPROVE → done.
If APPROVE-WITH-CHANGES → make the small fixes, commit, re-push.
If NEEDS-MAJOR-REVISION → triage findings before user runs notebook 61.

---

## Self-review checklist (per writing-plans skill)

**1. Spec coverage** (against §3.2 + §3.3 of design spec):

- §3.2 raw conversation pairs (~8K) — Task 5 ✓
- §3.2 metadata-as-query (47K) — Task 4 ✓
- §3.2 doc2query (~235K, optional) — Task 6 ✓
- §3.2 stratified split per source — Task 7 ✓
- §3.3 `format_query_for_sid_input` exact signature + format — Task 2 ✓
- §3.3 windowed history (last N=3 turn-pairs) — Task 3 ✓
- §3.3 used at BOTH training (W2) and inference (W4) — confirmed by single function in mcrs.sid.training_data, used both by builder (W2) and intended for inference (W4 will import same function)

**2. Placeholder scan**: searched for "TBD", "TODO", "implement later" — none. Every test has actual assertions; every code block is runnable.

**3. Type consistency**:
- `track_to_sid: dict[str, tuple[int, int, int]]` — used consistently in Tasks 4, 5, 6, 8
- Pair dict schema `{source, track_id, query, code_1, code_2, code_3}` — emitted identically by all 3 builders
- `SIDTrainingDataset.__getitem__ → (str, torch.Tensor[3, long])` — matches the train.parquet schema

**4. Sequencing**: each task only depends on artifacts from earlier tasks. Task 8 (orchestration) imports from Tasks 2-7 (training_data.py functions). Task 9 (dataloader) reads parquet that Task 8 writes. Task 10 (notebook) calls Task 8's script. Task 11 reviews everything.

**5. Plan-deviation hooks built in**:
- Task 6 (doc2query) gracefully handles parquet absence — W2 ships without it if user hasn't run notebook 54
- Task 8's --no-doc2query flag explicit override
- Stratified split tested against single-row-per-source edge case (Task 7)

**Plan complete and ready for execution.**

---

## Estimated wallclock

| Component | Time |
|---|---|
| Task 1 (setup) | 2 min |
| Tasks 2-7 (TDD pure functions) | ~15-20 min via subagent |
| Task 8 (orchestration script) | ~5 min via subagent |
| Task 9 (dataloader + tests) | ~5 min via subagent |
| Task 10 (notebook) | ~3 min via subagent |
| Task 11 (final review) | ~10 min agent + iteration |
| **Total my coordination time** | **~40-60 min** |
| User Colab time (notebook 61) | **~5-10 min** (no GPU needed) |

W2 is dramatically lighter than W1 — no GPU compute (just dataframe building + HF dataset reads), no failure modes from VQ training, no codebook collapse to debug. After this, W3 (generator fine-tune) is back to GPU territory.
