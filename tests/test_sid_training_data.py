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


def test_build_doc2query_pairs_handles_numpy_object_array_queries():
    """Regression: pd.read_parquet returns list[str] columns as numpy object arrays.
    Earlier code used `row.get('synthetic_queries') or []` which raised
    'ambiguous truth value' on numpy arrays. Builder must coerce to list first.
    """
    import numpy as np
    from mcrs.sid.training_data import build_doc2query_pairs

    rows = [
        {"track_id": "t_ok", "synthetic_queries": np.array(["q1", "q2", "q3"], dtype=object)},
        {"track_id": "t_empty", "synthetic_queries": np.array([], dtype=object)},
        {"track_id": "t_none", "synthetic_queries": None},
        {"track_id": "t_orphan", "synthetic_queries": np.array(["x"], dtype=object)},
    ]
    track_to_sid = {"t_ok": (1, 2, 3), "t_empty": (4, 5, 6), "t_none": (7, 8, 9)}

    pairs = build_doc2query_pairs(rows, track_to_sid)

    assert len(pairs) == 3
    assert all(p["track_id"] == "t_ok" for p in pairs)
    assert {p["query"] for p in pairs} == {"q1", "q2", "q3"}
    assert {p["source"] for p in pairs} == {"doc2query"}


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


def test_build_raw_conversation_pairs_emits_session_id():
    """Each raw pair carries a session_id derived from the input session dict."""
    from mcrs.sid.training_data import build_raw_conversation_pairs

    sessions = [
        {
            "session_id": "sess-A",
            "conversations": [
                {"role": "user", "content": "play indie rock"},
                {"role": "music", "content": "track-1"},
                {"role": "user", "content": "more"},
                {"role": "music", "content": "track-2"},
            ],
        },
        {
            "id": "sess-B",  # alternative key
            "conversations": [
                {"role": "user", "content": "jazz"},
                {"role": "music", "content": "track-3"},
            ],
        },
        {
            # no session id at all → synthetic fallback
            "conversations": [
                {"role": "user", "content": "x"},
                {"role": "music", "content": "track-1"},
            ],
        },
    ]
    track_to_sid = {"track-1": (1, 2, 3), "track-2": (4, 5, 6), "track-3": (7, 8, 9)}
    pairs = build_raw_conversation_pairs(sessions, track_to_sid, n_turns_window=3)

    sids = {p["session_id"] for p in pairs}
    assert "sess-A" in sids
    assert "sess-B" in sids
    # Synthetic fallback for the 3rd session (index 2)
    assert any(s.startswith("session_") for s in sids)
    # First session should have 2 pairs both tagged sess-A
    sess_a_pairs = [p for p in pairs if p["session_id"] == "sess-A"]
    assert len(sess_a_pairs) == 2


def test_build_metadata_pairs_emit_session_id_none():
    """Metadata pairs include session_id=None (uniform schema)."""
    from mcrs.sid.training_data import build_metadata_as_query_pairs

    rows = [{"track_id": "t1", "track_name": "Yesterday", "artist_name": "Beatles"}]
    pairs = build_metadata_as_query_pairs(rows, {"t1": (1, 2, 3)})
    assert pairs[0]["session_id"] is None


def test_build_doc2query_pairs_emit_session_id_none():
    """Doc2query pairs include session_id=None (uniform schema)."""
    from mcrs.sid.training_data import build_doc2query_pairs

    rows = [{"track_id": "t1", "synthetic_queries": ["q1", "q2"]}]
    pairs = build_doc2query_pairs(rows, {"t1": (1, 2, 3)})
    assert all(p["session_id"] is None for p in pairs)


def test_stratified_split_group_by_keeps_session_in_one_partition():
    """Group-level split: every session_id ends up entirely in train OR entirely in val,
    never split across both. This is the data-leakage fix."""
    import pandas as pd
    from mcrs.sid.training_data import stratified_split

    # 20 sessions × 5 turns each = 100 raw rows + 50 metadata rows
    rows = []
    for s in range(20):
        for t in range(5):
            rows.append({
                "source": "raw", "session_id": f"sess-{s}",
                "track_id": f"t{s}-{t}", "query": "q",
                "code_1": 1, "code_2": 2, "code_3": 3,
            })
    for m in range(50):
        rows.append({
            "source": "metadata", "session_id": None,
            "track_id": f"meta-{m}", "query": "mq",
            "code_1": 4, "code_2": 5, "code_3": 6,
        })
    df = pd.DataFrame(rows)

    train, val = stratified_split(
        df, val_frac=0.20, seed=42,
        group_by={"raw": "session_id"},
    )
    train_raw_sessions = set(train[train["source"] == "raw"]["session_id"])
    val_raw_sessions = set(val[val["source"] == "raw"]["session_id"])

    # No session appears in both partitions — the leakage fix
    assert train_raw_sessions.isdisjoint(val_raw_sessions), (
        f"Session leak: {train_raw_sessions & val_raw_sessions}"
    )
    # ~20% of 20 sessions = 4 in val
    assert len(val_raw_sessions) == 4

    # Metadata source uses row-level split (group_by doesn't apply)
    meta_train = (train["source"] == "metadata").sum()
    meta_val = (val["source"] == "metadata").sum()
    assert meta_train + meta_val == 50
    assert 9 <= meta_val <= 11  # ~20% of 50
