"""Tests for build_bi_encoder_training_data.py orchestration."""
import pytest


def test_build_triples_for_row_writes_query_pos_neg():
    """Each output row is a dict with 'query', 'pos' (list), 'neg' (list)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row

    row = {
        "session_id": "s1",
        "user_id": "u1",
        "turn_number": 1,
        "current_user_query": "play me 70s rock",
        "chat_history": [],
        "user_profile_raw": {"age": 30, "country_code": "US"},
        "conversation_goal": {"listener_goal": "discover music"},
        "track_id": "t_gold",
    }
    track_text_map = {
        "t_gold": "track_name: gold | artist: a | album: b | release_date: 2023 | tag_list: rock",
        "t_neg1": "track_name: neg1 | ...",
        "t_neg2": "track_name: neg2 | ...",
    }
    triple = build_triples_for_row(
        row, gold_track_id="t_gold", neg_track_ids=["t_neg1", "t_neg2"],
        track_text_map=track_text_map,
    )
    assert "query" in triple
    assert "pos" in triple and isinstance(triple["pos"], list) and len(triple["pos"]) == 1
    assert triple["pos"][0] == track_text_map["t_gold"]
    assert "neg" in triple and len(triple["neg"]) == 2
    assert triple["neg"][0] == track_text_map["t_neg1"]


def test_build_triples_for_row_drops_negs_not_in_map():
    """Negatives whose track_id isn't in the text map are silently dropped (catalog drift)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {"session_id": "s", "current_user_query": "q", "chat_history": [],
           "user_profile_raw": None, "conversation_goal": None, "track_id": "t1"}
    track_text_map = {"t1": "gold", "t_neg1": "n1"}  # t_neg2 missing
    triple = build_triples_for_row(
        row, gold_track_id="t1", neg_track_ids=["t_neg1", "t_neg2"],
        track_text_map=track_text_map,
    )
    assert len(triple["neg"]) == 1
    assert triple["neg"][0] == "n1"


def test_build_triples_for_row_default_mode_is_bge_m3_structured():
    """B-1 regression: training-time query format MUST match production
    config 180's query_preprocessing_mode (bge_m3_structured). If this
    drifts, the fine-tune is optimized for a distribution the runtime
    never sees → near-zero or negative leaderboard lift after ~10 GPU-hr."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "chat_history": [{"role": "user", "content": "I like 70s rock"}],
        "current_user_query": "play me upbeat",
        "user_profile_raw": {"age_group": "25-34", "country_code": "US"},
        "conversation_goal": {"listener_goal": "discover"},
        "track_id": "t_gold",
    }
    triple = build_triples_for_row(
        row, "t_gold", ["t_neg1"], {"t_gold": "G", "t_neg1": "N1"},
    )
    # Default mode must emit the structured 4-block format.
    assert "[USER]:" in triple["query"], \
        f"default mode is not bge_m3_structured: {triple['query']!r}"
    assert "[GOAL]:" in triple["query"]
    assert "[HISTORY]:" in triple["query"]
    assert "[QUERY]:" in triple["query"]


def test_build_triples_for_row_explicit_raw_mode_still_works():
    """Back-compat: callers can opt into the old raw mode via query_mode='raw'."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "chat_history": [], "current_user_query": "q",
        "user_profile_raw": None, "conversation_goal": None, "track_id": "t1",
    }
    triple = build_triples_for_row(
        row, "t1", ["n1"], {"t1": "G", "n1": "N"}, query_mode="raw",
    )
    assert "[USER]:" not in triple["query"]
    assert "user: q" in triple["query"]


def test_iter_conversation_turns_drops_empty_user_query():
    """I-2 regression: a 'user' turn with empty content followed by a 'music'
    turn must NOT produce a degenerate training row (would teach the model
    to map a blank query to a specific track)."""
    from scripts.build_bi_encoder_training_data import _iter_conversation_turns

    sessions = [{
        "session_id": "s1",
        "user_profile": None,
        "conversation_goal": None,
        "conversations": [
            {"role": "user", "content": ""},          # blank — must be skipped
            {"role": "music", "content": "track_X"},  # gold
            {"role": "user", "content": "real query"},
            {"role": "music", "content": "track_Y"},
        ],
    }]
    rows = _iter_conversation_turns(sessions)
    # Only ONE row should emit (the real query → track_Y); the blank→track_X
    # is dropped because its user content was empty.
    assert len(rows) == 1
    assert rows[0]["current_user_query"] == "real query"
    assert rows[0]["track_id"] == "track_Y"


def test_build_triples_for_row_emits_session_id():
    """Sub 2 fix: emit session_id so TripleJsonlDataset can split val
    SESSION-disjoint. Without this, row-shuffle val splits create 95%+
    session overlap with train (Sub 1 leak)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "session_id": "session_42",
        "chat_history": [{"role": "user", "content": "hi"}],
        "current_user_query": "play rock",
        "user_profile_raw": None,
        "conversation_goal": None,
        "track_id": "t_gold",
    }
    triple = build_triples_for_row(
        row, "t_gold", ["n1"], {"t_gold": "G", "n1": "N"},
    )
    assert "session_id" in triple, "build_triples_for_row must emit session_id"
    assert triple["session_id"] == "session_42"


def test_iter_conversation_turns_emits_user_id():
    """Δ1: rows must carry user_id from session['user_id']. Required for
    user-disjoint train/val split downstream (TripleJsonlDataset.split_key='user_id').
    Without this propagation, all sessions of one user can leak across train/val."""
    from scripts.build_bi_encoder_training_data import _iter_conversation_turns

    sessions = [{
        "session_id": "s_alpha",
        "user_id": "user_42",
        "user_profile": None,
        "conversation_goal": None,
        "conversations": [
            {"role": "user", "content": "play rock"},
            {"role": "music", "content": "track_A"},
        ],
    }]
    rows = _iter_conversation_turns(sessions)
    assert len(rows) == 1
    assert "user_id" in rows[0], "rows must carry user_id"
    assert rows[0]["user_id"] == "user_42"


def test_build_triples_for_row_emits_user_id():
    """Δ1: the JSONL triple carries user_id alongside session_id."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "session_id": "session_42",
        "user_id": "user_99",
        "chat_history": [{"role": "user", "content": "hi"}],
        "current_user_query": "play rock",
        "user_profile_raw": None,
        "conversation_goal": None,
        "track_id": "t_gold",
    }
    triple = build_triples_for_row(
        row, "t_gold", ["n1"], {"t_gold": "G", "n1": "N"},
    )
    assert "user_id" in triple, "build_triples_for_row must emit user_id"
    assert triple["user_id"] == "user_99"


def test_build_triples_for_row_emits_neg_tids_in_order():
    """Δ3 issue C (extension per RocketQAv2 / BGE-M3 §3.3): emit neg_tids
    parallel to neg texts so the in-batch loss can mask the
    cross-pos-into-neg-slot collision (this query's gold appearing as a
    hard negative for another query in the batch)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "session_id": "s1", "user_id": "u1",
        "chat_history": [], "current_user_query": "q",
        "user_profile_raw": None, "conversation_goal": None,
        "track_id": "t_gold",
    }
    text_map = {"t_gold": "G", "t_n1": "N1", "t_n2": "N2", "t_n3": "N3"}
    triple = build_triples_for_row(
        row, "t_gold", ["t_n1", "t_n2", "t_n3"], text_map,
    )
    assert "neg_tids" in triple, "must emit neg_tids alongside neg texts"
    # Order MUST match `neg` exactly so loss-time masking can map slot k → tid.
    assert triple["neg_tids"] == ["t_n1", "t_n2", "t_n3"]
    assert triple["neg"] == ["N1", "N2", "N3"]


def test_build_triples_for_row_neg_tids_filter_matches_neg_filter():
    """When a neg's tid is absent from track_text_map (catalog drift), it's
    dropped from BOTH `neg` and `neg_tids` so their lengths stay equal."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "session_id": "s", "user_id": "u",
        "chat_history": [], "current_user_query": "q",
        "user_profile_raw": None, "conversation_goal": None,
        "track_id": "t1",
    }
    # t_missing absent from map → must be dropped from BOTH lists.
    text_map = {"t1": "G", "t_n1": "N1"}
    triple = build_triples_for_row(
        row, "t1", ["t_n1", "t_missing"], text_map,
    )
    assert len(triple["neg"]) == 1
    assert len(triple["neg_tids"]) == 1
    assert triple["neg_tids"][0] == "t_n1"


def test_build_triples_for_row_query_is_non_trivial():
    """Regression: ensure the query field carries the user's actual content.

    Previously the script silently produced empty queries because the row dict
    didn't have the expected component keys. This test asserts the contract:
    when components are present, the query string carries them.
    """
    from scripts.build_bi_encoder_training_data import build_triples_for_row
    row = {
        "chat_history": [{"role": "user", "content": "I like 70s rock"}],
        "current_user_query": "play me something upbeat",
        "user_profile_raw": {"age": 30},
        "conversation_goal": {"listener_goal": "discover"},
        "track_id": "t_gold",
    }
    triple = build_triples_for_row(
        row, "t_gold", ["t_neg1"], {"t_gold": "G", "t_neg1": "N1"},
    )
    # The actual user content MUST appear in the query.
    assert "play me something upbeat" in triple["query"], \
        f"query is missing the current user content: {triple['query']!r}"
    assert len(triple["query"]) > 20, \
        f"query is suspiciously short (possible schema bug): {triple['query']!r}"
