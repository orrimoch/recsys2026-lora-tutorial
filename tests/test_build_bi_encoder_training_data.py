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
