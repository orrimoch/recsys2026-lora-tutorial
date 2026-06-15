"""Training-phase parity for the compact ColBERT query (Option B).

The ColBERT training data must be built with the SAME compact query as serve/Blind-A,
or the retrain reintroduces a train/serve skew. iter_positive_turns(compact_query=True)
builds each triple's query via the shared build_retrieval_query(mode="compact_colbert"),
so training and inference are byte-identical. Default (compact_query=False) keeps the
legacy full-dialog metadata rendering untouched.
"""
from mcrs.crs_baseline import build_retrieval_query
from scripts.build_colbert_train_data import iter_positive_turns, _prior_slice

_ID2META = lambda tid: f"meta:{tid}"

_SESSION = {
    "session_id": "s1", "user_id": "u1",
    "conversation_goal": {"listener_goal": "hard 90s hip hop"},
    "user_profile": {"preferred_musical_culture": "US Hip Hop"},
    "goal_progress_assessments": [
        {"turn_number": 1, "goal_progress_assessment": "MOVES_TOWARD_GOAL"},
        {"turn_number": 2, "goal_progress_assessment": "MOVES_TOWARD_GOAL"},
    ],
    "conversations": [
        {"turn_number": 1, "role": "user", "content": "play hard 90s hip hop"},
        {"turn_number": 1, "role": "music", "content": "tGOLD1"},
        {"turn_number": 2, "role": "user", "content": "more"},
        {"turn_number": 2, "role": "music", "content": "tGOLD2"},
    ],
}


def test_compact_query_turn1_is_goal_culture_last_user():
    rows = iter_positive_turns(_SESSION, _ID2META, compact_query=True)
    assert rows[0]["query"] == (
        "goal: hard 90s hip hop\nculture: US Hip Hop\nplay hard 90s hip hop")


def test_compact_query_warm_turn_drops_history_and_metadata():
    rows = iter_positive_turns(_SESSION, _ID2META, compact_query=True)
    warm = rows[1]["query"]                         # turn-2 row
    assert warm == "goal: hard 90s hip hop\nculture: US Hip Hop\nmore"
    assert "play hard 90s hip hop" not in warm      # prior history dropped
    assert "meta:" not in warm and "tGOLD" not in warm  # no metadata/UUID


def test_compact_query_matches_shared_serve_helper_exactly():
    # The parity guarantee: train builder == serve builder, same function.
    rows = iter_positive_turns(_SESSION, _ID2META, compact_query=True)
    prior = _prior_slice(_SESSION["conversations"], 2)
    expected = build_retrieval_query(
        prior, mode="compact_colbert", goal_text="hard 90s hip hop",
        user_profile=_SESSION["user_profile"])
    assert rows[1]["query"] == expected


def test_default_still_renders_full_dialog_metadata():
    # Back-compat: without compact_query, legacy build_dialog_query path is unchanged.
    rows = iter_positive_turns(_SESSION, _ID2META)
    assert "meta:tGOLD1" in rows[1]["query"]         # warm row renders music metadata
    assert "assistant:" in rows[1]["query"]
