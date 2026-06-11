"""Tests for the ColBERT fine-tune training-data builder (pure functions only;
pool building + catalog load + JSONL write are I/O, exercised in the notebook).

The parity-critical function is build_dialog_query: it MUST reproduce the exact
query nb74/nb82 builds (role: content lines + '\\ngoal: <listener_goal>'), or the
fine-tune trains on a different distribution than the re-probe reranks with.
"""
import pytest

from scripts.build_colbert_train_data import (
    build_colbert_triple,
    build_dialog_query,
    goal_progress_label,
    iter_positive_turns,
    select_hard_negatives,
)

# Fake catalog renderer (stands in for MusicCatalogDB.id_to_metadata).
_ID2META = lambda tid: f"meta:{tid}"

_SESSION = {
    "session_id": "s1",
    "user_id": "u1",
    "conversation_goal": {"listener_goal": "hard 90s hip hop"},
    "goal_progress_assessments": [
        {"turn_number": 1, "goal_progress_assessment": "MOVES_TOWARD_GOAL"},
        {"turn_number": 2, "goal_progress_assessment": "DOES_NOT_MOVE_TOWARD_GOAL"},
    ],
    "conversations": [
        {"turn_number": 1, "role": "user", "content": "play hard 90s hip hop"},
        {"turn_number": 1, "role": "music", "content": "tGOLD1"},
        {"turn_number": 2, "role": "user", "content": "more"},
        {"turn_number": 2, "role": "music", "content": "tGOLD2"},
    ],
}


class TestGoalProgressLabel:
    def test_returns_label_for_labeled_turn(self):
        assert goal_progress_label(_SESSION, 1) == "MOVES_TOWARD_GOAL"
        assert goal_progress_label(_SESSION, 2) == "DOES_NOT_MOVE_TOWARD_GOAL"

    def test_returns_none_for_unlabeled_turn(self):
        assert goal_progress_label(_SESSION, 3) is None

    def test_handles_missing_assessments_key(self):
        assert goal_progress_label({"conversations": []}, 1) is None


class TestBuildDialogQuery:
    def test_turn1_is_user_turn_plus_goal(self):
        # Turn-1 prior slice = just the current user turn (no earlier history).
        prior = [{"turn_number": 1, "role": "user", "content": "play hard 90s hip hop"}]
        q = build_dialog_query(prior, "hard 90s hip hop", _ID2META)
        assert q == "user: play hard 90s hip hop\ngoal: hard 90s hip hop"

    def test_music_turns_render_as_assistant_metadata(self):
        # nb74 parity: music role -> 'assistant', content -> id_to_metadata(tid).
        prior = [
            {"turn_number": 1, "role": "user", "content": "play hard 90s hip hop"},
            {"turn_number": 1, "role": "music", "content": "tGOLD1"},
            {"turn_number": 2, "role": "user", "content": "more"},
        ]
        q = build_dialog_query(prior, "hard 90s hip hop", _ID2META)
        assert q == (
            "user: play hard 90s hip hop\n"
            "assistant: meta:tGOLD1\n"
            "user: more\n"
            "goal: hard 90s hip hop"
        )

    def test_empty_goal_appends_nothing(self):
        prior = [{"turn_number": 1, "role": "user", "content": "hey"}]
        assert build_dialog_query(prior, "", _ID2META) == "user: hey"


class TestIterPositiveTurns:
    def test_yields_only_moves_toward_goal_turns(self):
        rows = iter_positive_turns(_SESSION, _ID2META)
        assert len(rows) == 1  # turn 2 is DOES_NOT_MOVE -> dropped; turn-1 always kept
        r = rows[0]
        assert r["gold_tid"] == "tGOLD1"
        assert r["turn_number"] == 1
        assert r["session_id"] == "s1"
        assert r["query"] == "user: play hard 90s hip hop\ngoal: hard 90s hip hop"

    def test_turn1_positive_synthesized_even_without_label(self):
        # REGRESSION (RCA #1): real turn-1 has NO goal_progress_assessment, so the
        # MOVES_TOWARD_GOAL filter dropped 100% of turn-1 rows — yet the gate is 100%
        # turn-1. Turn-1 must be synthesized (gold-direct) regardless of label.
        session = {
            "session_id": "s2",
            "conversation_goal": {"listener_goal": "calm jazz"},
            "goal_progress_assessments": [
                {"turn_number": 2, "goal_progress_assessment": "MOVES_TOWARD_GOAL"},
            ],  # NOTE: no turn-1 entry — mirrors real data
            "conversations": [
                {"turn_number": 1, "role": "user", "content": "play calm jazz"},
                {"turn_number": 1, "role": "music", "content": "tGOLD1"},
                {"turn_number": 2, "role": "user", "content": "more"},
                {"turn_number": 2, "role": "music", "content": "tGOLD2"},
            ],
        }
        rows = iter_positive_turns(session, _ID2META)
        tns = sorted(r["turn_number"] for r in rows)
        assert tns == [1, 2]  # turn-1 synthesized despite no label; turn-2 via MOVES
        r1 = next(r for r in rows if r["turn_number"] == 1)
        assert r1["gold_tid"] == "tGOLD1"

    def test_turn_gt1_without_moves_label_still_dropped(self):
        # The tn==1 bypass must NOT relax the filter for warm turns.
        session = {
            "session_id": "s3",
            "conversation_goal": {"listener_goal": "x"},
            "goal_progress_assessments": [
                {"turn_number": 2, "goal_progress_assessment": "DOES_NOT_MOVE_TOWARD_GOAL"},
            ],
            "conversations": [
                {"turn_number": 1, "role": "user", "content": "a"},
                {"turn_number": 1, "role": "music", "content": "g1"},
                {"turn_number": 2, "role": "user", "content": "b"},
                {"turn_number": 2, "role": "music", "content": "g2"},
            ],
        }
        rows = iter_positive_turns(session, _ID2META)
        assert sorted(r["turn_number"] for r in rows) == [1]  # turn-2 dropped


class TestSelectHardNegatives:
    def test_returns_top_k_non_gold_in_pool_order(self):
        pool = ["tA", "tGOLD1", "tB", "tC"]
        assert select_hard_negatives(pool, "tGOLD1", k=2) == ["tA", "tB"]

    def test_returns_none_when_gold_not_in_pool(self):
        # Recall miss -> the ranker can't recover it -> skip this training row.
        assert select_hard_negatives(["tA", "tB"], "tGOLD1", k=2) is None


class TestBuildColbertTriple:
    def test_triple_shape(self):
        triple = build_colbert_triple(
            query="q", pos_text="meta:tGOLD1", neg_texts=["meta:tA", "meta:tB"],
            pos_tid="tGOLD1", neg_tids=["tA", "tB"],
            session_id="s1", turn_number=1,
        )
        assert triple["query"] == "q"
        assert triple["positive"] == "meta:tGOLD1"
        assert triple["negatives"] == ["meta:tA", "meta:tB"]
        assert triple["pos_tid"] == "tGOLD1"
        assert triple["neg_tids"] == ["tA", "tB"]
        assert triple["turn_number"] == 1
