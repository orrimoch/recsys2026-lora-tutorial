"""Tests for mcrs/rerank/listwise_llm.py — the LLM listwise reranker (stage 2).

Stage 1 is the trained K2 LGBM (coarse order over the fused 500-pool). Stage 2
hands the LGBM top-`window` (default 50) to an LLM that reorders them listwise
(RankGPT-style), so the final top-20 can promote a gold the LGBM put at rank
21-50 (the addressable nDCG bucket). The LLM call is INJECTED (`generate_fn`)
so every unit here is pure + offline; only make_gemini_generate_fn is live-API.

Invariants the tests pin down:
- the reorder is a permutation of the window (no hallucinated / dropped / duped ids);
- the tail (ranks window+1..) is never touched — it can't reach top-20 anyway;
- ANY failure (empty generation, unparseable order) falls back to the LGBM order,
  never an exception and never a shrunk list.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcrs.contracts import Candidate, RankedList, TurnContext, UserProfile  # noqa: E402
import mcrs.rerank.listwise_llm as lw  # noqa: E402


# --- build_listwise_prompt -------------------------------------------------

def test_build_prompt_numbers_candidates_and_asks_for_ranking():
    p = lw.build_listwise_prompt("user: something upbeat", "discover new artists",
                                 ["Song A by Artist A [rock]", "Song B by Artist B [jazz]"])
    assert "[1] Song A by Artist A [rock]" in p
    assert "[2] Song B by Artist B [jazz]" in p
    assert "user: something upbeat" in p
    assert "discover new artists" in p           # goal included when present
    assert "[" in p and "]" in p                 # asks for bracketed ranking output


def test_build_prompt_omits_goal_line_when_absent():
    p = lw.build_listwise_prompt("user: hi", None, ["Song A by Artist A"])
    assert "Listener goal" not in p


# --- parse_ranking ---------------------------------------------------------

def test_parse_ranking_reads_bracketed_order():
    assert lw.parse_ranking("[3] > [1] > [2]", 3) == [3, 1, 2]


def test_parse_ranking_dedupes_preserving_first_occurrence():
    assert lw.parse_ranking("[2] [2] [1] [2]", 3) == [2, 1]


def test_parse_ranking_drops_out_of_range_indices():
    # 25 and 0 are not valid 1..n positions -> ignored, not crash
    assert lw.parse_ranking("[2] > [25] > [0] > [1]", 3) == [2, 1]


def test_parse_ranking_falls_back_to_bare_integers():
    assert lw.parse_ranking("3 > 1 > 2", 3) == [3, 1, 2]


def test_parse_ranking_returns_none_on_garbage():
    assert lw.parse_ranking("no numbers here", 3) is None
    assert lw.parse_ranking("", 3) is None


# --- listwise_reorder ------------------------------------------------------

def test_reorder_applies_order_to_window_and_keeps_tail():
    ids = ["a", "b", "c", "d", "e"]          # window=3 -> reorder a,b,c; keep d,e
    out = lw.listwise_reorder(ids, [3, 1, 2], window=3)
    assert out == ["c", "a", "b", "d", "e"]


def test_reorder_appends_window_ids_missing_from_order():
    # order only mentions 2 of the 3 window items -> the unmentioned one is kept
    # (appended in original order), never dropped.
    ids = ["a", "b", "c", "d"]
    out = lw.listwise_reorder(ids, [3], window=3)
    assert out == ["c", "a", "b", "d"]
    assert sorted(out) == sorted(ids)        # permutation, nothing lost


def test_reorder_empty_order_is_identity_fallback():
    ids = ["a", "b", "c"]
    assert lw.listwise_reorder(ids, [], window=3) == ids
    assert lw.listwise_reorder(ids, None, window=3) == ids


def test_reorder_window_larger_than_list_is_safe():
    ids = ["a", "b"]
    assert lw.listwise_reorder(ids, [2, 1], window=50) == ["b", "a"]


# --- rerank_ids (orchestrator, generate_fn injected) -----------------------

def _describe(tid):
    return f"track {tid}"


def test_rerank_ids_reorders_via_generated_order():
    ids = ["a", "b", "c", "d"]
    gen = lambda prompt: "[3] > [1] > [2]"       # noqa: E731 — reorder the window of 3
    out = lw.rerank_ids("ctx", None, ids, _describe, gen, window=3)
    assert out == ["c", "a", "b", "d"]


def test_rerank_ids_falls_back_when_generation_empty():
    ids = ["a", "b", "c"]
    out = lw.rerank_ids("ctx", None, ids, _describe, lambda p: None, window=3)
    assert out == ids                            # None generation -> LGBM order kept


def test_rerank_ids_falls_back_when_order_unparseable():
    ids = ["a", "b", "c"]
    out = lw.rerank_ids("ctx", None, ids, _describe, lambda p: "garbage", window=3)
    assert out == ids


def test_rerank_ids_empty_pool_is_safe():
    assert lw.rerank_ids("ctx", None, [], _describe, lambda p: "[1]", window=3) == []


def test_rerank_ids_passes_window_descriptions_to_prompt():
    ids = ["a", "b", "c", "d", "e"]
    seen = {}
    def gen(prompt):
        seen["prompt"] = prompt
        return "[1]"
    lw.rerank_ids("ctx", None, ids, _describe, gen, window=2)
    assert "track a" in seen["prompt"] and "track b" in seen["prompt"]
    assert "track c" not in seen["prompt"]       # only the window is shown


# --- listwise_rerank (RankedList wrapper) ----------------------------------

def _turn(utterances, goal=None):
    n = len(utterances)
    return TurnContext(
        session_id="s1", user_id="u1", turn_number=n, utterances=list(utterances),
        goal=goal, user_profile=UserProfile("u1", None, None, None, []),
        history_tids=[], segment="warm")


def test_listwise_rerank_reorders_rankedlist_items_by_id():
    turn = _turn(["user: i want energetic rock"], goal="discover")
    items = [Candidate("a"), Candidate("b"), Candidate("c")]
    ranked = RankedList(turn, items)
    gen = lambda prompt: "[3] > [2] > [1]"       # noqa: E731
    out = lw.listwise_rerank(ranked, _describe, gen, window=3)
    assert [c.track_id for c in out.items] == ["c", "b", "a"]
    assert out.turn is turn


def test_listwise_rerank_feeds_conversation_and_goal_to_the_llm():
    turn = _turn(["user: chill piano please"], goal="relax tonight")
    ranked = RankedList(turn, [Candidate("a"), Candidate("b")])
    seen = {}
    def gen(prompt):
        seen["p"] = prompt
        return "[1] > [2]"
    lw.listwise_rerank(ranked, _describe, gen, window=2)
    assert "chill piano please" in seen["p"]
    assert "relax tonight" in seen["p"]


def test_listwise_rerank_fallback_preserves_all_candidates():
    turn = _turn(["user: hi"])
    items = [Candidate("a"), Candidate("b"), Candidate("c")]
    ranked = RankedList(turn, items)
    out = lw.listwise_rerank(ranked, _describe, lambda p: None, window=3)
    assert [c.track_id for c in out.items] == ["a", "b", "c"]


# --- cache_generate_fn (rerun-free dev; pure wrapper) ----------------------

def test_cache_generate_fn_hits_second_time(tmp_path):
    calls = {"n": 0}
    def gen(prompt):
        calls["n"] += 1
        return "[1] > [2]"
    cached = lw.cache_generate_fn(gen, str(tmp_path))
    assert cached("same prompt") == "[1] > [2]"
    assert cached("same prompt") == "[1] > [2]"
    assert calls["n"] == 1                        # second call served from disk


def test_cache_generate_fn_does_not_cache_empty(tmp_path):
    calls = {"n": 0}
    def gen(prompt):
        calls["n"] += 1
        return None
    cached = lw.cache_generate_fn(gen, str(tmp_path))
    cached("p"); cached("p")
    assert calls["n"] == 2                         # a failed (None) generation is re-tried


def test_cache_generate_fn_no_dir_always_calls(tmp_path):
    calls = {"n": 0}
    def gen(prompt):
        calls["n"] += 1
        return "[1]"
    cached = lw.cache_generate_fn(gen, None)
    cached("p"); cached("p")
    assert calls["n"] == 2


# --- in-session liked tracks rendered into the prompt (bug #5) --------------
# The warm final-turn proxy is bound by within-session continuation. The listwise
# reranker must SEE the tracks the user already liked this session (turn.history_tids),
# not just the user utterances — it's the single strongest warm-continuation signal.

def test_render_conversation_includes_liked_tracks():
    turn = _turn(["user: more like that"], goal=None)
    object.__setattr__(turn, "history_tids", ["a", "b"]) if False else None
    # build a turn that actually carries history
    turn = TurnContext(session_id="s", user_id="u", turn_number=1,
                       utterances=["user: more like that"], goal=None,
                       user_profile=UserProfile("u", None, None, None, []),
                       history_tids=["a", "b"], segment="warm")
    out = lw.render_conversation(turn, _describe)
    assert "user: more like that" in out
    assert "track a" in out and "track b" in out         # liked tracks rendered


def test_render_conversation_no_history_is_just_utterances():
    turn = _turn(["user: hello"])
    assert lw.render_conversation(turn, _describe) == "user: hello"


def test_render_conversation_tolerates_describe_errors():
    turn = TurnContext(session_id="s", user_id="u", turn_number=1, utterances=["user: hi"],
                       goal=None, user_profile=UserProfile("u", None, None, None, []),
                       history_tids=["bad"], segment="warm")
    def boom(tid): raise KeyError(tid)
    assert lw.render_conversation(turn, boom) == "user: hi"   # bad describe -> skipped, no crash


def test_listwise_rerank_feeds_liked_tracks_to_the_llm():
    turn = TurnContext(session_id="s", user_id="u", turn_number=1,
                       utterances=["user: keep it going"], goal=None,
                       user_profile=UserProfile("u", None, None, None, []),
                       history_tids=["x"], segment="warm")
    ranked = RankedList(turn, [Candidate("a"), Candidate("b")])
    seen = {}
    def gen(prompt):
        seen["p"] = prompt
        return "[1] > [2]"
    lw.listwise_rerank(ranked, _describe, gen, window=2)
    assert "track x" in seen["p"]                          # the liked track reached the prompt
