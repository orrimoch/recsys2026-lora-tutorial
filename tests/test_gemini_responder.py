"""Tests for scripts/gemini_responder.py.

The Gemini responder is a post-processor: it takes an existing prediction.json
(with predicted_track_ids already produced by the retrieval+rerank pipeline),
rejoins the conversation context + recommended-track metadata, and regenerates
predicted_response via the Gemini API — leaving predicted_track_ids untouched.
It is the responder-side twin of scripts/gemini_judge_responses.py.

CPU-testable units (no live API): context rendering, track formatting (top_n
truncation), prompt construction, the output-row merge (track_ids preserved,
response replaced), and the generate-with-fallback retry path (Gemini mocked).
The dataset/API wiring in main() is integration-only.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gemini_responder as gr  # noqa: E402


# --- fixtures --------------------------------------------------------------

@pytest.fixture
def item_db_meta():
    return {
        "trk-A": {"track_name": "Song A", "artist_name": "Artist A",
                  "album_name": "Album A", "tags": ["indie", "mellow"]},
        "trk-B": {"track_name": "Song B", "artist_name": "Artist B",
                  "album_name": "Album B", "tags": ["techno"]},
        "trk-C": {"track_name": "Song C", "artist_name": "Artist C",
                  "album_name": "Album C", "tags": []},
    }


@pytest.fixture
def conversations():
    # Turn 1: full (user / music / assistant). Turn 2: only the user query
    # (the turn we are responding to — no assistant/music yet on a blind row).
    return [
        {"turn_number": 1, "role": "user", "content": "I want something chill"},
        {"turn_number": 1, "role": "music", "content": "trk-A"},
        {"turn_number": 1, "role": "assistant", "content": "Try this mellow track."},
        {"turn_number": 2, "role": "user", "content": "Now something with more energy"},
    ]


class FakeModel:
    """Stand-in for genai.GenerativeModel. `behaviors` is a list of either a
    string (returned as resp.text) or an Exception (raised). The last entry
    repeats for any extra calls."""

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    def generate_content(self, prompt, **kwargs):  # accepts generation_config etc.
        b = self.behaviors[min(self.calls, len(self.behaviors) - 1)]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        return types.SimpleNamespace(text=b)


# --- render_context --------------------------------------------------------

def test_render_context_expands_music_and_ends_at_target_user_turn(conversations, item_db_meta):
    ctx = gr.render_context(conversations, item_db_meta, target_turn=2)
    # history from turn 1 is present, with the music turn expanded to a name
    assert "I want something chill" in ctx
    assert "Song A" in ctx and "Artist A" in ctx
    assert "Try this mellow track." in ctx
    # the target user query is included
    assert "Now something with more energy" in ctx


def test_render_context_enriches_prior_tracks_with_attributes(item_db_meta):
    # A prior in-session rec must carry its attributes so the reply can connect
    # the new pick to what the user liked via a shared trait (continuity).
    convs = [
        {"turn_number": 1, "role": "user", "content": "something chill"},
        {"turn_number": 1, "role": "music", "content": "trk-A"},
        {"turn_number": 1, "role": "assistant", "content": "a calm one for you"},
        {"turn_number": 2, "role": "user", "content": "now a bit more upbeat"},
    ]
    ctx = gr.render_context(convs, item_db_meta, target_turn=2)
    assert "Song A" in ctx and "Artist A" in ctx
    assert "indie" in ctx and "mellow" in ctx  # attributes now surfaced


def test_build_prompt_has_continuity_directive():
    p = gr.build_prompt(context="user: more like that", tracks_str="X by Y", listener_goal="")
    low = p.lower()
    assert "earlier" in low and "liked" in low  # references prior liked tracks


def test_build_prompt_explanation_bridges_without_echoing():
    p = gr.build_prompt(context="user: x", tracks_str="A by B", listener_goal="")
    low = p.lower()
    assert "echo" in low                   # anti-parroting guard (avoid formulaic mirroring)
    assert "concrete track detail" in low  # protect the grounded explanation (don't over-compress)


def test_build_prompt_has_cold_start_directive():
    p = gr.build_prompt(context="user: play me something", tracks_str="X by Y", listener_goal="")
    low = p.lower()
    # cold-start path: lean on the current request + goal, don't fabricate history
    assert "no prior history" in low and ("current request" in low or "stated goal" in low)


def test_render_context_excludes_target_turn_assistant_reply(item_db_meta):
    convs = [
        {"turn_number": 1, "role": "user", "content": "hello"},
        {"turn_number": 1, "role": "assistant", "content": "LEAKED_GOLD_REPLY"},
    ]
    # responding to turn 1: its own assistant reply must NOT leak into context
    ctx = gr.render_context(convs, item_db_meta, target_turn=1)
    assert "hello" in ctx
    assert "LEAKED_GOLD_REPLY" not in ctx


def test_render_context_surfaces_prior_track_thought(item_db_meta):
    # Each prior 'music' turn carries the recommender's 'thought' (why it picked
    # that track). Surfacing it gives the responder real continuity reasoning to
    # build on instead of inventing it. Prior turns only -> leak-safe.
    convs = [
        {"turn_number": 1, "role": "user", "content": "something chill"},
        {"turn_number": 1, "role": "music", "content": "trk-A",
         "thought": "warm mellow guitar matches the chill request"},
        {"turn_number": 1, "role": "assistant", "content": "a calm one"},
        {"turn_number": 2, "role": "user", "content": "now upbeat"},
    ]
    ctx = gr.render_context(convs, item_db_meta, target_turn=2)
    assert "warm mellow guitar matches the chill request" in ctx
    assert "Song A" in ctx  # still carries the track name + attributes


def test_render_context_excludes_target_turn_thought(item_db_meta):
    # The target turn's own 'thought' is reasoning about the gold we must
    # generate — it must NOT leak into the context.
    convs = [
        {"turn_number": 1, "role": "user", "content": "hi"},
        {"turn_number": 1, "role": "music", "content": "trk-A",
         "thought": "LEAK_THOUGHT_FOR_THE_GOLD"},
    ]
    ctx = gr.render_context(convs, item_db_meta, target_turn=1)
    assert "LEAK_THOUGHT_FOR_THE_GOLD" not in ctx


def test_render_context_tolerates_missing_thought(item_db_meta):
    # Music turns without a thought (or NaN after DataFrame coercion) must not
    # crash or emit a literal 'nan'.
    convs = [
        {"turn_number": 1, "role": "user", "content": "chill"},
        {"turn_number": 1, "role": "music", "content": "trk-A"},
        {"turn_number": 2, "role": "user", "content": "more"},
    ]
    ctx = gr.render_context(convs, item_db_meta, target_turn=2)
    assert "Song A" in ctx
    assert "nan" not in ctx.lower()


# --- era / popularity grounding on the recommended track block -------------

def test_format_tracks_includes_era_decade():
    # release_date -> a citable era attribute (the prompt only lets the model
    # mention attributes present in the track block).
    meta = {"trk-A": {"track_name": "Song A", "artist_name": "Artist A",
                      "album_name": "Album A", "tags": ["indie"],
                      "release_date": "1994-05-01"}}
    s = gr.format_tracks(["trk-A"], meta, n=1)
    assert "1990s" in s


def test_format_tracks_tolerates_missing_or_bad_release_date():
    meta = {
        "trk-A": {"track_name": "A", "artist_name": "B", "tags": []},
        "trk-B": {"track_name": "C", "artist_name": "D", "tags": [],
                  "release_date": ""},
    }
    s = gr.format_tracks(["trk-A", "trk-B"], meta, n=2)
    assert "A by B" in s and "C by D" in s  # no crash, no bogus era


def test_decade_helper():
    assert gr._decade("1994-05-01") == "1990s"
    assert gr._decade("2007") == "2000s"
    assert gr._decade(["1981-01-01"]) == "1980s"  # HF singleton-list field
    assert gr._decade("") is None
    assert gr._decade(None) is None
    assert gr._decade("not-a-date") is None


# --- user context (stated prefs only, no inferred demographics) ------------

def test_render_user_context_includes_only_stated_prefs():
    up = {"preferred_musical_culture": "Western Alternative Rock",
          "preferred_language": "English",
          "age": 20, "gender": "female", "country_name": "Brazil"}
    s = gr.render_user_context(up)
    assert "Western Alternative Rock" in s
    assert "English" in s
    # inferred demographics must NOT be injected (stereotyping guardrail)
    assert "20" not in s and "female" not in s and "Brazil" not in s


def test_render_user_context_empty_when_no_stated_prefs():
    assert gr.render_user_context({}) == ""
    assert gr.render_user_context(None) == ""
    assert gr.render_user_context({"age": 30, "gender": "male"}) == ""


def test_build_prompt_includes_user_context():
    p = gr.build_prompt(context="user: x", tracks_str="A by B", listener_goal="",
                        user_context="Preferred musical culture: Western Alternative Rock")
    assert "Western Alternative Rock" in p


def test_build_prompt_omits_user_context_line_when_absent():
    p = gr.build_prompt(context="user: x", tracks_str="A by B", listener_goal="")
    assert "musical culture" not in p.lower()


# --- goal progress tally (prior picks only) --------------------------------

def test_goal_progress_tally_counts_prior_only():
    gpa = [
        {"turn_number": 1, "goal_progress_assessment": None},
        {"turn_number": 2, "goal_progress_assessment": "MOVES_TOWARD_GOAL"},
        {"turn_number": 3, "goal_progress_assessment": "DOES_NOT_MOVE_TOWARD_GOAL"},
        {"turn_number": 4, "goal_progress_assessment": "MOVES_TOWARD_GOAL"},
    ]
    # target turn 4: only turns 1-3 count -> 1 moved toward, 1 missed (turn1 null skipped)
    s = gr.goal_progress_tally(gpa, target_turn=4)
    assert "1 moved toward" in s and "1 missed" in s


def test_goal_progress_tally_empty_when_no_prior_assessments():
    gpa = [{"turn_number": 1, "goal_progress_assessment": None}]
    assert gr.goal_progress_tally(gpa, target_turn=1) == ""
    assert gr.goal_progress_tally([], target_turn=3) == ""
    assert gr.goal_progress_tally(None, target_turn=3) == ""


def test_build_prompt_includes_goal_progress():
    p = gr.build_prompt(context="user: x", tracks_str="A by B", listener_goal="",
                        goal_progress="Goal progress so far: 2 moved toward, 1 missed")
    assert "2 moved toward" in p


# --- format_tracks (top_n truncation) --------------------------------------

def test_format_tracks_respects_top_n(item_db_meta):
    s = gr.format_tracks(["trk-A", "trk-B", "trk-C"], item_db_meta, n=1)
    assert "Song A" in s
    assert "Song B" not in s and "Song C" not in s


def test_format_tracks_includes_artist_and_tags(item_db_meta):
    s = gr.format_tracks(["trk-A"], item_db_meta, n=3)
    assert "Artist A" in s
    assert "indie" in s  # grounding signal for the explanation axis


def test_format_tracks_handles_missing_metadata(item_db_meta):
    # unknown id must not crash and must not emit a bare 'None'
    s = gr.format_tracks(["trk-A", "unknown-id"], item_db_meta, n=3)
    assert "Song A" in s
    assert "None by None" not in s


# --- build_prompt ----------------------------------------------------------

def test_build_prompt_contains_context_tracks_and_goal():
    p = gr.build_prompt(context="user: play jazz",
                        tracks_str="Song A by Artist A [jazz]",
                        listener_goal="discover new jazz artists")
    assert "user: play jazz" in p
    assert "Song A by Artist A [jazz]" in p
    assert "discover new jazz artists" in p


def test_build_prompt_includes_few_shot_examples():
    p = gr.build_prompt(context="user: play jazz", tracks_str="Song A by Artist A",
                        listener_goal="")
    # few-shot style reference must be present, and labeled so the model treats
    # the example tracks as illustrations (not tracks to reuse).
    assert "EXAMPLES" in p
    assert "do not reuse" in p.lower()


def test_build_prompt_omits_goal_line_when_absent():
    p = gr.build_prompt(context="user: play jazz",
                        tracks_str="Song A by Artist A",
                        listener_goal="")
    # no empty/dangling goal label when there is no goal
    assert "Listener goal:" not in p


# --- build_output_row (track_ids preserved, response replaced) -------------

def test_build_output_row_preserves_track_ids_and_replaces_response():
    pred = {
        "session_id": "sess-1", "user_id": "user-1", "turn_number": 2,
        "predicted_track_ids": ["trk-A", "trk-B"],
        "predicted_response": "OLD RESPONSE",
    }
    out = gr.build_output_row(pred, "NEW RESPONSE")
    assert out["predicted_track_ids"] == ["trk-A", "trk-B"]
    assert out["predicted_response"] == "NEW RESPONSE"
    assert out["session_id"] == "sess-1"
    assert out["user_id"] == "user-1"
    assert out["turn_number"] == 2


def test_build_output_row_has_exactly_the_submission_fields():
    pred = {
        "session_id": "s", "user_id": "u", "turn_number": 1,
        "predicted_track_ids": ["trk-A"], "predicted_response": "x",
        "extra_field": "should be dropped",
    }
    out = gr.build_output_row(pred, "y")
    assert set(out.keys()) == {
        "session_id", "user_id", "turn_number",
        "predicted_track_ids", "predicted_response",
    }


# --- generate_response (retry + fallback, Gemini mocked) -------------------

def test_generate_response_returns_model_text_on_success():
    model = FakeModel(["  Here's a great pick for you.  "])
    out = gr.generate_response(model, "prompt", fallback="ORIG")
    assert out == "Here's a great pick for you."  # stripped


def test_generate_response_falls_back_on_persistent_failure():
    model = FakeModel([RuntimeError("boom")])
    out = gr.generate_response(model, "prompt", fallback="ORIG",
                               max_attempts=3, sleep_fn=lambda *_: None)
    assert out == "ORIG"  # never drops the row's response


def test_generate_response_retries_then_succeeds_on_rate_limit():
    model = FakeModel([
        RuntimeError("429 ResourceExhausted: retry in 1s"),
        RuntimeError("429 ResourceExhausted: retry in 1s"),
        "recovered reply",
    ])
    out = gr.generate_response(model, "prompt", fallback="ORIG",
                               max_attempts=6, sleep_fn=lambda *_: None)
    assert out == "recovered reply"
    assert model.calls == 3


def test_build_structured_prompt_asks_for_json_axes_and_reply():
    p = gr.build_structured_prompt(context="user: play jazz",
                                   tracks_str="Song A by Artist A [jazz]",
                                   listener_goal="discover new jazz")
    assert "user: play jazz" in p
    assert "Song A by Artist A [jazz]" in p
    assert "discover new jazz" in p
    assert '"reply"' in p          # the field we extract
    assert '"user_state"' in p     # the personalization axes block
    assert "mood" in p


def test_parse_structured_reply_extracts_reply_field():
    raw = '{"user_state": {"mood": "chill"}, "fit": "x", "reply": "Try this mellow track."}'
    assert gr.parse_structured_reply(raw) == "Try this mellow track."


def test_parse_structured_reply_handles_code_fence_and_whitespace():
    raw = '```json\n{"reply": "  Here you go.  "}\n```'
    assert gr.parse_structured_reply(raw) == "Here you go."


def test_parse_structured_reply_returns_none_on_garbage_or_missing_key():
    assert gr.parse_structured_reply("not json at all") is None
    assert gr.parse_structured_reply('{"no_reply_key": 1}') is None
    assert gr.parse_structured_reply("") is None


def test_parse_structured_reply_recovers_from_trailing_junk_after_object():
    # greedy {.*} over-grabs -> json.loads fails -> regex fallback recovers reply.
    raw = '{"user_state": {"mood": "hi"}, "reply": "Spin this."} <-- note, not json'
    assert gr.parse_structured_reply(raw) == "Spin this."


def test_parse_structured_reply_does_not_leak_analysis():
    # core safety property: only `reply` may surface, never user_state/fit.
    raw = ('{"user_state": {"mood": "SECRET_MOOD"}, "fit": "SECRET_FIT", '
           '"reply": "Enjoy this mellow pick."}')
    out = gr.parse_structured_reply(raw)
    assert out == "Enjoy this mellow pick."
    assert "SECRET_MOOD" not in out and "SECRET_FIT" not in out


def test_parse_structured_reply_prefers_outer_reply_when_malformed_and_nested():
    # malformed (so the regex-fallback path runs) AND a nested decoy `reply`:
    # must return the real top-level reply, not the nested one.
    raw = '{"user_state": {"reply": "WRONG"}, "reply": "RIGHT", bad,}'
    assert gr.parse_structured_reply(raw) == "RIGHT"


def test_generate_response_structured_success_extracts_reply():
    model = FakeModel(['{"user_state": {"mood":"hype"}, "reply": "Crank this one."}'])
    out = gr.generate_response(model, "prompt", fallback="ORIG",
                               parse_fn=gr.parse_structured_reply)
    assert out == "Crank this one."


def test_generate_response_structured_parse_failure_falls_back():
    # model ignored the JSON instruction -> we must NOT submit raw text; fall back.
    model = FakeModel(["totally not json"])
    out = gr.generate_response(model, "prompt", fallback="ORIG",
                               parse_fn=gr.parse_structured_reply)
    assert out == "ORIG"


# --- best-of-N self-judged --------------------------------------------------

def test_parse_judge_score_sums_axes_or_none():
    assert gr.parse_judge_score('{"personalization": 4, "explanation_quality": 3}') == 7.0
    assert gr.parse_judge_score('{"personalization": 4}') is None  # missing axis
    assert gr.parse_judge_score("garbage") is None
    assert gr.parse_judge_score("") is None


def test_judge_instructions_anchored_1_to_5_with_examples():
    j = gr.JUDGE_INSTRUCTIONS
    low = j.lower()
    # both named axes (exact JSON keys the parser expects)
    assert "personalization" in low and "explanation_quality" in low
    # 1-5 scale, both ends anchored
    assert "1-5" in j
    assert "5 =" in j and "1 =" in j
    # concrete anchors / examples of success vs failure
    assert "example" in low
    # scored independently (matches the official 'two independent dimensions')
    assert "independent" in low
    # still asks for the JSON object the parser needs
    assert '"personalization"' in j and '"explanation_quality"' in j


def test_build_judge_prompt_contains_reply_and_rubric():
    p = gr.build_judge_prompt("ctx", "Song A by Artist A", "my reply text")
    assert "my reply text" in p
    assert "personalization" in p.lower() and "explanation" in p.lower()


def test_pick_best_prefers_highest_score_and_handles_all_none():
    assert gr.pick_best([("a", 3.0), ("b", 7.0), ("c", 1.0)]) == "b"
    assert gr.pick_best([("a", None), ("b", None)]) == "a"  # no scores -> first
    assert gr.pick_best([]) is None


def test_generate_best_of_n_picks_highest_scored_candidate():
    gen = FakeModel(["weak reply", "strong reply"])
    judge = FakeModel(['{"personalization": 2, "explanation_quality": 2}',
                       '{"personalization": 5, "explanation_quality": 4}'])
    out = gr.generate_best_of_n(gen, judge, "prompt", "ctx", "tracks",
                                fallback="ORIG", n=2, temperatures=[0.5, 1.0],
                                sleep_fn=lambda *_: None)
    assert out == "strong reply"
    assert gen.calls == 2 and judge.calls == 2


def test_parse_pairwise_verdict():
    assert gr.parse_pairwise_verdict("A") == "A"
    assert gr.parse_pairwise_verdict("B") == "B"
    assert gr.parse_pairwise_verdict("Answer: A") == "A"
    assert gr.parse_pairwise_verdict(" b ") == "B"
    assert gr.parse_pairwise_verdict("neither is clear") is None
    assert gr.parse_pairwise_verdict("") is None


def test_build_pairwise_prompt_has_both_replies_and_axes():
    p = gr.build_pairwise_prompt("ctx", "Song X", "REPLY_ALPHA", "REPLY_BETA")
    assert "REPLY_ALPHA" in p and "REPLY_BETA" in p
    low = p.lower()
    assert "personalization" in low and "explanation" in low
    assert " a " in f" {low} " or "'a'" in low  # asks for an A/B verdict


class ContentJudge:
    """Order-invariant fake judge: whichever slot's reply contains 'BEST' wins,
    regardless of A/B position. Counts calls (to assert comparison counts)."""

    def __init__(self):
        self.calls = 0

    def generate_content(self, prompt, **kw):
        self.calls += 1
        a = re.search(r"=== REPLY A ===\n(.*?)\n\n=== REPLY B ===", prompt, re.DOTALL)
        b = re.search(r"=== REPLY B ===\n(.*?)\n\n", prompt, re.DOTALL)
        atext = a.group(1) if a else ""
        btext = b.group(1) if b else ""
        letter = "A" if "BEST" in atext else ("B" if "BEST" in btext else "A")
        return types.SimpleNamespace(text=letter)


def test_select_pairwise_koth_picks_best_and_does_n_minus_1_compares():
    judge = ContentJudge()
    cands = ["c1", "c2", "c3 BEST", "c4", "c5", "c6"]
    out = gr.select_pairwise_koth(judge, "ctx", "tracks", cands, sleep_fn=lambda *_: None)
    assert out == "c3 BEST"
    assert judge.calls == len(cands) - 1  # king-of-the-hill = N-1 comparisons


def test_select_round_robin_picks_best_and_does_all_pairs():
    judge = ContentJudge()
    cands = ["c1", "c2 BEST", "c3", "c4"]
    out = gr.select_round_robin(judge, "ctx", "tracks", cands, sleep_fn=lambda *_: None)
    assert out == "c2 BEST"
    assert judge.calls == 6  # C(4,2)


def test_generate_best_of_n_pairwise_selects_best():
    gen = FakeModel(["c1", "c2 BEST", "c3"])
    judge = ContentJudge()
    out = gr.generate_best_of_n(gen, judge, "prompt", "ctx", "tracks", fallback="ORIG",
                                n=3, temperatures=[0.5, 0.8, 1.0], sleep_fn=lambda *_: None,
                                select="pairwise")
    assert out == "c2 BEST"


def test_generate_best_of_n_falls_back_when_all_generation_fails():
    gen = FakeModel([RuntimeError("boom")])
    judge = FakeModel(['{"personalization": 5, "explanation_quality": 5}'])
    out = gr.generate_best_of_n(gen, judge, "prompt", "ctx", "tracks",
                                fallback="ORIG", n=3, temperatures=[0.5],
                                sleep_fn=lambda *_: None)
    assert out == "ORIG"


def test_generate_response_falls_back_when_resp_text_raises():
    # google.generativeai raises (not returns None) on a blocked/empty
    # candidate when you access resp.text. That is non-retryable -> fallback.
    class _Blocked:
        @property
        def text(self):
            raise ValueError("Invalid operation: response.text requires a single candidate")

    class _Model:
        def generate_content(self, prompt):
            return _Blocked()

    out = gr.generate_response(_Model(), "prompt", fallback="ORIG",
                               max_attempts=3, sleep_fn=lambda *_: None)
    assert out == "ORIG"
