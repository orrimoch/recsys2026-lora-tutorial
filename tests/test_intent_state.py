"""Tests for the intent_state Q* rewriter (plan §3/§5).

The pure JSON-parse + prompt-build core and the cached orchestration are tested with
an injected fake client — no Gemini API. The live Gemini call is integration-only.
"""
from mcrs.query_rewriters.intent_state import (
    IntentStateRewriter,
    build_intent_user_content,
    parse_intent_state,
)


class _FakeClient:
    """Stands in for GeminiClient — counts calls, returns a preset JSON string."""
    def __init__(self, out='{"query_star": "clean q", "state": {"era": "90s"}, "info_content": 0.6}'):
        self.out = out
        self.calls = 0

    def generate(self, system_instruction, user_content):
        self.calls += 1
        return self.out


class TestParseIntentState:
    def test_parses_clean_json(self):
        raw = ('{"state": {"genre": "hip-hop", "era": "90s"}, '
               '"query_star": "hard 90s east coast hip hop", "info_content": 0.8}')
        r = parse_intent_state(raw, fallback_query="raw q")
        assert r["query_star"] == "hard 90s east coast hip hop"
        assert r["state"]["era"] == "90s"
        assert r["info_content"] == 0.8

    def test_strips_markdown_code_fences(self):
        raw = '```json\n{"query_star": "calm jazz", "state": {}, "info_content": 0.5}\n```'
        assert parse_intent_state(raw, "")["query_star"] == "calm jazz"

    def test_malformed_falls_back_to_raw_query(self):
        r = parse_intent_state("not json at all", fallback_query="play hard hip hop")
        assert r["query_star"] == "play hard hip hop"  # channel still works on the raw query
        assert r["state"] == {}
        assert r["info_content"] == 0.0

    def test_missing_or_empty_query_star_falls_back(self):
        assert parse_intent_state('{"state": {}, "info_content": 0.3}', "fb")["query_star"] == "fb"
        assert parse_intent_state('{"query_star": "  ", "info_content": 0.3}', "fb")["query_star"] == "fb"


class TestBuildIntentUserContent:
    def test_includes_goal_recent_history_profile_blocks(self):
        s = build_intent_user_content(
            history_text="user: hi\nassistant: track X",
            recent_text="more like that but for a workout",
            goal_text="hard 90s east coast hip hop",
            profile={"preferred_musical_culture": "Anglo-American Rock", "age_group": "30s"},
        )
        assert "[GOAL]" in s and "hard 90s east coast hip hop" in s
        assert "[RECENT]" in s and "more like that but for a workout" in s
        assert "[HISTORY]" in s and "track X" in s
        assert "Anglo-American Rock" in s  # profile taste prior


class TestIntentStateRewriter:
    def test_rewrite_parses_client_output(self, tmp_path):
        fc = _FakeClient()
        r = IntentStateRewriter(client=fc, cache_dir=str(tmp_path))
        out = r.rewrite("s1", 1, history_text="", recent_text="raw", goal_text="g", profile={})
        assert out["query_star"] == "clean q"
        assert out["state"]["era"] == "90s"
        assert fc.calls == 1

    def test_caches_by_session_and_turn(self, tmp_path):
        fc = _FakeClient()
        r = IntentStateRewriter(client=fc, cache_dir=str(tmp_path))
        r.rewrite("s1", 1, "", "raw", "g", {})
        r.rewrite("s1", 1, "", "raw", "g", {})  # same (session, turn) -> cache hit
        assert fc.calls == 1

    def test_client_error_falls_back_to_raw_query(self, tmp_path):
        class Boom:
            def generate(self, system_instruction, user_content):
                raise RuntimeError("api down")
        r = IntentStateRewriter(client=Boom(), cache_dir=str(tmp_path))
        out = r.rewrite("s2", 1, "", "play calm jazz", "calm jazz", {},
                        fallback_query="play calm jazz\ngoal: calm jazz")
        assert out["query_star"] == "play calm jazz\ngoal: calm jazz"
