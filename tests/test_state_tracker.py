"""Tests for music-crs-baselines/mcrs/query_rewriters/state_tracker.py (W1, Gap 8).

Tests the parser + cache + fallback logic without loading a real model.
We provide a fake LM with .tokenizer / .lm / .device that returns canned
strings so we can exercise the failure-recovery branches deterministically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = REPO_ROOT / "music-crs-baselines"
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from mcrs.query_rewriters.state_tracker import (  # noqa: E402
    PREFIX_PRIME,
    StateTracker,
    parse_user_state,
)

PROMPT_PATH = BASELINES / "mcrs" / "system_prompts" / "state_extraction.txt"


# ---------------------------------------------------------------------------
# parse_user_state — strict + permissive cases
# ---------------------------------------------------------------------------

class TestParseUserState:
    def test_strict_envelope(self):
        text = "<user_state>\nmood: calm\nintent: explore\n</user_state>"
        state = parse_user_state(text)
        assert state == {"mood": "calm", "intent": "explore"}

    def test_strict_envelope_with_text_around(self):
        text = "Sure, here's the state:\n<user_state>\nmood: calm\n</user_state>\nhope this helps"
        state = parse_user_state(text)
        assert state == {"mood": "calm"}

    def test_closing_tag_only_permissive(self):
        # Common Qwen-1.5B failure mode — no opening tag, but closing present
        text = "user state:\nmood: upbeat\nenergy: high\n</user_state>"
        state = parse_user_state(text)
        assert state == {"mood": "upbeat", "energy": "high"}

    def test_no_tags_with_header(self):
        text = "user state:\nmood: calm\nintent: explore\n\nuser: next track"
        state = parse_user_state(text)
        # Stops at "\n\n" or "\nuser:" boundary
        assert state == {"mood": "calm", "intent": "explore"}

    def test_no_tags_no_header_fails(self):
        text = "mood: calm\nintent: explore"
        # Neither tags nor "user state:" header → returns None
        assert parse_user_state(text) is None

    def test_unknown_keys_dropped(self):
        text = "<user_state>mood: calm\nfoo: bar\nintent: explore</user_state>"
        state = parse_user_state(text)
        assert state == {"mood": "calm", "intent": "explore"}

    def test_empty_text(self):
        assert parse_user_state("") is None
        assert parse_user_state(None) is None  # type: ignore[arg-type]

    def test_envelope_but_no_allowed_keys(self):
        text = "<user_state>foo: bar\nbaz: qux</user_state>"
        # Falls through to permissive paths, also empty → None
        assert parse_user_state(text) is None

    def test_all_six_keys(self):
        text = (
            "<user_state>\n"
            "mood: calm\nintent: explore\nenergy: low\n"
            "sonic_pref: folk\nera_pref: 2010s\nfamiliarity: familiar\n"
            "</user_state>"
        )
        state = parse_user_state(text)
        assert set(state.keys()) == {
            "mood", "intent", "energy", "sonic_pref", "era_pref", "familiarity"
        }


# ---------------------------------------------------------------------------
# Fake LMs — exercise the StateTracker without loading real weights
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Tokenizer stub that pretends to apply a chat template + tokenize.

    For these tests we don't actually need real tokens; the FakeHFLm
    bypasses tokenization in its generate() override.
    """
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Produce a recognisable prompt string.
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: "

    def __call__(self, text, return_tensors=None):
        # Return a stub object with input_ids/attention_mask attributes that look ok.
        # The fake model.generate ignores them anyway.
        class _Inputs:
            input_ids = _FakeTensor([0])
            attention_mask = _FakeTensor([1])
        return _Inputs()

    def encode(self, text, add_special_tokens=False):
        return [0]

    def batch_decode(self, tokens, skip_special_tokens=True):
        # Return whatever was stuffed into the output list.
        if isinstance(tokens, _FakeTensor) and getattr(tokens, "_decoded_payload", None) is not None:
            return [tokens._decoded_payload]
        return [""]


class _FakeTensor:
    def __init__(self, data, decoded=None):
        self._data = data
        self._decoded_payload = decoded
        self.shape = (1, len(data))

    def to(self, device):
        return self

    def __getitem__(self, idx):
        # Slice for outputs[:, input_ids.shape[1]:]
        return self


class _FakeHFLm:
    """Stand-in for LLAMA_MODEL.lm (the HF causal LM).

    The script's _generate_hf calls model.generate(...) then
    tokenizer.batch_decode(outputs[:, input_ids.shape[1]:]). We bypass
    real generation by encoding the canned output into the fake tensor,
    which our fake tokenizer.batch_decode then unpacks.
    """
    def __init__(self, canned_outputs):
        self._canned = list(canned_outputs)
        self._call_count = 0

    def generate(self, input_ids, **kwargs):
        out = self._canned[self._call_count] if self._call_count < len(self._canned) else self._canned[-1]
        self._call_count += 1
        return _FakeTensor([0], decoded=out)


class FakeLM:
    """Stand-in for LLAMA_MODEL — exposes .tokenizer / .lm / .device."""
    def __init__(self, canned_outputs):
        self.tokenizer = _FakeTokenizer()
        self.lm = _FakeHFLm(canned_outputs)
        self.device = "cpu"


# ---------------------------------------------------------------------------
# StateTracker — basic flow
# ---------------------------------------------------------------------------

class TestStateTrackerBasic:
    def test_extract_first_try_success(self, tmp_path):
        canned = ["mood: calm\nintent: explore\nenergy: low\n</user_state>"]
        lm = FakeLM(canned)
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        state = tracker.extract("sess1", 1, "I want something calm", "")
        assert state == {"mood": "calm", "intent": "explore", "energy": "low"}
        assert tracker.stats["ok_first_try"] == 1
        assert tracker.stats["ok_after_retry"] == 0
        assert tracker.stats["fallback_to_prior"] == 0
        assert tracker.stats["drop"] == 0

    def test_extract_strict_envelope_with_prefix_prime(self, tmp_path):
        # The tracker prepends PREFIX_PRIME to whatever the LM returns;
        # so even "x: y\n</user_state>" becomes "<user_state>\nx: y\n</user_state>".
        canned = ["mood: upbeat\n</user_state>"]
        lm = FakeLM(canned)
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        state = tracker.extract("sess1", 1, "I want something upbeat", "")
        assert state == {"mood": "upbeat"}

    def test_extract_first_try_fails_retry_succeeds(self, tmp_path):
        # First attempt: pure noise (parse fails).
        # Second attempt: valid output.
        canned = [
            "I'm sorry I can't help with that",
            "mood: relaxed\n</user_state>",
        ]
        lm = FakeLM(canned)
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path,
                               debug_failures=True)
        state = tracker.extract("sess2", 1, "explain something", "")
        assert state == {"mood": "relaxed"}
        assert tracker.stats["ok_first_try"] == 0
        assert tracker.stats["ok_after_retry"] == 1


# ---------------------------------------------------------------------------
# StateTracker — caching
# ---------------------------------------------------------------------------

class TestStateTrackerCache:
    def test_cache_hit_after_first_call(self, tmp_path):
        canned = ["mood: calm\n</user_state>"]
        lm = FakeLM(canned)
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        s1 = tracker.extract("sess1", 1, "I want something calm", "")
        s2 = tracker.extract("sess1", 1, "I want something calm", "")
        assert s1 == s2
        assert tracker.stats["cache_hits"] == 1
        # Only one model call (the cache hit didn't generate again).
        assert lm.lm._call_count == 1

    def test_cache_persists_across_tracker_instances(self, tmp_path):
        canned = ["mood: calm\n</user_state>"]
        lm1 = FakeLM(canned)
        t1 = StateTracker(lm=lm1, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        s1 = t1.extract("sess1", 1, "calm please", "")

        # New tracker instance reading from the same cache dir
        lm2 = FakeLM(["this would never be used"])
        t2 = StateTracker(lm=lm2, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        s2 = t2.extract("sess1", 1, "calm please", "")
        assert s1 == s2
        assert lm2.lm._call_count == 0  # disk cache hit, no LM call

    def test_cache_uses_proper_filename(self, tmp_path):
        canned = ["mood: calm\n</user_state>"]
        lm = FakeLM(canned)
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        tracker.extract("sess-uuid-A", 3, "hi", "")
        cache_files = list((tmp_path / "state").glob("*.json"))
        assert len(cache_files) == 1
        assert "sess-uuid-A" in cache_files[0].name
        assert "__3.json" in cache_files[0].name


# ---------------------------------------------------------------------------
# StateTracker — fallback chain (Gap 10)
# ---------------------------------------------------------------------------

class TestStateTrackerFallback:
    def test_fallback_to_prior_after_two_parse_failures(self, tmp_path):
        # Turn 1: succeeds.
        # Turn 2: both attempts fail → fall back to turn-1 state.
        canned = [
            "mood: calm\nintent: explore\n</user_state>",  # turn 1 ok
            "garbage no envelope",                           # turn 2 attempt 1
            "still garbage",                                 # turn 2 attempt 2
        ]
        lm = FakeLM(canned)
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        s1 = tracker.extract("sess-x", 1, "first", "")
        s2 = tracker.extract("sess-x", 2, "second", "")
        assert s1 is not None
        assert s2 is not None
        # s2 should carry the fallback marker AND the original state values
        assert s2.get("_fallback") == "prior_turn_state"
        assert s2["mood"] == "calm"
        assert s2["intent"] == "explore"
        assert tracker.stats["fallback_to_prior"] == 1

    def test_was_fallback_helper(self):
        # Public contract: callers detect fallback via was_fallback(state).
        assert StateTracker.was_fallback({"_fallback": "prior_turn_state"}) is True
        assert StateTracker.was_fallback({"mood": "calm"}) is False
        assert StateTracker.was_fallback(None) is False
        assert StateTracker.was_fallback({}) is False

    def test_drop_when_no_prior_state(self, tmp_path):
        # First turn fails twice, no prior state to fall back to.
        canned = ["garbage", "still garbage"]
        lm = FakeLM(canned)
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        s = tracker.extract("sess-y", 1, "broken", "")
        assert s is None
        assert tracker.stats["drop"] == 1
        assert tracker.stats["fallback_to_prior"] == 0


# ---------------------------------------------------------------------------
# StateTracker — backend detection (Gap 2)
# ---------------------------------------------------------------------------

class TestBackendDetection:
    def test_hf_backend_detected(self, tmp_path):
        lm = FakeLM(["mood: calm\n</user_state>"])
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        # FakeLM has .device + .lm.generate → detected as HF
        assert tracker._backend == "hf"

    def test_vllm_class_name_routes_vllm(self, tmp_path):
        # Construct a fake whose CLASS NAME contains "vllm" (case-insensitive).
        # The detection prefers explicit class-name match.
        class FakeVLLM_MODEL:
            def __init__(self):
                self.tokenizer = _FakeTokenizer()
                # Mock vllm.LLM-like: has .lm with .generate (but we won't call it
                # in this test — backend detection only).
                self.lm = _FakeHFLm(["mood: calm\n</user_state>"])
                # No .device attribute (matches real VLLM_MODEL)

        lm = FakeVLLM_MODEL()
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        assert tracker._backend == "vllm"


# ---------------------------------------------------------------------------
# StateTracker — stats / report
# ---------------------------------------------------------------------------

class TestStateTrackerReport:
    def test_report_shape(self, tmp_path):
        lm = FakeLM(["mood: calm\n</user_state>"])
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        tracker.extract("s", 1, "hi", "")
        rep = tracker.report()
        assert "calls" in rep
        assert "ok_first_try" in rep
        assert "parse_validity_excl_cache" in rep
        assert rep["parse_validity_excl_cache"] == 1.0

    def test_parse_validity_with_cache_only(self, tmp_path):
        lm = FakeLM(["mood: calm\n</user_state>"])
        tracker = StateTracker(lm=lm, prompt_path=PROMPT_PATH, cache_dir=tmp_path)
        tracker.extract("s", 1, "hi", "")
        tracker.extract("s", 1, "hi", "")  # cache hit
        # 1 real call, 1 cache hit, 1 ok_first_try → parse_validity 1.0
        assert tracker.stats["calls"] == 2
        assert tracker.stats["cache_hits"] == 1
        assert tracker.parse_validity() == 1.0
