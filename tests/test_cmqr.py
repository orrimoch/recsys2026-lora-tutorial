"""Tests for music-crs-baselines/mcrs/query_rewriters/cmqr.py (W2).

Covers:
  - parse_rewrites: numbered-rewrite parser, length filtering, format tolerance
  - rrf_fuse: Reciprocal Rank Fusion math + weights
  - dedupe_keep_first: contract
  - format_state_for_prompt: state injection
  - CMQR_REWRITER end-to-end with a fake LM + fake retriever
  - Cache hit behavior
  - Backend detection (HF vs vLLM)
  - Parse-failure fallback (degrades to original query, never crashes)
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

from mcrs.query_rewriters.cmqr import (  # noqa: E402
    CMQR_REWRITER,
    _detect_backend,
    dedupe_keep_first,
    format_state_for_prompt,
    parse_rewrites,
    rrf_fuse,
)

PROMPT_PATH = BASELINES / "mcrs" / "system_prompts" / "cmqr_rewrites.txt"


# ---------------------------------------------------------------------------
# parse_rewrites
# ---------------------------------------------------------------------------

class TestParseRewrites:
    def test_clean_four_rewrites(self):
        text = (
            "1. dreamy slow songs for winding down\n"
            "2. calm reflective folk tracks with vocals\n"
            "3. low-energy 2010s indie with texture\n"
            "4. mellow familiar music for focus"
        )
        rs = parse_rewrites(text, n=4)
        assert len(rs) == 4
        assert rs[0] == "dreamy slow songs for winding down"
        assert rs[3] == "mellow familiar music for focus"

    def test_paren_style(self):
        text = "1) first rewrite words here\n2) second rewrite words here"
        rs = parse_rewrites(text, n=4)
        assert len(rs) == 2

    def test_colon_style(self):
        text = "1: first rewrite words here\n2: second rewrite words here"
        rs = parse_rewrites(text, n=4)
        assert len(rs) == 2

    def test_filters_too_short(self):
        # 'ok' is below MIN_WORDS=3 → dropped
        text = "1. ok\n2. dreamy slow songs for winding\n3. calm reflective folk tracks"
        rs = parse_rewrites(text, n=4)
        assert len(rs) == 2

    def test_filters_too_long(self):
        # 25-word rewrite > MAX_WORDS=20 → dropped
        long = " ".join(["word"] * 25)
        text = f"1. {long}\n2. dreamy slow songs for winding"
        rs = parse_rewrites(text, n=4)
        assert len(rs) == 1
        assert rs[0] == "dreamy slow songs for winding"

    def test_dedupe_repeated_numbers(self):
        # Same number twice — keep first seen, ignore duplicates
        text = "1. first rewrite words\n1. duplicate version words\n2. second rewrite words"
        rs = parse_rewrites(text, n=4)
        assert rs == ["first rewrite words", "second rewrite words"]

    def test_noisy_preamble_and_suffix(self):
        text = (
            "Sure, here are the rewrites:\n\n"
            "1. dreamy slow songs for winding\n"
            "2. calm reflective folk tracks\n\n"
            "Hope these help!"
        )
        rs = parse_rewrites(text, n=4)
        assert len(rs) == 2

    def test_empty_input(self):
        assert parse_rewrites("", n=4) == []
        assert parse_rewrites(None, n=4) == []  # type: ignore[arg-type]

    def test_no_numbered_lines(self):
        assert parse_rewrites("Just regular text, no numbers anywhere.", n=4) == []

    def test_caps_at_n(self):
        text = "1. one one one\n2. two two two\n3. three three three\n4. four four four\n5. five five five"
        rs = parse_rewrites(text, n=3)
        assert len(rs) == 3
        assert rs[-1] == "three three three"


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------

class TestRrfFuse:
    def test_single_list(self):
        out = rrf_fuse([["a", "b", "c"]], topk=3, k=60)
        assert out == ["a", "b", "c"]

    def test_two_identical_lists(self):
        out = rrf_fuse([["a", "b", "c"], ["a", "b", "c"]], topk=3, k=60)
        # Same ranks; same total scores; relative ordering preserved
        assert out == ["a", "b", "c"]

    def test_disjoint_lists(self):
        # No overlap — first list dominates by sort stability of dict iteration
        out = rrf_fuse([["a", "b"], ["c", "d"]], topk=4, k=60)
        # Each id appears in exactly one list at rank 1 or 2; rank-1 ids fuse highest
        # 'a' and 'c' both have rank 1 score 1/61; 'b' and 'd' have rank 2 score 1/62
        assert set(out[:2]) == {"a", "c"}
        assert set(out[2:]) == {"b", "d"}

    def test_overlapping_lifts_to_top(self):
        # Doc that appears in BOTH lists should outrank a doc appearing in only one
        out = rrf_fuse([["a", "b", "c"], ["b", "x", "y"]], topk=3, k=60)
        # 'b' in both: 1/62 + 1/61. 'a' only first: 1/61.
        assert out[0] == "b"

    def test_weights_change_order(self):
        # Heavily weight the second list — its rank-1 should win
        out = rrf_fuse(
            [["a", "b"], ["b", "a"]],
            topk=2, k=60, weights=[1.0, 100.0],
        )
        # Second list dominates: rank-1 of second is 'b'
        assert out[0] == "b"

    def test_topk_truncation(self):
        out = rrf_fuse([["a", "b", "c", "d", "e"]], topk=3, k=60)
        assert out == ["a", "b", "c"]

    def test_empty_lists(self):
        assert rrf_fuse([], topk=5, k=60) == []
        assert rrf_fuse([[], []], topk=5, k=60) == []

    def test_weights_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="weights length"):
            rrf_fuse([["a"], ["b"]], topk=2, k=60, weights=[1.0])


# ---------------------------------------------------------------------------
# dedupe_keep_first
# ---------------------------------------------------------------------------

class TestDedupe:
    def test_basic(self):
        assert dedupe_keep_first(["a", "b", "a", "c", "b", "d"]) == ["a", "b", "c", "d"]

    def test_empty(self):
        assert dedupe_keep_first([]) == []

    def test_no_dupes(self):
        assert dedupe_keep_first(["a", "b", "c"]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# format_state_for_prompt
# ---------------------------------------------------------------------------

class TestFormatState:
    def test_full_state(self):
        state = {
            "mood": "calm", "intent": "explore", "energy": "low",
            "sonic_pref": "folk", "era_pref": "2010s", "familiarity": "familiar",
        }
        out = format_state_for_prompt(state)
        for key in ["mood", "intent", "energy", "sonic_pref", "era_pref", "familiarity"]:
            assert f"{key}:" in out

    def test_filters_unknown(self):
        state = {"mood": "calm", "energy": "unknown", "sonic_pref": ""}
        out = format_state_for_prompt(state)
        assert "mood: calm" in out
        assert "energy" not in out  # filtered: value == 'unknown'
        assert "sonic_pref" not in out  # filtered: empty string

    def test_filters_unknown_keys(self):
        # Keys outside ALLOWED_STATE_KEYS get dropped
        state = {"mood": "calm", "song_title": "hallucinated"}
        out = format_state_for_prompt(state)
        assert "mood: calm" in out
        assert "song_title" not in out

    def test_empty_or_none(self):
        assert format_state_for_prompt(None) == "(none)"
        assert format_state_for_prompt({}) == "(none)"
        assert format_state_for_prompt({"mood": "unknown"}) == "(none)"


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

class _FakeHFLM:
    pass


class _FakeVLLMLM:
    pass


_FakeVLLMLM.__name__ = "VLLM_MODEL"  # mimic the real class name


class TestBackendDetection:
    def test_hf_default(self):
        assert _detect_backend(_FakeHFLM()) == "hf"

    def test_vllm_class_name(self):
        assert _detect_backend(_FakeVLLMLM()) == "vllm"


# ---------------------------------------------------------------------------
# CMQR_REWRITER end-to-end with a fake LM + fake retriever
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Reduce messages to a deterministic string so generate() can switch on it
        return "::".join(m["content"][:40] for m in messages)

    def __call__(self, text, return_tensors=None):
        # Return a tiny fake tensor envelope; _generate_hf calls input_ids.to(device)
        import torch
        return _Box({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

    def batch_decode(self, ids, skip_special_tokens=True):
        # The hook below stashes the next decoded payload on the LM
        return [getattr(self._owner, "_next_decode", "")]


class _Box(dict):
    """dict that exposes keys as attributes (for tokenizer output)."""
    def __getattr__(self, k):
        return self[k]


class _FakeModel:
    def __init__(self, decode_outputs):
        self._decode_outputs = list(decode_outputs)
        self._call_idx = 0

    def generate(self, input_ids, attention_mask=None, **kwargs):
        # Record the next decoded string on the parent LM so the tokenizer can
        # return it, then return a tiny tensor that has the right shape for slicing.
        import torch
        out = self._decode_outputs[self._call_idx % len(self._decode_outputs)]
        self._call_idx += 1
        # We stash via the lm reference; tokenizer reads from self._owner._next_decode
        self._next_decode = out
        # Return tensor where output shape > input shape (so slicing [:, n:] works)
        return torch.tensor([[1, 2, 3, 4, 5, 6]])


class _FakeLM:
    """LM wrapper with the LLAMA_MODEL surface that CMQR uses."""
    def __init__(self, decode_outputs):
        self.tokenizer = _FakeTokenizer()
        self.lm = _FakeModel(decode_outputs)
        self.device = "cpu"
        # Wire up so tokenizer.batch_decode returns the model's next output.
        self.tokenizer._owner = self.lm
        # The _generate_hf code re-prepends "1." to the decoded output.


class _FakeRetriever:
    """Retriever that returns deterministic ranked lists per query."""
    def __init__(self, mapping):
        # mapping: dict of (query_substring, ranked_list)
        self.mapping = mapping
        self.calls = []

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None):
        self.calls.append((list(queries), topk, list(user_ids) if user_ids else None))
        out = []
        for q in queries:
            for substr, ranked in self.mapping.items():
                if substr in q:
                    out.append(ranked[:topk])
                    break
            else:
                # Default: return a stable per-query result based on hash
                out.append([f"track_{abs(hash(q)) % 100}_{i}" for i in range(topk)])
        return out


class TestCMQREndToEnd:
    @pytest.fixture
    def cmqr_setup(self, tmp_path):
        """Build a CMQR with fake LM that emits a clean 4-rewrite block."""
        # The fake LM's _generate_hf re-prepends "1.", so the decoded string
        # starts at the body of rewrite #1.
        decoded = (
            " dreamy slow songs for winding down\n"
            "2. calm reflective folk tracks here\n"
            "3. low-energy indie with texture\n"
            "4. mellow familiar music for focus"
        )
        lm = _FakeLM([decoded])
        retriever = _FakeRetriever({
            "dreamy slow": ["t_dream_1", "t_dream_2", "t_dream_3"],
            "calm reflective": ["t_calm_1", "t_calm_2", "t_dream_1"],
            "low-energy indie": ["t_indie_1", "t_calm_1", "t_dream_1"],
            "mellow familiar": ["t_mellow_1", "t_dream_1"],
        })
        cmqr = CMQR_REWRITER(
            lm=lm,
            inner_retriever=retriever,
            prompt_path=PROMPT_PATH,
            cache_dir=str(tmp_path / "cache"),
            n_rewrites=4,
            topk_per_rewrite=10,
            rrf_k=60,
            debug=True,
        )
        return cmqr, retriever

    def test_full_pipeline_emits_and_fuses(self, cmqr_setup):
        cmqr, retriever = cmqr_setup
        cmqr.set_batch_context(
            session_ids=["sess_A"],
            turn_numbers=[1],
            extracted_states=[{"mood": "calm", "energy": "low"}],
        )
        out = cmqr.batch_text_to_item_retrieval(
            ["I'm winding down, want something dreamy"], topk=5,
        )
        assert len(out) == 1
        assert isinstance(out[0], list)
        # t_dream_1 appears in 3 of 4 rewrite results — should fuse to top
        assert out[0][0] == "t_dream_1"

    def test_inner_called_with_batched_rewrites(self, cmqr_setup):
        cmqr, retriever = cmqr_setup
        cmqr.set_batch_context(
            session_ids=["sess_A"],
            turn_numbers=[1],
            extracted_states=[None],
        )
        cmqr.batch_text_to_item_retrieval(["original query for fake LM"], topk=5)
        # One batched call to the inner; queries == 4 rewrites
        assert len(retriever.calls) == 1
        call_queries, call_topk, _ = retriever.calls[0]
        assert len(call_queries) == 4  # 4 rewrites
        assert call_topk == 10  # topk_per_rewrite

    def test_cache_hit_skips_lm(self, cmqr_setup, tmp_path):
        cmqr, retriever = cmqr_setup
        cmqr.set_batch_context(
            session_ids=["sess_A"],
            turn_numbers=[1],
            extracted_states=[{"mood": "calm"}],
        )
        # First call — populates cache + bumps stats
        cmqr.batch_text_to_item_retrieval(["first call"], topk=5)
        first_lm_calls = cmqr.lm.lm._call_idx

        # Second call same context — should hit cache, NOT call the LM
        cmqr.set_batch_context(
            session_ids=["sess_A"],
            turn_numbers=[1],
            extracted_states=[{"mood": "calm"}],
        )
        cmqr.batch_text_to_item_retrieval(["second call same ctx"], topk=5)
        assert cmqr.lm.lm._call_idx == first_lm_calls  # no LM hit
        assert cmqr.stats["cache_hits"] == 1

    def test_parse_failure_degrades_to_original(self, tmp_path):
        # LM emits empty output — once "1." is prepended, body is empty so
        # parse_rewrites returns []. CMQR should fall back to [original_query].
        lm = _FakeLM([""])
        retriever = _FakeRetriever({"original query": ["t_orig_1", "t_orig_2"]})
        cmqr = CMQR_REWRITER(
            lm=lm,
            inner_retriever=retriever,
            prompt_path=PROMPT_PATH,
            cache_dir=str(tmp_path / "cache"),
            n_rewrites=4,
            topk_per_rewrite=10,
            debug=True,
        )
        cmqr.set_batch_context(
            session_ids=["sess_X"],
            turn_numbers=[1],
            extracted_states=[None],
        )
        out = cmqr.batch_text_to_item_retrieval(["original query"], topk=5)
        # Inner retriever was called once with one query (the original)
        assert len(retriever.calls) == 1
        assert len(retriever.calls[0][0]) == 1
        assert out[0][0] == "t_orig_1"
        assert cmqr.stats["rewrites_dropped"] == 1
        assert len(cmqr.failures) == 1


# ---------------------------------------------------------------------------
# Misc — set_batch_context length validation
# ---------------------------------------------------------------------------

class TestContextValidation:
    def test_mismatched_lengths_raise(self, tmp_path):
        cmqr = CMQR_REWRITER(
            lm=_FakeLM([""]),
            inner_retriever=_FakeRetriever({}),
            prompt_path=PROMPT_PATH,
            cache_dir=str(tmp_path / "cache"),
        )
        with pytest.raises(ValueError, match="lengths must match"):
            cmqr.set_batch_context(
                session_ids=["a", "b"],
                turn_numbers=[1],
                extracted_states=[None, None],
            )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_shape(self, tmp_path):
        cmqr = CMQR_REWRITER(
            lm=_FakeLM([""]),
            inner_retriever=_FakeRetriever({}),
            prompt_path=PROMPT_PATH,
            cache_dir=str(tmp_path / "cache"),
            n_rewrites=4,
            topk_per_rewrite=50,
            rrf_k=60,
        )
        rep = cmqr.report()
        for key in ["calls", "cache_hits", "rewrites_ok", "rewrites_partial",
                    "rewrites_dropped", "full_rewrite_rate",
                    "n_rewrites_target", "topk_per_rewrite", "rrf_k"]:
            assert key in rep
        assert rep["n_rewrites_target"] == 4
        assert rep["topk_per_rewrite"] == 50
        assert rep["rrf_k"] == 60
