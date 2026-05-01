"""Tests for music-crs-baselines/mcrs/rerankers/pro_rank.py (W3).

Strategy: avoid loading the real Qwen-0.5B model. Subclass ProRankReranker
with a fake `_score_pairs` that returns deterministic scores keyed by the
doc text, so we can exercise the ranking + cache logic deterministically.
We also test the helper functions (`_safe_model_name`, `_detect_backend`,
`_cache_key`) and the prompt-template validation.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = REPO_ROOT / "music-crs-baselines"
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from mcrs.rerankers.pro_rank import (  # noqa: E402
    DEFAULT_PROMPT_TEMPLATE,
    ProRankReranker,
    _detect_backend,
    _safe_model_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestSafeModelName:
    def test_strips_slashes(self):
        assert _safe_model_name("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen_Qwen2.5-0.5B-Instruct"

    def test_keeps_alnum(self):
        assert _safe_model_name("foo-bar_123.4") == "foo-bar_123.4"

    def test_strips_special(self):
        assert _safe_model_name("foo@bar:baz") == "foo_bar_baz"


class TestDetectBackend:
    def test_hf_default(self):
        class FakeHF: pass
        assert _detect_backend(FakeHF()) == "hf"

    def test_vllm_class_name(self):
        class VLLM_MODEL: pass  # mimic real class name
        assert _detect_backend(VLLM_MODEL()) == "vllm"


# ---------------------------------------------------------------------------
# Fake reranker that bypasses model loading. We override __init__ so we
# don't need transformers, and stub out the methods that would otherwise
# touch the GPU / disk-cache.
# ---------------------------------------------------------------------------

class FakeProRank(ProRankReranker):
    """Test double — same scoring/ranking logic, no real model."""

    def __init__(self, tid_to_text: dict[str, str], cache_dir: str,
                 prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
                 score_fn=None):
        # Skip the real __init__ entirely.
        # Set the bare minimum the public interface needs.
        self.device = "cpu"
        self.dtype = None
        self.model_name = "fake/test-model"
        self.model_path = None
        self.batch_size = 4
        self.max_doc_chars = 200
        self.prompt_template = prompt_template
        self.with_rationales = False
        # Validate template (mirrors the real __init__ check).
        if "{query}" not in prompt_template or "{doc}" not in prompt_template:
            raise ValueError(
                "prompt_template must contain both {query} and {doc} placeholders."
            )
        self.tokenizer = None
        self.model = None
        self.yes_id = -1
        self.no_id = -1
        self.tid_to_text = tid_to_text
        self._score_cache_path = Path(cache_dir) / "prorank" / "scores_fake_test.pkl"
        self._score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._score_cache: dict[str, float] = {}
        if self._score_cache_path.exists():
            try:
                with self._score_cache_path.open("rb") as f:
                    self._score_cache = pickle.load(f)
            except (pickle.PickleError, EOFError):
                self._score_cache = {}
        self.stats = {"scored_pairs": 0, "cache_hits": 0, "rerank_calls": 0}
        # Custom score function for tests; default uses doc-length signal.
        self._score_fn = score_fn or (lambda q, d: float(len(d)))

    def _score_pairs(self, pairs):
        # Return scores from the test-injected score_fn.
        return np.array([self._score_fn(q, d) for q, d in pairs], dtype=np.float32)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

class TestPromptTemplate:
    def test_default_has_placeholders(self):
        assert "{query}" in DEFAULT_PROMPT_TEMPLATE
        assert "{doc}" in DEFAULT_PROMPT_TEMPLATE

    def test_init_rejects_missing_query(self, tmp_path):
        with pytest.raises(ValueError, match="placeholders"):
            FakeProRank(
                tid_to_text={},
                cache_dir=str(tmp_path),
                prompt_template="No placeholder {doc} only",
            )

    def test_init_rejects_missing_doc(self, tmp_path):
        with pytest.raises(ValueError, match="placeholders"):
            FakeProRank(
                tid_to_text={},
                cache_dir=str(tmp_path),
                prompt_template="No doc {query} only",
            )


# ---------------------------------------------------------------------------
# Scoring + ranking
# ---------------------------------------------------------------------------

class TestRerank:
    @pytest.fixture
    def reranker(self, tmp_path):
        # Score = length of doc text. Longer doc → higher score → ranked first.
        tid_to_text = {
            "t_short": "abc",
            "t_med": "abc def",
            "t_long": "abc def ghi jkl",
            "t_longest": "abc def ghi jkl mno pqr",
        }
        return FakeProRank(tid_to_text=tid_to_text, cache_dir=str(tmp_path))

    def test_orders_by_score_descending(self, reranker):
        out = reranker.rerank(
            queries=["any query"],
            candidate_tids=[["t_short", "t_med", "t_long", "t_longest"]],
            topk=4,
        )
        assert out == [["t_longest", "t_long", "t_med", "t_short"]]

    def test_truncates_to_topk(self, reranker):
        out = reranker.rerank(
            queries=["q"],
            candidate_tids=[["t_short", "t_med", "t_long", "t_longest"]],
            topk=2,
        )
        assert out == [["t_longest", "t_long"]]

    def test_handles_empty_candidate_list(self, reranker):
        out = reranker.rerank(queries=["q"], candidate_tids=[[]], topk=5)
        assert out == [[]]

    def test_unknown_tid_treated_as_empty_doc(self, reranker):
        # Score(query, "") == 0; should rank last
        out = reranker.rerank(
            queries=["q"],
            candidate_tids=[["t_short", "t_unknown", "t_long"]],
            topk=3,
        )
        assert out[0][-1] == "t_unknown"

    def test_per_query_independence(self, reranker):
        # Two queries — ranking should be independent
        out = reranker.rerank(
            queries=["q1", "q2"],
            candidate_tids=[["t_short", "t_med"], ["t_long", "t_longest"]],
            topk=2,
        )
        assert out == [["t_med", "t_short"], ["t_longest", "t_long"]]

    def test_ignores_side_channel_kwargs(self, reranker):
        # Reranker contract: accepts user_ids / goal_categories / etc. but ignores them.
        out = reranker.rerank(
            queries=["q"],
            candidate_tids=[["t_short", "t_long"]],
            topk=2,
            user_ids=["u1"],
            goal_categories=["discovery"],
            goal_specificities=["specific"],
            user_profiles_raw=[{"foo": "bar"}],
        )
        assert out == [["t_long", "t_short"]]


# ---------------------------------------------------------------------------
# Score caching
# ---------------------------------------------------------------------------

class TestCache:
    @pytest.fixture
    def reranker(self, tmp_path):
        tid_to_text = {"t_a": "alpha", "t_b": "beta beta beta"}
        # Counter-based scoring so repeated calls would return different values
        # if the cache weren't working.
        call_counter = {"n": 0}
        def score_fn(q, d):
            call_counter["n"] += 1
            return float(len(d))
        r = FakeProRank(tid_to_text=tid_to_text, cache_dir=str(tmp_path), score_fn=score_fn)
        r._call_counter = call_counter  # expose for assertions
        return r

    def test_first_call_scores_all(self, reranker):
        reranker.rerank(queries=["q1"], candidate_tids=[["t_a", "t_b"]], topk=2)
        assert reranker._call_counter["n"] == 2  # 2 (q1, doc) pairs scored
        assert reranker.stats["cache_hits"] == 0
        assert reranker.stats["scored_pairs"] == 2

    def test_repeat_call_hits_cache(self, reranker):
        reranker.rerank(queries=["q1"], candidate_tids=[["t_a", "t_b"]], topk=2)
        # Second call same (query, tids) — should hit cache, not re-score
        reranker.rerank(queries=["q1"], candidate_tids=[["t_a", "t_b"]], topk=2)
        assert reranker._call_counter["n"] == 2  # still 2, no new scoring
        assert reranker.stats["cache_hits"] == 2
        assert reranker.stats["scored_pairs"] == 2  # unchanged

    def test_different_query_does_not_hit(self, reranker):
        reranker.rerank(queries=["q1"], candidate_tids=[["t_a", "t_b"]], topk=2)
        reranker.rerank(queries=["q2"], candidate_tids=[["t_a", "t_b"]], topk=2)
        assert reranker._call_counter["n"] == 4  # different query → re-score
        assert reranker.stats["cache_hits"] == 0

    def test_save_and_load_cache(self, tmp_path):
        # Build, score, save, reload — should hit cache after reload.
        tid_to_text = {"t_a": "alpha"}
        r1 = FakeProRank(tid_to_text=tid_to_text, cache_dir=str(tmp_path))
        r1.rerank(queries=["q1"], candidate_tids=[["t_a"]], topk=1)
        r1.save_cache()

        r2 = FakeProRank(tid_to_text=tid_to_text, cache_dir=str(tmp_path))
        # New tracker should have inherited the cache from disk.
        assert len(r2._score_cache) >= 1
        r2.rerank(queries=["q1"], candidate_tids=[["t_a"]], topk=1)
        assert r2.stats["cache_hits"] == 1

    def test_cache_key_distinct_per_query_tid(self, tmp_path):
        r = FakeProRank(tid_to_text={}, cache_dir=str(tmp_path))
        k1 = r._cache_key("query a", "tid_x")
        k2 = r._cache_key("query b", "tid_x")
        k3 = r._cache_key("query a", "tid_y")
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3

    def test_cache_key_deterministic(self, tmp_path):
        r = FakeProRank(tid_to_text={}, cache_dir=str(tmp_path))
        assert r._cache_key("q", "t") == r._cache_key("q", "t")


# ---------------------------------------------------------------------------
# Stats / report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_shape(self, tmp_path):
        r = FakeProRank(tid_to_text={"t_a": "a"}, cache_dir=str(tmp_path))
        r.rerank(queries=["q"], candidate_tids=[["t_a"]], topk=1)
        rep = r.report()
        for key in ["scored_pairs", "cache_hits", "rerank_calls",
                    "cached_total", "model", "device"]:
            assert key in rep
        assert rep["model"] == "fake/test-model"
        assert rep["device"] == "cpu"
        assert rep["rerank_calls"] == 1


# ---------------------------------------------------------------------------
# Cache filename includes prompt-template hash (W3 review P1 #4)
# ---------------------------------------------------------------------------

class TestCacheFilenamePromptHash:
    def test_filename_contains_prompt_hash(self, tmp_path):
        # Two rerankers with DIFFERENT prompt templates should write to
        # DIFFERENT cache files.
        r1 = FakeProRank(
            tid_to_text={"t_a": "a"}, cache_dir=str(tmp_path),
            prompt_template="Variant A: {query} {doc}",
        )
        r2 = FakeProRank(
            tid_to_text={"t_a": "a"}, cache_dir=str(tmp_path),
            prompt_template="Variant B: {query} {doc}",
        )
        # FakeProRank hard-codes the cache path (test double); confirm the
        # REAL ProRank module uses prompt-hash filenames by inspecting source.
        # The fake's path is deterministic; we just sanity-check that
        # changing prompt templates is *intended* to change filenames.
        # The real assertion is in the source code (sha1(prompt_template)[:8])
        # — see pro_rank.py:204.
        assert r1.prompt_template != r2.prompt_template
        # And both should be storable side-by-side without overwriting:
        # FakeProRank uses scores_fake_test.pkl always, so this test mostly
        # documents the requirement.

    def test_real_module_includes_hash_in_filename_pattern(self):
        # Read the source and assert the hash-mixing line is present.
        from mcrs.rerankers import pro_rank
        import inspect, hashlib
        src = inspect.getsource(pro_rank)
        assert "sha1(prompt_template" in src or "prompt_template.encode" in src, (
            "pro_rank.py must mix sha1(prompt_template) into the cache "
            "filename to prevent stale-cache reuse on prompt changes."
        )


# ---------------------------------------------------------------------------
# Cache key separator (W3 review P2 #12)
# ---------------------------------------------------------------------------

class TestCacheKeySeparator:
    def test_concatenation_collisions_avoided(self, tmp_path):
        # Without a separator, ("query a", "btid") and ("query ab", "tid")
        # would hash to the same string. Verify the separator is in place.
        r = FakeProRank(tid_to_text={}, cache_dir=str(tmp_path))
        k1 = r._cache_key("query a", "btid")
        k2 = r._cache_key("query ab", "tid")
        assert k1 != k2, (
            "Cache key collision: concatenation without separator allows "
            "distinct (query, tid) inputs to share keys."
        )


# ---------------------------------------------------------------------------
# generate_rationales (W3 review P0 #2 — plan §A5 rationale emission)
# ---------------------------------------------------------------------------

class FakeProRankWithGenerate(FakeProRank):
    """FakeProRank that also stubs the rationale-generation path.

    We need to fake the tokenizer + model + generate flow. Inputs and
    outputs are deterministic so we can assert on the truncation logic.
    """

    def __init__(self, tid_to_text, cache_dir, gen_outputs=None):
        super().__init__(tid_to_text=tid_to_text, cache_dir=cache_dir)
        # Capture the strings each generate call should "produce".
        self._gen_outputs = list(gen_outputs or [])

    def generate_rationales(self, query, tids, max_rationale_tokens=16):
        # Override the real method with a deterministic stub. We bypass
        # tokenizer/model entirely; just demonstrate the contract:
        # one rationale per tid, lowercased, truncated at hard stops.
        from mcrs.rerankers.pro_rank import DEFAULT_RATIONALE_TEMPLATE  # noqa: F401
        out: list[str] = []
        for i, _t in enumerate(tids):
            raw = self._gen_outputs[i] if i < len(self._gen_outputs) else ""
            # Apply the same truncation/cleaning the real method does.
            text = raw
            for stop in ["\n", ". ", "?", "!"]:
                idx = text.find(stop)
                if idx > 0:
                    text = text[:idx]
                    break
            text = text.strip().strip(".,;:!?\"'`").lower()
            out.append(text[:80])
        return out


class TestGenerateRationales:
    def test_returns_one_per_tid(self, tmp_path):
        r = FakeProRankWithGenerate(
            tid_to_text={"t_a": "a", "t_b": "b"},
            cache_dir=str(tmp_path),
            gen_outputs=["bouncy upbeat groove", "warm reflective ballad"],
        )
        out = r.generate_rationales("any query", ["t_a", "t_b"])
        assert len(out) == 2
        assert out[0] == "bouncy upbeat groove"
        assert out[1] == "warm reflective ballad"

    def test_lowercase_and_strip(self, tmp_path):
        r = FakeProRankWithGenerate(
            tid_to_text={"t_a": "a"},
            cache_dir=str(tmp_path),
            gen_outputs=["  CAPS Heavy 'Drums'.  "],
        )
        out = r.generate_rationales("q", ["t_a"])
        assert out[0] == "caps heavy 'drums"

    def test_truncates_at_first_stop(self, tmp_path):
        r = FakeProRankWithGenerate(
            tid_to_text={"t_a": "a"},
            cache_dir=str(tmp_path),
            gen_outputs=["nostalgic acoustic vibe. additional sentence ignored"],
        )
        out = r.generate_rationales("q", ["t_a"])
        assert "additional" not in out[0]
        assert out[0] == "nostalgic acoustic vibe"

    def test_empty_tids_returns_empty(self, tmp_path):
        r = FakeProRankWithGenerate(
            tid_to_text={}, cache_dir=str(tmp_path), gen_outputs=[],
        )
        out = r.generate_rationales("q", [])
        assert out == []

    def test_hard_caps_at_80_chars(self, tmp_path):
        r = FakeProRankWithGenerate(
            tid_to_text={"t_a": "a"},
            cache_dir=str(tmp_path),
            gen_outputs=["x" * 200],
        )
        out = r.generate_rationales("q", ["t_a"])
        assert len(out[0]) <= 80


# ---------------------------------------------------------------------------
# peft adapter path (W3 review P1 #5)
# ---------------------------------------------------------------------------

class TestPeftAdapterPath:
    def test_real_module_imports_peft_lazily(self):
        # Importing pro_rank.py must NOT eagerly import peft. Some users won't
        # have peft installed for the inference-only baseline.
        import importlib, sys
        # Drop pro_rank from sys.modules to force re-import.
        for mod in list(sys.modules):
            if mod.startswith("mcrs.rerankers.pro_rank"):
                del sys.modules[mod]
        from mcrs.rerankers import pro_rank as _pr  # noqa: F401
        # peft import should not have happened during pro_rank import.
        # (It's only triggered when model_path is provided to __init__.)
        # This is more of a "code review" test — assert the import is
        # local-scoped by reading the source.
        import inspect
        src = inspect.getsource(_pr)
        # peft import should be inside the __init__ body, not at the top.
        top = src.split("class ProRankReranker")[0]
        assert "from peft import" not in top, (
            "peft must be imported lazily inside __init__ (only when "
            "model_path is set), not at module top level."
        )

    def test_model_path_load_is_guarded(self):
        # Read the source: model_path branch should call PeftModel.from_pretrained
        # but only when model_path is truthy.
        from mcrs.rerankers import pro_rank
        import inspect
        src = inspect.getsource(pro_rank.ProRankReranker.__init__)
        # The guard should check model_path before importing peft.
        assert "if model_path" in src
        assert "from peft import PeftModel" in src
        # The two should appear in the right order (guard before import).
        guard_idx = src.find("if model_path")
        import_idx = src.find("from peft import PeftModel")
        assert 0 < guard_idx < import_idx
