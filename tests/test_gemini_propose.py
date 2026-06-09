"""Tier-1 #3.5 Gemini backend for propose-then-ground.

The real Gemini API call is integration (Colab). These test the new logic with a
FAKE client (dependency-injected): prompt build, response->proposals parsing,
graceful degradation on API error, on-disk caching, and pg_model routing — none
of which need google-genai installed or an API key.
"""
import pytest

from mcrs.query_rewriters.propose_ground import ProposeGenerator
from mcrs.query_rewriters.gemini_propose import (
    GeminiProposeGenerator,
    build_propose_generator,
)


def _prompt(tmp_path):
    p = tmp_path / "propose.txt"
    p.write_text("List N tracks as 'Artist - Title'.", encoding="utf-8")
    return str(p)


class _FakeClient:
    """Duck-typed Gemini client: .generate(system, user) -> str."""

    def __init__(self, response="1. A - x\n2. B - y", raises=False):
        self.response = response
        self.raises = raises
        self.calls = []

    def generate(self, system_instruction, user_content):
        self.calls.append((system_instruction, user_content))
        if self.raises:
            raise RuntimeError("api error")
        return self.response


def test_calls_client_and_parses_proposals(tmp_path):
    client = _FakeClient("1. Radiohead - Creep\n2. Muse - Bliss")
    gen = GeminiProposeGenerator(client, _prompt(tmp_path), cache_dir=str(tmp_path),
                                 n_proposals=20, batch_size=2)
    out = gen.generate_batch(["i want sad rock"])
    assert out == [{"proposals": ["Radiohead - Creep", "Muse - Bliss"]}]
    # system instruction carries the proposal count; user carries the conversation
    sys_i, user = client.calls[0]
    assert "20" in sys_i
    assert "i want sad rock" in user


def test_api_error_degrades_to_empty(tmp_path):
    client = _FakeClient(raises=True)
    gen = GeminiProposeGenerator(client, _prompt(tmp_path), cache_dir=str(tmp_path),
                                 batch_size=2, max_retries=2)
    out = gen.generate_batch(["q"])
    assert out == [{"proposals": []}]          # channel degrades, never crashes
    assert len(client.calls) == 2              # retried max_retries times first


def test_caches_responses(tmp_path):
    client = _FakeClient("1. A - x")
    gen = GeminiProposeGenerator(client, _prompt(tmp_path), cache_dir=str(tmp_path))
    gen.generate_batch(["same q"])
    gen.generate_batch(["same q"])             # second call hits the on-disk cache
    assert len(client.calls) == 1


def test_n_proposals_truncates(tmp_path):
    client = _FakeClient("1. A - x\n2. B - y\n3. C - z")
    gen = GeminiProposeGenerator(client, _prompt(tmp_path), cache_dir=str(tmp_path),
                                 n_proposals=2)
    assert gen.generate_batch(["q"]) == [{"proposals": ["A - x", "B - y"]}]


def test_build_propose_generator_routes_gemini(tmp_path):
    gen = build_propose_generator("gemini-2.5-flash-lite", _prompt(tmp_path),
                                  str(tmp_path), n_proposals=20, batch_size=8)
    assert isinstance(gen, GeminiProposeGenerator)


def test_build_propose_generator_routes_hf_without_loading_model(tmp_path):
    class _FakeLM:
        pass
    gen = build_propose_generator("Qwen/Qwen2.5-7B-Instruct", _prompt(tmp_path),
                                  str(tmp_path), lm_factory=lambda: _FakeLM())
    assert isinstance(gen, ProposeGenerator)
    assert not isinstance(gen, GeminiProposeGenerator)
