"""Gemini API backend for the propose-then-ground channel (Tier-1 #3.5).

Swaps the local Qwen-7B for a hosted Gemini model (e.g. gemini-2.5-flash-lite).
Only the raw text generation differs — `parse_proposals`, the on-disk hash cache,
and `generate_batch` are inherited unchanged from `ProposeGenerator`, since they
are backend-agnostic. Cheap/fast for the 80-query Blind-A set and avoids loading
the 7B locally.

`GeminiClient` lazily imports `google-genai` (only on the first real call), so the
factory can construct + route without the SDK or an API key installed — those are
needed only when generation actually runs (Colab).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from .propose_ground import ProposeGenerator


class GeminiClient:
    """Thin wrapper over the google-genai SDK. `.generate(system, user) -> str`.

    Deterministic by default (temperature 0). Key from GEMINI_API_KEY /
    GOOGLE_API_KEY unless passed explicitly. The SDK import is deferred so that
    merely constructing this (e.g. during factory routing) needs neither the
    package nor a key.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite", api_key: str | None = None,
                 temperature: float = 0.0, max_output_tokens: int = 320,
                 thinking_budget: int | None = None):
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        # thinking_budget=0 disables "thinking" tokens (cheaper + no budget-eating on
        # non-reasoning tasks like catalog enrichment). None = SDK default (unchanged).
        self.thinking_budget = thinking_budget
        self._api_key = (api_key or os.environ.get("GEMINI_API_KEY")
                         or os.environ.get("GOOGLE_API_KEY"))
        self._client = None

    def _ensure(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "Gemini API key not set (GEMINI_API_KEY / GOOGLE_API_KEY).")
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, system_instruction: str, user_content: str) -> str:
        from google.genai import types
        cfg = dict(
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        if self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        resp = self._ensure().models.generate_content(
            model=self.model,
            contents=user_content,
            config=types.GenerateContentConfig(**cfg),
        )
        return getattr(resp, "text", None) or ""


class GeminiProposeGenerator(ProposeGenerator):
    """ProposeGenerator whose `_generate_raw` calls a Gemini client instead of a
    local HF model. `client` is any object exposing `.generate(system, user) -> str`
    (dependency-injected for tests). Inherits caching + parsing unchanged."""

    def __init__(self, client, system_prompt_path, cache_dir: str = "./cache",
                 n_proposals: int = 20, max_new_tokens: int = 320,
                 batch_size: int = 8, max_retries: int = 3):
        super().__init__(None, system_prompt_path, cache_dir=cache_dir,
                         n_proposals=n_proposals, max_new_tokens=max_new_tokens,
                         batch_size=batch_size)
        self.client = client
        self.max_retries = int(max_retries)

    def _generate_one(self, conversation: str) -> str:
        # Mirror build_messages: system carries the proposal count, user the convo.
        system = self.system_prompt.replace("N", str(self.n_proposals))
        user = f"Conversation:\n{conversation.strip()}\n"
        for _ in range(max(1, self.max_retries)):
            try:
                return self.client.generate(system, user)
            except Exception:
                continue  # transient API error -> retry, then degrade to empty
        return ""

    def _generate_raw(self, conversations: list[str]) -> list[str]:
        if not conversations:
            return []
        workers = max(1, min(self.batch_size, len(conversations)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._generate_one, conversations))


def build_propose_generator(pg_model: str, system_prompt_path, cache_dir: str, *,
                            n_proposals: int = 20, batch_size: int = 8,
                            lm_factory=None):
    """Route by model name: `gemini-*` -> Gemini API backend (no local model load);
    anything else -> the local HF `ProposeGenerator` (LLAMA_MODEL). `lm_factory`
    overrides the HF model construction (used by tests to avoid loading a 7B)."""
    if pg_model.lower().startswith("gemini"):
        return GeminiProposeGenerator(
            GeminiClient(model=pg_model), system_prompt_path, cache_dir=cache_dir,
            n_proposals=n_proposals, batch_size=batch_size)
    if lm_factory is None:
        from ..lm_modules.llama import LLAMA_MODEL
        def lm_factory():
            return LLAMA_MODEL(model_name=pg_model, attn_implementation="sdpa")
    return ProposeGenerator(lm_factory(), system_prompt_path, cache_dir=cache_dir,
                            n_proposals=n_proposals, batch_size=batch_size)
