"""Tests that reranker_max_output_tokens is correctly threaded from
load_reranker_module through to LLMListwiseReranker.__init__.

No real Gemini client, no DB load: the LLMListwiseReranker constructor is
monkeypatched so we can capture the exact kwargs it receives.
"""
import pytest
import mcrs.rerankers as R


class _CapturingReranker:
    """Records the kwargs passed to it so the test can assert on them."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # Expose max_output_tokens at instance level (mirroring the real class).
        self.max_output_tokens = kwargs.get("max_output_tokens", 512)


@pytest.fixture(autouse=False)
def patch_llm_listwise(monkeypatch):
    """Replace LLMListwiseReranker inside the llm_listwise_rerank module.

    load_reranker_module does `from .llm_listwise_rerank import LLMListwiseReranker`
    lazily inside the function body, so the symbol we must patch lives in the
    llm_listwise_rerank sub-module — not in mcrs.rerankers itself.
    """
    import mcrs.rerankers.llm_listwise_rerank as _llm_mod
    monkeypatch.setattr(_llm_mod, "LLMListwiseReranker", _CapturingReranker)
    yield


def test_max_output_tokens_forwarded_to_llm_listwise(tmp_path, patch_llm_listwise):
    """load_reranker_module passes max_output_tokens=2048 to LLMListwiseReranker."""
    rr = R.load_reranker_module(
        reranker_type="llm_listwise",
        item_db_name="fake_db",
        track_split_types=["all_tracks"],
        corpus_types=[],
        cache_dir=str(tmp_path),
        model_path="gemini-2.5-flash",
        max_output_tokens=2048,
    )
    assert isinstance(rr, _CapturingReranker), "Expected _CapturingReranker instance"
    assert rr.max_output_tokens == 2048, (
        f"Expected max_output_tokens=2048, got {rr.max_output_tokens}"
    )


def test_max_output_tokens_default_is_512(tmp_path, patch_llm_listwise):
    """load_reranker_module uses default max_output_tokens=512 when not specified."""
    rr = R.load_reranker_module(
        reranker_type="llm_listwise",
        item_db_name="fake_db",
        track_split_types=["all_tracks"],
        corpus_types=[],
        cache_dir=str(tmp_path),
        model_path="gemini-2.5-flash",
        # max_output_tokens NOT passed — must default to 512
    )
    assert isinstance(rr, _CapturingReranker)
    assert rr.max_output_tokens == 512, (
        f"Expected default max_output_tokens=512, got {rr.max_output_tokens}"
    )


def test_max_output_tokens_forwarded_in_kwargs(tmp_path, patch_llm_listwise):
    """The kwarg shows up explicitly in the captured kwargs dict."""
    rr = R.load_reranker_module(
        reranker_type="llm_listwise",
        item_db_name="fake_db",
        track_split_types=["all_tracks"],
        corpus_types=[],
        cache_dir=str(tmp_path),
        model_path="gemini-2.5-flash",
        max_output_tokens=1024,
    )
    assert "max_output_tokens" in rr.kwargs, (
        "max_output_tokens must be explicitly passed as a kwarg to LLMListwiseReranker"
    )
    assert rr.kwargs["max_output_tokens"] == 1024
