"""Tests that reranker_k (the LLM listwise reranker WINDOW) is threaded from
load_reranker_module through to LLMListwiseReranker.__init__.

EXP-010 found the reranker window k=50->100 is the highest-EV lever (it lets the
reranker see wall golds at union rank 51-100). Serve was hardwired to k=50; this
wires it so a config can ship k=100. Mirrors test_reranker_max_output_tokens_wiring.

No real Gemini client / DB load: LLMListwiseReranker is monkeypatched to capture kwargs.
"""
import pytest
import mcrs.rerankers as R


class _CapturingReranker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.k = kwargs.get("k", 50)   # mirror the real class default


@pytest.fixture
def patch_llm_listwise(monkeypatch):
    import mcrs.rerankers.llm_listwise_rerank as _llm_mod
    monkeypatch.setattr(_llm_mod, "LLMListwiseReranker", _CapturingReranker)
    yield


_COMMON = dict(reranker_type="llm_listwise", item_db_name="fake_db",
               track_split_types=["all_tracks"], corpus_types=[],
               model_path="gemini-2.5-flash")


def test_k_forwarded_to_llm_listwise(tmp_path, patch_llm_listwise):
    rr = R.load_reranker_module(cache_dir=str(tmp_path), k=100, **_COMMON)
    assert isinstance(rr, _CapturingReranker)
    assert rr.k == 100


def test_k_default_is_50(tmp_path, patch_llm_listwise):
    # not passed -> must default to 50 (bit-identical to every shipped config)
    rr = R.load_reranker_module(cache_dir=str(tmp_path), **_COMMON)
    assert rr.k == 50


def test_k_in_kwargs(tmp_path, patch_llm_listwise):
    rr = R.load_reranker_module(cache_dir=str(tmp_path), k=100, **_COMMON)
    assert "k" in rr.kwargs and rr.kwargs["k"] == 100
