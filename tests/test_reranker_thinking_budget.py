"""Plumbing test for the listwise-reranker thinking_budget (flash/pro no-thinking).

reranker_thinking_budget flows config -> load_crs_baseline -> CRS_BASELINE ->
load_reranker_module -> LLMListwiseReranker -> GeminiClient. None=thinking on (default),
0=thinking off. It's also part of the reranker cache key so think/no-think don't collide.
"""
import inspect


def test_thinking_budget_plumbed_through_signatures():
    from mcrs.rerankers.llm_listwise_rerank import LLMListwiseReranker
    from mcrs.rerankers import load_reranker_module
    from mcrs.crs_baseline import CRS_BASELINE
    from mcrs import load_crs_baseline
    assert "thinking_budget" in inspect.signature(LLMListwiseReranker.__init__).parameters
    assert "thinking_budget" in inspect.signature(load_reranker_module).parameters
    assert "reranker_thinking_budget" in inspect.signature(CRS_BASELINE.__init__).parameters
    assert "reranker_thinking_budget" in inspect.signature(load_crs_baseline).parameters
    # default must be None (thinking on) so existing configs are unchanged
    assert inspect.signature(LLMListwiseReranker.__init__).parameters["thinking_budget"].default is None
    assert inspect.signature(load_crs_baseline).parameters["reranker_thinking_budget"].default is None


def test_thinking_budget_in_reranker_cache_key(tmp_path):
    # think vs no-think must NOT share a cache file (else a stale flash-thinking
    # result would be served for the no-thinking config).
    from mcrs.rerankers.llm_listwise_rerank import LLMListwiseReranker
    common = dict(meta_lookup={}, system_prompt="x", cache_dir=str(tmp_path), client=object())
    on = LLMListwiseReranker(thinking_budget=None, **common)
    off = LLMListwiseReranker(thinking_budget=0, **common)
    assert on.thinking_budget is None and off.thinking_budget == 0
    assert on._cache_path("q", ["t1", "t2"]) != off._cache_path("q", ["t1", "t2"])
