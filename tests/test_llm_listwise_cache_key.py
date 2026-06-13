import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "music-crs-baselines"))
sys.path.insert(0, str(REPO))

from mcrs.rerankers.llm_listwise_rerank import LLMListwiseReranker


def _mk(tmp_path, max_tokens):
    # meta_lookup + client supplied -> no DB / no Gemini client created in __init__
    return LLMListwiseReranker(
        "dummy-db", ["all_tracks"], None, str(tmp_path),
        model_path="gemini-2.5-flash", client=object(), meta_lookup={},
        system_prompt="p", max_output_tokens=max_tokens,
    )


def test_cache_key_includes_max_output_tokens(tmp_path):
    r512 = _mk(tmp_path, 512)
    r2048 = _mk(tmp_path, 2048)
    q, head = "a query", ["t1", "t2", "t3"]
    # different token budgets must NOT collide in the cache (output depends on the budget)
    assert r512._cache_path(q, head) != r2048._cache_path(q, head)
    # same budget + same query/head must be stable
    assert r512._cache_path(q, head) == _mk(tmp_path, 512)._cache_path(q, head)
    assert r2048.max_output_tokens == 2048
