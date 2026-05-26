from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_union_specs_default_has_three_channels():
    specs = _wrrf_union_v1_specs({})
    types = [s["type"] for s in specs]
    assert types == ["bm25", "dense_metadata_qwen3", "same_artist"]


def test_union_specs_add_hyde_when_enabled():
    specs = _wrrf_union_v1_specs({"use_hyde": True, "w_hyde": 0.8})
    types = [s["type"] for s in specs]
    assert "hyde_qwen3" in types
    hyde = next(s for s in specs if s["type"] == "hyde_qwen3")
    assert hyde["weight"] == 0.8
    assert hyde["topk_internal"] == 100
    assert hyde["extra_config"]["hyde_model"] == "Qwen/Qwen2.5-7B-Instruct"
