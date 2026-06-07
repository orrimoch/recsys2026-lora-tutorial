"""Tier-1 #3.1a: attributes-qwen3 (instruct) as an opt-in second dense channel.

Orthogonal to the metadata dense field (0.62 same-track cosine, 3.6% neighbor
overlap), so it can surface new-artist / cold golds the metadata channel misses.
Opt-in via use_attributes so the shipped config 194 is unchanged (regression
safety for Tier 0). Mirrors the use_lyrics / use_clap_recall gating idiom.
"""
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_attributes_channel_off_by_default():
    types = [s["type"] for s in _wrrf_union_v1_specs({})]
    assert "dense_attributes_qwen3_instruct" not in types


def test_use_attributes_appends_instruct_channel():
    specs = _wrrf_union_v1_specs({"use_attributes": True})
    attr = [s for s in specs if s["type"] == "dense_attributes_qwen3_instruct"]
    assert len(attr) == 1
    assert attr[0]["weight"] == 0.4  # default
    assert attr[0]["topk_internal"] == 100


def test_w_attributes_overrides_weight():
    specs = _wrrf_union_v1_specs({"use_attributes": True, "w_attributes": 0.7})
    attr = [s for s in specs if s["type"] == "dense_attributes_qwen3_instruct"]
    assert attr[0]["weight"] == 0.7


def test_attributes_is_added_alongside_metadata_not_swapped():
    # Scope decision: ADD, do not replace the metadata dense channel.
    types = [s["type"] for s in _wrrf_union_v1_specs({"use_attributes": True})]
    assert "dense_metadata_qwen3_instruct" in types
    assert "dense_attributes_qwen3_instruct" in types
