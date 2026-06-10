"""Track B — use_doc_enriched union wiring (plan: drastically improve nDCG@20).
The doc-side enriched dense channel is opt-in and off by default, so the shipped
config-194 union is unchanged."""
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_doc_enriched_off_by_default():
    types = [s["type"] for s in _wrrf_union_v1_specs({})]
    assert "dense_doc_enriched_local" not in types


def test_doc_enriched_added_when_enabled_with_defaults():
    specs = _wrrf_union_v1_specs({"use_doc_enriched": True})
    s = next(s for s in specs if s["type"] == "dense_doc_enriched_local")
    assert s["weight"] == 0.7            # PRIMARY content channel, not a 0.4 side view
    assert s["topk_internal"] == 100
    assert s["extra_config"]["embed_model"] == "BAAI/bge-m3"
    assert s["extra_config"]["embed_label"] == "doc-enriched-v1"
    assert s["extra_config"]["instruct"] is False


def test_doc_enriched_passes_through_overrides():
    specs = _wrrf_union_v1_specs({
        "use_doc_enriched": True, "w_doc_enriched": 1.0,
        "doc_enriched_model": "Qwen/Qwen3-Embedding-4B",
        "doc_enriched_label": "doc-enriched-v2", "doc_enriched_instruct": True})
    s = next(s for s in specs if s["type"] == "dense_doc_enriched_local")
    assert s["weight"] == 1.0
    assert s["extra_config"]["embed_model"] == "Qwen/Qwen3-Embedding-4B"
    assert s["extra_config"]["embed_label"] == "doc-enriched-v2"
    assert s["extra_config"]["instruct"] is True
