"""Item 7 (plan §1/§4): bge-base-en diffuse-context dense channel union gating.

The plan pairs ColBERT's SHARP current-intent matching with bge's DIFFUSE dialog-context
matching — a complementary dense view (bge-base-en-v1.5, 768d) distinct from the Qwen3
dense channel. Opt-in via use_bge; reuses DENSE_LOCAL (catalog embeddings computed +
cached on first run). Default off.
"""
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_bge_off_by_default():
    assert "dense_metadata_bge_base_local" not in [s["type"] for s in _wrrf_union_v1_specs({})]


def test_use_bge_appends_channel_with_default_weight():
    cb = [s for s in _wrrf_union_v1_specs({"use_bge": True})
          if s["type"] == "dense_metadata_bge_base_local"]
    assert len(cb) == 1
    assert cb[0]["weight"] == 0.5
    assert cb[0]["topk_internal"] == 100


def test_w_bge_overrides_weight():
    cb = [s for s in _wrrf_union_v1_specs({"use_bge": True, "w_bge": 0.8})
          if s["type"] == "dense_metadata_bge_base_local"]
    assert cb[0]["weight"] == 0.8
