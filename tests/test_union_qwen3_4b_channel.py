"""EXP-008: Qwen3-Embedding-4B dense channel union gating.

The provided dense channel runs on Qwen3-Embedding-0.6B (the weak link). This
opt-in channel swaps in the 4B sibling (same family -> the asymmetric music
instruct prefix + DENSE_LOCAL loader are reused; catalog embeddings computed
once via scripts/embed_catalog.py). Tests "is the dense just under-powered?" =
the recall-wall probe. Default off; gate on DEV recall@100 + nDCG before Blind.
"""
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_qwen3_4b_off_by_default():
    assert "dense_metadata_qwen3_4b_local" not in [s["type"] for s in _wrrf_union_v1_specs({})]


def test_use_qwen3_4b_appends_channel_with_default_weight():
    cb = [s for s in _wrrf_union_v1_specs({"use_qwen3_4b": True})
          if s["type"] == "dense_metadata_qwen3_4b_local"]
    assert len(cb) == 1
    assert cb[0]["weight"] == 0.7  # mirrors the 0.6B dense default (w_qwen)
    assert cb[0]["topk_internal"] == 100


def test_w_qwen3_4b_overrides_weight():
    cb = [s for s in _wrrf_union_v1_specs({"use_qwen3_4b": True, "w_qwen3_4b": 1.0})
          if s["type"] == "dense_metadata_qwen3_4b_local"]
    assert cb[0]["weight"] == 1.0


# --- REPLACEMENT arm (RecSys review): drop the 0.6B base dense so the dense-text
#     axis isn't double-weighted in RRF when 4B is added ---

def test_base_dense_on_by_default():
    types = [s["type"] for s in _wrrf_union_v1_specs({})]
    assert "dense_metadata_qwen3_instruct" in types  # backward-compat: default unchanged


def test_use_base_dense_false_drops_the_06b_dense():
    types = [s["type"] for s in _wrrf_union_v1_specs({"use_base_dense": False})]
    assert "dense_metadata_qwen3_instruct" not in types
    assert "dense_metadata_qwen3" not in types
    assert "bm25" in types and "same_artist" in types  # other channels intact


def test_replacement_arm_has_4b_but_not_06b():
    # EXP-008 replacement arm: 0.6B -> 4B, single dense-text channel (no double-weight)
    types = [s["type"] for s in
             _wrrf_union_v1_specs({"use_base_dense": False, "use_qwen3_4b": True})]
    assert "dense_metadata_qwen3_4b_local" in types
    assert "dense_metadata_qwen3_instruct" not in types


# --- factory wiring (code-review): the spec type string must route to DENSE_LOCAL
#     with the right model/label/instruct (the load-bearing string match) ---

def test_factory_routes_qwen3_4b_to_dense_local(monkeypatch):
    import mcrs.retrieval_modules as rm

    captured = {}

    class _FakeDenseLocal:
        def __init__(self, dataset_name, track_split_types, corpus_types, cache_dir,
                     model_name=None, embed_label=None, instruct=None, instruct_label=None):
            captured.update(model_name=model_name, embed_label=embed_label,
                            instruct=instruct)

    monkeypatch.setattr(rm, "DENSE_LOCAL", _FakeDenseLocal)
    rm.load_retrieval_module("dense_metadata_qwen3_4b_local", "ds", ["all_tracks"],
                             ["track_name"], "/tmp/cache")
    assert captured["model_name"] == "Qwen/Qwen3-Embedding-4B"
    assert captured["embed_label"] == "qwen3-4b-metadata"
    assert captured["instruct"] == rm.QWEN3_MUSIC_INSTRUCT  # asymmetric prefix applied
