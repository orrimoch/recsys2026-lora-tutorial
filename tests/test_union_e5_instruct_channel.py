"""EXP-009: instruction-tuned ASYMMETRIC retriever dense channel union gating.

The instruct-prefix gave the 0.6B dense a 2x recall jump (nb74 Stage 7), so query
framing — not encoder param count — is the demonstrated lever for this asymmetric
conversation->track-metadata task. multilingual-e5-large-instruct is asymmetric by
design (instruct on query, raw passage) + multilingual (non-ASCII catalog names).
Same DENSE_LOCAL machinery as use_qwen3_4b. Default off; DEV-gated like EXP-008.
"""
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_e5_instruct_off_by_default():
    assert "dense_metadata_e5_instruct_local" not in [s["type"] for s in _wrrf_union_v1_specs({})]


def test_use_e5_instruct_appends_channel_with_default_weight():
    cb = [s for s in _wrrf_union_v1_specs({"use_e5_instruct": True})
          if s["type"] == "dense_metadata_e5_instruct_local"]
    assert len(cb) == 1
    assert cb[0]["weight"] == 0.7
    assert cb[0]["topk_internal"] == 100


def test_w_e5_instruct_overrides_weight():
    cb = [s for s in _wrrf_union_v1_specs({"use_e5_instruct": True, "w_e5_instruct": 1.0})
          if s["type"] == "dense_metadata_e5_instruct_local"]
    assert cb[0]["weight"] == 1.0


def test_e5_replacement_arm_has_e5_but_not_06b():
    # EXP-009 replacement arm: 0.6B -> e5-instruct, single dense-text channel
    types = [s["type"] for s in
             _wrrf_union_v1_specs({"use_base_dense": False, "use_e5_instruct": True})]
    assert "dense_metadata_e5_instruct_local" in types
    assert "dense_metadata_qwen3_instruct" not in types


def test_factory_routes_e5_instruct_to_dense_local(monkeypatch):
    import mcrs.retrieval_modules as rm

    captured = {}

    class _FakeDenseLocal:
        def __init__(self, dataset_name, track_split_types, corpus_types, cache_dir,
                     model_name=None, embed_label=None, instruct=None, instruct_label=None):
            captured.update(model_name=model_name, embed_label=embed_label,
                            instruct=instruct, instruct_label=instruct_label)

    monkeypatch.setattr(rm, "DENSE_LOCAL", _FakeDenseLocal)
    rm.load_retrieval_module("dense_metadata_e5_instruct_local", "ds", ["all_tracks"],
                             ["track_name"], "/tmp/cache")
    assert captured["model_name"] == "intfloat/multilingual-e5-large-instruct"
    assert captured["embed_label"] == "e5-mli-metadata"
    assert captured["instruct"] == rm.E5_MUSIC_INSTRUCT  # asymmetric query-side prefix
    assert captured["instruct_label"] == "e5-instruct-music-v1"
