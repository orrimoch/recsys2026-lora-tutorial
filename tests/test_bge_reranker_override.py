"""Test that BGE_RERANKER accepts a model_name override via constructor."""
import inspect

import pytest


def test_bge_reranker_constructor_accepts_model_name_kwarg():
    """The constructor signature includes `model_name` so the factory can pass
    `reranker_model_path` through to it."""
    from mcrs.rerankers.bge_reranker import BGE_RERANKER
    sig = inspect.signature(BGE_RERANKER.__init__)
    assert "model_name" in sig.parameters, \
        "BGE_RERANKER.__init__ must accept a model_name kwarg for fine-tuned model loading"


def test_reranker_factory_passes_model_path_to_bge_reranker(monkeypatch, tmp_path):
    """When reranker_type='bge_reranker_v2_m3' and model_path is set, the factory
    forwards model_path → BGE_RERANKER(model_name=...)."""
    from mcrs.rerankers import load_reranker_module

    captured = {}

    class _Stub:
        def __init__(self, item_db_name, track_split_types, corpus_types, cache_dir, model_name=None):
            captured["model_name"] = model_name
            captured["called"] = True

    import mcrs.rerankers.bge_reranker as bge_mod
    monkeypatch.setattr(bge_mod, "BGE_RERANKER", _Stub)

    load_reranker_module(
        reranker_type="bge_reranker_v2_m3",
        item_db_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        track_split_types=["all_tracks"],
        corpus_types=["track_name"],
        cache_dir=str(tmp_path),
        model_path="OrRim123/recsys2026-bge-reranker-music-v1",
    )
    assert captured["called"]
    assert captured["model_name"] == "OrRim123/recsys2026-bge-reranker-music-v1"
