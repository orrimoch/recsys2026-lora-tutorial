"""Tests for the reranker factory (load_reranker_module).

Covers the multimodal_cross_encoder registration's input validation. The
success path constructs MULTIMODAL_RERANKER (loads a 570M model + artifacts), so
only the error/dispatch paths are unit-tested here; the model path is exercised
on Colab.
"""
import pytest

from mcrs.rerankers import load_reranker_module

_COMMON = dict(
    item_db_name="db",
    track_split_types=["all_tracks"],
    corpus_types=["track_name"],
)


def test_falsy_type_returns_none():
    assert load_reranker_module(None, **_COMMON) is None
    assert load_reranker_module("", **_COMMON) is None


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unsupported reranker type"):
        load_reranker_module("no_such_reranker", **_COMMON)


def test_multimodal_cross_encoder_requires_model_path():
    with pytest.raises(ValueError, match="reranker_model_path"):
        load_reranker_module("multimodal_cross_encoder", **_COMMON)


def test_multimodal_cross_encoder_requires_artifacts():
    with pytest.raises(ValueError, match="reranker_multimodal_artifacts"):
        load_reranker_module(
            "multimodal_cross_encoder", model_path="some/model/dir", **_COMMON
        )
