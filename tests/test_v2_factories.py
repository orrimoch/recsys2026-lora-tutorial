"""Tests for the v2 wRRF factory variant that uses the fine-tuned BGE-M3."""
import pytest


def test_wrrf_bm25_dense_lyrics_bge_m3_ft_v1_factory_recognizes_type(tmp_path):
    """New factory recognizes the retrieval_type key (does NOT raise 'Unsupported')."""
    from mcrs.retrieval_modules import load_retrieval_module

    try:
        load_retrieval_module(
            retrieval_type="wrrf_bm25_dense_lyrics_bge_m3_ft_v1",
            dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            track_split_types=["all_tracks"],
            corpus_types=["track_name", "artist_name", "album_name"],
            cache_dir=str(tmp_path),
            extra_config={"bge_m3_hub_repo": "OrRim123/recsys2026-bge-m3-music-v1-merged"},
        )
    except ValueError as e:
        # Acceptable: sub-retrievers can't fully init in a test sandbox (no
        # cached embeddings) — but the factory MUST recognize the type.
        assert "Unsupported retrieval type" not in str(e), \
            f"factory failed to register the new retrieval_type: {e}"
    except FileNotFoundError:
        # Also acceptable: DENSE_LOCAL needs precomputed catalog pickles that
        # don't exist in tmp_path. Reaching this branch proves the factory
        # dispatched correctly.
        pass
