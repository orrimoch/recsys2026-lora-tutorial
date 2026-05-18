"""Tests for cross-encoder triple-builder."""
import pytest


def test_build_ce_triple_shapes():
    """Cross-encoder triple has query, pos list (len 1), neg list (len 7)."""
    from scripts.build_cross_encoder_training_data import build_ce_triple
    triple = build_ce_triple(
        query="play me jazz",
        gold_track_text="track_name: So What | ...",
        neg_track_texts=["track" + str(i) for i in range(7)],
    )
    assert triple["query"] == "play me jazz"
    assert triple["pos"] == ["track_name: So What | ..."]
    assert len(triple["neg"]) == 7


def test_build_ce_triple_preserves_neg_order():
    """Neg list order is preserved (matters for downstream pair construction)."""
    from scripts.build_cross_encoder_training_data import build_ce_triple
    triple = build_ce_triple(query="q", gold_track_text="g", neg_track_texts=["n1", "n2", "n3"])
    assert triple["neg"] == ["n1", "n2", "n3"]
