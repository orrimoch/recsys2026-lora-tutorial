"""Tests for SIDQuantizer wrapper class (RQ-VAE encoder/decoder + Sinkhorn loss)."""
import numpy as np
import pytest
import torch


@pytest.fixture
def small_quantizer():
    """A tiny quantizer for fast unit tests: 16-dim input -> 3 levels x 8 codes."""
    from mcrs.sid.quantizer import SIDQuantizer
    return SIDQuantizer(input_dim=16, latent_dim=8, num_levels=3, codebook_size=8, seed=42)


def test_sid_quantizer_encode_returns_3_token_tuple_per_input(small_quantizer):
    """encode() of a single embedding returns a tuple of 3 ints in [0, codebook_size)."""
    emb = np.random.randn(16).astype(np.float32)
    sid = small_quantizer.encode(emb)
    assert isinstance(sid, tuple)
    assert len(sid) == 3
    for code in sid:
        assert isinstance(code, int)
        assert 0 <= code < 8


def test_sid_quantizer_encode_batch_returns_array_shape_N_3(small_quantizer):
    """encode_batch() of N embeddings returns ndarray of shape (N, 3)."""
    embs = np.random.randn(50, 16).astype(np.float32)
    sids = small_quantizer.encode_batch(embs)
    assert sids.shape == (50, 3)
    assert sids.dtype in (np.int32, np.int64)


def test_sid_quantizer_save_load_round_trip(small_quantizer, tmp_path):
    """Saving and loading the quantizer preserves SID assignments deterministically."""
    from mcrs.sid.quantizer import SIDQuantizer

    embs = np.random.randn(20, 16).astype(np.float32)
    sids_before = small_quantizer.encode_batch(embs)

    path = tmp_path / "quantizer.pt"
    small_quantizer.save(path)
    loaded = SIDQuantizer.load(path)
    sids_after = loaded.encode_batch(embs)

    np.testing.assert_array_equal(sids_before, sids_after)


def test_sid_quantizer_train_step_returns_loss_dict(small_quantizer):
    """train_step on a batch returns a dict with the 3 loss components."""
    embs = torch.randn(8, 16, dtype=torch.float32)
    losses = small_quantizer.train_step(embs)
    assert "mse_recon" in losses
    assert "commitment" in losses
    assert "sinkhorn" in losses
    for v in losses.values():
        assert torch.is_tensor(v)


def test_sid_quantizer_encode_is_deterministic_given_seed(small_quantizer):
    """Same seed + same input = same SID."""
    from mcrs.sid.quantizer import SIDQuantizer

    quantizer_2 = SIDQuantizer(input_dim=16, latent_dim=8, num_levels=3, codebook_size=8, seed=42)
    emb = np.random.RandomState(0).randn(16).astype(np.float32)

    sid_1 = small_quantizer.encode(emb)
    sid_2 = quantizer_2.encode(emb)
    assert sid_1 == sid_2
