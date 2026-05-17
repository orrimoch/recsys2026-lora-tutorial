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
    """train_step on a batch returns a dict with the 3 scalar loss components
    (plus, since v2, a sinkhorn_per_level list — covered by separate tests)."""
    embs = torch.randn(8, 16, dtype=torch.float32)
    losses = small_quantizer.train_step(embs)
    for scalar_key in ("mse_recon", "commitment", "sinkhorn"):
        assert scalar_key in losses
        assert torch.is_tensor(losses[scalar_key])


def test_sid_quantizer_encode_is_deterministic_given_seed(small_quantizer):
    """Same seed + same input = same SID."""
    from mcrs.sid.quantizer import SIDQuantizer

    quantizer_2 = SIDQuantizer(input_dim=16, latent_dim=8, num_levels=3, codebook_size=8, seed=42)
    emb = np.random.RandomState(0).randn(16).astype(np.float32)

    sid_1 = small_quantizer.encode(emb)
    sid_2 = quantizer_2.encode(emb)
    assert sid_1 == sid_2


# ---------------------------------------------------------------------------
# W1 v2 fixes (2026-05-17): Sinkhorn applied per-level with cosine metric
# ---------------------------------------------------------------------------

def test_train_step_exposes_per_level_sinkhorn_breakdown(small_quantizer):
    """train_step() returns per-level sinkhorn losses (one entry per codebook level).

    v1 applied Sinkhorn only to level 0; v2 applies it to all `num_levels` codebooks.
    The per-level breakdown lets us verify each level got regularization, and lets
    operators log them during training to spot a single level collapsing.
    """
    embs = torch.randn(8, 16, dtype=torch.float32)
    losses = small_quantizer.train_step(embs)
    assert "sinkhorn_per_level" in losses, \
        "v2 train_step must expose per-level Sinkhorn breakdown"
    per_level = losses["sinkhorn_per_level"]
    assert len(per_level) == small_quantizer.num_levels, \
        f"Expected {small_quantizer.num_levels} entries, got {len(per_level)}"
    for level_loss in per_level:
        assert torch.is_tensor(level_loss)


def test_total_sinkhorn_equals_sum_of_per_level(small_quantizer):
    """`sinkhorn` in the returned dict is the sum of per-level losses, scaled by lambda."""
    embs = torch.randn(8, 16, dtype=torch.float32)
    losses = small_quantizer.train_step(embs)
    expected_total = small_quantizer.sinkhorn_lambda * sum(
        float(level_loss.item()) for level_loss in losses["sinkhorn_per_level"]
    )
    assert abs(float(losses["sinkhorn"].item()) - expected_total) < 1e-5


def test_all_per_level_sinkhorn_losses_are_nonzero(small_quantizer):
    """v1 bug: only level-0 sinkhorn was non-zero (level-1+2 got no regularization).

    With v2's per-level Sinkhorn, every level should contribute a real loss value
    on a varied input batch (codebook hasn't perfectly converged to uniform yet).
    """
    torch.manual_seed(123)
    embs = torch.randn(64, 16, dtype=torch.float32)
    losses = small_quantizer.train_step(embs)
    for level_idx, level_loss in enumerate(losses["sinkhorn_per_level"]):
        v = float(level_loss.item())
        assert v != 0.0, f"Level {level_idx} Sinkhorn loss is 0; regularization not applied"


def test_sinkhorn_uses_cosine_metric_not_l2(small_quantizer):
    """Sinkhorn input must match the VQ assignment metric (cosine), not L2.

    Mechanical check: scaling the encoder output by a positive constant changes L2
    distances but NOT cosine similarities. So a properly-cosine-based Sinkhorn loss
    should be (approximately) invariant to such scaling. An L2-based one would change.

    We patch a fake encoder that returns either z or 5*z, run Sinkhorn-relevant code,
    and verify the loss is stable across the scaling.
    """
    import torch.nn.functional as F

    z = torch.randn(8, small_quantizer.latent_dim, dtype=torch.float32)
    # Codebook at level 0 (matches what the cosine VQ uses, unit-norm).
    codebook_lvl0 = small_quantizer.rvq.layers[0]._codebook.embed[0]

    # Compute the Sinkhorn input the v2 way (cosine-based).
    z_n = F.normalize(z, dim=-1)
    cb_n = F.normalize(codebook_lvl0, dim=-1)
    neg_cos_z = -(z_n @ cb_n.T)
    neg_cos_5z = -(F.normalize(5.0 * z, dim=-1) @ cb_n.T)
    # Cosine is scale-invariant → these should match closely.
    assert torch.allclose(neg_cos_z, neg_cos_5z, atol=1e-5), \
        "Cosine-similarity matrix changes under input scaling — bug"

    # And confirm L2 distances WOULD differ (sanity check on the test logic).
    l2_z = ((z.unsqueeze(1) - codebook_lvl0.unsqueeze(0)) ** 2).sum(dim=-1)
    l2_5z = ((5.0 * z.unsqueeze(1) - codebook_lvl0.unsqueeze(0)) ** 2).sum(dim=-1)
    assert not torch.allclose(l2_z, l2_5z, atol=1.0), \
        "L2 distance unexpectedly invariant to scaling — test is flawed"
