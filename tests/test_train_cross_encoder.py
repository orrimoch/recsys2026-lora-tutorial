"""Unit tests for the Stage B cross-encoder training helpers (Phase 6c)."""
import torch

from scripts.train_cross_encoder import pairwise_bce_loss


def test_pairwise_bce_lower_for_correct_ranking():
    """Loss is lower when positives score high and negatives score low than
    when the predictions are inverted — confirms the loss drives the intended
    BCE(pos→1) + BCE(neg→0) objective."""
    pos_logits = torch.tensor([5.0, 4.0])
    neg_logits = torch.tensor([-5.0, -4.0, -3.0])

    good = pairwise_bce_loss(pos_logits, neg_logits)
    bad = pairwise_bce_loss(-pos_logits, -neg_logits)  # inverted predictions

    assert torch.isfinite(good)
    assert good < bad


def test_pairwise_bce_neg_weights_scale_negative_term():
    """Per-negative weights scale the negative BCE term — zero-weighting all
    negatives leaves only the positive term (used later for v2 rank-weighting)."""
    pos_logits = torch.tensor([2.0])
    neg_logits = torch.tensor([2.0, 2.0])

    unweighted = pairwise_bce_loss(pos_logits, neg_logits)
    zero_negs = pairwise_bce_loss(
        pos_logits, neg_logits, neg_weights=torch.zeros_like(neg_logits)
    )

    # Zeroing the negative weights must drop the (positive) negative-term loss.
    assert zero_negs < unweighted
    assert torch.isfinite(zero_negs)
