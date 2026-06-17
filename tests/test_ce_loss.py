# tests/test_ce_loss.py
import math, torch
from mcrs.training.ce_loss import masked_listwise_ce

def test_perfect_scores_low_loss():
    # group A: [gold=10, neg=0, neg=0]; group B (ragged): [gold=10, neg=0, PAD=-1e9]
    logits = torch.tensor([10., 0., 0., 10., 0., -1e9])     # index 5 is a pad slot in group B
    group_sizes = [3, 3]                                     # group B's 3rd slot is the pad
    loss = masked_listwise_ce(logits, group_sizes, group_weights=[1.0, 1.0])
    assert loss.item() < 0.01

def test_pad_excluded_from_denominator():
    # if the pad (-inf) leaked into the softmax it would change the loss vs a clean 2-item group
    logits = torch.tensor([2., 1., -1e9])
    clean = masked_listwise_ce(torch.tensor([2., 1.]), [2], group_weights=[1.0])
    padded = masked_listwise_ce(logits, [3], group_weights=[1.0])
    assert abs(clean.item() - padded.item()) < 1e-5

def test_group_weight_scales_loss():
    logits = torch.tensor([0., 5.])                          # gold loses -> high loss
    full = masked_listwise_ce(logits, [2], group_weights=[1.0])
    half = masked_listwise_ce(logits, [2], group_weights=[0.5])
    assert abs(half.item() - full.item()) < 1e-5             # normalized: weight cancels for 1 group
