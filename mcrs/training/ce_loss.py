# mcrs/training/ce_loss.py
"""K3b — grouped listwise-softmax (LCE) loss with ragged groups + per-group weight."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_listwise_ce(logits: torch.Tensor, group_sizes: list[int],
                       group_weights: list[float]) -> torch.Tensor:
    """logits: flat (sum(group_sizes),) with the gold at the FIRST position of each group.

    Per group: softmax over its real members (pads encoded as -inf are excluded), CE to target 0,
    then a weighted mean across groups: Σ w_g · L_g / Σ w_g. Per-group max is subtracted by
    F.log_softmax for fp/bf16 stability.
    """
    assert sum(group_sizes) == logits.shape[0], (
        f"group_sizes sum {sum(group_sizes)} != logits length {logits.shape[0]}")
    assert len(group_sizes) == len(group_weights), "one weight per group required"
    device = logits.device
    offset = 0
    losses, weights = [], []
    for size, w in zip(group_sizes, group_weights):
        g = logits[offset:offset + size]
        offset += size
        # pads encoded as -inf contribute 0 to the softmax DENOMINATOR (exp(-inf)=0) and are excluded;
        # log_softmax(-inf) is -inf but that lands only on pad positions, never on the gold at index 0.
        logp = F.log_softmax(g, dim=0)
        losses.append(-logp[0])                  # target index 0 = the gold
        weights.append(w)
    losses = torch.stack(losses)
    w = torch.tensor(weights, device=device, dtype=losses.dtype)
    return (losses * w).sum() / w.sum().clamp(min=1e-8)
