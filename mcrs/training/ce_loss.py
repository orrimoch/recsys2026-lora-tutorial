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
    device = logits.device
    offset = 0
    losses, weights = [], []
    for size, w in zip(group_sizes, group_weights):
        g = logits[offset:offset + size]
        offset += size
        logp = F.log_softmax(g, dim=0)           # -inf pads contribute 0 to the sum
        losses.append(-logp[0])                  # target index 0 = the gold
        weights.append(w)
    losses = torch.stack(losses)
    w = torch.tensor(weights, device=device, dtype=losses.dtype)
    return (losses * w).sum() / w.sum().clamp(min=1e-8)
