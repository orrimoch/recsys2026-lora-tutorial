"""Intent->content two-tower model (Tier-1 #3.3).

Item tower = ItemFusion over frozen multimodal catalog embeddings (reused from
sasrec_model). Query tower = an MLP head over a frozen query embedding (e.g.
Qwen3-Embedding-0.6B over the retrieval query). Both towers L2-normalize their
output so the score is a cosine; trained with in-batch InfoNCE on (intent->gold)
pairs. Pure model + loss — no data I/O (see scripts/train_two_tower.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec_model import ItemFusion


class TwoTowerModel(nn.Module):
    def __init__(self, item_modality_dims: list[int], q_in_dim: int = 1024,
                 d: int = 256, hidden: int = 1536, dropout: float = 0.3,
                 temperature: float = 0.07):
        super().__init__()
        self.d = d
        self.temperature = float(temperature)
        self.item_modality_dims = list(item_modality_dims)
        self.item_fusion = ItemFusion(item_modality_dims, d, hidden=hidden,
                                      dropout=dropout)
        self.query_head = nn.Sequential(
            nn.LayerNorm(q_in_dim),
            nn.Linear(q_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def encode_item(self, feats: torch.Tensor) -> torch.Tensor:
        """frozen item feats (..., sum(modality_dims)) -> L2-normalized (..., d)."""
        return F.normalize(self.item_fusion(feats), dim=-1)

    def encode_query(self, q_emb: torch.Tensor) -> torch.Tensor:
        """frozen query embedding (..., q_in_dim) -> L2-normalized (..., d)."""
        return F.normalize(self.query_head(q_emb), dim=-1)

    def score(self, q_vecs: torch.Tensor, item_vecs: torch.Tensor) -> torch.Tensor:
        """cosine(query, item) / temperature. q_vecs (B,d), item_vecs (N,d) ->
        (B,N). Both are expected L2-normalized (encode_* output)."""
        return (q_vecs @ item_vecs.t()) / self.temperature


def info_nce_loss(model: TwoTowerModel, q_emb: torch.Tensor,
                  pos_feats: torch.Tensor,
                  extra_neg_feats: torch.Tensor | None = None) -> torch.Tensor:
    """In-batch InfoNCE. q_emb (B,q_in_dim), pos_feats (B,sum_mods): item i is the
    positive for query i; the other B-1 batch items are negatives. extra_neg_feats
    (M,sum_mods) appends shared hard negatives. Cross-entropy with diagonal targets.
    """
    q = model.encode_query(q_emb)                  # (B,d)
    pos = model.encode_item(pos_feats)             # (B,d)
    item_bank = pos
    if extra_neg_feats is not None and len(extra_neg_feats):
        item_bank = torch.cat([pos, model.encode_item(extra_neg_feats)], dim=0)
    logits = model.score(q, item_bank)             # (B, B[+M])
    targets = torch.arange(q.shape[0], device=q.device)  # positive i is at column i
    return F.cross_entropy(logits, targets)
