"""Dialog-conditioned, content-fused SASRec: pure sequence builder + the
PyTorch model + the training loss. No data I/O (see scripts/train_sasrec.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_session_examples(track_seqs: list[list[int]], max_len: int) -> list[tuple[list[int], int]]:
    """For each session's ordered track-index list, yield (prefix, target) for
    every position t (0-based): prefix = the up-to-max_len tracks before t,
    target = track at t. Position 0 has an empty prefix (predict the first
    track from the dialog alone)."""
    examples: list[tuple[list[int], int]] = []
    for seq in track_seqs:
        for t in range(len(seq)):
            prefix = seq[max(0, t - max_len):t]
            examples.append((prefix, seq[t]))
    return examples


class ItemFusion(nn.Module):
    """Project a concatenated frozen multimodal track feature vector -> d-dim
    item representation. Only this MLP is trained; the inputs are frozen."""

    def __init__(self, in_dim: int, d: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class SasrecModel(nn.Module):
    """Causal self-attention over [context_token, item_1..item_t].

    Position 0 is the dialog context token (a projection of the dialog
    embedding); positions 1..t are the played-track item-reprs. Positional
    embeddings index the played-track subsequence position (NOT turn_number).
    The session state is the hidden state at the last real position, so an
    empty prefix (turn 1) yields the context-token state.
    """

    def __init__(self, item_in_dim: int, ctx_in_dim: int = 768, d: int = 128,
                 n_layers: int = 2, n_heads: int = 2, max_len: int = 50,
                 temperature: float = 0.07):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.temperature = temperature
        self.item_fusion = ItemFusion(item_in_dim, d)
        self.ctx_proj = nn.Linear(ctx_in_dim, d)
        self.pos_emb = nn.Embedding(max_len + 1, d)  # +1 for the context slot
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            batch_first=True, activation="gelu", dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def encode(self, ctx_emb: torch.Tensor, item_feats: torch.Tensor,
               lengths: torch.Tensor) -> torch.Tensor:
        """ctx_emb (B, ctx_in_dim); item_feats (B, L, item_in_dim);
        lengths (B,) = number of real items per row (0..L).
        Returns the session state (B, d) = hidden at the last real position."""
        B, L, _ = item_feats.shape
        ctx = self.ctx_proj(ctx_emb).unsqueeze(1)                 # (B,1,d)
        items = self.item_fusion(item_feats)                      # (B,L,d)
        seq = torch.cat([ctx, items], dim=1)                      # (B,L+1,d)
        pos = torch.arange(L + 1, device=seq.device)
        seq = seq + self.pos_emb(pos).unsqueeze(0)
        causal = torch.triu(
            torch.full((L + 1, L + 1), float("-inf"), device=seq.device), diagonal=1)
        idx = torch.arange(L + 1, device=seq.device).unsqueeze(0)  # (1,L+1)
        key_padding = idx > lengths.unsqueeze(1)                   # True = pad
        h = self.encoder(seq, mask=causal, src_key_padding_mask=key_padding)
        state = h[torch.arange(B, device=seq.device), lengths]    # last real pos
        return state

    def score(self, state: torch.Tensor, item_matrix: torch.Tensor) -> torch.Tensor:
        return (state @ item_matrix.t()) / self.temperature


def next_item_loss(model: SasrecModel, ctx_emb, item_feats, lengths, target_idx,
                   item_matrix) -> torch.Tensor:
    """Full-catalog softmax cross-entropy for next-item prediction.
    item_matrix (N, d) is model.item_fusion applied to all N catalog items."""
    state = model.encode(ctx_emb, item_feats, lengths)
    logits = model.score(state, item_matrix)
    return F.cross_entropy(logits, target_idx)
