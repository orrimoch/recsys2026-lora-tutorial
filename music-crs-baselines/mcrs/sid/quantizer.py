"""SIDQuantizer — RQ-VAE wrapper using vector-quantize-pytorch + Sinkhorn loss term.

Spec §2.2 architecture:
  L2-norm(text) ⊕ L2-norm(CF) ⊕ L2-norm(audio)
    → MLP encoder (1664 → 512 → 256)
    → ResidualVQ × 3 levels, codebook=256 each
    → MLP decoder (256 → 512 → 1664)
  Loss = MSE_recon + 0.25·commitment_loss + λ·sinkhorn_uniform_loss

W1 v2 fix history (2026-05-17):
  v1 Sinkhorn applied L2-distance regularization to LEVEL 0 ONLY, which
  (a) doesn't match the cosine-VQ assignment metric, and
  (b) leaves residual layers (L1, L2) un-regularized — driving the
      "hourglass" codebook collapse we observed in the W3 v1 diagnostic.
  v2 applies cosine-similarity-based Sinkhorn to ALL `num_levels` codebooks,
  walking the residual chain so each level's anti-collapse pressure matches
  what that level's codebook is meant to capture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


def _set_seed(seed: int) -> None:
    """Pin all RNG state for reproducible RQ-VAE training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sinkhorn_uniform_loss(
    costs: torch.Tensor,
    n_iters: int = 5,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Approximate Sinkhorn-uniform loss over codebook assignments (LC-Rec recipe motivation).

    Encourages each code to receive roughly 1/K of the total assignment mass,
    preventing codebook collapse. `costs` is a (B, K) matrix where LOWER = more
    affinity (e.g., L2 squared distance, OR negative cosine similarity for
    cosine-VQ setups). We Sinkhorn-normalize and penalize deviation from uniform
    code marginals.

    Renamed from `distances` to `costs` (v2): cosine-VQ uses negative cosine sim
    as the cost, not L2. The function is metric-agnostic; the caller picks the
    cost matrix to match whatever metric the VQ uses.
    """
    B, K = costs.shape
    log_q = -costs / max(epsilon, 1e-6)
    log_q = log_q - log_q.logsumexp(dim=1, keepdim=True)  # row-normalize
    for _ in range(n_iters):
        col_marg = log_q.logsumexp(dim=0)
        log_q = log_q - col_marg + np.log(B / K)
        row_marg = log_q.logsumexp(dim=1, keepdim=True)
        log_q = log_q - row_marg
    final_col = log_q.logsumexp(dim=0)
    target = torch.full_like(final_col, fill_value=np.log(B / K))
    return (final_col.exp() * (final_col - target)).sum()


def _cosine_cost_matrix(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Cost matrix for cosine-VQ Sinkhorn: -cos(z, code) per (input, code) pair.

    Both z and codebook are L2-normalized first so the cost is in [-1, 1] (lower =
    higher cosine similarity = more affinity). Matches the metric the cosine-VQ
    uses internally for assignment.
    """
    z_n = F.normalize(z, dim=-1)
    cb_n = F.normalize(codebook, dim=-1)
    return -(z_n @ cb_n.T)


class SIDQuantizer:
    """RQ-VAE wrapper that encodes embeddings into N-token SIDs.

    Default config matches spec §2.2:
      input_dim=1664, latent_dim=256, num_levels=3, codebook_size=256
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        num_levels: int = 3,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        sinkhorn_lambda: float = 0.10,
        seed: int = 42,
    ) -> None:
        _set_seed(seed)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.sinkhorn_lambda = sinkhorn_lambda
        self.seed = seed

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(latent_dim * 2, 64)),
            nn.GELU(),
            nn.Linear(max(latent_dim * 2, 64), latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim * 2, 64)),
            nn.GELU(),
            nn.Linear(max(latent_dim * 2, 64), input_dim),
        )
        # Anti-collapse configuration (2026-05-16, after 2 full-scale runs showed
        # chronic level-1 collapse to 2/256 codes even with dead-code revival):
        #
        # ROOT CAUSE: our input is L2-normalized per modality (text + CF + audio).
        # It lives near a unit hypersphere. L2-distance VQ on hypersphere data
        # collapses everything to the centroid; codes pull toward the centroid and
        # the encoder learns to map all points to ~the same latent direction.
        # Dead-code revival can't recover because revived codes get pulled back.
        #
        # FIX (canonical for normalized embeddings):
        # - use_cosine_sim=True: VQ assignment uses cosine similarity instead of L2.
        #   Codebook entries are forced to unit-norm, naturally spread on the sphere.
        # - orthogonal_reg_weight=1.0: penalizes pairs of codes that are similar,
        #   driving them apart. Lowered from 10.0 -> 1.0 (2026-05-16 v3.1) because
        #   too-aggressive ortho_reg starves residual layers (level 2/3) of natural
        #   structure to capture.
        # - threshold_ema_dead_code=2 + decay=0.8: standard dead-code revival.
        # - kmeans_iters=20: more thorough init.
        self.rvq = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_levels,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            use_cosine_sim=True,
            orthogonal_reg_weight=1.0,
            kmeans_init=True,
            kmeans_iters=20,
            threshold_ema_dead_code=2,
            decay=0.8,
        )

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.rvq.parameters())

    def to(self, device):
        self.encoder.to(device); self.decoder.to(device); self.rvq.to(device)
        return self

    def train_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """One training step: encode → quantize → decode → compute losses.

        v2 (2026-05-17): Sinkhorn applied per-level using cosine cost (matches the
        cosine-VQ assignment metric) and walks the residual chain so each level's
        codebook gets anti-collapse pressure on the input distribution that level
        is actually meant to quantize.
        """
        z = self.encoder(batch)
        z_q, indices, commitment_loss = self.rvq(z)
        recon = self.decoder(z_q)
        mse = ((recon - batch) ** 2).mean()

        # Per-level Sinkhorn over cosine cost. Walk the residual chain so the
        # cost matrix at level L uses the residual the L-th codebook actually sees.
        sinkhorn_per_level: list[torch.Tensor] = []
        z_residual = z
        for level in range(self.num_levels):
            codebook = self.rvq.layers[level]._codebook.embed[0]  # (K, D)
            cost = _cosine_cost_matrix(z_residual, codebook)
            sinkhorn_per_level.append(_sinkhorn_uniform_loss(cost))
            # Subtract this level's assigned (unit-norm) code to form the next residual.
            assigned_idx = indices[..., level] if indices.dim() > 1 else indices
            assigned_code = F.normalize(codebook, dim=-1)[assigned_idx]
            z_residual = z_residual - assigned_code

        sinkhorn_total = torch.stack(sinkhorn_per_level).sum()
        return {
            "mse_recon": mse,
            "commitment": commitment_loss.mean() if commitment_loss.dim() > 0 else commitment_loss,
            "sinkhorn": self.sinkhorn_lambda * sinkhorn_total,
            "sinkhorn_per_level": sinkhorn_per_level,
        }

    @torch.no_grad()
    def encode_batch(self, embs: np.ndarray) -> np.ndarray:
        """Encode N embeddings → (N, num_levels) array of code indices.

        Forces eval() mode on encoder + rvq before encoding so codebook EMA
        and dead-code revival don't mutate state during inference (otherwise
        save→encode→load→encode produces different SIDs because the codebook
        shifts in the first encode pass).
        """
        device = next(self.encoder.parameters()).device
        x = torch.from_numpy(np.asarray(embs, dtype=np.float32)).to(device)
        was_training_enc = self.encoder.training
        was_training_rvq = self.rvq.training
        self.encoder.eval()
        self.rvq.eval()
        try:
            z = self.encoder(x)
            _, indices, _ = self.rvq(z)
            return indices.detach().cpu().numpy().astype(np.int64)
        finally:
            if was_training_enc: self.encoder.train()
            if was_training_rvq: self.rvq.train()

    def encode(self, emb: np.ndarray) -> tuple[int, ...]:
        """Encode a single embedding → tuple of num_levels code indices."""
        sids = self.encode_batch(np.asarray(emb)[None, :])
        return tuple(int(c) for c in sids[0])

    def save(self, path) -> None:
        """Save the quantizer state (weights + hyperparameters)."""
        torch.save({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "commitment_weight": self.commitment_weight,
            "sinkhorn_lambda": self.sinkhorn_lambda,
            "seed": self.seed,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "rvq": self.rvq.state_dict(),
        }, path)

    @classmethod
    def load(cls, path) -> "SIDQuantizer":
        """Load a saved quantizer; restores deterministic encoding."""
        ckpt = torch.load(path, weights_only=False)
        q = cls(
            input_dim=ckpt["input_dim"],
            latent_dim=ckpt["latent_dim"],
            num_levels=ckpt["num_levels"],
            codebook_size=ckpt["codebook_size"],
            commitment_weight=ckpt["commitment_weight"],
            sinkhorn_lambda=ckpt["sinkhorn_lambda"],
            seed=ckpt["seed"],
        )
        q.encoder.load_state_dict(ckpt["encoder"])
        q.decoder.load_state_dict(ckpt["decoder"])
        q.rvq.load_state_dict(ckpt["rvq"])
        return q
