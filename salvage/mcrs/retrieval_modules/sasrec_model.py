"""Dialog-conditioned, content-fused SASRec: pure sequence builder + the
PyTorch model + the training loss. No data I/O (see scripts/train_sasrec.py)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_item_feats_mode(feats, modality_dims, mode: str = "content",
                          seed: int = 42):
    """Transform the concatenated catalog item features for the content-fusion
    ablation. `feats` is (N, sum(modality_dims)).

    Modes (one ablation axis = the item representation source):
      content  — unchanged; real metadata + audio. The production channel.
      metadata — keep only the first modality block (Qwen3 text).
      audio    — keep only the trailing modality block(s) (CLAP).
      random   — replace every value with a seeded random vector of the SAME
                 shape. Each item gets a unique fixed vector, so the fusion MLP
                 can only memorize a per-item embedding — an ID-only proxy with
                 no shared content structure. Isolates how much real content
                 (vs memorized ids) buys cold-item recall.

    Returns (feats, modality_dims), both adjusted to the mode.
    """
    if mode == "content":
        return feats, list(modality_dims)
    if mode == "metadata":
        d0 = modality_dims[0]
        return feats[:, :d0].copy(), [d0]
    if mode == "audio":
        d0 = modality_dims[0]
        return feats[:, d0:].copy(), list(modality_dims[1:])
    if mode == "random":
        rng = np.random.RandomState(seed)
        return rng.randn(*feats.shape).astype(np.float32), list(modality_dims)
    raise ValueError(f"unknown item-feats mode: {mode!r}")


def prior_turns(df, turn_number):
    """The SASRec conditioning slice for a target music turn: every turn BEFORE
    `turn_number` (all roles) plus the current-turn USER request. The current
    music (gold) turn and all future turns are excluded.

    Single source of truth for train/serve/feature-build parity: train_sasrec
    (training), build_lgbm_features (reranker features), and the nb74 dev harness
    all slice via this so they cannot drift. Operates on the passed dataframe's
    own API (no pandas import here — this module stays dependency-light)."""
    return df[(df["turn_number"] < turn_number)
              | ((df["turn_number"] == turn_number) & (df["role"] == "user"))]


def build_user_dialog(turns) -> str:
    """User-turns-only dialog text: newline-join the `content` of turns whose
    role is 'user', in order. `turns` is an iterable of dict-like rows with
    'role' and 'content'. Played-track ('music') turns are intentionally
    excluded — they are represented in the item sequence, not the context token."""
    return "\n".join(
        str(t["content"]) for t in turns if t.get("role") == "user"
    )


def inpool_target_index(pool_tids, gold_tid):
    """Column index of the gold within the recall pool, or None if the gold is
    NOT in the pool -> SKIP the sample (SASRec_Improved_Plan.md §4). At serve the
    ranker only reorders the pool, so a gold recall missed is unrecoverable; it is
    a recall failure, not a ranking one, and must not be a training target."""
    try:
        return list(pool_tids).index(gold_tid)
    except ValueError:
        return None


def build_sasrec_context(turns, goal_text: str | None = None) -> str:
    """SASRec context = user-dialog + optional listener_goal. SINGLE SOURCE for
    train/dev-gate/serve parity (SASRec_Improved_Plan.md): build it identically
    everywhere or the gate won't predict Blind. Empty/whitespace/None goal
    degrades to goal-less IDENTICALLY — mirrors build_retrieval_query's
    'raw_with_goal' format (`f"{base}\\ngoal: {gt}" if gt else base`)."""
    base = build_user_dialog(turns)
    gt = (goal_text or "").strip()
    return f"{base}\ngoal: {gt}" if gt else base


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
    """Project per-modality frozen track features -> d-dim item representation.

    The frozen multimodal track embeddings come from different sources at
    wildly different scales (Qwen3-metadata ~unit-norm, CLAP unnormalized, etc.).
    Each modality gets its OWN LayerNorm before concat so the MLP doesn't burn
    capacity discovering scale. After concat we expand to a wide hidden, LN +
    GELU + dropout, then project to d. Only the LNs and the MLP are trained;
    the input modalities are frozen catalog vectors.
    """

    def __init__(self, modality_dims: list[int], d: int = 256,
                 hidden: int = 1536, dropout: float = 0.3):
        super().__init__()
        self.modality_dims = list(modality_dims)
        in_dim = sum(self.modality_dims)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(m) for m in self.modality_dims])
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (..., sum(modality_dims)). Split per modality, LN each, concat, MLP.
        parts = torch.split(feats, self.modality_dims, dim=-1)
        normed = torch.cat(
            [norm(part) for norm, part in zip(self.norms, parts)], dim=-1)
        return self.net(normed)


class SasrecModel(nn.Module):
    """Causal self-attention over [context_token, item_1..item_t].

    Position 0 is the dialog context token (a projection of the dialog
    embedding); positions 1..t are the played-track item-reprs. Positional
    embeddings index the played-track subsequence position (NOT turn_number).
    The session state is the hidden state at the last real position, so an
    empty prefix (turn 1) yields the context-token state.
    """

    def __init__(self, item_modality_dims: list[int], ctx_in_dim: int = 768,
                 d: int = 256, n_layers: int = 2, n_heads: int = 2,
                 max_len: int = 50, temperature: float = 0.07):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.temperature = temperature
        self.item_modality_dims = list(item_modality_dims)
        self.item_fusion = ItemFusion(item_modality_dims, d)
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
        # Bool causal mask matches the dtype of `src_key_padding_mask` below, so
        # nn.TransformerEncoder doesn't emit the "mismatched mask" deprecation
        # warning and won't break on a future PyTorch upgrade.
        causal = torch.triu(
            torch.ones(L + 1, L + 1, dtype=torch.bool, device=seq.device), diagonal=1)
        idx = torch.arange(L + 1, device=seq.device).unsqueeze(0)  # (1,L+1)
        key_padding = idx > lengths.unsqueeze(1)                   # True = pad
        h = self.encoder(seq, mask=causal, src_key_padding_mask=key_padding)
        state = h[torch.arange(B, device=seq.device), lengths]    # last real pos
        return state

    def score(self, state: torch.Tensor, item_matrix: torch.Tensor) -> torch.Tensor:
        # L2-normalize state and items so the dot product is a cosine; the
        # temperature is then well-calibrated (standard contrastive practice).
        # Without normalization the init-scale dot products at d=128 are O(±20),
        # which divided by 0.07 makes the softmax near one-hot from step 0 —
        # killing the early-epoch gradient signal.
        state = F.normalize(state, dim=-1)
        item_matrix = F.normalize(item_matrix, dim=-1)
        return (state @ item_matrix.t()) / self.temperature


def next_item_loss(model: SasrecModel, ctx_emb, item_feats, lengths, target_idx,
                   item_matrix, label_smoothing: float = 0.05) -> torch.Tensor:
    """Full-catalog softmax cross-entropy for next-item prediction.
    item_matrix (N, d) is model.item_fusion applied to all N catalog items.

    label_smoothing default 0.05: with the full-catalog softmax (N ~ 60k) the
    target-only one-hot is overconfident and was the dominant driver of the
    val_loss flatlining while train_loss kept dropping. Tests that need to
    measure pure cross-entropy convergence (the overfit-batch test) pass 0.
    """
    state = model.encode(ctx_emb, item_feats, lengths)
    logits = model.score(state, item_matrix)
    return F.cross_entropy(logits, target_idx, label_smoothing=label_smoothing)


def inpool_loss(model: SasrecModel, ctx_emb, played_feats, lengths,
                pool_feats, target_in_pool, label_smoothing: float = 0.0) -> torch.Tensor:
    """In-pool contrastive loss (SASRec_Improved_Plan.md): softmax cross-entropy
    over each sample's OWN candidate POOL, not the full catalog. Trains the model
    on the exact serve task — rank the gold #1 among the ~K plausible tracks recall
    surfaced — which the full-catalog `next_item_loss` never does.

    Same dual-encoder paradigm as `score`: encode the session once, cosine vs each
    pool item, softmax. Differs only in that each row has its OWN K candidates, so
    scoring is a per-sample bmm instead of one shared item matrix.

    Args:
      ctx_emb      (B, ctx_in_dim)      dialog (+goal) context embedding
      played_feats (B, L, item_in_dim)  played-track features (the sequence)
      lengths      (B,)                 real items per row
      pool_feats   (B, K, item_in_dim)  frozen catalog features of the K pool candidates
      target_in_pool (B,)               column index of the gold within each row's pool
    """
    state = model.encode(ctx_emb, played_feats, lengths)        # (B,d)
    cand = model.item_fusion(pool_feats)                        # (B,K,d)
    state = F.normalize(state, dim=-1)
    cand = F.normalize(cand, dim=-1)                            # cosine (matches model.score)
    logits = torch.bmm(cand, state.unsqueeze(-1)).squeeze(-1) / model.temperature  # (B,K)
    return F.cross_entropy(logits, target_in_pool, label_smoothing=label_smoothing)
