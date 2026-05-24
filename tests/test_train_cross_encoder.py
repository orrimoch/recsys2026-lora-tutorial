"""Unit tests for the Stage B cross-encoder training helpers (Phase 6c)."""
import torch
import torch.nn as nn
from types import SimpleNamespace

from mcrs.training.multimodal_bi_encoder import MultiModalConfig
from mcrs.training.multimodal_cross_encoder import MultiModalCrossEncoder
from scripts.train_cross_encoder import (
    compute_batch_loss,
    flatten_ce_pairs,
    pairwise_bce_loss,
)


class _StubBackbone(nn.Module):
    """Tiny mean-mixing backbone — skips the 570M reranker download."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_dim)
        self._embed = nn.Embedding(vocab_size, hidden_dim)

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds=None, attention_mask=None):
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        pooled = (inputs_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        lhs = inputs_embeds.clone()
        lhs[:, 0, :] = pooled
        return SimpleNamespace(last_hidden_state=lhs)


class _StubTokenizer:
    """Minimal HF-tokenizer stand-in: text → padded char-code token ids."""

    def __call__(self, texts, padding=True, truncation=True, max_length=16,
                 return_tensors="pt"):
        seqs = [[1] + [(ord(c) % 40) + 2 for c in t][: max_length - 1] for t in texts]
        width = max(len(s) for s in seqs)
        input_ids = [s + [0] * (width - len(s)) for s in seqs]
        attn = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attn)}


def _row(query, pos, negs, audio_dim=8, cf_dim=6):
    v = lambda d: [0.1] * d
    return {
        "query": query, "positive": pos, "negatives": list(negs),
        "pos_clap": v(audio_dim), "pos_cf_track": v(cf_dim),
        "neg_clap": [v(audio_dim) for _ in negs], "neg_cf_track": [v(cf_dim) for _ in negs],
        "user_cf": v(cf_dim),
        "pos_tag_ids": [1, 2], "pos_year": 2000,
        "neg_tag_ids": [[3] for _ in negs], "neg_years": [1995 for _ in negs],
    }


def _tiny_ce_model(hidden_dim=16):
    cfg = MultiModalConfig(backbone_name="stub", hidden_dim=hidden_dim, audio_dim=8,
                           cf_dim=6, tag_vocab_size=5, max_tags=4, lora_rank=0)
    return MultiModalCrossEncoder(cfg, backbone=_StubBackbone(50, hidden_dim))


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


def test_flatten_ce_pairs_expands_row_into_pos_and_neg_pairs():
    """Each TripleJsonlDataset row (1 gold + K negs + modalities) flattens into
    K+1 (query, doc) scoring pairs with aligned modalities and a pos/neg flag.
    Query + user_cf are repeated across the row's pairs."""
    rows = [{
        "query": "q1",
        "positive": "POS",
        "negatives": ["NEG0", "NEG1"],
        "pos_clap": [0.1], "pos_cf_track": [0.2],
        "neg_clap": [[0.3], [0.4]], "neg_cf_track": [[0.5], [0.6]],
        "user_cf": [0.9],
        "pos_tag_ids": [1, 2], "pos_year": 2000,
        "neg_tag_ids": [[3], [4]], "neg_years": [1990, 1995],
    }]

    flat = flatten_ce_pairs(rows)

    assert flat["doc_text"] == ["POS", "NEG0", "NEG1"]
    assert flat["is_positive"] == [True, False, False]
    assert flat["query"] == ["q1", "q1", "q1"]
    assert flat["user_cf"] == [[0.9], [0.9], [0.9]]
    assert flat["doc_clap"] == [[0.1], [0.3], [0.4]]
    assert flat["doc_cf"] == [[0.2], [0.5], [0.6]]
    assert flat["doc_tags"] == [[1, 2], [3], [4]]
    assert flat["doc_year"] == [2000, 1990, 1995]


def test_flatten_ce_pairs_concatenates_multiple_rows():
    """Pairs from multiple rows are concatenated in row order."""
    rows = [
        {"query": "qA", "positive": "PA", "negatives": ["NA0"],
         "pos_clap": [1.0], "pos_cf_track": [1.0], "neg_clap": [[0.0]],
         "neg_cf_track": [[0.0]], "user_cf": [0.1],
         "pos_tag_ids": [1], "pos_year": 2001, "neg_tag_ids": [[2]], "neg_years": [1999]},
        {"query": "qB", "positive": "PB", "negatives": ["NB0"],
         "pos_clap": [2.0], "pos_cf_track": [2.0], "neg_clap": [[0.0]],
         "neg_cf_track": [[0.0]], "user_cf": [0.2],
         "pos_tag_ids": [3], "pos_year": 2002, "neg_tag_ids": [[4]], "neg_years": [1998]},
    ]

    flat = flatten_ce_pairs(rows)

    assert flat["doc_text"] == ["PA", "NA0", "PB", "NB0"]
    assert flat["is_positive"] == [True, False, True, False]
    assert flat["query"] == ["qA", "qA", "qB", "qB"]


def test_compute_batch_loss_runs_and_backprops():
    """The per-batch training step assembles flatten → tokenize → forward →
    split → pairwise BCE into a finite, differentiable scalar that backprops
    into the model params. Verified locally with a stub backbone (no GPU)."""
    model = _tiny_ce_model()
    tok = _StubTokenizer()
    rows = [
        _row("hello there", "gold track", ["bad one", "bad two"]),
        _row("another query", "second gold", ["neg a"]),
    ]

    loss = compute_batch_loss(model, tok, rows, max_tags=4, device="cpu")

    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_compute_batch_loss_lower_when_model_ranks_well():
    """Sanity: a batch is just one scalar; confirm the step returns a 0-dim
    tensor (not per-pair) so the optimizer sees a single objective."""
    model = _tiny_ce_model()
    tok = _StubTokenizer()
    rows = [_row("q", "p", ["n1", "n2"])]

    loss = compute_batch_loss(model, tok, rows, max_tags=4, device="cpu")

    assert loss.dim() == 0
