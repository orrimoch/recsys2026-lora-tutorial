# Content-fused, Dialog-conditioned SASRec Recall Channel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dialog-conditioned, content-fused SASRec next-item model and expose it as an opt-in 4th channel of `wrrf_union_v1`, then measure its recall@100 lift on full dev.

**Architecture:** A SASRec encoder runs causal self-attention over `[context_token, played-item-reprs]`. The context token is the Qwen3 embedding of the dialog (works from turn 1); each item-repr is a trained MLP over the concatenated frozen multimodal track embeddings (metadata-qwen3 1024 + CLAP audio 512 + cf-bpr 128), so brand-new-artist tracks are scorable. The same model serves as a recall channel (rank all 47K item-reprs); the P1 reranker feature is a separate plan.

**Tech Stack:** PyTorch (torch 2.10, CPU-testable / Colab-GPU-trained), numpy, datasets, pytest. Reuses the frozen-embedding loaders and the Qwen3-Embedding-0.6B text encoder from `dense_precomputed.py`.

**Spec:** `docs/superpowers/specs/2026-05-27-sasrec-recall-channel-design.md`. Scope is P0 (model + channel + recall ablation). P1 (feed the model's score into the LGBM) is a separate, gated plan.

**Test convention:** tests in `tests/`, import `from mcrs...`, run from repo root: `python -m pytest tests/<file> -v` (pytest.ini sets `pythonpath = music-crs-baselines`). Model tests use tiny dims on CPU.

**Where it runs:** All development and unit tests run locally on CPU (tiny dims). The model is trained ONLY on Colab GPU — `scripts/train_sasrec.py` is never executed locally; local checks for it are limited to `ast.parse` + the CPU unit tests. Tasks 1–6 are local (code + TDD); Task 7 (real training + recall ablation) is Colab-only.

---

## File Structure

- Create `music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py` — pure sequence builder, the `ItemFusion` + `SasrecModel` nn.Modules, and `next_item_loss`. No I/O.
- Create `music-crs-baselines/mcrs/retrieval_modules/sasrec_seq.py` — `SasrecRetriever`, the retriever-interface channel (holds a trained model + precomputed item-repr matrix + a text-encode fn; serves top-K).
- Create `scripts/train_sasrec.py` — load frozen embeddings (+impute), build the item-feature matrix, build training examples + dialog embeddings, train with full-softmax CE, save weights + the item-feature matrix + track-id order.
- Modify `music-crs-baselines/mcrs/retrieval_modules/__init__.py` — add a `sasrec_seq` build branch and a `use_sasrec` gate in `_wrrf_union_v1_specs`.
- Modify `colab/72_build_lgbm_features_train.ipynb` — add a recall-ablation cell (union without vs with the SASRec channel).
- Tests: `tests/test_sasrec_model.py`, `tests/test_sasrec_channel.py`, `tests/test_union_sasrec_wiring.py`.

---

### Task 1: Session example builder (pure)

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py` (this function only)
- Test: `tests/test_sasrec_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sasrec_model.py
from mcrs.retrieval_modules.sasrec_model import build_session_examples


def test_build_session_examples_includes_empty_prefix_first_turn():
    # session of 3 played tracks (indices) -> one example per position,
    # including position 0 (empty prefix -> predict first track from dialog).
    ex = build_session_examples([[10, 11, 12]], max_len=50)
    assert ex == [([], 10), ([10], 11), ([10, 11], 12)]


def test_build_session_examples_left_truncates_to_max_len():
    ex = build_session_examples([[1, 2, 3, 4]], max_len=2)
    # prefix for target 4 is [2,3] (last 2), for target 3 is [1,2]
    assert ([2, 3], 4) in ex
    assert ([1, 2], 3) in ex
    # single-track session yields only the empty-prefix example
    assert build_session_examples([[7]], max_len=50) == [([], 7)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sasrec_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcrs.retrieval_modules.sasrec_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py
"""Dialog-conditioned, content-fused SASRec: pure sequence builder + the
PyTorch model + the training loss. No data I/O (see scripts/train_sasrec.py)."""
from __future__ import annotations


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sasrec_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py tests/test_sasrec_model.py
git commit -m "feat(sasrec): session example builder"
```

---

### Task 2: ItemFusion module

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py`
- Test: `tests/test_sasrec_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sasrec_model.py  (append)
import torch
from mcrs.retrieval_modules.sasrec_model import ItemFusion


def test_item_fusion_projects_to_d_and_is_deterministic():
    fusion = ItemFusion(in_dim=16, d=8).eval()
    feats = torch.randn(5, 16)
    out1 = fusion(feats)
    out2 = fusion(feats)
    assert out1.shape == (5, 8)
    assert torch.allclose(out1, out2)  # deterministic in eval


def test_item_fusion_handles_extra_leading_dims():
    fusion = ItemFusion(in_dim=16, d=8).eval()
    out = fusion(torch.randn(3, 4, 16))  # (B, L, in_dim)
    assert out.shape == (3, 4, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sasrec_model.py -k item_fusion -v`
Expected: FAIL — `ImportError: cannot import name 'ItemFusion'`

- [ ] **Step 3: Write minimal implementation (append to sasrec_model.py)**

```python
# music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py  (append)
import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemFusion(nn.Module):
    """Project a concatenated frozen multimodal track feature vector -> d-dim
    item representation. Only this MLP is trained; the inputs are frozen."""

    def __init__(self, in_dim: int, d: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sasrec_model.py -k item_fusion -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py tests/test_sasrec_model.py
git commit -m "feat(sasrec): ItemFusion multimodal projection module"
```

---

### Task 3: SasrecModel (dialog context token + causal encoder) + loss

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py`
- Test: `tests/test_sasrec_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sasrec_model.py  (append)
from mcrs.retrieval_modules.sasrec_model import SasrecModel, next_item_loss


def _tiny_model():
    return SasrecModel(item_in_dim=16, ctx_in_dim=12, d=8, n_layers=1,
                       n_heads=2, max_len=5).eval()


def test_encode_returns_session_state_shape():
    m = _tiny_model()
    B, L = 4, 3
    ctx = torch.randn(B, 12)
    items = torch.randn(B, L, 16)
    lengths = torch.tensor([3, 2, 1, 0])  # row 3 = empty prefix (turn 1)
    state = m.encode(ctx, items, lengths)
    assert state.shape == (B, 8)
    assert torch.isfinite(state).all()  # empty-prefix row must be finite


def test_score_shape_against_item_matrix():
    m = _tiny_model()
    state = torch.randn(4, 8)
    item_matrix = torch.randn(20, 8)  # N=20 catalog items
    logits = m.score(state, item_matrix)
    assert logits.shape == (4, 20)


def test_next_item_loss_decreases_on_overfit_batch():
    torch.manual_seed(0)
    m = SasrecModel(item_in_dim=16, ctx_in_dim=12, d=8, n_layers=1, n_heads=2, max_len=5)
    all_item_feats = torch.randn(20, 16)          # frozen features for 20 items
    ctx = torch.randn(4, 12)
    items = torch.randn(4, 3, 16)
    lengths = torch.tensor([3, 3, 3, 3])
    target = torch.tensor([1, 5, 9, 13])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        item_matrix = m.item_fusion(all_item_feats)   # grads flow into fusion
        loss = next_item_loss(m, ctx, items, lengths, target, item_matrix)
        loss.backward(); opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 0.5  # the model can fit a fixed batch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sasrec_model.py -k "encode or score or loss" -v`
Expected: FAIL — `ImportError: cannot import name 'SasrecModel'`

- [ ] **Step 3: Write minimal implementation (append to sasrec_model.py)**

```python
# music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py  (append)
class SasrecModel(nn.Module):
    """Causal self-attention over [context_token, item_1..item_t].

    Position 0 is the dialog context token (a projection of the Qwen3 dialog
    embedding); positions 1..t are the played-track item-reprs. Positional
    embeddings index the played-track subsequence position (NOT turn_number).
    The session state is the hidden state at the last real position, so an
    empty prefix (turn 1) yields the context-token state.
    """

    def __init__(self, item_in_dim: int, ctx_in_dim: int = 1024, d: int = 128,
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
        # position 0 (ctx) always valid; item position j (1..L) valid if j <= lengths
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sasrec_model.py -v`
Expected: PASS (all model tests)

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/sasrec_model.py tests/test_sasrec_model.py
git commit -m "feat(sasrec): dialog-conditioned causal encoder + full-softmax loss"
```

---

### Task 4: SasrecRetriever channel

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/sasrec_seq.py`
- Test: `tests/test_sasrec_channel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sasrec_channel.py
import numpy as np
import torch
from mcrs.retrieval_modules.sasrec_model import SasrecModel
from mcrs.retrieval_modules.sasrec_seq import SasrecRetriever


def _build():
    torch.manual_seed(0)
    model = SasrecModel(item_in_dim=16, ctx_in_dim=12, d=8, n_layers=1,
                        n_heads=2, max_len=5).eval()
    track_ids = [f"t{i}" for i in range(20)]
    item_feats = torch.randn(20, 16)
    item_repr = model.item_fusion(item_feats).detach()        # (20, 8)
    # fake text encoder: returns a fixed (B, ctx_in_dim) array
    def text_encode(qs):
        return np.ones((len(qs), 12), dtype=np.float32)
    return SasrecRetriever(model, item_repr, track_ids, item_feats, text_encode,
                           max_len=5)


def test_channel_returns_topk_per_query():
    r = _build()
    out = r.batch_text_to_item_retrieval(
        ["q1", "q2"], topk=5,
        batch_context=[{"history_tids": ["t3", "t4"]}, {"history_tids": []}])
    assert len(out) == 2
    assert all(len(lst) == 5 for lst in out)
    assert all(t in {f"t{i}" for i in range(20)} for t in out[0])


def test_channel_handles_empty_history_turn_one():
    r = _build()
    out = r.batch_text_to_item_retrieval(
        ["q"], topk=3, batch_context=[{"history_tids": []}])
    assert len(out[0]) == 3  # works with no played tracks (dialog-only)


def test_channel_ignores_unknown_history_tids():
    r = _build()
    out = r.batch_text_to_item_retrieval(
        ["q"], topk=3, batch_context=[{"history_tids": ["UNKNOWN", "t2"]}])
    assert len(out[0]) == 3  # unknown ids dropped, not crashed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sasrec_channel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcrs.retrieval_modules.sasrec_seq'`

- [ ] **Step 3: Write minimal implementation**

```python
# music-crs-baselines/mcrs/retrieval_modules/sasrec_seq.py
"""SASRec recall channel: dialog + played history -> session state -> rank the
precomputed catalog item-repr matrix. Standard retriever interface so the wRRF
factory can use it as a union channel."""
from __future__ import annotations

import numpy as np
import torch


class SasrecRetriever:
    def __init__(self, model, item_repr, track_ids, item_feats, text_encode,
                 max_len: int = 50, batch_size: int = 256):
        self.model = model.eval()
        self.item_repr = item_repr                      # (N, d) tensor
        self.track_ids = track_ids
        self.item_feats = item_feats                    # (N, item_in_dim) tensor
        self.tid_to_idx = {t: i for i, t in enumerate(track_ids)}
        self.text_encode = text_encode                  # list[str] -> (B, ctx_in_dim) np
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self.item_in_dim = item_feats.shape[1]

    def _histories_to_feats(self, histories):
        """Build (B, L, item_in_dim) item-feature tensor + (B,) lengths from
        each row's history_tids (last max_len, unknown ids dropped)."""
        idx_lists = [
            [self.tid_to_idx[t] for t in (h or []) if t in self.tid_to_idx][-self.max_len:]
            for h in histories
        ]
        L = max((len(x) for x in idx_lists), default=0)
        L = max(L, 1)  # keep a non-zero time dim for the encoder
        feats = torch.zeros(len(histories), L, self.item_in_dim)
        lengths = torch.zeros(len(histories), dtype=torch.long)
        for b, ids in enumerate(idx_lists):
            lengths[b] = len(ids)
            if ids:
                feats[b, :len(ids)] = self.item_feats[ids]
        return feats, lengths

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None, batch_context=None):
        ctxs = batch_context or [{} for _ in queries]
        results: list[list[str]] = []
        for s in range(0, len(queries), self.batch_size):
            q_batch = queries[s:s + self.batch_size]
            h_batch = [c.get("history_tids", []) for c in ctxs[s:s + self.batch_size]]
            ctx_emb = torch.as_tensor(np.asarray(self.text_encode(q_batch)), dtype=torch.float32)
            feats, lengths = self._histories_to_feats(h_batch)
            with torch.no_grad():
                state = self.model.encode(ctx_emb, feats, lengths)     # (b, d)
                scores = self.model.score(state, self.item_repr)       # (b, N)
                top = torch.topk(scores, k=min(topk, scores.shape[1]), dim=1).indices
            for row in top.tolist():
                results.append([self.track_ids[i] for i in row])
        return results

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sasrec_channel.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/sasrec_seq.py tests/test_sasrec_channel.py
git commit -m "feat(sasrec): SasrecRetriever recall channel"
```

---

### Task 5: Factory wiring (`sasrec_seq` branch + gated union sub)

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/__init__.py`
- Test: `tests/test_union_sasrec_wiring.py`

The gating logic is tested without loading any model (mirrors the `use_hyde`
pattern). The full `sasrec_seq` build (loads weights + the dense encoder) is
integration, exercised on Colab in Task 7.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_union_sasrec_wiring.py
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_union_specs_default_has_no_sasrec():
    specs = _wrrf_union_v1_specs({})
    assert all(s["type"] != "sasrec_seq" for s in specs)


def test_union_specs_add_sasrec_when_enabled():
    specs = _wrrf_union_v1_specs({"use_sasrec": True, "w_sasrec": 0.9})
    sas = next(s for s in specs if s["type"] == "sasrec_seq")
    assert sas["weight"] == 0.9
    assert sas["topk_internal"] == 100
    assert sas["extra_config"]["model_dir"] == "sasrec_v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_union_sasrec_wiring.py -v`
Expected: FAIL — `KeyError`/`StopIteration` (no `sasrec_seq` spec yet)

- [ ] **Step 3: Implement — extend the helper and add the build branch**

In `_wrrf_union_v1_specs` (in `__init__.py`), append after the existing `use_hyde` block, before `return specs`:

```python
    if ec.get("use_sasrec"):
        specs.append({
            "type": "sasrec_seq", "topk_internal": 100,
            "weight": float(ec.get("w_sasrec", 1.0)),
            "extra_config": {
                "model_dir": ec.get("sasrec_model_dir", "sasrec_v1"),
                "max_len": int(ec.get("sasrec_max_len", 50)),
            },
        })
```

Add the build branch next to the `hyde_qwen3` branch in `load_retrieval_module`:

```python
    elif retrieval_type == "sasrec_seq":
        import os
        import torch
        from .sasrec_model import SasrecModel
        from .sasrec_seq import SasrecRetriever
        ec = extra_config or {}
        model_dir = os.path.join(cache_dir, "retrieval_v2", "sasrec",
                                 ec.get("model_dir", "sasrec_v1"))
        ckpt = torch.load(os.path.join(model_dir, "sasrec.pt"), map_location="cpu")
        model = SasrecModel(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        item_feats = torch.as_tensor(ckpt["item_feats"], dtype=torch.float32)
        track_ids = ckpt["track_ids"]
        with torch.no_grad():
            item_repr = model.item_fusion(item_feats)
        # reuse the shared Qwen3-Embedding-0.6B query encoder for the dialog token
        dense = load_retrieval_module(
            "dense_metadata_qwen3", dataset_name, track_split_types,
            corpus_types, cache_dir, extra_config={})
        return SasrecRetriever(
            model, item_repr, track_ids, item_feats,
            text_encode=lambda qs: dense._encode_queries(qs),
            max_len=int(ec.get("max_len", 50)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_union_sasrec_wiring.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full SASRec suite for regressions**

Run: `python -m pytest tests/test_sasrec_model.py tests/test_sasrec_channel.py tests/test_union_sasrec_wiring.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/__init__.py tests/test_union_sasrec_wiring.py
git commit -m "feat(sasrec): factory branch + gated union channel (use_sasrec)"
```

---

### Task 6: Training script + one-step training test

**Files:**
- Create: `scripts/train_sasrec.py`
- Test: `tests/test_sasrec_model.py` (the `next_item_loss` overfit test in Task 3 already covers the training step on synthetic data — no new unit test; the script is integration, smoke-tested on Colab in Task 7)

The script is integration code (loads HF embeddings, trains on GPU). Its core
math — the loss + a gradient step — is already unit-tested via
`test_next_item_loss_decreases_on_overfit_batch` (Task 3). This task writes the
orchestration; verify it by `python -c "import ast; ast.parse(open(...).read())"`
and by the Colab smoke in Task 7.

- [ ] **Step 1: Write the script**

```python
# scripts/train_sasrec.py
"""Train the dialog-conditioned content-fused SASRec and save it for the
sasrec_seq channel. Leakage-safe: train split only.

Saves to {cache_dir}/retrieval_v2/sasrec/{out}/sasrec.pt a dict with
state_dict, model_kwargs, item_feats (N, item_in_dim), track_ids (len N).
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "music-crs-baselines"))
from datasets import load_dataset, concatenate_datasets
from mcrs.retrieval_modules.sasrec_model import SasrecModel, build_session_examples, next_item_loss
from mcrs.db_item.music_catalog import MusicCatalogDB

TRACK_EMB = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
ITEM_DB = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
META_COL, CLAP_COL, CF_COL = "metadata-qwen3_embedding_0.6b", "laion_clap", "cf-bpr"


def _impute(mat, valid):
    """Replace invalid (all-zero / NaN) rows with the column mean of valid rows."""
    if valid.all():
        return mat
    gmean = mat[valid].mean(axis=0)
    mat = mat.copy()
    mat[~valid] = gmean
    return mat


def load_item_feats(splits):
    """Concatenate the 3 frozen modality embeddings -> (N, 1664), imputed,
    aligned to a single track_id order."""
    ds = concatenate_datasets([load_dataset(TRACK_EMB)[s] for s in splits])
    track_ids = list(ds["track_id"])
    parts = []
    for col, dim in [(META_COL, 1024), (CLAP_COL, 512), (CF_COL, 128)]:
        raw = ds[col]
        mat = np.zeros((len(track_ids), dim), dtype=np.float32)
        valid = np.zeros(len(track_ids), dtype=bool)
        for i, v in enumerate(raw):
            if v is not None and len(v) == dim:
                mat[i] = v
                valid[i] = True
        parts.append(_impute(mat, valid))
    feats = np.concatenate(parts, axis=1)  # (N, 1664)
    return track_ids, feats


def build_dialog_and_seqs(item_db, tid_to_idx, max_seq):
    """Walk train sessions -> per music turn: (dialog text up to t, prefix track
    indices, target index). Returns parallel lists."""
    tr = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    dialogs, prefixes, targets = [], [], []
    for sess in tr:
        df = pd.DataFrame(sess["conversations"])
        played_idx = []
        for _, m in df[df["role"] == "music"].iterrows():
            tn = int(m["turn_number"])
            tgt = tid_to_idx.get(m["content"])
            prior = df[(df["turn_number"] < tn) |
                       ((df["turn_number"] == tn) & (df["role"] == "user"))]
            lines = []
            for _, t in prior.iterrows():
                role = "assistant" if t["role"] == "music" else t["role"]
                content = t["content"]
                if t["role"] == "music":
                    try:
                        content = item_db.id_to_metadata(content)
                    except Exception:
                        content = str(content)
                lines.append(f"{role}: {content}")
            if tgt is not None:           # only supervise targets that are in the catalog
                dialogs.append("\n".join(lines))
                prefixes.append(played_idx[-max_seq:])
                targets.append(tgt)
            if tid_to_idx.get(m["content"]) is not None:
                played_idx.append(tid_to_idx[m["content"]])
    return dialogs, prefixes, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", default="sasrec_v1")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    item_db = MusicCatalogDB(ITEM_DB, ["all_tracks"], ["track_name", "artist_name", "album_name"])
    track_ids, feats = load_item_feats(["all_tracks"])
    tid_to_idx = {t: i for i, t in enumerate(track_ids)}
    feats_t = torch.as_tensor(feats, device=dev)
    print(f"[sasrec] item feats {feats.shape}")

    dialogs, prefixes, targets = build_dialog_and_seqs(item_db, tid_to_idx, args.max_seq)
    print(f"[sasrec] {len(targets)} training examples")

    # Encode dialogs ONCE with the shared Qwen3 encoder (reuse dense retriever).
    from mcrs.retrieval_modules import load_retrieval_module
    dense = load_retrieval_module("dense_metadata_qwen3", ITEM_DB, ["all_tracks"],
                                  ["track_name", "artist_name", "album_name"], args.cache_dir, extra_config={})
    ctx = np.zeros((len(dialogs), 1024), dtype=np.float32)
    for s in range(0, len(dialogs), 256):
        ctx[s:s + 256] = dense._encode_queries(dialogs[s:s + 256])
    ctx_t = torch.as_tensor(ctx, device=dev)

    model = SasrecModel(item_in_dim=feats.shape[1], ctx_in_dim=1024, d=args.d,
                        max_len=args.max_seq).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    order = np.arange(len(targets))
    for ep in range(args.epochs):
        np.random.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), args.batch_size):
            bi = order[s:s + args.batch_size]
            L = max((len(prefixes[i]) for i in bi), default=0); L = max(L, 1)
            fb = torch.zeros(len(bi), L, feats.shape[1], device=dev)
            ln = torch.zeros(len(bi), dtype=torch.long, device=dev)
            for j, i in enumerate(bi):
                pf = prefixes[i]; ln[j] = len(pf)
                if pf:
                    fb[j, :len(pf)] = feats_t[pf]
            cb = ctx_t[bi]
            tb = torch.as_tensor([targets[i] for i in bi], device=dev)
            opt.zero_grad()
            item_matrix = model.item_fusion(feats_t)        # (N, d), grads flow in
            loss = next_item_loss(model, cb, fb, ln, tb, item_matrix)
            loss.backward(); opt.step(); tot += float(loss)
        print(f"[sasrec] epoch {ep} mean_loss {tot / max(1, len(order)//args.batch_size):.4f}")

    out_dir = os.path.join(args.cache_dir, "retrieval_v2", "sasrec", args.out)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "model_kwargs": {"item_in_dim": feats.shape[1], "ctx_in_dim": 1024,
                         "d": args.d, "max_len": args.max_seq},
        "item_feats": feats, "track_ids": track_ids,
    }, os.path.join(out_dir, "sasrec.pt"))
    print(f"[sasrec] saved -> {out_dir}/sasrec.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check the script**

Run: `python -c "import ast; ast.parse(open('scripts/train_sasrec.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Confirm the model suite still passes (no import breakage)**

Run: `python -m pytest tests/test_sasrec_model.py tests/test_sasrec_channel.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/train_sasrec.py
git commit -m "feat(sasrec): training script (multimodal fusion + full-softmax CE)"
```

---

### Task 7: Colab — train + recall ablation (integration)

Runs on Colab GPU. Confirms the column key for CLAP, trains, and measures the
recall@100 lift of the SASRec channel in the union.

**Files:**
- Modify: `colab/72_build_lgbm_features_train.ipynb` (new cell after the HyDE cell)

- [ ] **Step 1: Confirm the CLAP column name against the dataset**

In a Colab cell:
```python
from datasets import load_dataset
ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Embeddings', split='all_tracks')
print([c for c in ds.column_names if 'clap' in c.lower() or 'audio' in c.lower()])
```
If the audio column is not literally `laion_clap`, set `CLAP_COL` in
`scripts/train_sasrec.py` to the printed name and re-commit that one-line fix.

- [ ] **Step 2: Train the model**

```python
!cd /content/recsys2026 && python -u scripts/train_sasrec.py \
    --cache-dir /content/recsys2026/experiments/cache --out sasrec_v1 --epochs 5
```
Expected: per-epoch decreasing `mean_loss`, then `saved -> .../sasrec_v1/sasrec.pt`.

- [ ] **Step 3: Add the recall-ablation cell**

```python
# 9) SASRec channel recall ablation: union without vs with use_sasrec.
# Reuses queries/golds/played/user_ids from cell 7.
import numpy as np
from mcrs.retrieval_modules import load_retrieval_module

def recall_at(cands, k):
    return float(np.mean([1.0 if g in c[:k] else 0.0 for c, g in zip(cands, golds)]))

base = load_retrieval_module('wrrf_union_v1', ITEM_DB, ['all_tracks'], CORPUS,
                             CACHE_DIR, extra_config={})
sas = load_retrieval_module('wrrf_union_v1', ITEM_DB, ['all_tracks'], CORPUS,
                            CACHE_DIR, extra_config={'use_sasrec': True, 'w_sasrec': 1.0})
ctx = [{'history_tids': p} for p in played]
cb = base.batch_text_to_item_retrieval(queries, topk=100, user_ids=user_ids, batch_context=ctx)
cs = sas.batch_text_to_item_retrieval(queries, topk=100, user_ids=user_ids, batch_context=ctx)
print('=== SASRec channel recall ablation (n=' + str(len(golds)) + ', FULL dev) ===')
print('  union (3-chan) : recall@20=' + str(round(recall_at(cb,20),4)) +
      ' @100=' + str(round(recall_at(cb,100),4)))
print('  union + SASRec : recall@20=' + str(round(recall_at(cs,20),4)) +
      ' @100=' + str(round(recall_at(cs,100),4)) + '   (G1 gate 0.46)')
print('  delta recall@100 :', round(recall_at(cs,100) - recall_at(cb,100), 4))
```

- [ ] **Step 4: Run smoke (N_EVAL=1500 in cell 7) then full dev (N_EVAL=None)**

Decision gate: if `union + SASRec` recall@100 lifts meaningfully over baseline (target ≥ +0.03, clears 0.46), proceed to P1 (the LGBM relevance feature). If flat, inspect the item-fusion / training (modality ablation) before abandoning.

- [ ] **Step 5: Commit the notebook**

```bash
git add colab/72_build_lgbm_features_train.ipynb
git commit -m "eval(sasrec): Phase-1 recall ablation cell (union +/- SASRec)"
```

---

## Self-Review

**Spec coverage:**
- Content-fused item tower (metadata+CLAP+cf-bpr, imputed) → Task 2 (`ItemFusion`) + Task 6 (`load_item_feats` + `_impute`). ✓
- Dialog context token (position 0, Qwen3 dialog emb, works turn 1) → Task 3 (`SasrecModel.encode` ctx slot; empty-prefix test) + Task 6 (dialog encoding). ✓
- Positional index = played-subsequence position (not turn_number) → Task 3 (`pos_emb` over the sequence; `build_session_examples` indexes by played position). ✓
- Causal self-attention, session state = last real position → Task 3. ✓
- Full-softmax CE loss (temperature) → Task 3 (`next_item_loss`, `temperature`) + Task 6. ✓
- Recall channel (rank 47K item-reprs), retriever interface → Task 4. ✓
- Factory `use_sasrec` gate, ensemble-first → Task 5. ✓
- Leakage-safe (train split only) → Task 6 (`split="train"`; eval uses dev only in Task 7). ✓
- conversation_goal excluded → not referenced anywhere in the model/training inputs. ✓ (by omission)
- Recall ablation validation → Task 7. ✓
- Hard-negative mining is the documented first refinement (out of scope for v1) — intentionally not a task; revisit after Task 7's recall readout.

**Placeholder scan:** No TBD/TODO. The one runtime unknown (CLAP column key) is an explicit Task-7 Step-1 verification with a defined fix, not a placeholder.

**Type consistency:** `SasrecModel(item_in_dim, ctx_in_dim, d, n_layers, n_heads, max_len, temperature)`, `.encode(ctx_emb, item_feats, lengths)->(B,d)`, `.score(state, item_matrix)->(B,N)`, `.item_fusion(feats)->(...,d)` — identical in Tasks 3, 4, 5, 6. `next_item_loss(model, ctx_emb, item_feats, lengths, target_idx, item_matrix)` — same in Task 3 test and Task 6 script. `SasrecRetriever(model, item_repr, track_ids, item_feats, text_encode, max_len, batch_size)` — Task 4 def matches the Task 5 factory construction. The saved checkpoint keys (`state_dict`, `model_kwargs`, `item_feats`, `track_ids`) written in Task 6 match those read in the Task 5 factory branch. Consistent.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-27-sasrec-recall-channel.md`.
