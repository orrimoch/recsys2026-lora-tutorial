"""Stage A: custom PEFT-LoRA fine-tune of BAAI/bge-m3 on conversation→track triples.

Why a custom loop (not FlagEmbedding's CLI):
  FlagEmbedding's master `unified_finetune` does NOT expose --use_lora /
  --lora_rank / --lora_alpha. A `torchrun -m FlagEmbedding...` invocation
  with those flags crashes at startup. This script uses sentence-transformers'
  underlying AutoModel + peft.LoraConfig + a small MultipleNegativesRanking
  loss to get equivalent training behavior with the exact LoRA settings the
  plan calls for (r=32 / alpha=64 over attention+FFN projections).

Hyperparameters (spec §6):
  - lr 5e-6, per-device bs 2, train_group_size 8 (1 pos + 7 negs),
    n_negatives_per_query 15 (all mined negs used; tiny per-row denominator hurts
    contrastive signal — ML reviewer I1, restored from spec §6),
    temperature 0.05, epochs 2, warmup_ratio 0.1.
  - LoRA r=32 alpha=64 over query/key/value/dense projections.
  - bf16 mixed precision via torch.cuda.amp.
  - After training: merge LoRA via peft_model.merge_and_unload(), push merged
    model to Hub.

Usage:
  python scripts/train_bi_encoder.py \
    --triples experiments/cache/retrieval_v2/triples_bge_m3.jsonl \
    --output-dir /content/bge_m3_finetune \
    --hub-repo OrRim123/recsys2026-bge-m3-music-v1 \
    --results-dir /content/drive/MyDrive/recsys2026_retrieval_v2_cache/results/bge_m3 \
    --merge --cleanup-after-push
"""
from __future__ import annotations

# CRITICAL: set BEFORE any other import. The `tensorboard` package (pulled by
# torch.utils.tensorboard.SummaryWriter), and `datasets` (which may pull JAX),
# preallocate GPU memory aggressively by default (TF grabs ~80%, JAX grabs 90%).
# On Blackwell-95GB this leaves PyTorch with ~20 GB → OOM at bs=32. Setting
# these env vars early disables that preallocation so PyTorch gets the full GPU.
import os as _os_early
_os_early.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
_os_early.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os_early.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Also: better PyTorch allocator behavior under fragmentation pressure.
_os_early.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional


# LoRA target modules for BGE-M3 (XLM-RoBERTa under the hood).
_BGE_M3_LORA_TARGETS = ["query", "key", "value", "dense"]


class TripleJsonlDataset:
    """Loads JSONL triples produced by scripts/build_bi_encoder_training_data.py.

    Schema per row (after Δ1):
      {query, pos:[str], neg:[str,...], pos_tid, user_id, session_id}

    On __getitem__, returns {query, positive, negatives, pos_tid, user_id,
    session_id}. Negatives are sampled WITHOUT REPLACEMENT from the row's
    mined neg list (per Δ3 issue E): rows whose `len(neg) < n_negatives`
    are DROPPED at load time, not padded with repeated negs. This eliminates
    the upsampling-with-replacement path that inflated negative gradients.

    Train/val split (Δ2): keyed on `split_key` ∈ {"user_id","session_id","row"}.
    Default "user_id" — every session of a given user lives in exactly one
    partition (the user-stated contract). Missing values for the chosen key
    raise loudly; no silent fallback (issue A).
    """

    def __init__(self, path: str, n_negatives: int = 15, seed: int = 42,
                 split: str = "all", val_fraction: float = 0.0,
                 split_key: str = "user_id"):
        """Args:
            path: JSONL produced by scripts/build_bi_encoder_training_data.py.
            n_negatives: number of negatives returned per __getitem__ call.
                Rows with fewer mined negatives than this are DROPPED (issue E).
            seed: governs both the train/val key shuffle AND per-call neg sampling.
            split: 'all' (default; load every row), 'train', or 'val'.
            val_fraction: only used when split ∈ {'train','val'}. Default 0.0
                preserves back-compat. The §6 hyperparameter table sets 0.10.
            split_key: 'user_id' (default; users' sessions all in one
                partition), 'session_id' (legacy), or 'row' (no group
                semantics — explicit opt-in).
        """
        import json as _json
        if split_key not in ("user_id", "session_id", "row"):
            raise ValueError(
                f"split_key must be one of 'user_id'|'session_id'|'row'; "
                f"got {split_key!r}"
            )
        all_rows = []
        with open(path) as f:
            for line in f:
                obj = _json.loads(line)
                if not obj.get("pos") or not obj.get("neg"):
                    continue
                all_rows.append(obj)

        # Issue E (§6.5): drop rows with fewer mined negs than n_negatives.
        # This eliminates the upsampling-with-replacement path that previously
        # padded short rows by random.choice — which duplicated negs in the
        # same row and inflated their gradient.
        n_neg_req = int(n_negatives)
        survivors = [r for r in all_rows if len(r["neg"]) >= n_neg_req]
        n_dropped = len(all_rows) - len(survivors)
        if n_dropped > 0:
            pct = 100.0 * n_dropped / max(1, len(all_rows))
            print(
                f"[TripleJsonlDataset] DROPPED {n_dropped} rows "
                f"({pct:.1f}%) with fewer than n_negatives={n_neg_req} "
                f"mined negatives (issue E: no upsampling with replacement).",
                file=sys.stderr,
            )
            if pct > 5.0:
                print(
                    f"[TripleJsonlDataset] WARNING: >5% of rows dropped. "
                    f"Consider lowering --n-negatives to match the mining "
                    f"distribution, or re-mine with larger --pool-size.",
                    file=sys.stderr,
                )
        all_rows = survivors

        if split == "all":
            self.rows = all_rows
        elif split in ("train", "val"):
            self.rows = self._split_by_key(
                all_rows, split=split, val_fraction=val_fraction,
                seed=seed, split_key=split_key,
            )
        else:
            raise ValueError(
                f"unknown split: {split!r} (expected 'all'|'train'|'val')"
            )
        self.n_negatives = n_neg_req
        self.split_key = split_key
        self.rng = random.Random(seed)

    @staticmethod
    def _split_by_key(all_rows, split, val_fraction, seed, split_key):
        """Issue A fix: inspect EVERY row (not just all_rows[0]) for the
        split key. Loud-fail when any row is missing it for the chosen
        split_key — no silent fallback."""
        from collections import defaultdict

        if split_key == "row":
            shuffle_rng = random.Random(seed)
            shuffled = list(all_rows)
            shuffle_rng.shuffle(shuffled)
            n_val = int(round(float(val_fraction) * len(shuffled)))
            n_val = max(0, min(n_val, len(shuffled) - 1))
            if split == "val":
                return shuffled[len(shuffled) - n_val:] if n_val > 0 else []
            return shuffled[: len(shuffled) - n_val]

        # split_key ∈ {"user_id", "session_id"} — check every row.
        missing = [i for i, r in enumerate(all_rows) if not r.get(split_key)]
        if missing:
            raise ValueError(
                f"split_key={split_key!r} requires every row to carry "
                f"a non-empty {split_key} field. {len(missing)} rows are "
                f"missing it (first indices: {missing[:5]}). Re-mine the "
                f"triples with the latest builder, or pass split_key="
                f"'session_id' (legacy) or 'row' (no grouping) explicitly."
            )

        by_key = defaultdict(list)
        for r in all_rows:
            by_key[str(r[split_key])].append(r)
        keys = sorted(by_key.keys())  # deterministic starting order
        shuffle_rng = random.Random(seed)
        shuffle_rng.shuffle(keys)
        n_val_keys = int(round(float(val_fraction) * len(keys)))
        n_val_keys = max(0, min(n_val_keys, len(keys) - 1))
        if split == "val":
            chosen = keys[-n_val_keys:] if n_val_keys > 0 else []
        else:  # train
            chosen = keys[:-n_val_keys] if n_val_keys > 0 else keys
        return [r for k in chosen for r in by_key[k]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        neg_texts = list(row["neg"])
        # neg_tids parallel to neg (issue C extension); empty list if legacy.
        neg_tids = list(row.get("neg_tids") or [])
        keep_tids = len(neg_tids) == len(neg_texts)
        # Δ3 (§6.5): WITHOUT-REPLACEMENT sampling, K_data fixed by n_negatives.
        # Rows with fewer negs than n_negatives were already dropped at __init__.
        if len(neg_texts) > self.n_negatives:
            picks = self.rng.sample(range(len(neg_texts)), self.n_negatives)
            neg_texts = [neg_texts[i] for i in picks]
            if keep_tids:
                neg_tids = [neg_tids[i] for i in picks]
        out = {
            "query": row["query"],
            "positive": row["pos"][0],
            "negatives": neg_texts,
        }
        if keep_tids and neg_tids:
            out["neg_tids"] = neg_tids
        # Optional fields surfaced for in-batch mask (issue C) and sampler.
        for opt_key in ("pos_tid", "user_id", "session_id"):
            if opt_key in row:
                out[opt_key] = row[opt_key]
        return out

    def pos_tids(self) -> list[str]:
        """All gold track_ids in row order. Returns [] if triples don't carry pos_tid."""
        return [r["pos_tid"] for r in self.rows if "pos_tid" in r]

    def user_ids(self) -> list:
        """All user_ids in row order; entries may be None for legacy triples."""
        return [r.get("user_id") for r in self.rows]

    def queries(self) -> list[str]:
        """All query strings in row order. Used for full-catalog val eval."""
        return [r["query"] for r in self.rows]


class UserDisjointBatchSampler:
    """BatchSampler that yields lists of row indices forming batches whose
    `user_id` values are pairwise distinct (Δ3 issue B fix).

    Constructed from a parallel `row_user_ids` list (one entry per dataset
    row, in dataset order) + `batch_size` + `seed`. Each iteration over the
    sampler is one epoch:

      - Every row index is yielded AT MOST ONCE (sampling without replacement).
      - Each yielded batch contains batch_size distinct users.
      - When the remaining pool cannot supply `batch_size` distinct users,
        the residual rows form a final short batch and the epoch ends.
      - Deterministic via (seed, epoch counter).

    Used as `DataLoader(batch_sampler=...)` — replaces `shuffle=True`.
    PyTorch-style: implements __iter__ and __len__.
    """

    def __init__(self, row_user_ids, batch_size: int, seed: int = 42,
                 fixed_seed: bool = False):
        """Args:
            fixed_seed: if True, the sampler does NOT advance its epoch
                counter on iteration → every `list(sampler)` call yields
                IDENTICAL batches. Used by val_loader so val_loss is
                directly comparable across opt-steps (same composition,
                model is the only variable). Train_loader leaves this
                False so train batches change across epochs (standard SGD).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        self._user_ids = list(row_user_ids)
        if not self._user_ids:
            raise ValueError("row_user_ids is empty")
        # Pre-bucket indices by user for fast distinct-user batching.
        from collections import defaultdict
        self._by_user = defaultdict(list)
        for i, uid in enumerate(self._user_ids):
            if uid is None:
                raise ValueError(
                    f"row {i} has user_id=None; UserDisjointBatchSampler "
                    f"requires every row to carry a non-empty user_id"
                )
            self._by_user[uid].append(i)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.fixed_seed = bool(fixed_seed)
        self.epoch = 0
        # Pre-compute the batch count for __len__: assemble one epoch once
        # using a probe seed so it matches what __iter__ would emit.
        self._cached_len = self._count_batches_for_seed(self.seed)

    def _build_epoch(self, epoch_seed: int):
        """Return the list-of-batches that __iter__ would emit for the
        given seed. Pulled out so __len__ can match exactly.

        Scheduling: heap-based greedy load-balancing — at each batch we
        pop the `batch_size` users with the MOST remaining rows. This
        maximizes batch fullness on imbalanced data (some users have
        many rows, others few): all batches are full as long as ≥
        batch_size distinct users still have unassigned rows; the tail
        may have ragged batches once the active-user count drops below
        batch_size.
        """
        import heapq

        rng = random.Random(epoch_seed)
        users = list(self._by_user.keys())
        rng.shuffle(users)
        queues = {u: list(self._by_user[u]) for u in users}
        for u in users:
            rng.shuffle(queues[u])
        # Heap entries: (-remaining_count, tiebreak, user). Negative count
        # gives a max-heap by remaining; tiebreak = stable per-user random
        # value to break ties deterministically without bias toward
        # insertion order.
        tiebreaks = {u: rng.random() for u in users}
        heap = [(-len(queues[u]), tiebreaks[u], u) for u in users if queues[u]]
        heapq.heapify(heap)

        batches: list[list[int]] = []
        while heap:
            picked: list[str] = []
            while heap and len(picked) < self.batch_size:
                _, _, u = heapq.heappop(heap)
                picked.append(u)
            batch = [queues[u].pop() for u in picked]
            batches.append(batch)
            for u in picked:
                if queues[u]:
                    heapq.heappush(heap, (-len(queues[u]), tiebreaks[u], u))
        return batches

    def _count_batches_for_seed(self, seed: int) -> int:
        return len(self._build_epoch(seed))

    def __iter__(self):
        if self.fixed_seed:
            # Val mode: same composition every iteration.
            epoch_seed = self.seed
            for batch in self._build_epoch(epoch_seed):
                yield batch
        else:
            # Train mode: advance epoch so batches differ across epochs.
            epoch_seed = self.seed + self.epoch
            for batch in self._build_epoch(epoch_seed):
                yield batch
            self.epoch += 1

    def __len__(self) -> int:
        return self._cached_len


def _collate_batch(batch: list[dict], q_tokenizer, d_tokenizer,
                   max_q_len: int, max_p_len: int):
    """Tokenize a list of {query, positive, negatives} rows into tensors.

    Issue G fix (§6.5): takes TWO tokenizer instances — `q_tokenizer`
    pre-configured with `truncation_side='left'` (preserves the [QUERY]:
    block at the end, which carries the current user turn), and
    `d_tokenizer` with `truncation_side='right'` (preserves track_name at
    the start of the doc text). NO runtime mutation of truncation_side —
    safe under any DataLoader worker config.

    Returns (q_enc, d_enc, n_per) where n_per = 1 + len(negatives) and
    `pos_tids` is also returned via the batch metadata for the in-batch
    false-positive mask (§6.5 issue C).
    """
    queries = [b["query"] for b in batch]
    # positives + negatives per row → (B * (1 + n_negs)) docs.
    docs: list[str] = []
    n_per = 1 + len(batch[0]["negatives"])
    for b in batch:
        docs.append(b["positive"])
        docs.extend(b["negatives"])

    q_enc = q_tokenizer(queries, max_length=max_q_len, padding=True,
                        truncation=True, return_tensors="pt")
    d_enc = d_tokenizer(docs, max_length=max_p_len, padding=True,
                        truncation=True, return_tensors="pt")
    return q_enc, d_enc, n_per


def _extract_pos_tids(batch: list[dict]) -> list:
    """Pull pos_tid off each row (or None if absent). Used by the masked
    in-batch loss (§6.5 issue C)."""
    return [b.get("pos_tid") for b in batch]


def _cls_pool(last_hidden: "torch.Tensor") -> "torch.Tensor":
    """L2-normalized CLS token. Matches BGE-M3 inference (CLS token, NOT mean).

    BGEM3FlagModel.encode() reads last_hidden_state[:, 0] at inference time, so
    training must pool the same way or LoRA-adapted weights won't be optimized
    for what production reads.
    """
    import torch.nn.functional as F

    cls = last_hidden[:, 0]
    return F.normalize(cls, p=2, dim=1)


def _info_nce_loss(q_emb: "torch.Tensor", d_emb: "torch.Tensor", n_per: int, temperature: float) -> "torch.Tensor":
    """Per-row InfoNCE: each query contrasted against ONLY its own 1 pos + (n_per-1) negs.
    Denominator = n_per (e.g., 16). Used when --in-batch-negs is disabled."""
    import torch
    import torch.nn.functional as F

    B = q_emb.size(0)
    d_emb = d_emb.view(B, n_per, -1)             # (B, n_per, D)
    scores = torch.einsum("bd,bnd->bn", q_emb, d_emb) / temperature  # (B, n_per)
    labels = torch.zeros(B, dtype=torch.long, device=scores.device)  # positive is index 0
    return F.cross_entropy(scores, labels)


def _info_nce_loss_in_batch(q_emb: "torch.Tensor", d_emb: "torch.Tensor",
                            n_per: int, temperature: float) -> "torch.Tensor":
    """In-batch-negative InfoNCE: each query contrasted against ALL docs in the
    micro-batch (= B * n_per), not just its own row. Standard recipe for
    modern dense retrievers (BGE-M3, E5, GTE).

    Denominator: B * n_per (e.g., 32 at bs=2). Per-query positive sits at
    column `i * n_per` of the score matrix (docs are laid out as
    [pos_0, neg_0_0..14, pos_1, neg_1_0..14, ...] by the collator).

    False-negative risk: if track X is the positive for query A and was mined
    as a negative for query B, the loss pushes X UP for A and DOWN for B
    simultaneously. In music CRS this happens for popular tracks. Literature
    accepts this — the signal boost (denominator size) dominates the noise.
    See `_info_nce_loss_in_batch_masked` for the issue-C-aware variant that
    masks duplicate `pos_tid` collisions.
    """
    import torch
    import torch.nn.functional as F

    B = q_emb.size(0)
    # d_emb is already (B * n_per, D) from the collator — no reshape needed.
    scores = (q_emb @ d_emb.T) / temperature      # (B, B * n_per)
    labels = torch.arange(B, device=scores.device) * n_per  # each query's positive idx
    return F.cross_entropy(scores, labels)


def _info_nce_loss_in_batch_masked(q_emb: "torch.Tensor", d_emb: "torch.Tensor",
                                   n_per: int, temperature: float,
                                   pos_tids: list,
                                   neg_tids_per_row=None) -> "torch.Tensor":
    """In-batch InfoNCE with false-positive collision masking (§6.5 issue C).

    Patches two label-noise modes (both well-known in dense-retrieval
    literature; cf. RocketQAv2 [Ren et al. EMNLP 2021] and BGE-M3 §3.3):

      (a) **Same-pos-tid collision**: queries i, j (i ≠ j) share the same
          gold track. The unmasked loss treats query j's positive column
          j·n_per as a NEGATIVE for query i — pure contradiction.
      (b) **Cross-pos-into-neg collision**: query i's gold appears as one
          of query j's mined hard negatives. That negative slot column
          j·n_per + 1 + k holds query i's own gold embedding, with a high
          score — inflating query i's denominator.

    Mask: for every cell (i, c) where c is a doc slot belonging to row
    j ≠ i AND that slot's track_id equals `pos_tids[i]`, set scores[i, c]
    to `-inf`. Cells on the diagonal (row i's own slots — its positive at
    i·n_per and its own mined negs at i·n_per+1..) are NEVER touched.

    Falls back to standard in-batch InfoNCE when `pos_tids` is empty or
    contains any None (legacy triple files without pos_tid). The
    cross-pos-into-neg mask is additionally gated on `neg_tids_per_row`
    being present.
    """
    import torch
    import torch.nn.functional as F

    if not pos_tids or any(t is None for t in pos_tids):
        return _info_nce_loss_in_batch(q_emb, d_emb, n_per, temperature)

    B = q_emb.size(0)
    K = n_per - 1  # negatives per row
    scores = (q_emb @ d_emb.T) / temperature      # (B, B * n_per)
    labels = torch.arange(B, device=scores.device) * n_per

    full_mask = torch.zeros_like(scores, dtype=torch.bool)
    has_neg_tids = (
        neg_tids_per_row is not None
        and len(neg_tids_per_row) == B
    )
    for i in range(B):
        target = pos_tids[i]
        for j in range(B):
            if i == j:
                continue  # never mask own-row slots
            # (a) same-pos-tid collision → mask col j*n_per for row i.
            if pos_tids[j] == target:
                full_mask[i, j * n_per] = True
            # (b) cross-pos-into-neg collision → mask negs of row j whose
            #     tid equals target.
            if has_neg_tids and neg_tids_per_row[j]:
                ntids_j = neg_tids_per_row[j]
                for k in range(min(K, len(ntids_j))):
                    if ntids_j[k] == target:
                        full_mask[i, j * n_per + 1 + k] = True

    if full_mask.any():
        scores = scores.masked_fill(full_mask, float("-inf"))

    return F.cross_entropy(scores, labels)


def _val_metrics_from_scores(scores: "torch.Tensor") -> tuple[float, float]:
    """Compute (top1_accuracy, mean_nDCG) given a (B, n_per) score matrix
    where column 0 is the positive. Pure tensor op; no model required.

    Extracted from _val_loss_now so we can unit-test the nDCG formula
    without spinning up a real encoder.
    """
    import torch

    preds = scores.argmax(dim=-1)
    top1 = float((preds == 0).float().mean().item())
    pos_scores = scores[:, 0:1]
    # rank of the positive = 1 + (# candidates with strictly higher score).
    ranks = (scores > pos_scores).sum(dim=-1).float() + 1.0
    # nDCG with single relevant item: 1 / log2(rank + 1). Ideal at rank 1 = 1.0.
    ndcg = 1.0 / torch.log2(ranks + 1.0)
    return top1, float(ndcg.mean().item())


def _train(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # Reproducibility seed — applied BEFORE LoRA init so the random LoRA-B
    # weight pattern is deterministic across runs with the same --seed. The
    # data split / batch sampler / per-row neg sampler ALSO read this seed
    # (forwarded explicitly below). Note: bf16 + cuDNN non-determinism
    # remains; bit-exact repro would require torch.backends.cudnn.deterministic=True
    # which costs ~30% wallclock on Blackwell — skipped for the production
    # recipe. Practical repro within ~0.01 dev nDCG is achieved with this seed.
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train-bi-encoder] device={device} seed={args.seed}", file=sys.stderr)

    # Issue G fix (§6.5): two pre-configured tokenizer instances so collate
    # never mutates truncation_side at call time. Queries left-truncate
    # (preserves [QUERY]: block at end); docs right-truncate (preserves
    # track_name at start).
    q_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    q_tokenizer.truncation_side = "left"
    d_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    d_tokenizer.truncation_side = "right"
    # Back-compat alias for any downstream code that still inspects `tokenizer`.
    tokenizer = d_tokenizer
    base_model = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    # Gradient checkpointing recomputes activations on backward → saves
    # memory but costs ~30% wallclock. Worth it at bs=2 on small GPUs;
    # wasted overhead at bs=8+ on Blackwell-95GB. Toggle via CLI flag.
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        print("[train-bi-encoder] gradient checkpointing: ON", file=sys.stderr)
    else:
        print("[train-bi-encoder] gradient checkpointing: OFF (~30% faster; needs more VRAM)", file=sys.stderr)

    # Warm-start: if --resume-from is set, load an existing LoRA adapter
    # instead of creating a fresh one. Use case: train 1 epoch → evaluate →
    # decide to train another epoch from that checkpoint. Optimizer +
    # scheduler restart fresh (intentionally: a new warmup is healthier than
    # bit-exact continuation, and we always know what schedule we ran).
    if args.resume_from:
        from peft import PeftModel
        print(f"[train-bi-encoder] WARM START from {args.resume_from}", file=sys.stderr)
        model = PeftModel.from_pretrained(base_model, args.resume_from, is_trainable=True)
    else:
        # Training does CLS-pool on last_hidden_state directly (see _cls_pool), so
        # XLM-RoBERTa's `pooler` submodule is never on the gradient path. Earlier
        # versions added `modules_to_save=["pooler"]` per spec line 140's wording,
        # but that wrapped a module the loss never touches AND collided with the
        # `dense` substring in `target_modules` (peft would also LoRA-wrap pooler.dense).
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=_BGE_M3_LORA_TARGETS,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(base_model, lora_cfg)
    model.to(device)
    model.print_trainable_parameters()

    # ML-reviewer N1 + §6.5 Δ2: user-disjoint train/val split. Every session
    # of a given user_id lives in exactly one partition.
    train_ds = TripleJsonlDataset(
        args.triples, n_negatives=args.n_negatives,
        split="train" if args.val_fraction > 0 else "all",
        val_fraction=args.val_fraction,
        split_key=args.split_key,
        seed=args.seed,
    )
    val_ds = (TripleJsonlDataset(
        args.triples, n_negatives=args.n_negatives,
        split="val", val_fraction=args.val_fraction,
        split_key=args.split_key,
        seed=args.seed,
    ) if args.val_fraction > 0 else None)
    n_val = len(val_ds) if val_ds is not None else 0
    print(f"[train-bi-encoder] {len(train_ds)} train triples, {n_val} val triples "
          f"(split_key={args.split_key!r})", file=sys.stderr)

    # Δ3 (§6.5): user-disjoint batch sampler. Each batch contains rows whose
    # user_id values are pairwise distinct → no false-negative collisions
    # from same-user batch rows.
    use_user_sampler = (
        args.split_key == "user_id"
        and all(u is not None for u in train_ds.user_ids())
    )
    if use_user_sampler:
        print(f"[train-bi-encoder] batch sampler: UserDisjointBatchSampler "
              f"(distinct users per batch, without-replacement)", file=sys.stderr)
    else:
        print(f"[train-bi-encoder] batch sampler: DataLoader(shuffle=True) "
              f"(no user grouping; split_key={args.split_key!r})", file=sys.stderr)

    # Δ3 (§6.5) issue C: select the masked vs unmasked in-batch loss.
    # We always use the masked variant when in_batch_negs is on — it falls
    # back to unmasked behavior automatically when pos_tids are absent.
    if args.in_batch_negs:
        _loss_uses_mask = True
        print(f"[train-bi-encoder] InfoNCE: IN-BATCH negatives WITH false-pos "
              f"collision mask (denominator before mask = "
              f"{args.per_device_batch_size * (1 + args.n_negatives)} "
              f"per query at bs={args.per_device_batch_size})", file=sys.stderr)
    else:
        _loss_uses_mask = False
        print(f"[train-bi-encoder] InfoNCE: per-row only "
              f"(denominator = {1 + args.n_negatives} per query)", file=sys.stderr)

    def _loss_fn(q_emb, d_emb, n_per, temperature, pos_tids=None,
                 neg_tids_per_row=None):
        if args.in_batch_negs:
            return _info_nce_loss_in_batch_masked(
                q_emb, d_emb, n_per, temperature,
                pos_tids=pos_tids or [],
                neg_tids_per_row=neg_tids_per_row,
            )
        return _info_nce_loss(q_emb, d_emb, n_per, temperature)

    # ML-reviewer I-3: full-catalog val nDCG@K. Requires val triples carrying
    # pos_tid (new builder schema). On every --val-full-catalog-every-n-steps
    # opt-step, encode the FULL catalog + all val queries, compute nDCG@K
    # against the actual ~50k corpus (not just the 16 mined cands).
    catalog_texts: Optional[list[str]] = None
    catalog_tids: Optional[list[str]] = None
    val_queries_text: Optional[list[str]] = None
    val_gold_tids: Optional[list[str]] = None
    full_cat_enabled = (
        args.val_full_catalog_every_n_steps > 0
        and val_ds is not None
        and len(val_ds.pos_tids()) == len(val_ds)
    )
    if args.val_full_catalog_every_n_steps > 0 and not full_cat_enabled:
        print("[train-bi-encoder] --val-full-catalog-every-n-steps requested but "
              "val triples lack pos_tid (older mining run?). Skipping full-catalog "
              "val. Re-mine with the latest builder to enable.", file=sys.stderr)
    if full_cat_enabled:
        from datasets import load_dataset as _load_dataset
        REPO_ROOT_LOCAL = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(REPO_ROOT_LOCAL / "music-crs-baselines"))
        from mcrs.retrieval_modules.bge_m3_format import format_track_text as _fmt
        print("[train-bi-encoder] loading catalog for full-catalog val eval",
              file=sys.stderr)
        tm = _load_dataset(args.track_meta_hf, split="all_tracks")
        catalog_tids = [r["track_id"] for r in tm]
        catalog_texts = [_fmt(
            r.get("track_name", "unknown"), r.get("artist_name"),
            r.get("album_name"), r.get("release_date"), r.get("tag_list"),
        ) for r in tm]
        val_queries_text = val_ds.queries()
        val_gold_tids = val_ds.pos_tids()
        print(f"[train-bi-encoder] full-catalog val ENABLED: "
              f"{len(catalog_tids)} catalog tracks × {len(val_queries_text)} "
              f"val queries, every {args.val_full_catalog_every_n_steps} opt-steps "
              f"(~{len(catalog_tids) / 64 * 0.05:.0f} sec/eval on Blackwell)",
              file=sys.stderr)
    # The collate returns (q_enc, d_enc, n_per). We also need pos_tids +
    # neg_tids out of the raw batch for the masked loss; wrap collate to
    # return both pieces.
    def _collate_with_meta(b):
        q_enc, d_enc, n_per = _collate_batch(
            b, q_tokenizer, d_tokenizer,
            args.query_max_len, args.passage_max_len,
        )
        meta = {
            "pos_tids": [row.get("pos_tid") for row in b],
            "neg_tids_per_row": [row.get("neg_tids") for row in b],
        }
        return q_enc, d_enc, n_per, meta

    if use_user_sampler:
        from torch.utils.data import BatchSampler  # noqa: F401  (type-check pin)
        train_sampler = UserDisjointBatchSampler(
            train_ds.user_ids(),
            batch_size=args.per_device_batch_size,
            seed=args.seed,
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=2,
            collate_fn=_collate_with_meta,
        )
    else:
        loader = DataLoader(
            train_ds,
            batch_size=args.per_device_batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=_collate_with_meta,
        )
    # Val_loader: use the SAME UserDisjointBatchSampler as train, with
    # fixed_seed=True so every val pass sees the same batch composition.
    # That makes val_loss directly comparable to train_loss on the TB curve
    # (both denominators are clean of same-user false negatives). Without
    # this, val_loader fell back to `shuffle=False` over user-grouped rows,
    # producing batches saturated with same-user collisions → val_loss
    # inflated even when generalization was fine.
    val_user_sampler_ok = (
        val_ds is not None
        and n_val > 0
        and use_user_sampler
        and all(u is not None for u in val_ds.user_ids())
    )
    if val_user_sampler_ok:
        val_sampler = UserDisjointBatchSampler(
            val_ds.user_ids(),
            batch_size=args.per_device_batch_size,
            seed=args.seed + 1,   # different seed than train (args.seed) for independence
            fixed_seed=True,      # same composition every val pass → reproducible
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=0,
            collate_fn=_collate_with_meta,
        )
        print(f"[train-bi-encoder] val_loader: UserDisjointBatchSampler "
              f"(fixed_seed=True; val_loss is now apples-to-apples with train_loss)",
              file=sys.stderr)
    elif val_ds is not None and n_val > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_with_meta,
        )
        print(f"[train-bi-encoder] val_loader: DataLoader(shuffle=False) "
              f"(legacy; val_loss may show same-user false-neg pollution)",
              file=sys.stderr)
    else:
        val_loader = None

    # Optimizer steps after grad-accum: total_micro_batches / accum_steps.
    accum = max(1, int(args.gradient_accumulation_steps))
    total_micro = len(loader) * args.epochs
    total_steps = max(1, total_micro // accum)
    # I4: only optimize trainable params (saves ~9GB of AdamW state on a 567M model).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    # I2: warmup_ratio 0.1 → 10% linear warmup, then linear decay.
    # N5: end_factor=0.1 (not 0.0) so the final ~10% of steps still updates.
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                     total_iters=max(1, total_steps - warmup_steps)),
        ],
        milestones=[warmup_steps],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    def _val_loss_now() -> Optional[tuple[float, float, float]]:
        """Forward-only pass over the full val_loader.

        Returns (mean_loss, top1_accuracy, mean_ndcg) where:
          - mean_loss: InfoNCE loss on the val triples (effective-batch scale).
          - top1_accuracy: fraction of val queries whose positive (always at
            index 0 of the 1+n_negatives candidates) is ranked first.
          - mean_ndcg: per-query nDCG over the 16 candidates, computed as
            1/log2(rank_of_positive + 1) and averaged. Direct retrieval-quality
            proxy that complements loss (loss can plateau while ranking still
            sharpens, or vice versa). Note: this is nDCG over the val-triple
            16-candidate set, NOT the full ~50k catalog — useful as a relative
            indicator of improvement, not directly comparable to the dev
            nDCG@20 cell that scores against the full catalog.

        Restores model.train() on exit.
        """
        if val_loader is None:
            return None
        model.eval()
        losses: list[float] = []
        n_correct = 0
        n_total = 0
        ndcg_sum = 0.0
        with torch.no_grad():
            for vq, vd, vn, vmeta in val_loader:
                vq = {k: v.to(device) for k, v in vq.items()}
                vd = {k: v.to(device) for k, v in vd.items()}
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                    vq_out = model(**vq)
                    vd_out = model(**vd)
                    vq_emb = _cls_pool(vq_out.last_hidden_state)
                    vd_emb = _cls_pool(vd_out.last_hidden_state)
                    vloss = _loss_fn(vq_emb, vd_emb, vn, args.temperature,
                                     pos_tids=vmeta["pos_tids"],
                                     neg_tids_per_row=vmeta["neg_tids_per_row"])
                # Re-compute per-query top-1 + nDCG from the same embeddings.
                B = vq_emb.size(0)
                d_emb_grouped = vd_emb.view(B, vn, -1)
                scores = torch.einsum("bd,bnd->bn", vq_emb, d_emb_grouped)
                batch_top1, batch_ndcg = _val_metrics_from_scores(scores)
                n_correct += int(round(batch_top1 * B))
                ndcg_sum += float(batch_ndcg) * B
                n_total += B
                losses.append(float(vloss.item()))
        model.train()
        if not losses:
            return None
        return (
            float(sum(losses) / len(losses)),
            float(n_correct) / max(1, n_total),
            float(ndcg_sum) / max(1, n_total),
        )

    def _encode_texts(texts: list[str], max_len: int,
                      truncation_side: str = "right") -> "torch.Tensor":
        """Forward-only encode for full-catalog val. Pools at the CLS token
        + L2-normalizes, matching the training-time pooling contract.

        Issue G fix: picks the pre-configured tokenizer (q_tokenizer for
        'left'-truncate queries, d_tokenizer for 'right'-truncate docs) —
        no runtime mutation of `truncation_side`.
        """
        model.eval()
        tok = q_tokenizer if truncation_side == "left" else d_tokenizer
        BATCH = max(1, args.val_encode_batch_size)
        out_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                chunk = texts[i:i + BATCH]
                enc = tok(chunk, max_length=max_len, padding=True,
                          truncation=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                    dtype=torch.bfloat16):
                    o = model(**enc)
                emb = _cls_pool(o.last_hidden_state).float().cpu()
                out_chunks.append(emb)
        model.train()
        return torch.cat(out_chunks, dim=0)

    def _val_full_catalog_ndcg(k: int = 20) -> Optional[float]:
        """Re-encode the full catalog + val queries with the current model,
        compute nDCG@K for each val query against the full catalog, return mean.

        This is the ML-reviewer I-3 fix: the per-row val/ndcg over 16 cands is
        a training-set echo; this metric scores against the real ~50k corpus.
        """
        if not full_cat_enabled:
            return None
        try:
            cat_emb = _encode_texts(catalog_texts, args.passage_max_len,
                                    truncation_side="right")  # (N, D)
            q_emb_full = _encode_texts(val_queries_text, args.query_max_len,
                                       truncation_side="left")  # (Q, D)
            sims = q_emb_full @ cat_emb.T  # (Q, N)
            # top-K indices per query (any order), then sort within top-K.
            topk_idx = torch.topk(sims, k=min(k, sims.size(1)), dim=1).indices  # (Q, k)
            tid_to_idx = {tid: i for i, tid in enumerate(catalog_tids)}
            import math as _math
            ndcgs: list[float] = []
            for qi, gold_tid in enumerate(val_gold_tids):
                if gold_tid not in tid_to_idx:
                    ndcgs.append(0.0)
                    continue
                gold_cat_idx = tid_to_idx[gold_tid]
                row = topk_idx[qi].tolist()
                if gold_cat_idx in row:
                    rank = row.index(gold_cat_idx) + 1
                    ndcgs.append(1.0 / _math.log2(rank + 1))
                else:
                    ndcgs.append(0.0)
            return float(sum(ndcgs) / len(ndcgs)) if ndcgs else None
        except Exception as e:
            print(f"[train-bi-encoder] WARN: full-catalog val failed: {e!r}",
                  file=sys.stderr)
            return None

    # Gradient-accumulation training loop. Loss is divided by `accum` so the
    # accumulated gradient matches what a single bs=(per_device*accum) step
    # would produce; optimizer steps only every `accum` micro-batches.
    micro_step = 0
    opt_step = 0
    best_val_loss = float("inf")
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for q_enc, d_enc, n_per, meta in loader:
            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            d_enc = {k: v.to(device) for k, v in d_enc.items()}
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                q_out = model(**q_enc)
                d_out = model(**d_enc)
                q_emb = _cls_pool(q_out.last_hidden_state)
                d_emb = _cls_pool(d_out.last_hidden_state)
                loss = _loss_fn(
                    q_emb, d_emb, n_per, args.temperature,
                    pos_tids=meta["pos_tids"],
                    neg_tids_per_row=meta["neg_tids_per_row"],
                ) / accum
                # Free training-side diagnostic: per-row (B, n_per) score
                # matrix on this micro-batch — the SAME shape that
                # val/ndcg uses, so train/ndcg_inbatch and val/ndcg are
                # directly comparable on the TB curve. Used to verify
                # train/val alignment (no leak signature).
                with torch.no_grad():
                    _B_train = q_emb.size(0)
                    _d_per_row = d_emb.view(_B_train, n_per, -1)
                    _train_scores = torch.einsum("bd,bnd->bn",
                                                 q_emb.float(),
                                                 _d_per_row.float())
                    _train_top1, _train_ndcg = _val_metrics_from_scores(_train_scores)
            loss.backward()
            micro_step += 1
            if micro_step % accum == 0:
                # Compute (and log) the gradient norm BEFORE optimizer.step()
                # so we see the unclipped magnitude. max_norm=inf → measure only.
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=float("inf")))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1
                if opt_step % args.logging_steps == 0:
                    # `loss.item()` is the per-micro-batch contribution; multiply
                    # back by accum to get the effective-batch loss for logging.
                    writer.add_scalar("train/loss", float(loss.item()) * accum, opt_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], opt_step)
                    writer.add_scalar("train/grad_norm", grad_norm, opt_step)
                    writer.add_scalar("train/top1_inbatch", _train_top1, opt_step)
                    writer.add_scalar("train/ndcg_inbatch", _train_ndcg, opt_step)
                    print(f"[train-bi-encoder] opt_step={opt_step}/{total_steps} "
                          f"loss={float(loss.item()) * accum:.4f} "
                          f"train_ndcg={_train_ndcg:.4f} "
                          f"grad_norm={grad_norm:.3f}", file=sys.stderr)
                # ML-reviewer N1: periodic val InfoNCE every --val-every-n-steps.
                # Fires AFTER the optimizer step so the loss reflects the latest
                # parameter update.
                if val_loader is not None and args.val_every_n_steps > 0 \
                        and opt_step % args.val_every_n_steps == 0:
                    vresult = _val_loss_now()
                    if vresult is not None:
                        vl, vacc, vndcg = vresult
                        writer.add_scalar("val/loss", vl, opt_step)
                        writer.add_scalar("val/top1_acc", vacc, opt_step)
                        writer.add_scalar("val/ndcg", vndcg, opt_step)
                        improved = vl < best_val_loss
                        if improved:
                            best_val_loss = vl
                        print(f"[train-bi-encoder] opt_step={opt_step}/{total_steps} "
                              f"val_loss={vl:.4f} val_top1={vacc:.3f} "
                              f"val_ndcg={vndcg:.4f}"
                              f"{' (new best)' if improved else ''}",
                              file=sys.stderr)
                # ML-reviewer I-3: full-catalog val nDCG@20 (much more honest
                # than the 16-cand val/ndcg). Slower (~20-40 sec on Blackwell);
                # gate behind --val-full-catalog-every-n-steps.
                if full_cat_enabled and args.val_full_catalog_every_n_steps > 0 \
                        and opt_step % args.val_full_catalog_every_n_steps == 0:
                    fc_ndcg = _val_full_catalog_ndcg(k=20)
                    if fc_ndcg is not None:
                        writer.add_scalar("val/full_catalog_ndcg_at_20",
                                          fc_ndcg, opt_step)
                        print(f"[train-bi-encoder] opt_step={opt_step}/{total_steps} "
                              f"val_full_ndcg@20={fc_ndcg:.4f} "
                              f"(over {len(val_gold_tids)} val queries × "
                              f"{len(catalog_tids)} catalog tracks)",
                              file=sys.stderr)
        # End-of-epoch checkpoint (warm-startable via --resume-from).
        if args.checkpoint_every_n_epochs > 0 \
                and (epoch + 1) % args.checkpoint_every_n_epochs == 0:
            ckpt_dir = output_dir / f"checkpoint_epoch_{epoch + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"[train-bi-encoder] checkpoint saved → {ckpt_dir} "
                  f"(use --resume-from {ckpt_dir} to continue from here)",
                  file=sys.stderr)
    # Flush any partial accumulation at end of training.
    if micro_step % accum != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Final val pass (logged at opt_step so it lands on the same x-axis
    # as the periodic val curve). Useful when --val-every-n-steps would have
    # missed the very last step of training.
    if val_loader is not None:
        final_result = _val_loss_now()
        if final_result is not None:
            fvl, fvacc, fvndcg = final_result
            writer.add_scalar("val/loss", fvl, opt_step)
            writer.add_scalar("val/top1_acc", fvacc, opt_step)
            writer.add_scalar("val/ndcg", fvndcg, opt_step)
            improved = fvl < best_val_loss
            if improved:
                best_val_loss = fvl
            print(f"[train-bi-encoder] FINAL opt_step={opt_step} "
                  f"val_loss={fvl:.4f} val_top1={fvacc:.3f} "
                  f"val_ndcg={fvndcg:.4f} (best_loss={best_val_loss:.4f})",
                  file=sys.stderr)
    if full_cat_enabled:
        final_fc = _val_full_catalog_ndcg(k=20)
        if final_fc is not None:
            writer.add_scalar("val/full_catalog_ndcg_at_20", final_fc, opt_step)
            print(f"[train-bi-encoder] FINAL val_full_ndcg@20={final_fc:.4f}",
                  file=sys.stderr)
    writer.close()
    # Save the adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train-bi-encoder] adapter saved → {output_dir}", file=sys.stderr)


def _merge_and_push(args):
    """Merge LoRA → base, copy sentence-transformers scaffolding from BAAI/bge-m3
    so the merged repo deploys with CLS-pool + L2-normalize, then upload the whole
    folder (not just the HF model files) to the Hub.

    Why the scaffolding copy: training pools `last_hidden_state[:, 0]` (CLS),
    matching BGE-M3's native head. But `merged.push_to_hub` uploads only the
    HF AutoModel artifacts. Without `modules.json` + `1_Pooling/` + `2_Normalize/`
    in the repo, `SentenceTransformer(<hub_repo>)` silently falls back to mean-pool
    + no-normalize defaults → inference uses mean-pool while training used CLS,
    destroying the fine-tune signal.
    """
    import shutil

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    print("[train-bi-encoder] merging LoRA → base", file=sys.stderr)
    base = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, args.output_dir)
    merged = peft_model.merge_and_unload()
    merged_dir = Path(args.output_dir) / "merged"
    merged.save_pretrained(str(merged_dir))
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(str(merged_dir))
    print(f"[train-bi-encoder] merged → {merged_dir}", file=sys.stderr)

    # Pull only the sentence-transformers scaffolding files from the base repo
    # (model weights/tokenizer already saved by save_pretrained above).
    print(f"[train-bi-encoder] copying ST scaffolding from {args.base_model}",
          file=sys.stderr)
    st_src = snapshot_download(
        args.base_model,
        allow_patterns=[
            "modules.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
            "1_Pooling/*",
            "2_Normalize/*",
        ],
    )
    for fname in ("modules.json", "sentence_bert_config.json",
                  "config_sentence_transformers.json"):
        src = Path(st_src) / fname
        if src.exists():
            shutil.copy(src, merged_dir / fname)
    for subdir in ("1_Pooling", "2_Normalize"):
        src = Path(st_src) / subdir
        if src.is_dir():
            shutil.copytree(src, merged_dir / subdir, dirs_exist_ok=True)
    print("[train-bi-encoder] ST scaffolding present:",
          sorted(p.name for p in merged_dir.iterdir() if p.name.startswith(("1_", "2_", "modules", "sentence", "config_sentence"))),
          file=sys.stderr)

    # ML-reviewer I3: pin the pool mode in case the upstream BGE-M3 config
    # ever changes. Training optimizes CLS; if the scaffolding ever ships with
    # mean-pool, the merged model deploys wrong.
    import json as _json
    pool_cfg_path = merged_dir / "1_Pooling" / "config.json"
    if pool_cfg_path.exists():
        pool_cfg = _json.loads(pool_cfg_path.read_text())
        assert pool_cfg.get("pooling_mode_cls_token") is True, (
            f"BAAI/bge-m3 ST scaffolding shipped with non-CLS pooling: "
            f"{pool_cfg}. Training used CLS; deploy would mismatch."
        )
        print(f"[train-bi-encoder] pooling pinned: CLS=True ({pool_cfg_path})",
              file=sys.stderr)

    hub_target = f"{args.hub_repo}-merged"
    print(f"[train-bi-encoder] uploading merged folder → {hub_target}", file=sys.stderr)
    api = HfApi()
    api.create_repo(repo_id=hub_target, exist_ok=True, private=False)
    api.upload_folder(folder_path=str(merged_dir), repo_id=hub_target, repo_type="model")
    return merged_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triples", required=True)
    p.add_argument("--base-model", default="BAAI/bge-m3")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hub-repo", required=True,
                   help="HF Hub repo prefix (the merged model is pushed to <hub_repo>-merged).")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--cleanup-after-push", action="store_true")
    p.add_argument("--results-dir", default=None)
    # Hyperparameters (spec §6)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--n-negatives", type=int, default=15)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16,
                   help="Micro-batches accumulated per optimizer step. With "
                        "per-device-batch-size=2, default 16 → effective batch 32. "
                        "ML reviewer I2: bs=2 + no in-batch negs → very noisy "
                        "InfoNCE signal; accumulating recovers a usable denominator.")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--query-max-len", type=int, default=512)
    p.add_argument("--passage-max-len", type=int, default=256)
    p.add_argument("--lora-rank", type=int, default=64,
                   help="LoRA rank. Sub 2 default 64 (was 32 in Sub 1). Bumped "
                        "alongside the 6x data expansion (~50-65k triples) to "
                        "give more capacity for richer patterns. ~28M trainable "
                        "params (~5%% of base model).")
    p.add_argument("--lora-alpha", type=int, default=128,
                   help="LoRA alpha. Convention is 2*rank (so 128 for r=64).")
    p.add_argument("--logging-steps", type=int, default=50)
    # ML-reviewer I-1: in-batch negatives for stronger contrastive signal.
    p.add_argument("--in-batch-negs", dest="in_batch_negs", action="store_true",
                   default=True,
                   help="Use in-batch InfoNCE (each query contrasted against ALL "
                        "docs in the micro-batch, not just its own 1 pos + 15 negs). "
                        "Standard modern recipe; default ON. Disable via "
                        "--no-in-batch-negs to reproduce per-row contrastive.")
    p.add_argument("--no-in-batch-negs", dest="in_batch_negs", action="store_false",
                   help="See --in-batch-negs.")
    p.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_true", default=True,
                   help="Enable gradient checkpointing (saves VRAM, ~30%% slower). "
                        "Default ON for back-compat with bs=2. Disable via "
                        "--no-gradient-checkpointing when running bs=8+ on "
                        "Blackwell-95GB to recover the wallclock.")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false",
                   help="See --gradient-checkpointing.")
    # ML-reviewer I-3: full-catalog val nDCG.
    p.add_argument("--val-full-catalog-every-n-steps", type=int, default=0,
                   help="Periodically encode the FULL ~50k catalog + val queries "
                        "and compute val/full_catalog_ndcg_at_20 vs the actual "
                        "corpus (not just the 16 mined cands). 0 disables. "
                        "Recommended 200 for production (~20-40 sec/eval). Requires "
                        "val triples carrying pos_tid (new builder schema).")
    p.add_argument("--track-meta-hf", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                   help="Catalog dataset for full-catalog val eval. Ignored "
                        "unless --val-full-catalog-every-n-steps > 0.")
    p.add_argument("--val-encode-batch-size", type=int, default=64,
                   help="Batch size for val-time encoding (full catalog + val "
                        "queries). 64 is safe on Blackwell-95GB.")
    # Reproducibility seed — drives data split, batch samplers, per-row neg
    # sampling, and LoRA init. Practical repro within ~0.01 dev nDCG; not
    # bit-exact due to bf16 + cuDNN non-determinism (see _train() seeding).
    p.add_argument("--seed", type=int, default=42,
                   help="Master seed for reproducibility (defaults to 42; "
                        "data split, samplers, LoRA init all keyed on this).")
    # §6.5 Δ2: user-disjoint train/val split key. Every session of a given
    # user lives in exactly one partition.
    p.add_argument("--split-key", type=str, default="user_id",
                   choices=["user_id", "session_id", "row"],
                   help="Train/val split key. 'user_id' (default) = strongest "
                        "leakage discipline; user's sessions all live in one "
                        "partition. 'session_id' = legacy; 'row' = no group "
                        "semantics (explicit opt-in for triples lacking ids).")
    # ML-reviewer N1: held-out val InfoNCE during training.
    # §6.5 issue D: default bumped 0.05 → 0.10 for stable metric.
    p.add_argument("--val-fraction", type=float, default=0.10,
                   help="Fraction of triples held out from training for periodic "
                        "val-InfoNCE eval. 0.0 disables val tracking. Default 0.10 "
                        "(0.05 was too noisy at typical mine sizes).")
    p.add_argument("--val-every-n-steps", type=int, default=100,
                   help="Run val pass every N optimizer steps. 0 disables. "
                        "Default 100 → roughly every ~10 minutes on Blackwell "
                        "for the production config.")
    # Checkpointing / warm-start.
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=1,
                   help="Save an adapter checkpoint to "
                        "{output_dir}/checkpoint_epoch_{N}/ at the end of every "
                        "Nth epoch. 0 disables. Default 1 (every epoch). "
                        "Each checkpoint is ~30 MB; use --resume-from <ckpt_dir> "
                        "later to warm-start another training run from it.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a previously saved adapter checkpoint (e.g. "
                        "{output_dir}/checkpoint_epoch_3/). When set, loads "
                        "that adapter as the starting point instead of creating "
                        "a fresh LoRA. Optimizer + scheduler restart fresh "
                        "(intentional: clean warmup is healthier than bit-exact "
                        "continuation; --epochs counts ADDITIONAL epochs).")
    args = p.parse_args()

    _train(args)

    if args.merge:
        _merge_and_push(args)

    if args.results_dir is not None:
        rd = Path(args.results_dir)
        rd.mkdir(parents=True, exist_ok=True)
        runs_src = Path(args.output_dir) / "runs"
        if runs_src.exists():
            shutil.copytree(runs_src, rd / "runs", dirs_exist_ok=True)

    if args.cleanup_after_push and args.merge:
        merged_dir = Path(args.output_dir) / "merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)


if __name__ == "__main__":
    main()
