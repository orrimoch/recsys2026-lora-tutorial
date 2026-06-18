"""Stage 2 — fine-tune ColBERT with the ORIGINAL feedback: PyLate `losses.Contrastive` over the
(query, positive, negative) triples from `colbert_data`, with a turn-1 dev-recall callback that
selects the single best checkpoint (never a train-internal loss — every train-side metric has been
anti-correlated with dev in this campaign).

Ports `scripts/train_colbert.py` (recall-union-lgbm) onto fresh-start. Pure functions here are
unit-tested; `make_dev_eval_callback` + `train_colbert` are GPU integration (run on Colab).

Leak discipline: train on TRAIN-split triples; select on TEST-split (dev) turn-1 — a session-disjoint
split (memory: the K3b leak was selecting on train sessions).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from mcrs.retrieval.colbert_channel import maxsim_score


# --------------------------------------------------------------------------- #
# Pure functions (unit-tested).
# --------------------------------------------------------------------------- #
def triples_to_contrastive_rows(triples: Sequence[dict]) -> list[dict[str, str]]:
    """Explode each builder triple {query, positive, negatives[]} into flat
    (query, positive, negative) rows — one per negative — for PyLate's Contrastive loss.
    Triples with no negatives are dropped (no pair to form)."""
    rows: list[dict[str, str]] = []
    for t in triples:
        for neg in (t.get("negatives") or []):
            rows.append({"query": t["query"], "positive": t["positive"], "negative": neg})
    return rows


def wall_recall_at_k(ranked_lists: Sequence[Sequence[str]], golds: Sequence[str],
                     wall: Sequence[bool], k: int) -> float:
    """recall@k over the wall subset only (rows where wall[i] is True). Returns 0.0 when there are
    no wall rows (safe sentinel so best-model max-tracking never sees NaN). At turn-1 every row is
    wall, so this == plain turn-1 recall@k."""
    idx = [i for i, w in enumerate(wall) if w]
    if not idx:
        return 0.0
    hits = sum(1.0 for i in idx if golds[i] in ranked_lists[i][:k])
    return hits / len(idx)


def existing_artifact_blocks(path: str, force: bool) -> bool:
    """True if `path` exists and --force was NOT passed → SKIP to avoid clobbering a live artifact
    in place (the 2026-06-15 accident overwrote the 0.50 ColBERT). Pure → unit-tested."""
    return bool(path) and os.path.exists(path) and not force


def rerank_pool(query_emb: np.ndarray, pool_embs: Sequence[np.ndarray],
                pool_tids: Sequence[str], topk: int) -> list[str]:
    """Rerank a candidate pool by MaxSim; return the top-`topk` track ids. Deterministic:
    descending score, ties broken by ascending pool index (matches ColBERTChannel)."""
    scores = np.array([maxsim_score(query_emb, d) for d in pool_embs], dtype=np.float32)
    order = np.lexsort((np.arange(len(scores)), -scores))[:topk]
    return [pool_tids[i] for i in order]


# --------------------------------------------------------------------------- #
# Dev-eval callback (GPU integration; run on Colab, not unit-tested).
# --------------------------------------------------------------------------- #
def make_dev_eval_callback(pack: dict, eval_steps: int, dev_subset: int, k: int,
                           out_dir: str, seed: int):  # pragma: no cover
    """TrainerCallback: every `eval_steps`, rerank a dev turn-1 subset with the LIVE model (same
    MaxSim re-probe as the gate) and save+override the single best checkpoint by dev recall@k.

    `pack` (built offline from the dev split + fusion pool) has: queries, pools (list[list[tid]]),
    golds, wall, tid_to_text. We encode the live model's query/doc tokens and rerank each pool —
    pool-reranking (not full-catalog) keeps eval cheap during training."""
    import random
    import sys

    import torch
    from transformers import TrainerCallback

    n = len(pack["queries"])
    order = list(range(n))
    random.Random(seed).shuffle(order)
    sub = sorted(order[: min(dev_subset, n)])
    queries = [pack["queries"][i] for i in sub]
    pools = [pack["pools"][i] for i in sub]
    golds = [pack["golds"][i] for i in sub]
    wall = [pack["wall"][i] for i in sub]
    need_tids = sorted({t for pool in pools for t in pool})
    need_texts = [pack["tid_to_text"][t] for t in need_tids]
    print(f"[dev-eval] subset={len(queries)} turn-1 queries, {len(need_tids)} pool docs, "
          f"every {eval_steps} steps -> recall@{k} (best -> {out_dir})", file=sys.stderr)

    state = {"best": -1.0}

    def _evaluate(model) -> float:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            doc_emb = {t: np.asarray(e, dtype=np.float32) for t, e in
                       zip(need_tids, model.encode(need_texts, is_query=False, show_progress_bar=False))}
            q_emb = model.encode(queries, is_query=True, show_progress_bar=False)
            ranked = []
            for qe, pool in zip(q_emb, pools):
                tids = [t for t in pool if t in doc_emb]
                ranked.append(rerank_pool(np.asarray(qe, dtype=np.float32),
                                          [doc_emb[t] for t in tids], tids, topk=100))
        if was_training:
            model.train()
        return wall_recall_at_k(ranked, golds, wall, k)

    class DevRecallCallback(TrainerCallback):
        def _eval_and_save(self, model, step, tag):
            score = _evaluate(model)
            best = score > state["best"]
            if best:
                state["best"] = score
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(out_dir)
            print(f"[dev-eval] step {step} ({tag}) recall@{k}={score:.4f}"
                  f"{'  <- NEW BEST, saved (override)' if best else ''}", file=sys.stderr)

        def on_step_end(self, args, st, control, **kwargs):
            if st.global_step > 0 and st.global_step % eval_steps == 0:
                self._eval_and_save(kwargs["model"], st.global_step, "periodic")

        def on_train_end(self, args, st, control, **kwargs):
            self._eval_and_save(kwargs["model"], st.global_step, "final")
            print(f"[dev-eval] DONE best recall@{k}={state['best']:.4f} -> {out_dir}", file=sys.stderr)

    return DevRecallCallback()


def train_colbert(triples: Sequence[dict], out_dir: str, *, base_model: str = "lightonai/GTE-ModernColBERT-v1",
                  dev_eval_pack: Optional[dict] = None, epochs: float = 2.0, batch_size: int = 32,
                  lr: float = 1e-5, weight_decay: float = 0.01, warmup_ratio: float = 0.1,
                  eval_steps: int = 500, dev_subset: int = 300, q_len: int = 96, d_len: int = 96,
                  k: int = 20, seed: int = 42, force: bool = False) -> str:  # pragma: no cover
    """Fine-tune ColBERT with PyLate Contrastive on the builder triples. Selects the best checkpoint
    via the dev-recall callback when `dev_eval_pack` is given, else final-only save. Returns out_dir.

    `base_model` defaults to GTE-ModernColBERT (the channel/probe model — far stronger than the
    original colbertv2.0 default); pass another to switch. Skips if out_dir exists unless `force`
    (guards the in-place-overwrite accident)."""
    import sys

    if existing_artifact_blocks(out_dir, force):
        print(f"[train-colbert] {out_dir} exists — SKIPPING (pass force=True to retrain)", file=sys.stderr)
        return out_dir

    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from pylate import losses, models, utils

    rows = triples_to_contrastive_rows(triples)
    print(f"[train-colbert] {len(rows)} contrastive rows | base={base_model}", file=sys.stderr)
    train_dataset = Dataset.from_list(rows)

    model = models.ColBERT(model_name_or_path=base_model, query_length=q_len, document_length=d_len)
    train_loss = losses.Contrastive(model=model)

    callbacks = []
    if dev_eval_pack is not None:
        callbacks.append(make_dev_eval_callback(dev_eval_pack, eval_steps, dev_subset, k, out_dir, seed))
    else:
        print("[train-colbert] no dev_eval_pack: final-only save, NO dev selection", file=sys.stderr)

    targs = SentenceTransformerTrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio, max_grad_norm=1.0,
        bf16=True, seed=seed, logging_steps=50, save_strategy="no")  # callback owns saving
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train_dataset, loss=train_loss,
        data_collator=utils.ColBERTCollator(model.tokenize), callbacks=callbacks)
    trainer.train()

    if dev_eval_pack is None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
    print(f"[train-colbert] best/final model -> {out_dir}", file=sys.stderr)
    return out_dir
