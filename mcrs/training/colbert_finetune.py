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


def triples_to_distillation_rows(triples: Sequence[dict]) -> list[dict]:
    """Builder triples -> PyLate knowledge-distillation rows for `losses.Distillation` (T2.1).

    Each row is {query, documents: [positive] + negatives, scores: [pos_score] + neg_scores} — the
    teacher's relevance over the document list is the soft label the student MaxSim learns to match.
    `documents[0]` is always the positive, aligned with `scores[0]`. Triples WITHOUT teacher scores
    (the plain contrastive path) are skipped — KD has nothing to distill from them. Malformed rows
    (scores/negatives length mismatch) are skipped rather than silently misaligned."""
    rows: list[dict] = []
    for t in triples:
        if t.get("pos_score") is None or t.get("neg_scores") is None:
            continue
        negs = list(t.get("negatives") or [])
        neg_scores = list(t["neg_scores"])
        if len(negs) != len(neg_scores):
            continue
        rows.append({
            "query": t["query"],
            "documents": [t["positive"]] + negs,
            "scores": [float(t["pos_score"])] + [float(s) for s in neg_scores],
        })
    return rows


def count_query_collisions(batch_rows: Sequence[dict]) -> int:
    """Number of rows whose `query` repeats an earlier row in the same batch. 0 means every
    (query, positive) gold is unique within the batch, so no query's own gold becomes an in-batch
    false negative under Contrastive's in-batch negatives. Used to VERIFY the NO_DUPLICATES sampler
    on the first real batch (exploded triples repeat (query, positive) across negatives)."""
    seen: set = set()
    dups = 0
    for r in batch_rows:
        q = r["query"]
        if q in seen:
            dups += 1
        seen.add(q)
    return dups


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


def update_best_state(score: float, state: dict, patience: int) -> tuple[bool, bool]:
    """Update best-tracking state in place; return (is_best, should_stop). `state` is
    {"best": float, "since_improve": int}. A strictly-greater score is a new best (ties keep the
    earlier checkpoint) and resets the counter; otherwise the no-improve counter increments and we
    stop once it reaches `patience` (patience<=0 disables early stop). Pure → unit-tested; the
    GPU callback delegates here so the early-stop decision is covered."""
    is_best = score > state["best"]
    if is_best:
        state["best"] = score
        state["since_improve"] = 0
    else:
        state["since_improve"] += 1
    should_stop = bool(patience) and state["since_improve"] >= patience
    return is_best, should_stop


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
                           seed: int, save_best_fn, early_stop_patience: int = 0,
                           metrics_path: Optional[str] = None):  # pragma: no cover
    """TrainerCallback: every `eval_steps`, rerank a dev turn-1 subset with the LIVE model (same
    MaxSim re-probe as the gate); on a new best dev recall@k call `save_best_fn(model)` to persist it.

    `save_best_fn` is injected so the caller decides WHAT to persist (full model for full-FT, or just
    the LoRA adapter). `early_stop_patience` > 0 halts training after that many consecutive evals
    without improvement (the dev curve here peaks early then decays — don't burn hours past the peak).

    `pack` (built offline from the dev split + fusion pool) has: queries, pools (list[list[tid]]),
    golds, wall, tid_to_text. We encode the live model's query/doc tokens and rerank each pool —
    pool-reranking (not full-catalog) keeps eval cheap during training."""
    import json
    import random
    import sys

    import torch
    from transformers import TrainerCallback

    def _dump(record: dict):
        """Append one metric record as a JSONL line to `metrics_path` (a Drive path) so the loss /
        recall history SURVIVES a VM kill — cell output + in-memory state are lost on preemption.
        Never let logging crash training."""
        if not metrics_path:
            return
        try:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[dev-eval] metrics dump skipped: {e!r}", file=sys.stderr)

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
          f"every {eval_steps} steps -> recall@{k} (patience={early_stop_patience})", file=sys.stderr)

    state = {"best": -1.0, "since_improve": 0}

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
        def _eval_and_save(self, model, step, tag, control=None):
            score = _evaluate(model)
            is_best, should_stop = update_best_state(score, state, early_stop_patience)
            if is_best:
                save_best_fn(model)
            tail = "  <- NEW BEST, saved" if is_best else f"  ({state['since_improve']} eval(s) since best)"
            print(f"[dev-eval] step {step} ({tag}) recall@{k}={score:.4f}{tail}", file=sys.stderr)
            _dump({"kind": "eval", "step": step, "tag": tag, f"recall@{k}": score,
                   "is_best": is_best, "best": state["best"]})
            if should_stop and control is not None:
                control.should_training_stop = True
                print(f"[dev-eval] EARLY STOP: {state['since_improve']} evals without improvement "
                      f"(best recall@{k}={state['best']:.4f})", file=sys.stderr)

        def on_log(self, args, st, control, logs=None, **kwargs):
            # HF Trainer emits train loss / lr / epoch here every logging_steps -> persist them too.
            if logs:
                _dump({"kind": "train", "step": st.global_step, **logs})

        def on_step_end(self, args, st, control, **kwargs):
            if st.global_step > 0 and st.global_step % eval_steps == 0:
                self._eval_and_save(kwargs["model"], st.global_step, "periodic", control)

        def on_train_end(self, args, st, control, **kwargs):
            self._eval_and_save(kwargs["model"], st.global_step, "final")
            print(f"[dev-eval] DONE best recall@{k}={state['best']:.4f}", file=sys.stderr)

    return DevRecallCallback()


def train_colbert(triples: Sequence[dict], out_dir: str, *, base_model: str = "lightonai/GTE-ModernColBERT-v1",
                  dev_eval_pack: Optional[dict] = None, epochs: float = 2.0, batch_size: int = 32,
                  lr: Optional[float] = None, weight_decay: float = 0.01, warmup_ratio: float = 0.1,
                  eval_steps: int = 500, dev_subset: int = 300, q_len: int = 96, d_len: int = 96,
                  k: int = 20, seed: int = 42, force: bool = False,
                  use_lora: bool = True, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
                  lora_target_modules="all-linear", early_stop_patience: int = 2,
                  use_distillation: bool = False, metrics_path: Optional[str] = None) -> str:  # pragma: no cover
    """Fine-tune ColBERT with PyLate Contrastive on the builder triples. Selects the best checkpoint
    via the dev-recall callback (with early stopping) when `dev_eval_pack` is given. Returns out_dir.

    LoRA (default): adapt only LoRA adapters on the backbone, FREEZE the ColBERT projection — the
    frozen base can't catastrophically forget on small/narrow data (the full-FT run peaked at step
    ~1500 then decayed). LoRA's nominal `lr` is higher than full-FT (1e-4 vs 1e-5) but its EFFECT is
    far more conservative. After training, the best adapter is merged into a fresh base so `out_dir`
    is a PLAIN ColBERT the gate loads with no PEFT (avoids the adapter-silently-ignored load bug).

    `base_model` defaults to GTE-ModernColBERT. Skips if out_dir exists unless `force`."""
    import sys

    if existing_artifact_blocks(out_dir, force):
        print(f"[train-colbert] {out_dir} exists — SKIPPING (pass force=True to retrain)", file=sys.stderr)
        return out_dir

    if lr is None:                                # LoRA tolerates a higher LR; full-FT needs ~1e-5 or it
        lr = 1e-4 if use_lora else 1e-5          # diverges/forgets — resolve by mode so it can't be mismatched
        # NOTE behavior change: the old default was an unconditional 1e-4. A caller that omits lr on a
        # full-FT (use_lora=False) run now gets 1e-5, not 1e-4 — the resolved lr is logged below.

    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers
    from pylate import losses, models, utils

    if use_distillation:
        rows = triples_to_distillation_rows(triples)
        if not rows:
            raise ValueError("use_distillation=True but no triple carries teacher scores — build the "
                             "triples with a teacher_score_fn (see build_colbert_train_data) first")
        # Guard a PARTIAL teacher-score cache (e.g. a teacher batch failed mid-build): KD silently
        # training on a fraction of the data would look fine but learn little. Surface the ratio loudly.
        if len(rows) < 0.5 * len(triples):
            raise ValueError(f"only {len(rows)}/{len(triples)} triples carry teacher scores (<50%) — "
                             "the KD score cache looks partial; rebuild the triples (FORCE_REBUILD).")
        # Decorrelate session/turn-ordered rows: HF Trainer shuffles, but pre-shuffle deterministically
        # so KD batches never depend on the trainer's sampler defaults (review H1).
        import random as _random
        _random.Random(seed).shuffle(rows)
    else:
        rows = triples_to_contrastive_rows(triples)
    mode = "distillation (KD)" if use_distillation else "contrastive"
    print(f"[train-colbert] {len(rows)} {mode} rows | base={base_model} | "
          f"{'LoRA r=%d a=%d' % (lora_r, lora_alpha) if use_lora else 'full-FT'} | lr={lr}", file=sys.stderr)
    train_dataset = Dataset.from_list(rows)

    model = models.ColBERT(model_name_or_path=base_model, query_length=q_len, document_length=d_len)

    adapter_dir = out_dir.rstrip("/") + "__adapter"
    if use_lora:
        from peft import LoraConfig, get_peft_model
        for p in model[1].parameters():           # freeze the ColBERT Dense projection (extra-conservative)
            p.requires_grad_(False)
        peft_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=lora_target_modules, bias="none")
        model[0].auto_model = get_peft_model(model[0].auto_model, peft_cfg)
        model[0].auto_model.enable_input_require_grads()
        model[0].auto_model.print_trainable_parameters()
        save_best_fn = lambda m: m[0].auto_model.save_pretrained(adapter_dir)   # noqa: E731 (cheap adapter)
    else:
        def save_best_fn(m):  # full-FT: persist the whole model
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            m.save_pretrained(out_dir)

    # KD (T2.1): distill the K3b cross-encoder teacher's graded relevance into the late-interaction
    # student — the standard ColBERTv2-style recipe (KL/margin-MSE on teacher vs student scores).
    # VERIFY ON COLAB for the pinned pylate version: some versions expect the KD dataset pre-processed
    # via Dataset.map(utils.KDProcessing(...)) and/or a distillation-specific collator. The data shape
    # produced above ({query, documents, scores}) is the inline form; adjust the two lines below if the
    # installed pylate's losses.Distillation expects the id-based (query_id, document_ids) form.
    train_loss = losses.Distillation(model=model) if use_distillation else losses.Contrastive(model=model)

    callbacks = []
    if dev_eval_pack is not None:
        callbacks.append(make_dev_eval_callback(dev_eval_pack, eval_steps, dev_subset, k, seed,
                                                save_best_fn, early_stop_patience=early_stop_patience,
                                                metrics_path=metrics_path))
    else:
        print("[train-colbert] no dev_eval_pack: final-only save, NO dev selection", file=sys.stderr)

    targs = SentenceTransformerTrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio, max_grad_norm=1.0,
        bf16=True, seed=seed, logging_steps=50, save_strategy="no",  # callback owns saving
        # Contrastive: exploded triples -> the same (query, positive) appears K_NEGS times; Contrastive
        # uses IN-BATCH negatives, so two copies in a batch would make a query's own gold an in-batch
        # negative. NO_DUPLICATES drops a row whose any column value already appears in the batch. The
        # guard below VERIFIES this on the first batch. Distillation rows are one-per-query with an
        # explicit documents+scores list (no in-batch-negative reuse), so the default sampler is correct.
        batch_sampler=(BatchSamplers.BATCH_SAMPLER if use_distillation else BatchSamplers.NO_DUPLICATES))
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train_dataset, loss=train_loss,
        data_collator=utils.ColBERTCollator(model.tokenize), callbacks=callbacks)

    # VERIFY the NO_DUPLICATES sampler on the first real batch: count repeated queries (in-batch false
    # negatives). 0 = safe. If >0, the sampler did not dedup and the v3 one-row-per-triple fix is needed.
    # (Contrastive only — distillation rows are one-per-query, no in-batch-negative collision to check.)
    if not use_distillation:
        try:
            first_idx = next(iter(trainer.get_train_dataloader().batch_sampler))
            dups = count_query_collisions([train_dataset[int(i)] for i in first_idx])
            warn = "" if dups == 0 else "  WARNING: NO_DUPLICATES did NOT dedup — apply the v3 fix"
            print(f"[train-colbert] first-batch query collisions: {dups} (0 = no in-batch false negatives)"
                  f"{warn}", file=sys.stderr)
        except Exception as e:                        # never block training on the diagnostic
            print(f"[train-colbert] first-batch collision check skipped: {e!r}", file=sys.stderr)

    trainer.train()

    # Finalize: produce a PLAIN ColBERT at out_dir (gate loads it with no PEFT).
    if use_lora:
        import os
        from peft import PeftModel
        if dev_eval_pack is not None and os.path.exists(adapter_dir):
            # merge the BEST adapter (saved by the callback) into a fresh base + its (frozen) projection
            base = models.ColBERT(model_name_or_path=base_model, query_length=q_len, document_length=d_len)
            base[0].auto_model = PeftModel.from_pretrained(base[0].auto_model, adapter_dir).merge_and_unload()
            base.save_pretrained(out_dir)
        else:                                     # no dev pack -> merge the final in-memory adapter
            model[0].auto_model = model[0].auto_model.merge_and_unload()
            model.save_pretrained(out_dir)
    elif dev_eval_pack is None:
        save_best_fn(model)
    print(f"[train-colbert] best/final model -> {out_dir}", file=sys.stderr)
    return out_dir
