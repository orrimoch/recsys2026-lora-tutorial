"""K3b — cross-encoder fine-tune training loop + OOF orchestration."""
from __future__ import annotations

import math, random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from mcrs.training.ce_data import assign_session_folds
from mcrs.training.ce_loss import masked_listwise_ce


def oof_ce_scores(turns, *, folds, seed, fit_fn, score_fn):
    """Leak-free per-CANDIDATE CE scores via k-fold session-disjoint cross-fitting (spec §2/§4.6).

    turns: list of (session_id, turn_number). fit_fn(train_rows, fold=)->model. score_fn(model, row)
    returns a {track_id: score} dict for that turn's candidate pool. Each held-out row is scored ONLY
    by a model trained on the OTHER folds. Returns {(session_id, turn_number, track_id): score} — the
    per-candidate granularity K2 needs to stack `ce_ft_score` as a K1 feature.
    """
    sids = [s for s, _ in turns]
    fold_of = assign_session_folds(sids, k=folds, seed=seed)
    rows = [{"session_id": s, "turn": t, "fold": f} for (s, t), f in zip(turns, fold_of)]
    out = {}
    for held in range(folds):
        train_rows = [(r["session_id"], r["turn"], r["fold"]) for r in rows if r["fold"] != held]
        model = fit_fn(train_rows, fold=held)               # trained on the OTHER folds only
        for r in rows:
            if r["fold"] == held:
                for tid, s in score_fn(model, r).items():   # per-candidate scores for this turn's pool
                    out[(r["session_id"], r["turn"], tid)] = s
    return out


class CEGroupDataset(Dataset):
    """Holds [(query, [pos_doc, neg...], group_weight)]; tokenizes pairs lazily. `group_len` is a cheap
    length proxy (query token count) so the sampler can batch similar-length groups (less pad waste)."""
    def __init__(self, groups, tokenizer, max_length, max_doc_tokens, doc_budget_fn, truncate_fn):
        self.groups = groups
        self.tok = tokenizer
        self.max_length = max_length
        self.max_doc_tokens = max_doc_tokens
        self.doc_budget_fn = doc_budget_fn
        self.truncate_fn = truncate_fn
        self._qlen = [len(self._enc(g[0])) for g in groups]   # cached query token counts

    def _enc(self, s):
        return self.tok.encode(s, add_special_tokens=False, truncation=True, max_length=self.max_length)

    def group_len(self, i): return self._qlen[i]              # length proxy for the sampler
    def __len__(self): return len(self.groups)

    def __getitem__(self, i):
        q, docs, gw = self.groups[i]
        budget = self.doc_budget_fn(self._qlen[i], self.max_length, self.max_doc_tokens)
        pairs = [(q, self.truncate_fn(self._enc, self.tok.decode, d, budget)) for d in docs]
        return {"pairs": pairs, "group_weight": gw, "size": len(pairs)}


class LengthGroupedSampler(Sampler):
    """Order indices so each micro-batch holds similar-length groups (minimizes padding -> faster GPU).
    Shuffles within length-bucketed mega-batches so epochs still differ."""
    def __init__(self, dataset, batch_size, shuffle=True, mega=50):
        self.lengths = [dataset.group_len(i) for i in range(len(dataset))]
        self.shuffle, self.mega = shuffle, max(1, mega) * batch_size

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(idx)
        out = []
        for s in range(0, len(idx), self.mega):                # sort each mega-block by length
            out.extend(sorted(idx[s:s + self.mega], key=lambda i: self.lengths[i]))
        return iter(out)

    def __len__(self): return len(self.lengths)


def _collate(batch, tokenizer, max_length):
    flat = [p for ex in batch for p in ex["pairs"]]
    sizes = [ex["size"] for ex in batch]
    weights = [ex["group_weight"] for ex in batch]
    feats = tokenizer([q for q, _ in flat], [d for _, d in flat],
                      padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return feats, sizes, weights


def finetune_cross_encoder(groups_train, groups_val, *, base_model, lora_cfg,
                           max_length=2048, max_doc_tokens=1100, dtype="auto",
                           train_cfg, logger, out_dir, val_eval_fn=None):
    """LoRA fine-tune of bge-reranker-v2-m3 with masked listwise-softmax, GPU-optimized for a 16GB T4/G4.

    Memory/throughput: gradient CHECKPOINTING (fits seq=2048 on 16GB) + mixed precision (bf16 where the GPU
    supports it, else fp16 + GradScaler — a T4/G4 has no bf16) + length-grouped batching. STABILITY: a large
    EFFECTIVE batch via gradient ACCUMULATION (effective = batch_groups * grad_accum) without OOM. EARLY
    STOPPING on val nDCG@20 with `early_stop_patience` epochs; the best-val checkpoint is the returned adapter.
    NOTE: no in-batch negatives (a cross-encoder can't reuse them cheaply) — negatives-per-gold is `N` from
    sampling (the contrast knob); accumulation grows the gradient batch (the stability knob). They are distinct.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from mcrs.rerank.cross_encoder import doc_token_budget, truncate_doc_tokens, _resolve_dtype

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prec = _resolve_dtype(dtype, dev == "cuda" and torch.cuda.is_bf16_supported())    # bf16|fp16|fp32
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[prec]
    use_amp = (dev == "cuda" and prec != "fp32")
    use_scaler = (prec == "fp16")                              # GradScaler only for fp16

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    model.gradient_checkpointing_enable()                     # trade compute for memory (key for 2048 on T4)
    model.config.use_cache = False                            # required with checkpointing
    peft_cfg = LoraConfig(r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                          target_modules=lora_cfg["target_modules"], modules_to_save=["classifier"])
    model = get_peft_model(model, peft_cfg)
    model.enable_input_require_grads()                        # so checkpointing tracks LoRA grads
    model = model.to(dev)                                     # master weights fp32; autocast does the compute

    micro = train_cfg["batch_groups"]                         # micro-batch (groups) sized to fit memory
    accum = max(1, train_cfg.get("grad_accum", 1))            # EFFECTIVE batch = micro * accum (stability)
    patience = train_cfg.get("early_stop_patience", 1)        # stop after N epochs without val improvement

    def make_loader(groups, shuffle):
        ds = CEGroupDataset(groups, tok, max_length, max_doc_tokens, doc_token_budget, truncate_doc_tokens)
        sampler = (LengthGroupedSampler(ds, micro, shuffle=shuffle)
                   if train_cfg.get("group_by_length", True) else None)
        return DataLoader(ds, batch_size=micro, sampler=sampler, shuffle=(shuffle and sampler is None),
                          collate_fn=lambda b: _collate(b, tok, max_length), pin_memory=(dev == "cuda"))

    from transformers import get_cosine_schedule_with_warmup
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0))
    n_micro = math.ceil(len(groups_train) / micro)
    total_steps = max(1, math.ceil(n_micro / accum) * train_cfg["epochs"])   # OPTIMIZER steps over the run
    warmup_steps = int(train_cfg.get("warmup", 0.05) * total_steps)          # ~5% linear warmup
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)  # then cosine decay to ~0
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    best, best_metric, since_improved, step, loss_ema = None, -math.inf, 0, 0, None
    for epoch in range(train_cfg["epochs"]):
        model.train(); opt.zero_grad()
        for i, (feats, sizes, weights) in enumerate(make_loader(groups_train, True)):
            feats = {k: v.to(dev, non_blocking=True) for k, v in feats.items()}
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                logits = model(**feats).logits.squeeze(-1)
                loss = masked_listwise_ce(logits, sizes, weights) / accum     # scale for accumulation
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:                           # optimizer step once per `accum` micro-batches
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
                step += 1
                cur = float(loss) * accum                       # undo the accumulation scaling for logging
                loss_ema = cur if loss_ema is None else 0.98 * loss_ema + 0.02 * cur   # smooth the noisy curve
                if step % train_cfg.get("log_every", 50) == 0:  # EMA is display-only; never feeds optimization
                    logger.log({"train_loss": cur, "train_loss_ema": loss_ema,
                                "lr": sched.get_last_lr()[0], "epoch": epoch}, step=step)
        # ---- validation + early stopping ----
        if dev == "cuda": torch.cuda.empty_cache()
        model.eval()
        with torch.inference_mode():
            vl = []
            for feats, sizes, weights in make_loader(groups_val, False):
                feats = {k: v.to(dev) for k, v in feats.items()}
                with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                    vl.append(float(masked_listwise_ce(model(**feats).logits.squeeze(-1), sizes, weights)))
            val_loss = sum(vl) / max(len(vl), 1)
        val_ndcg = val_eval_fn(model, tok) if val_eval_fn else -val_loss      # dev nDCG@20 (real metric)
        logger.log({"val_loss": val_loss, "val_ndcg@20": val_ndcg, "epoch": epoch}, step=step)
        if val_ndcg > best_metric:                             # improved -> checkpoint best, reset patience
            best_metric, since_improved, best = val_ndcg, 0, out_dir
            model.save_pretrained(out_dir)
        else:
            since_improved += 1
            if since_improved >= patience:                     # EARLY STOP
                logger.log({"early_stop_epoch": epoch}, step=step)
                break
        if dev == "cuda": torch.cuda.empty_cache()
    return best
