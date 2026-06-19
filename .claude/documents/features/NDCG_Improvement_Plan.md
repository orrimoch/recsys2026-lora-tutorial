# nDCG@20 Improvement Plan — per-model levers

## Context

The metric we are pushing is nDCG@20 on the Blind-A leaderboard. Blind eval scores only the
final turn of each session (one row/session, 80 rows), so calibrate against a dev *final-turn*
proxy, not the dev all-turns average. The stack is fixed (no architecture change):

```
RRF fusion (BM25 + dense + cknn_audio + cknn_attr + cf + same_artist + related_artist + ColBERT)
  -> K2 (LGBM LambdaMART)  [frozen-CE score is one K2 feature]
  -> K3b (bge-reranker-v2-m3 + LoRA, top-K re-score, chained after K2)
```

This plan proposes a small number of improvements per model — cross-encoder (K3b), LGBM (K2),
ColBERT — each aimed at the single biggest lever for that model, plus a few ranked supporting
changes. No change touches the spine: ColBERT stays a late-interaction RRF channel, K2 stays an
LGBM LambdaMART reranker, K3b stays a LoRA cross-encoder re-scoring the top-K.

## Guiding principle: where nDCG@20 actually comes from

nDCG@20 = (is the gold in the pool at all?) × (is it ranked into the top-20?).

- recall@pool is the ceiling. Today fused recall ≈ 0.65–0.69 @500; ~31–35% of golds are not even
  reachable. Raising the ceiling is ColBERT's job.
- ranking-within-pool is the realised score under that ceiling. That is K2 + K3b's job.

So the two families attack different parts and compound. Estimates below are hypotheses to be
measured on the dev final-turn proxy — treat them as priority signals, not promises.

Dependency note (informs the suggested order at the end): K3b is also the teacher for ColBERT
distillation and the source of K2's frozen-CE feature. Improving K3b first cascades into both
other models.

A note on the obvious lever: more epochs / more training data (larger `TRAIN_SUBSET`, more
sessions, full-train retrain) is a valid, low-risk baseline for all three models — and the cheapest
thing to try. It is deliberately listed as a supporting item in each section rather than a
"biggest lever", because it scales with compute but has diminishing returns and does not change the
shape of the supervision. The biggest-lever picks below are the less-obvious, higher-EV changes
(deeper re-score window, cross-encoder distillation, richer discriminative features). Run the
obvious lever in parallel as the floor; chase the headline levers for the real gains.

---

## 1. Cross-encoder (K3b) — biggest lever: deeper re-score window

### Current recipe (grounded)
- Base `BAAI/bge-reranker-v2-m3`; LoRA r=16, α=32, dropout=0.05, targets `[query, value]` + classifier head (`mcrs/training/ce_finetune.py:112-113`).
- Loss: masked listwise softmax-CE, per-group goal-progress weight, gold at index 0 (`mcrs/training/ce_loss.py:9-33`).
- Negatives: `N_NEG=15`, rank-stratified from the fused pool, same-artist soft-downweighted 0.25, near-dup denoised (`mcrs/training/ce_data.py:48-90`).
- Re-score depth: `CROSS_ENCODER_K=100` at both train and serve (`mcrs/rerank/neural.py:38`, `mcrs/training/ce_data.py:169`).
- `MAX_LEN=2048`, `MAX_DOC_TOK=1100`, query side never truncated (`mcrs/rerank/cross_encoder.py:103-107`).
- Goal-progress group weights MOVES=1.0 / DOES_NOT=0.3 (`mcrs/training/ce_data.py:112`).
- Val/early-stop on real nDCG@20 (`mcrs/training/ce_finetune.py:155`), patience small.
- `TRAIN_SUBSET=4000` of 15,199 sessions (~26%).

### Biggest lever — raise CROSS_ENCODER_K 100 → 200 (→500 on A100), with matched negatives
K3b can only reorder candidates it actually scores: `top = candidates[:cross_encoder_k]`
(`mcrs/rerank/neural.py:51-52`). With K=100, any gold that K2 ranked at position 101–500 is
invisible to the CE and can never reach the top-20. At K=200 the CE both trains on harder negatives
sampled deeper in the pool and rescues mid-pool golds at serve. Train and serve K must move
together (train==serve parity) and `N_NEG` should rise with K (see below) so the deeper window is
actually populated with contrast.

Files: `nb/phase2_ce_finetune.ipynb` (config `CROSS_ENCODER_K`); already plumbed through
`mcrs/training/ce_data.py:115,151,169` and `mcrs/rerank/neural.py:38,51`. Cost scales linearly in K
(pairs/turn), so K=500 is an A100 item; K=200 is the T4-feasible first step. Estimated +0.01–0.03.

### Supporting improvements (ranked)
| # | Change | Files | Why | Est. |
|---|--------|-------|-----|------|
| 1 | `N_NEG` 15 → 30 (toward the serve window) | `nb/phase2_ce_finetune.ipynb`, sampler `mcrs/training/ce_data.py:48-90` | More contrast per gold; only pays off once K is large enough to contain them — pair with the K lever | +0.005–0.015 |
| 2 | Full `TRAIN_SUBSET` (→15,199) on A100 | `nb/phase2_ce_finetune.ipynb` | 26% → 100% coverage of the train distribution; combine with K=500 | +0.015–0.04 |
| 3 | False-negative denoise of sampled negs | `mcrs/training/ce_data.py` (negative selection) | Drop negatives the *frozen* base CE scores very high (likely unlabeled positives) before training — stops penalising true matches | +0.003–0.01 |

### Avoid
- Raising `MAX_LEN` past 2048 (measured token p99 ≈ 1790; query preserved, doc capped — already right) — buys OOM, not signal.
- Removing/equalising goal-progress weights (it is a real, leak-safe per-group signal; gate, don't drop — see §4).
- Raw (non-enriched) docs at train or serve — train/serve token-distribution skew; spec §2 hard-fails this.
- Raising LoRA lr above 1e-4 (B-matrix zero-init + tiny adapter → instability).

---

## 2. LGBM (K2) — biggest lever: stronger discriminative supervision

K2's job is separating the one gold from ~150 hard in-pool negatives. It currently learns from a
fairly thin signal: per-channel rank-inverses + `rrf_score`/`n_channels_hit` + a few catalog
features + two strong score features (`dense_cos`, frozen `ce_score`), with binary labels
(`mcrs/rerank/features.py:33-49`, one positive per group, `mcrs/rerank/lgbm.py`). Two levers raise
the ranking signal; lead with the lower-risk one.

### Current recipe (grounded)
- 19 features (`mcrs/rerank/features.py:43-48` + injected `dense_cos`, `ce_score`).
- Binary labels, one gold/group; gold-not-in-pool groups skipped.
- `objective=lambdarank`, `metric=ndcg`, `eval_at=[20]`; `label_gain` and `lambdarank_truncation_level` left at defaults.
- Negatives: random cap `NEG_CAP=150` (random, not top-N, by design — top-N would teach an inverted score↔gold correlation), `mcrs/rerank/lgbm.py:47-60`.
- Regularised params already tuned (lr 0.05, num_leaves 15, min_child_samples 30, L1/L2, subsample 0.8), early stop 50 on a session-disjoint val.

### Biggest lever (high confidence, low risk) — richer discriminative features + LambdaMART nDCG@20 tuning
Add interaction/consensus features that the model cannot currently express, plus calibrate the two
score features per turn, plus point LambdaMART at the metric horizon:
- Consensus/interaction features in `mcrs/rerank/features.py:build()`: e.g. `top5_bm25_and_dense`,
  `consensus_3plus` (# channels ranking the cand ≤10), `dense_cf_agree`. Multi-view agreement is a
  strong, leak-free gold signal built from existing per-channel ranks.
- Per-turn min–max calibration of `dense_cos` (and re-confirm `ce_score`, already pool-normalised in
  `mcrs/rerank/neural.py`) so magnitudes are comparable across turns.
- Set `lambdarank_truncation_level=20` (align pair generation with nDCG@20) in `mcrs/rerank/lgbm.py:fit()`.

Estimated +0.01–0.03 combined; low risk (pure feature/objective config, no new data source).

### Second lever (gated) — graded/weighted relevance via goal-progress
Mirror K3b: weight the gold per group by its turn's goal-progress assessment (MOVES vs DOES_NOT)
via LambdaMART `label_gain` / per-group weighting, threading a grade through
`mcrs/rerank/train.py:build_rerank_groups` and `mcrs/rerank/lgbm.py:_xy/fit`. Caveat that downgrades
this from "biggest" to "gated": there is a known train/test goal-progress distribution shift
(train ≈ 44% MOVES / 43% DOES_NOT vs test ≈ 77% / 10%), and Blind-A scores only the final turn.
Gate before shipping (see §4): verify goal-progress is present at serve, and that the
`DOES_NOT_MOVE_TOWARD_GOAL` slice does not regress. Estimated +0.01–0.03 if it holds, 0 (or revert) if not.

### Supporting improvements (ranked)
| # | Change | Files | Why | Est. |
|---|--------|-------|-----|------|
| 1 | Recompute the frozen `ce_score` feature at the new K3b depth | `nb/phase2_rerank.ipynb` (CE-stack block), `mcrs/rerank/neural.py` | Keeps K2's strongest feature aligned with the deeper, better CE from §1 | +0.005–0.015 |
| 2 | Popularity-percentile + recency features (debias) | `mcrs/rerank/features.py` | Lets K2 down-weight popularity except when the query matches — helps niche golds and diversity | +0.005–0.015 |
| 3 | Per-segment (cold/warm) boosters | `mcrs/rerank/lgbm.py:fit/rerank` | Cold turns lack personalization features → different feature importance; specialise | +0.005–0.01 |

### Avoid
- Top-N (instead of random) negative capping — teaches an inverted score↔gold correlation (`mcrs/rerank/lgbm.py:47-60`).
- Re-encoding RRF order as a new feature (`rrf_rank_inv`) — redundant with existing per-channel rank-inverses; known prior dud (spec 51 §4.1).
- Fine-tuning a CE *inside* K2 (in-sample leak) — the frozen-CE stack is the leak-safe form; fine-tuned CE belongs in K3b with OOF.
- More HP tuning — already regularised; the gap is not under/over-fit.

---

## 3. ColBERT — biggest lever: distillation from the K3b cross-encoder teacher

ColBERT is the recall-ceiling lever. The LoRA fine-tune already beats dense at every depth
(fused recall lift ≈ +3.6pts@200). Its biggest remaining lever is supervision quality, not size.

### Current recipe (grounded)
- Base `lightonai/GTE-ModernColBERT-v1`; LoRA r=16/α=32, projection frozen (`mcrs/training/colbert_finetune.py:214-215`).
- Loss: PyLate `losses.Contrastive` — binary positive/negative only (`mcrs/training/colbert_finetune.py:227`). No teacher scores anywhere.
- Hard negs: `POOL_SIZE=100`, `K_NEGS=15` from the fused pool (`mcrs/training/colbert_data.py:32-37`).
- Query: focused `QueryBuilder(recency_window=1)` (last utterance + goal) — correct for MaxSim's additive semantics (`mcrs/retrieval/query.py:34-39`).
- `Q_LEN=96`, `D_LEN=300` (enriched-doc p99 ≈ 866 → ~14% of docs truncated); expansion-first keeps doc2query under the cap (`mcrs/retrieval/colbert_channel.py:50-62`).
- Checkpoint selection: turn-1 dev recall@20; held-out G3 gate on sessions 500:1000.

### Biggest lever — knowledge distillation (D3), Contrastive → Distillation
Contrastive loss is binary: it cannot teach that a top-5 non-gold is closer to the gold than a
bottom-10 non-gold. Distilling the K3b cross-encoder's fine-grained relevance into the
late-interaction student transfers exactly that ranking signal — the most reliably documented
recall/ranking lever for ColBERT, and already scoped-but-deferred in `Colbert_Improved.md` (D3).
Concretely:
- Extend the triple schema to carry per-negative teacher scores (`mcrs/training/colbert_data.py:42-53`).
- Add an offline scoring step: score `(query, [gold + hard_negs])` with the K3b CE, store in the JSONL.
- Switch the loss to `pylate.losses.Distillation` (KL / margin-MSE on normalised teacher vs student
  scores) when teacher scores are present, Contrastive as fallback (`mcrs/training/colbert_finetune.py:227`).
- New config flag `USE_DISTILLATION` in `nb/phase2_colbert_finetune.ipynb`.

Estimated +0.015–0.03 nDCG@20 ceiling (recall@pool +2–4% relative, per the PyLate/JaColBERT line).
Use the improved §1 K3b as the teacher.

### Supporting improvements (ranked)
| # | Change | Files | Why | Est. |
|---|--------|-------|-----|------|
| 1 | `D_LEN` 300 → 512 | `nb/phase2_colbert_finetune.ipynb`, index `mcrs/training/colbert_index.py` | ~14% of enriched docs are truncated today, dropping high-value doc2query/tags; ModernBERT trains-short/serves-long | +0.005–0.02 |
| 2 | Faceted / multi-aspect query (A3) | `mcrs/retrieval/query.py:_build_enriched`, notebook wiring | MaxSim scores aspects independently; a genre+era+mood+artist query exploits that and fills the underused Q_LEN budget | +0.01–0.03 |
| 3 | False-negative filtering of hard negs via teacher | `mcrs/training/colbert_data.py:32-37` | Drop pool negatives the CE scores above threshold (likely unlabeled positives) — same denoise as §1.3, cleaner contrast | +0.005–0.01 |
| 4 | `K_NEGS` 15 → 25 | `nb/phase2_colbert_finetune.ipynb` | More contrast per positive; cheap | +0.005–0.01 |

### Avoid
- Full-dialogue (non-focused) query — junk/stale tokens only add to MaxSim and blur ranking (R8 §3.2).
- Chasing a different ANN backend — current PLAID retrieval is end-to-end and fine; the prize is recall/ranking, not latency.
- Running dense (R4) and ColBERT both as query channels — redundant; ColBERT replaces R4's query-dense role.
- Expecting to break the ~0.66@500 wall — it is content-fundamental; ColBERT's realistic prize is cold-segment unique recall + a stronger reranker base.

---

## 4. Validation protocol (every change)

- Measure on the dev final-turn proxy (one trailing turn per dev session) — this is how Blind-A is
  scored. Use `nb/phase3_dev_experiments.ipynb` for slot-free retrieval/rerank levers, and each
  model's own training notebook for that model.
- ColBERT changes additionally pass the held-out G3 recall gate (`nb/phase2_colbert_finetune.ipynb`):
  beats dense @200, or fused lift ≥ +0.01 @200, or unique-recall ≥ 0.02.
- K2 / K3b changes ship only on a measured dev nDCG@20 lift over the current model, sliced by
  segment (cold/warm). K3b also keeps the spec §4.3 guard: abort the goal-progress lever if the
  `DOES_NOT_MOVE_TOWARD_GOAL` slice regresses.
- K2 goal-progress lever gate (before any train): confirm `goal_progress_assessments` exist on the
  serve/blind rows; if absent, do not enable (train/serve mismatch).
- Keep train==serve parity for every CE/ColBERT knob (K, N_NEG don't affect serve; `MAX_LEN`,
  `MAX_DOC_TOK`, `D_LEN`, `Q_LEN`, query construction must match between train and the index/serve).

## 5. Suggested sequencing (because the models share signals)

1. K3b first (§1): deeper K + more negatives (+ full data on A100). This improves the teacher.
2. ColBERT (§3): distil from the improved K3b; add `D_LEN`=512 and faceted query. Raises the ceiling.
3. K2 (§2): retrain with the richer feature set + nDCG@20 tuning, recompute the frozen-CE feature at
   the new K3b depth, then optionally the gated goal-progress lever and per-segment models.

Each section is independently runnable in its own notebook, but this order lets the cascade compound.

## 6. Impact / effort summary

| Model | Biggest lever | Confidence | Effort | Est. nDCG@20 |
|-------|---------------|------------|--------|--------------|
| K3b | re-score depth K 100→200(→500) + matched negs | high | low (config) → med (A100 full data) | +0.01–0.03 |
| ColBERT | CE→ColBERT distillation (D3) | high | med (new loss + teacher scoring step) | +0.015–0.03 ceiling |
| K2 | consensus/interaction features + nDCG@20 tuning | high | low | +0.01–0.03 |
| K2 | graded goal-progress labels | medium (gated) | low–med | +0.01–0.03 or revert |
