# K3b — Cross-Encoder LoRA Fine-Tune (bge-reranker-v2-m3) + OOF Stacking

> Phase-2, optional. Implements the LoRA fine-tune option of K3 (`52_K3_neural_reranker.md` §4).
> Fine-tunes `BAAI/bge-reranker-v2-m3` on Train with denoised hard negatives and a grouped listwise-softmax
> (contrastive) loss, then serves the adapter two ways: (a) final-stage rerank inside `ChainReranker(K2, K3)`,
> and (b) an out-of-fold `ce_ft_score` feature stacked back into K1 → K2.
> Gated: ships only on a measured dev nDCG@20 lift over K2. See `52_K3_neural_reranker.md`, `000_INDEX.md`.

## 1. Purpose
Adapt the off-the-shelf cross-encoder to the music-CRS query/doc distribution with a cheap LoRA fine-tune,
sharpening the top-1..3 ordering K2 can't reach. Because the fine-tuned model has seen the gold labels, its
stacked score must be produced out-of-fold — the OOF constraint the frozen K3 model did not need.

## 2. Interface / contract
New module `mcrs/training/ce_finetune.py`:

```python
def build_doc(catalog, track_id, *, max_doc_chars=2000) -> str
    # The SINGLE doc builder, used by train positives, train negatives, AND serve (NeuralReranker._doc):
    # id_to_metadata(enriched=True)[:max_doc_chars], then doc-side token truncation. Same string train==serve.

def build_ce_training_groups(query_builder, fusion, turns, gold_fn, *, catalog, cross_encoder_k,
        n_negatives=15, sampling="rank_strat", same_artist="soft_downweight",   # soft_downweight | drop | keep
        denoise_near_dup=True, skip_top_rank=False, k_min=4, seed) -> list[CEGroup]
    # CEGroup = (query_text, [pos_doc, neg_doc_1..n], group_weight). Positive at index 0.
    # Keeps only turns where gold ∈ R7 top cross_encoder_k (gold-in-pool); drops the rest.

def finetune_cross_encoder(groups_train, groups_val, *, base_model, lora_cfg,
        max_length=2048, max_doc_tokens, dtype="bf16", train_cfg, logger, out_dir) -> str
    # Grouped listwise-softmax CE; logs train/val loss + val nDCG@20; checkpoints best; returns adapter revision.

def oof_ce_scores(query_builder, fusion, train_turns, gold_fn, *, folds=3, **ft_kwargs) -> dict[(sid,turn,tid)->float]
    # k-fold session-disjoint fine-tunes; leak-free, within-pool-normalized ce_ft_score for every Train candidate.
```

Extensions to existing code:
- `mcrs/rerank/cross_encoder.py::build_cross_encoder_score_fn(..., lora_adapter=None, max_length=2048,
  max_doc_tokens=...)` — after loading the `CrossEncoder`, apply/merge the PEFT adapter onto `ce.model`. Its
  `max_length`/`max_doc_tokens`/dtype must equal the training config's (the shipped default is 512, which would
  silently truncate the dialogue against a 2048-trained adapter). `NeuralReranker`/`ChainReranker` are unchanged.
- `mcrs/retrieval/query.py::QueryBuilder` — add the enriched template (§4.1). This one builder is used by the
  train driver (`mcrs/rerank/train.py`), serve (`NeuralReranker.score`), and this fine-tune, so train==serve is
  enforced by shared code.
- `mcrs/data/catalog.py::_raw_doc` — include `tags` (and doc2query parity), so an enriched-missing track is not
  served a structurally different doc than the adapter trained on. For the fine-tune, require 100% enriched
  coverage of the pool (hard-fail or backfill on a miss).
- Notebook `nb/phase2_ce_finetune.ipynb` — orchestrates build-groups → fine-tune → eval → OOF → push adapter.

## 3. Dependencies
F1 (`id_to_metadata(enriched=True)`, `canonical_track_id`, catalog metadata for artist/title denoise), F2
(`TurnContext`/`Candidate`/`RankedList`/`Reranker`), F3 (`score_official`: nDCG@20 + hit-rank), K1 (the
`ce_ft_score` stacking feature), K2 (`build_rerank_groups`, `LGBMReranker._session_split`, retrain with the OOF
feature), R7 (`RRFFusion.fuse` candidate pool), A1 (enriched docs / doc2query), P0 (`cross_encoder_k`).
Libraries: `BAAI/bge-reranker-v2-m3`, `transformers` (`AutoModelForSequenceClassification`, num_labels=1),
`peft` (LoRA), `sentence-transformers` (serve wrapper), `huggingface_hub` (adapter by revision), Trackio
(logging). GPU on Colab (MPS too slow for full runs). Recipe-only reference (recoverable from the
old git branches `recall-union-lgbm`, `stage-b-cross-encoder`, `fresh-model`, `exp/*`): the prior
full-FT + multimodal cross-encoder trainer — reference, not ported.

## 4. Design & logic

### 4.1 Input format (baked into train==serve — changing it requires re-training)
The model scores one `(query, doc)` text pair → one relevance logit. No instruction prefix (that belongs to the
gemma/minicpm reranker family, not m3). Field markers are plain words, not special tokens; the LoRA learns them.

Query template — a single template; the `taste:` line is omitted entirely for cold users (empty history):
```
request: <latest utterance>
context: <older utterances, recency-windowed, oldest-dropped-first>
goal:    <conversation goal>
taste:   <≤5 recent "artist – title", newest-first>     # omitted when history is empty
```
- `request:` = `ctx.utterances[-1]`; `context:` = `ctx.utterances[:-1]` joined; `goal:` = `ctx.goal`; `taste:` =
  `ctx.history_tids` mapped to catalog artist/title. The content is all in the data; only the labels are added.
- The latest utterance is first and never truncated (`_apply_cap` already protects latest + goal), so the current
  intent dominates. The query is not otherwise token-capped — the dialogue is the long side and a small cap would
  gut it (see §4.2).
- Causal and leak-safe: built only from turns 1..t and `ctx.history_tids` (in-session prior golds), never the
  gold. No cross-session history exists in the data (`user_profile.history_tids` is `[]`), so the `taste:` clause
  cannot drag in stale taste from other sessions — it reflects only what the user accepted earlier in this dialogue.
- Excluded from the CE string: demographics (age/gender/country) and popularity. These have no lexical-match value
  and only dilute the dialogue; they belong in the K1/LGBM feature layer.
- The `taste:` clause is enabled, and also run as a with/without ablation (§6) — kept only if it wins dev nDCG@20.

Doc template — the enriched catalog text: `name: …, artist: …, album: …, tags: …, year: … | <doc2query×4>`
(tags + year + doc2query already present on the enriched path). Built via `build_doc` so train and serve produce
byte-identical strings.

### 4.2 Token budget & max_length
Measured on 8k train turns / 20k tracks with the real bge tokenizer: the query is the long side, not the doc.

| Side | p50 | p95 | p99 | max |
|---|---|---|---|---|
| Query (warm, enriched) | 360 | 708 | 866 | 1752 |
| Query (cold) | — | — | 117 | 460 |
| Doc (enriched, 2000-char-cap upper bound) | 849 | 911 | 924 | 957 |

Full query + full doc need ~1790 tokens at p99, so they do not both fit in 1024. `max_length=2048` fits p99 of
both with headroom; nothing truncates except the rare >p99 outlier, so the query needs no cap. bge-m3 supports up
to 8192 (extended position embeddings); fine-tuning at 2048 makes the serve length equal the train length.

2048 is a ceiling, not a fixed cost: shorter pairs are dynamically padded only to the batch maximum (not 2048) and
pad tokens are attention-masked out, so they never affect the score and average compute tracks the real ~800–1300
token lengths. Use length-grouped batching so a rare long pair doesn't inflate a whole batch's padding. Doc-side
truncation reuses `truncate_doc_tokens`/`doc_token_budget` and only bites past 2048; the query is never truncated.

The enriched-doc p99 (924) is a conservative upper bound (a 2000-char-padded doc); real enriched docs are likely
~400–500 tokens. Re-measure on the real A1 corpus, then pin `max_length` once for train+serve — if real-doc-p99 +
query-p99 ≈ 1366, drop to 1536 (cheaper). `max_doc_tokens` = `max_length − query budget − 4 specials`.

### 4.3 Positive / negative construction (the crux)
Positive: the single gold track's doc (one gold per turn). Keep a turn only if `gold ∈ R7 top cross_encoder_k`
(gold-in-pool) — K3 only reorders the retrieved slice at serve, so a missed gold is R7's problem, not K3's. This
matches K2's `build_training_data` policy and the serve distribution.

Negatives: hard negatives drawn from that turn's R7 top-`cross_encoder_k` pool (the distribution K3 reranks at
serve), minus the gold, with denoising:
- Near-duplicate titles (normalized-title match to the gold): dropped — genuine false negatives.
- Same-artist as the gold: down-weighted at sampling time (given a lower selection probability), not dropped, so
  these tracks still appear as negatives but don't dominate the group. At serve, same-artist candidates are in the
  pool and must be ranked, so dropping them would train an artist-disjoint contrast and deploy on an artist-
  inclusive one, discarding the hardest signal. Doing it in the sampler (rather than as a per-negative loss weight)
  keeps the loss a clean masked softmax (§4.4). Drop-vs-down-weight-vs-keep is a gated ablation; outright-drop only
  same-artist near-duplicates.
- `skip_top_rank` defaults off: rank-1 is the item the current system already prefers over the gold — the
  highest-leverage negative to learn to demote. Skipping it is expected to hurt; gated, not a default.
- Sampling is rank-stratified (some top-of-pool, some mid) so the training contest approximates the serve pool near
  the top, and deterministic (sort the pool, seed per `(fold, session_id, turn)` à la `LGBMReranker._cap`). Start
  N=15 and tune upward toward the serve pool size (`cross_encoder_k` ≈ 50–200); 15 is a starting point, not a fixed
  default — see the objective↔metric note in §4.4.

Ragged groups: after denoise a turn may have fewer than N negatives. Do not pad with dummy docs (that pollutes the
softmax denominator). Use a per-group masked softmax (variable size; absent positions masked to `-inf`). Drop turns
with fewer than `k_min` real negatives and report how many were dropped (silent dropping shifts the distribution
toward easy-pool turns).

Positive-quality weighting via `goal_progress_assessments` (gated lever): there is no explicit like/dislike field —
the gold is the played track, not necessarily a good rec. Coverage is 87.5% (the missing 12.5% are turn-1 golds,
null by design); the distribution shifts from train 44% MOVES / 43% DOES_NOT to test 77% / 10%. The signal is
attached only to the played gold, so it can weight positives but cannot label negatives.
- Implementation: a per-group sample weight on the loss (not a feature, not a filter). Map MOVES→1.0,
  DOES_NOT→`w_low` (≈0.3), null→1.0. Total loss = `Σ_t w_t·L_t / Σ_t w_t` (normalized so batch gradient scale is
  weight-distribution-invariant), applied after the per-group softmax-CE.
- Soft-weight, never hard-drop: dropping the 43% DOES_NOT golds discards half the data and teaches nothing about
  the 10% of test golds that are off-goal yet still must rank #1 (the eval gold is the played track regardless).
- No serve leak: it is a label-time weight, never a model input. `assessment[t]` is label-adjacent — usable only as
  a training weight, never a feature for turn t. Prior-turn (1..t−1) goal-progress may feed a causal K1 feature.
- Guards: do not tune `w_low` to reproduce the test ratio (fitting toward the test label distribution is mild
  leakage and confounded by curation). Use a fixed modest down-weight; let dev decide. Report dev nDCG@20 split by
  the gold's goal-progress label and abort the lever if off-goal-gold nDCG regresses even when the mean rises.
- The natural-language feedback ("I love them, but I want something new") is already in the full-dialogue query, so
  it needs no extra plumbing.

### 4.4 Loss — grouped listwise-softmax cross-entropy (contrastive)
Per turn, score the group `[gold, neg_1..n]` to logits, take a masked softmax over the group (subtract the
per-group max for numerical stability), and apply `CrossEntropyLoss(target=0)` with the optional goal-progress
group weight. This is localized contrastive estimation — the FlagEmbedding / bge-reranker recipe. There are no
in-batch negatives (a cross-encoder would have to re-score the full query×doc cross product); each gold is
contrasted only against its own sampled hard negatives.

Objective↔metric gap (documented, not a bug): with one gold per turn, official nDCG@20 reduces to `1/log2(rank+1)`
if the gold ranks ≤20 else 0 — graded-MRR on the gold's rank in the serve pool (~50–200). LCE over a small sampled
group optimizes pairwise dominance over those negatives, not full-pool rank, and is blind to the top-20 cutoff.
Mitigate with rank-stratified negatives and N chosen near the serve pool; report training-group nDCG, not just
loss; and rely on the dev nDCG@20 gate as the real check.

### 4.5 Model & training
- Base `AutoModelForSequenceClassification(bge-reranker-v2-m3, num_labels=1)`; the logit is the score
  (sentence-transformers `CrossEncoder` wraps the same model, so serve is compatible).
- PEFT LoRA: r=16, α=32, dropout=0.05, target attention `query`/`value`; the 1-logit head is trained too. A few M
  trainable params.
- Optimizer/LR: AdamW, lr=1e-4 (LoRA-appropriate — higher than full-FT's ~1e-5 since only the adapter+head train),
  weight_decay=0. Cosine schedule with ~5% linear warmup, stepped per OPTIMIZER step (per accumulation cycle); the
  warmup keeps the zero-init LoRA-B and untrained head from destabilizing early steps.
- Epochs & early stopping: epochs=3 is the upper bound (cross-encoder rerankers fit fast and overfit past 1–3
  epochs); early-stop on val nDCG@20 with patience=1, and the best-val-nDCG@20 epoch is the returned adapter.
- GPU strategy (tuned for a 16GB Colab T4/G4):
  - Mixed precision auto via `_resolve_dtype` — bf16 where the GPU supports it, else **fp16 + GradScaler** (a T4/G4
    is Turing and has **no bf16**). The masked softmax subtracts the per-group max, so it is fp16-stable. Serve
    dtype is resolved the same way, so train==serve on the same hardware.
  - **Gradient checkpointing** (`use_cache=False`, `enable_input_require_grads`) — the key lever that lets seq=2048
    fit in 16GB.
  - **Gradient accumulation** — large EFFECTIVE batch (`batch_groups × grad_accum`, e.g. 2×16=32 groups) for a
    stable gradient without OOM; the micro-batch is sized to memory.
  - **Length-grouped batching** so similar-length groups share a micro-batch (less pad waste, faster).
  - `inference_mode` + `empty_cache` around validation.
- Batch vs negatives (distinct knobs): there are **no in-batch negatives** (a cross-encoder can't reuse other
  groups' docs without re-scoring them). Negatives-per-gold is `N` from sampling (the contrast/quality knob, §4.3);
  the effective batch (accumulation) is the gradient-stability knob. Raising the batch does not add negatives.
- Build docs via `build_doc` and tokenize with the same truncation helpers and the same `max_length`/
  `max_doc_tokens` values as serve (asserted, not assumed).

### 4.6 Splits & OOF
- Session-disjoint splitting (reuse `LGBMReranker._session_split`): all turns of a session stay on one side,
  removing the within-session leak (turn 3's gold is turn 5's history). Not user-disjoint: although 34% of train
  users repeat, the official test set is 74% seen-users, so session-disjoint matches the eval distribution and
  gives an unbiased val estimate (user-disjoint would bias it low). The CE takes no `user_id` input, so there is no
  user-identity leak channel.
- Near-dup dedup (mandatory at every boundary — train/val and each OOF fold): drop near-identical `(query → gold)`
  examples that would otherwise straddle a split, since a near-dup in two folds means the model trained on one has
  effectively seen the other's held-out example, leaking the "leak-free" `ce_ft_score`.
- Within Train: a session-disjoint train/val split drives the loss curves and val nDCG@20. Dev is held clean for
  the final K2-vs-K2+K3 comparison (never used for early stopping). Also report a held-out-user val slice as a
  seen-vs-new-user diagnostic (not for selection).
- OOF stacking: k-fold session-disjoint over Train (k=3 for the first run; raise to 5 later if it helps). Each fold
  is scored by a model fine-tuned on the other k−1 folds → leak-free `ce_ft_score` for every Train candidate, which
  K2 consumes as a K1 feature (injected like the existing `dense_cos`). One additional fine-tune on all of Train
  produces the serve adapter. Cost = k+1 fine-tunes (k=3 → four).
- Invariant: the all-train (serve) adapter is used only at Dev/serve inference and never produces a train-row
  `ce_ft_score` — every train-row value comes from that row's held-out fold model (asserted by a test).
- Feature calibration: each fold's adapter emits logits on its own scale, so a raw `ce_ft_score` column would be
  fold-dependent and LightGBM (one column) cannot correct it. Rank/min-max-normalize the score within each turn's
  candidate pool before stacking — a within-group transform invariant to fold-wise logit scale.

### 4.7 Logging, checkpointing, serve
- Logging (Trackio): per optimizer step, the raw train loss plus an EMA-smoothed train loss (the per-step loss is
  noisy with small micro-batches + hard negatives; the EMA is display-only and never feeds optimization) and the
  current LR. Per epoch: val loss + val nDCG@20 via `score_official` (the real ranking metric). All decisions
  (early stopping, checkpoint selection) use val nDCG@20 — never the noisy train loss. Console/JSON fallback if
  Trackio is unavailable.
- Checkpointing: the best-val-nDCG@20 LoRA adapter is saved to disk and pushed to HF Hub by revision; that revision
  flows into `NeuralReranker(model_revision=...)` and `build_cross_encoder_score_fn(lora_adapter=...)`, and D1
  records the hash as the train==serve pin.
- Serve: `build_cross_encoder_score_fn(lora_adapter=<revision>)` applies the adapter onto `ce.model`;
  `ChainReranker(K2, K3_ft)` runs the final-stage rerank. The OOF `ce_ft_score` additionally feeds K1 → K2.

## 5. Reuse
Reuse `build_rerank_groups` (pool construction, train==serve), the `cross_encoder.py` truncation helpers,
`LGBMReranker._session_split`, `score_official`, and `QueryBuilder` (extend, don't fork). Port behind the gate.
The prior multimodal cross-encoder trainer and notebooks `82_*`/`90_*` (recoverable from the old git
branches) are recipe references only (full-FT + multimodal) — this module is text-only LoRA; no carried-over numbers.

## 6. Eval & acceptance gate
No zero-shot CE baseline is required (per decision). Ships only if dev nDCG@20 improves: (a) K2 vs K2 + K3(fine-
tuned) final-stage, and (b) K2 vs K2 retrained with the OOF `ce_ft_score`. Report per-segment (cold/warm),
per-goal-progress-label, `mean_hit_rank`, and cost/turn within the K3 per-turn budget. No lift → does not ship.

## 7. Tests
- Negatives: near-dup titles dropped; same-artist down-weighted (not dropped); rank-stratified and deterministic
  given the seed; gold-in-pool skip drops gold-absent turns.
- Ragged group: 3 real negatives → a length-4 masked softmax (target 0), not a dummy-padded length-16 one; turns
  with `< k_min` negatives are dropped and counted.
- Train==serve: `build_doc` yields byte-identical strings for train positives, train negatives, and serve;
  enriched-missing hard-fails (no silent raw fallback); train and serve `max_length`/`max_doc_tokens`/dtype values
  are equal (assert values, not just shared helpers).
- Query: cold turns omit the `taste:` clause (no dangling marker); empty `context:`/`goal:` lines omitted.
- OOF / leak: the k-fold `ce_ft_score` is cross-fit (an in-sample variant fails the test); cross-fold near-dup
  dedup is enforced; train-row scores trace to fold models, never the all-train adapter; the score is within-pool
  normalized before stacking.
- Serve: loading the adapter changes scores vs base; deterministic given seed + pinned revision.
- Catalog: `_raw_doc` includes `tags` (the fallback is no longer degraded vs the fine-tune's doc distribution).

## 8. Failure modes & guards
- False negatives → near-dup hard-drop + same-artist soft down-weight (the model still learns the same-artist
  neighborhood it sees at serve).
- Objective↔metric gap → rank-stratified negatives, N near the serve pool, report training-group nDCG, gate on dev.
- dtype skew / fp16 softmax instability → train bf16, pin serve dtype to train, subtract per-group max.
- In-sample stacking leak → k-fold session-disjoint OOF + cross-fold dedup + the all-train-adapter invariant, all
  tested.
- Train/serve skew → single `QueryBuilder` and `build_doc`; pinned `max_length`/`max_doc_tokens`/dtype/revision;
  raw-fallback tag fix; D1 records the hash.
- Recency lost in flat history → the `taste:` clause is recency-ordered, newest-first, capped at 5.
- Overfit to "history present" → check the cold/warm balance of the gold-in-pool training groups.
- doc2query expansion noise → already length-capped; an optional relevance filter is a future lever.
- Cost blowup → k=3 first; LoRA-only params; bf16; length-grouped batching; score only the top-`cross_encoder_k`
  slice; reuse the `max_pairs_per_turn` budget assert. 2048 is a ceiling, so true cost tracks the real lengths.

## 9. Config knobs
`rerank.neural.lora.{enabled,r,alpha,dropout,target_modules}`, `.max_length` (2048; pin after the A1 re-measure),
`.max_doc_tokens`, `.dtype` (auto → bf16 where supported else fp16; train==serve), `.adapter_revision`;
`.negatives.{n,k_min,sampling,same_artist,denoise_near_dup,skip_top_rank}`; `.goal_progress.{enabled,w_low}`;
`.oof.{folds,dedup_cross_fold,score_norm}`; `.split.{key,dedup_near_dup}`;
`.train.{epochs(3),lr(1e-4),weight_decay,warmup(0.05,cosine),batch_groups,grad_accum,early_stop_patience(1),
group_by_length,gradient_checkpointing,log_every,seed}`; `query.{markers,taste_items}`;
`logging.trackio.{enabled,project}`. Defaults/types from the F2 config loader.

## 10. Definition of Done
- [ ] `build_doc` + `build_ce_training_groups`: gold-in-pool groups, rank-stratified denoised negatives, masked
      ragged softmax, goal-progress weights; tested.
- [ ] Enriched `QueryBuilder` template (markers, cold-omitted `taste:` clause, demographics excluded); cold path
      tested; train==serve via the single builder.
- [ ] `finetune_cross_encoder` (PEFT LoRA, listwise-softmax, bf16) logs train/val loss + val nDCG@20 (Trackio) and
      checkpoints the best adapter to the Hub by revision.
- [ ] `build_cross_encoder_score_fn(lora_adapter=...)` serves the adapter with `max_length`/`max_doc_tokens`/dtype
      equal to training (tested); single `build_doc` train+serve; 100% enriched coverage; raw-fallback fix landed.
- [ ] OOF `ce_ft_score` (k=3 session-disjoint, cross-fold dedup, within-pool normalized) stacked into K1; K2
      retrained; in-sample-fails-a-test guard + all-train-adapter invariant test.
- [ ] Dev nDCG@20 lift measured: K2 vs K2+K3(ft) and K2 vs K2+OOF-feature; per-segment + per-goal-progress; within
      cost budget. Ships only on a lift.
- [ ] Code review approved.

Decisions (locked 2026-06-17): (1) same-artist negatives = soft down-weight, gated ablation; (2) goal-progress
positive weighting = enabled, gated, with guards; (3) negatives = N=15 rank-stratified, tune toward the serve pool;
(4) OOF k=3 for the first run; (5) max_length=2048 now, revisit 1536 after the A1 re-measure; (6) taste clause
enabled (+ with/without ablation); (7) logging = Trackio; (8) LoRA r=16, α=32, dropout=0.05, target q/v, head trained.

## 11. Build order & dependencies
Built after K2 (the GBDT pool + feature/stacking harness), A1 (enriched docs), P0 (`cross_encoder_k`), and R7.
Extends K3 (`52`, §4 LoRA option). Blocks nothing hard — an optional precision lever feeding L1, whose OOF
`ce_ft_score` loops back into K1/K2. GPU/Colab. On the "push to 0.55" path.
