# nDCG-Stretch Retrieval Plan — Design Spec

**Date**: 2026-05-18
**Owner**: Or Rimoch
**Branch**: `fresh-model`
**Status**: Draft — awaiting user review before implementation plan

---

## 1. Goal

Substantially improve **nDCG@20** on the RecSys 2026 Music CRS Challenge Blind-A leaderboard
by replacing the current dense retrieval + reranking stages with a domain-tuned bi-encoder +
cross-encoder + LightGBM stack, using parallel-fusion (wRRF) architecture.

**Numerical target**: nDCG@20 ≥ 0.35 (current Blind-A baseline 0.06; literature-stretch ceiling
on this catalog scale is ~0.40 given Qwen-3B responder + cost constraint of ~15–20 GPU-hours).

### Non-goals

- **LLM-score axis**: deferred to Phase 2 (responder upgrade). The v5-kto-3B Qwen responder
  stays unchanged throughout this plan. LLM-score is expected to lift *passively* as top-N
  candidate quality improves; not actively optimized here.
- **CatDiv axis**: saturated at ~0.03 across all teams per `project_cat_div_saturated.md`.
  Out of scope.
- **LexDiv axis**: already competitive (0.77–0.78). Out of scope.
- **SID-style generative retrieval**: closed per `project_sid_closed_2026_05_18.md`. Not revisited.

### Phase 2 (later — separate spec)

Per `project_llm_responder_optimization_strategies.md`, Tier-1 responder wins
(top_n_for_prompt tuning, response length, state tracker ablation) → Tier-2 7B + best-of-N.
Expected composite lift: +0.05 to +0.13. Out of scope for this spec.

---

## 2. Target framing — literature-anchored

Why 0.35+ is ambitious-but-defensible:

| Stage | Predicted incremental nDCG@20 | Source |
|---|---|---|
| Zero-shot BGE-M3 + BM25 fusion | 0.10–0.14 | BGE-M3 paper [Chen 2024]; matches MS-MARCO BM25→BGE-M3 published lift of ~2× |
| Domain fine-tuned BGE-M3 | +0.05–0.10 | NV-Retriever §5 [Moreira 2024] reports +4.5% relative; Sionic AI 2024 fine-tune blog reports similar |
| Fine-tuned cross-encoder rerank | +0.05–0.15 | BGE-reranker paper [Chen & Xu 2023]; Tonellotto NIR §4 (2022) — cross-encoders consistently add this margin |
| LightGBM LambdaRank with ~50 features | +0.03–0.10 | RecSys 2024 Challenge LGBM-Ranker baseline (cited above) |
| **Stacked total** | **0.23–0.49** | All stages near their literature ceiling |

The **0.35 mid-range** is what we target as the stretch upper-confidence bound; **0.25 is the
modest gate**; below 0.20 we abort and reassess. The top-team 0.51 nDCG@20 (per leaderboard
inspection on 2026-05-18) likely uses a 7B-scale dense encoder + heavier reranking — out of
this plan's cost envelope.

---

## 3. Architecture

Three-stage parallel-fusion retrieval, mirroring the existing
`wrrf_bm25_dense_lyrics_bge_m3_v1` factory pattern (already in
`mcrs/retrieval_modules/__init__.py:145-201`).

```
Query (chat_history + current_user_query + CMQR rewrites, per existing online format)
   │
   ├──► Stage A: Parallel retrieve top-200 each
   │       BM25 (existing 5-field corpus)
   │       dense_lyrics (existing — KEEP as complementary signal)
   │       BGE-M3 bi-encoder (NEW, fine-tuned, merged-and-pushed to Hub)
   │
   ├──► wRRF fusion (k=60) → top-200 candidate pool
   │
   ├──► Stage B: BGE-reranker-base cross-encoder (NEW, fine-tuned)
   │       Score every (query, candidate) → top-50
   │
   ├──► Stage C: LightGBM LambdaRank (NEW, ~50 features) → top-20
   │
   └──► v5-kto-3B Qwen responder (UNCHANGED — Phase 1 scope)
```

**Critical alignment with existing infrastructure**:
- The new BGE-M3 plugs into the existing `wrrf_bm25_dense_lyrics_bge_m3_v1` factory by
  changing its `model_name` constant from `BAAI/bge-m3` (zero-shot) to
  `OrRim123/recsys2026-bge-m3-music-v1-merged` (fine-tuned + merged).
- Stage B replaces ProRank in the inference pipeline via a new `retrieval_type` value
  `wrrf_bm25_dense_lyrics_bge_m3_ce_v1` or via the existing `reranker_type` config field.
- Stage C is a NEW post-rerank step; appended to `crs_baseline.py:batch_chat` after
  the existing rerank stage.

---

## 4. Existing infrastructure to reuse

| Component | Path | Reuse strategy |
|---|---|---|
| Query construction | `mcrs/crs_baseline.py:batch_chat` builds query from `chat_history` + `current_user_query` + CMQR | **REUSE AS-IS**. Fine-tune BGE-M3 with the EXACT same query string the inference pipeline produces — no new formatter. |
| BM25 retriever | `mcrs/retrieval_modules/bm25.py` | reuse unchanged |
| dense_lyrics retriever | `mcrs/retrieval_modules/dense.py` | reuse unchanged |
| wRRF aggregator | `mcrs/retrieval_modules/rrf.py:RRF_MODEL` | reuse unchanged; new sub-spec drops into its `sub_specs` list |
| LGBM feature builder | `scripts/build_lgbm_features.py` (currently 14 features) | **EXTEND** to ~50 features; don't rewrite |
| Submission validator | `scripts/precheck_prediction.py` + `scripts/validate_prediction.py` | reuse unchanged |
| Score tracker | `scripts/blind_a_score_tracker.py` | reuse — log every submission |
| TensorBoard pattern | W3's `--results-dir` flag pattern | reuse |
| Colab setup | notebooks 60–66 setup-cell pattern (Drive + HF_TOKEN + git clone + symlinks) | reuse |

---

## 5. Data sources

| Source | Rows | Purpose | Location |
|---|---|---|---|
| W2 train.parquet (raw subset) | 121,592 conversation→track pairs | Stage A + B fine-tune supervision | Drive: `recsys2026_sid_training_cache/train.parquet`, filter `source == 'raw'` |
| Held-out 20% of train sessions | ~30K turns | Stage C LGBM training (avoid dev leakage) | derived offline, session-disjoint split with `seed=42` |
| W2 val.parquet | 3,114 | Diagnostic offline eval (raw + metadata) | Drive: `recsys2026_sid_training_cache/val.parquet` |
| Dev set | 1,000 sessions × 8 turns = 8,000 | Composite gate (per `feedback_offline_eval_must_match_online.md`) | HF: `talkpl-ai/...-Dataset` (dev split) |
| Blind-A | 80 × 1 | CodaBench submission only | HF: `talkpl-ai/...-Blind-A` (test) |
| Track corpus | 47,071 | Item universe | HF: `talkpl-ai/...-Track-Metadata` |
| Pre-computed track embeddings | 47K | LGBM features (cosine similarities) | HF: `talkpl-ai/...-Track-Embeddings` |

Data schema fields we use (confirmed exposed via `huggingface_hub` API + existing
`scripts/build_lgbm_features.py:147-148`):

- `conversation_goal.category` (one of TalkPlayData 2 categories A–K)
- `conversation_goal.specificity` (LL / LH / HL / HH)
- `conversation_goal.listener_goal` (free-text)
- `goal_progress_assessments[]` (per-turn MOVES_TOWARD_GOAL / DOES_NOT_MOVE_TOWARD_GOAL)
- `user_profile_raw.{age, country_code, preferred_musical_culture}`
- `chat_history[]` (list of `{role, content}`)
- `current_user_query`, `turn_number`, `session_id`, `user_id`, `track_id`

Fields the TalkPlayData 2 paper documents but our HF dataset does NOT expose (verified by
reviewer): `target_turn_count`, `listener_expertise`. **Excluded from feature set.**

---

## 6. Stage A — BGE-M3 bi-encoder fine-tune

### Model + recipe

- **Base**: `BAAI/bge-m3` (567M, XLM-RoBERTa, 8192 context)
- **PEFT**: LoRA r=32, alpha=64 on attention modules; `modules_to_save=["pooler"]`. After
  training, **merge LoRA into the base and push to Hub** as
  `OrRim123/recsys2026-bge-m3-music-v1-merged`. This avoids extending `dense_local.py` to
  load PEFT adapters — the merged model loads via standard
  `AutoModel.from_pretrained`. Pattern mirrors `project_responder_merge_pattern.md`.
- **Library**: `FlagEmbedding` unified_finetune
- **Loss**: `m3_kd_loss` (BGE-M3 dense + sparse heads). **ColBERT head dropped** — colbert
  distillation expects sub-token alignment that 47K-track corpus doesn't exhibit cleanly.
- **Self-distillation after step 500**: enable per BGE-M3 paper §3.3.

### Hyperparameters (sources: Sionic AI 2024 blog + BGE-M3 paper §3.3)

| Param | Value |
|---|---|
| `learning_rate` | 5e-6 |
| `per_device_train_batch_size` | 2 |
| `train_group_size` | 8 (1 positive + 7 negatives) |
| `temperature` | 0.05 |
| `num_train_epochs` | 2 |
| `warmup_ratio` | 0.1 |
| `max_seq_length` | 512 query / 256 passage (capped from BGE-M3 default 8192 to fit memory) |
| precision | bf16 |

### Hard-negative mining (NV-Retriever-style, adapted for small catalog)

- **Mining model**: zero-shot BGE-M3 (encode 121K queries + 47K tracks)
- **Per query**: top-200 candidates by cosine sim
- **Filter**: TopK-PercPos at **threshold = 0.80** (NOT 0.95 — at 47K catalog scale, 0.95
  rejects almost everything per reviewer finding. 0.80 keeps ~5–10 valid negatives per query
  on average.)
- **Sample**: up to 15 negatives per query from filtered list (rank 2–200)
- **Static mining** (one-shot before training, not iterative)
- **Output cache**: `experiments/cache/sid_training/triples_bge_m3.jsonl`

### Query format

Use the EXACT query string `crs_baseline.py:batch_chat` produces at inference (current chat
history concatenation + CMQR rewrites). Do NOT use the SID format
`format_query_for_sid_input`. This guarantees train/eval feature parity per
`feedback_no_data_leakage.md` §5-6.

### Track text format

`track_name | artist_name | album_name | release_date | tag_list` (matches BM25's 5-field
corpus; deliberate single-axis change vs. existing dense_metadata-Qwen3 which uses only the
pre-computed 1024-dim embedding from 3 fields).

### Wallclock

| Step | Time |
|---|---|
| Zero-shot encode 121K queries + 47K tracks | ~1.5 hr Blackwell |
| HN mining (filter + sample) | ~0.5 hr |
| Fine-tune (2 epochs × 121K × group=8) | ~6–8 hr |
| Catalog re-embed with fine-tuned model | ~0.5 hr |
| Offline eval on val + dev | ~0.5 hr |
| **Stage A total** | **~9–11 hr Blackwell** |

---

## 7. Stage B — BGE-reranker-base cross-encoder fine-tune

### Model + recipe

- **Base**: `BAAI/bge-reranker-base` (110M, XLM-RoBERTa, BCE head)
- **Full fine-tune** (no LoRA — 110M is small enough; LoRA on cross-encoders historically
  underperforms full FT per BGE/FlagEmbedding maintainers)
- **Loss**: cross-entropy on pairwise (pos, neg) triples
- **Library**: FlagEmbedding `examples/reranker`

### Hyperparameters

| Param | Value |
|---|---|
| `learning_rate` | 2e-5 |
| `per_device_train_batch_size` | 16 |
| `num_train_epochs` | 3 |
| `warmup_ratio` | 0.1 |
| `max_seq_length` | 512 (query + doc combined) |
| precision | bf16 |

### Hard negatives (Stage A's fine-tuned BGE-M3, in-distribution)

- Use Stage A's merged Hub model to retrieve top-100 per query
- PercPos filter at 0.80
- Sample 7 negatives per query
- Output: `experiments/cache/sid_training/triples_reranker.jsonl`

### Hub push

`OrRim123/recsys2026-bge-reranker-music-v1`

### Wallclock

| Step | Time |
|---|---|
| HN re-mining via fine-tuned BGE-M3 | ~0.5 hr |
| Fine-tune (3 epochs × 121K triples × bs=16) | ~2.5 hr |
| Offline eval | ~0.3 hr |
| **Stage B total** | **~3.3 hr Blackwell** |

---

## 8. Stage C — LightGBM LambdaRank

### Model + recipe

- **Library**: `lightgbm`, CPU-only
- **Objective**: `lambdarank` (directly optimizes nDCG)
- **Eval metric**: `ndcg@20`

### Hyperparameters (sources: RecSys 2024 LGBM-Ranker + standard L2R configs)

| Param | Value |
|---|---|
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `min_data_in_leaf` | 100 |
| `n_estimators` | 1000 (early stop patience=50 on val nDCG@20) |
| `feature_fraction` | 0.8 |
| `bagging_fraction` | 0.8, `bagging_freq=5` |
| `lambda_l2` | 1.0 |
| `group` structure | one group per (session_id, turn_number) |

### Training data (held-out TRAIN slice, NOT dev — per reviewer finding)

- Split: 80/20 random over TRAIN sessions with `seed=42` (session-disjoint to prevent leakage)
- 80% slice (~12,000 train sessions × ~6 raw turns avg = ~70K turns) → Stage B top-50 → 3.5M
  (turn, candidate) rows
- 20% slice (~3,000 train sessions) → val for early stopping
- Output: `experiments/cache/lgbm/features_train.parquet` + `features_val.parquet`

### Feature set (~50 features — extend existing `scripts/build_lgbm_features.py`)

The existing script already has 14 features: `wrrf_rank, cfbpr_score, pop_log, recency_years,
tag_count, artist_in_query, goal_category, goal_specificity, user_age_group, user_country,
user_gender, label, query_id, candidate_tid`.

**Add ~36 new features** in five groups:

#### Retrieval scores (+5)
- `bge_m3_cosine`, `bm25_score`, `wrrf_fused_score`, `ce_logit`, `ce_rank`

#### Track structured (+10)
- `release_year_sin`, `release_year_cos`, `decade_one_hot` (6 buckets), `artist_pop_rank`,
  `track_name_token_len`

#### Query × track interactions (+8)
- Cosine vs precomputed `metadata-qwen3_embedding_0.6b`
- Cosine vs precomputed `lyrics-qwen3_embedding_0.6b`
- Cosine vs precomputed `audio-laion_clap`
- Tag overlap (query terms ∩ track tag_list)
- BM25 score against CMQR top-1 rewrite
- Token overlap with `current_user_query`
- Embedding distance (BGE-M3) between current query and prior chat-history turns (drift signal)
- Mean popularity rank of CMQR top-5 hits (proxy for whether query is "popular-direction")

#### Conversation features (+10)
- `last_turn_moved_toward_goal` (binary, derived from `goal_progress_assessments`)
- `consecutive_negative_progress` (count)
- `prior_recommendations_count` (parsed from `chat_history`)
- `distinct_artists_recommended_so_far`
- `query_drift_score` (cosine sim BGE-M3 embedding turn-1 vs turn-N)
- `turn_number` (1–8 ordinal)
- `chat_history_n_messages`, `chat_history_n_tokens`
- `query_has_question_mark`, `is_first_turn`

#### User-side (+3)
- `preferred_musical_culture` (one-hot top-10)
- `user_listening_history_top_decade` (from User-Metadata if available)
- `user_listening_history_top_tag` (from User-Metadata if available)

**Excluded** (paper-only, not in our HF dataset): `target_turn_count`, `listener_expertise`.

### Wallclock

| Step | Time |
|---|---|
| Feature extraction (3.5M rows) | ~1 hr CPU |
| LightGBM training (1000 estimators, early stop typically 200–400) | ~30 min CPU |
| Offline eval | ~15 min CPU |
| **Stage C total** | **~1.75 hr CPU (zero GPU)** |

---

## 9. Submission cadence + offline gates

Per `feedback_offline_eval_must_match_online.md`: offline gates use the **full composite** via
the official evaluator on the dev set, not bare nDCG@20.

| # | Stack | Diagnostic offline (val nDCG@20) | **Gate (dev composite, official eval)** | Abort rule |
|---|---|---|---|---|
| **0** | Zero-shot BGE-M3 baseline via existing `wrrf_bm25_dense_lyrics_bge_m3_v1` config (already wired) | ≥ 0.08 | ≥ 0.20 | Sanity check only — if zero-shot regresses vs current 0.21, the design's BGE-M3 choice is wrong, halt entirely |
| 1 | + Fine-tuned BGE-M3 (replaces zero-shot in same factory) | ≥ 0.15 | ≥ 0.21 (no regression vs v5-kto baseline) | **If composite < 0.18 → halt cycle, do not run Submissions 2/3** |
| 2 | + Fine-tuned BGE-reranker (replaces ProRank) | ≥ 0.25 | ≥ 0.23 | If composite regresses vs Submission 1 → revert to Submission 1 stack |
| 3 | + LightGBM LambdaRank | ≥ 0.35 | ≥ 0.25 | If composite regresses vs Submission 2 → revert and ship Submission 2 |

**Stretch target**: Submission 3 nDCG@20 ≥ 0.35 if all stages perform at their literature
upper bound. Conservative outcome: Submission 3 nDCG@20 ≥ 0.25.

CodaBench daily quota: 1 submission/day. Stage 0 is offline-only (no CodaBench), so the cycle
costs 3 quota days. We have ~5 weeks until Blind-B (2026-06-23).

---

## 10. Risk and abort criteria

| Symptom | Likely cause | Mitigation |
|---|---|---|
| Submission 1 composite < 0.18 (vs 0.21 baseline) | Fine-tune over-fits or query format mismatch | Halt cycle. Inspect train/eval feature parity. |
| Submission 2 regresses vs 1 | Cross-encoder over-fits to in-distribution hard negatives | Reduce CE epochs to 2; increase HN count to 10 |
| Submission 3 regresses vs 2 | LGBM overfitting; train/val leakage despite session-disjoint split | Audit feature pipeline for label leakage; tighten LGBM regularization |
| LLM-score axis tanks > 0.20 (mirroring SID failure) | Top-N candidate change polluted responder context | Expected risk; Phase 2 responder upgrade is the corrective workstream |
| Zero-shot BGE-M3 (Stage 0) underperforms | BGE-M3 isn't a fit for this catalog | Halt entirely; revisit encoder choice |
| Stage A HN mining yields < 3 negatives per query on average | PercPos threshold still too strict | Drop threshold to 0.70; expand pool to top-500 |

---

## 11. Testing strategy

Per project working principles + the SID-closing memory's smoke-test-gate lesson:

| Layer | What | Tool |
|---|---|---|
| Unit tests (per new module) | HN miner, query formatter, LGBM feature extractor, eval functions | pytest, TDD before integration |
| **Integration smoke (per stage)** | **5-query end-to-end run before declaring stage shipped** | A 5-min smoke cell in each notebook |
| Pre-submission smoke | 20 dev rows through full pipeline before any CodaBench upload | Cell in `73_run_blindset_retrieval_v2.ipynb` |
| Training health | TensorBoard event files persisted to Drive `results/<run_id>/runs/` | Same pattern as W3 |
| Submission precheck | `scripts/precheck_prediction.py` before every zip | Existing W6 SOP |
| Score logging | `scripts/blind_a_score_tracker.py append` after every score | Existing SOP |

---

## 12. Notebook structure (mirrors notebooks 60–66 setup pattern)

| Notebook | Purpose | GPU? | Wallclock |
|---|---|---|---|
| `70_train_bi_encoder.ipynb` | HN mining + Stage A fine-tune + merge-and-push + offline eval | yes | ~9–11 hr Blackwell |
| `71_train_cross_encoder.ipynb` | Stage B HN re-mining + fine-tune + push + eval | yes | ~3 hr |
| `72_build_lgbm_features_train.ipynb` | Feature extraction + LightGBM training | CPU | ~1.75 hr |
| `73_run_blindset_retrieval_v2.ipynb` | Blind-A inference with new stack | yes | ~60–100 min |
| `74_compare_v1_v2.ipynb` | Diagnostic: v1 vs v2 on dev set + val | yes | ~30 min |

Each notebook reuses the Drive + HF_TOKEN + `git clone -b fresh-model` setup-cell pattern.

---

## 13. Out of scope (explicit)

- SID-style generative retrieval (closed)
- CatDiv axis optimization (saturated)
- Responder model changes (Phase 2)
- Encoder-decoder T5 swap (cost-prohibitive)
- 7B-scale dense encoders (cost-prohibitive)
- Retrieval-only experiments without composite evaluation (per `feedback_no_retrieval_only.md`)
- W1 quantizer re-tuning (W1 v2 is shipped; no SID downstream)

---

## 14. Phase 2 transition (separate spec, post-Phase-1)

After Phase 1 freezes (Submission 3 evaluated, decision locked):

1. Read `project_llm_responder_optimization_strategies.md` Tier 1 menu.
2. Spin up `75_responder_tier_1_experiments.ipynb` with prompt/response-length/state-tracker
   ablations against the Phase 1 frozen retrieval stack.
3. Submit a Phase-2 Blind-A with the same retrieval stack + new responder; isolate the
   LLM-axis lift.

---

## 15. Cross-references

- `project_blind_a_first_results.md` — baseline 0.21 composite reference
- `project_blind_a_axis_interpretation.md` — composite formula
- `project_sid_strategic_priorities_2026_05_17.md` — workstream ROI ranking
- `project_sid_closed_2026_05_18.md` — SID-closed lessons (latent W4 bugs, smoke-gate
  requirement)
- `project_w1_v2_success_2026_05_17.md` — recent W1 state (for posterity; W1 not used here)
- `feedback_no_data_leakage.md` — train/eval split discipline
- `feedback_offline_eval_must_match_online.md` — composite eval gate requirement
- `feedback_experiment_ablation_discipline.md` — single-axis-per-submission rationale
- `feedback_ensemble_first_mindset.md` — additive-signal philosophy
- `project_codabench_submission.md` — daily quota + zip layout
- Reviewer findings memo: see commit message for design-review iteration on this spec

---

## Decisions locked in

- ✅ Three-stage parallel-fusion architecture (wRRF + cross-encoder + LightGBM)
- ✅ BGE-M3 (567M) + BGE-reranker-base (110M) as the model backbones
- ✅ Fine-tune via FlagEmbedding `unified_finetune` (dropping ColBERT head)
- ✅ NV-Retriever-style HN mining at PercPos-0.80 threshold (NOT 0.95)
- ✅ Three Blind-A submissions, composite-gated
- ✅ LightGBM trained on held-out TRAIN slice (not dev)
- ✅ Existing `build_lgbm_features.py` extended (14 → ~50 features)
- ✅ Stage 0 zero-shot baseline as sanity check before fine-tune investment
- ✅ Composite kill-switch at < 0.18

## Open items for the implementation plan

- Specific hard-negative-miner Python script structure (mirror `eval_sid_generator.py` style?)
- Whether to mine on raw conversation pairs only or include metadata-source pairs
- Exact LGBM hyperparameter grid (defaults proposed; tune offline)
- Whether to also fine-tune BGE-M3's sparse head (out per current scope)
