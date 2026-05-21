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

## 6.5. Split + batch composition discipline (amendment 2026-05-21)

This subsection patches gaps surfaced by the Sub-1 leak incident (commit
`ed18ee2`: in-training val nDCG 0.2425 vs HF dev nDCG 0.1120 — a 0.13 gap
driven by 95% session overlap between train and val under row-level shuffle).
The reactive fix added session-disjoint splitting in `TripleJsonlDataset`.
This amendment formalizes the discipline:

**Split contract (user-stated):** train / val / test partitions are
**user-disjoint**. Every session belonging to a given `user_id` lives in
exactly one partition. The HF dataset's `train`/`test` splits are assumed
user-disjoint (asserted defensively in nb 70 cell 7); the internal
train→train/val split inside mined triples is enforced by
`TripleJsonlDataset` keyed on `user_id`.

This amendment addresses three weaknesses the reactive session-disjoint fix
did not cover (problems table below) AND patches the ML-hygiene issues A–H
surfaced by end-to-end review (see §6.5.10).

### 6.5.1 Problems being closed

| Problem | Today | Risk |
|---|---|---|
| **Val split key is per-session, not per-user.** A single user has ~5 sessions on average; sessions from the same user share that user's taste vocabulary. Session-disjoint val still leaks user-level memorization signal into the val metric. | `TripleJsonlDataset.session_disjoint=True` (post-`ed18ee2`) | Val nDCG remains an over-estimate of generalization. Sub 2 results are read off a still-inflated number. |
| **Batches are composed by `DataLoader(shuffle=True)`** with no grouping discipline. In-batch InfoNCE treats every other row's positive as a negative; if two batch rows are from the same user, their "false negatives" are real taste collisions — pushing apart tracks that should be near. | `loader = DataLoader(train_ds, shuffle=True, ...)` (`scripts/train_bi_encoder.py:381-387`) | Quietly weakens contrastive signal proportional to (probability two batch rows share a user). At bs=32 and average ~5 rows/user, that probability is non-trivial. |
| **Effective batch is bound by per-device VRAM.** Recent commit log: `bs=2/accum=16` → `bs=16/accum=2` → `bs=32/accum=1` (OOM at 93/95 GB) → reverted to `bs=16/accum=2`. Effective batch = 32; in-batch denominator ≈ 256 candidates/query. Modern dense-retriever recipes (BGE-M3 paper, GTE, E5) use effective bs ≥ 256 → denom ≥ 4096. | Native PyTorch in-batch InfoNCE in `_info_nce_loss_in_batch` | Contrastive signal is weaker than what published BGE-M3 fine-tunes train against. |

### 6.5.2 Δ1 — Propagate `user_id` through the data pipeline

The HF conversation dataset row carries `user_id` at the session level
(verified: row keys are `['session_id', 'user_id', 'session_date',
'user_profile', 'conversation_goal', 'conversations',
'goal_progress_assessments']`). Today `_iter_conversation_turns` reads
`session_id` and `user_profile` but drops `user_id`.

**Builder change** (`scripts/build_bi_encoder_training_data.py`):
- `_iter_conversation_turns` reads `session.get("user_id")` and emits it on
  each row alongside `session_id`.
- `build_triples_for_row` emits `user_id` in the JSONL triple, next to
  `session_id`.
- JSONL schema gains one field: `user_id`. Back-compat: absent field is
  treated as `None`.

### 6.5.3 Δ2 — User-disjoint train/val split

**`TripleJsonlDataset` change** (`scripts/train_bi_encoder.py:56`):
- Replace boolean `session_disjoint` with an explicit `split_key` arg
  taking values `"user_id"` | `"session_id"` | `"row"`. Default
  `"user_id"`.
- Resolution at load time:
  - If `split_key="user_id"` AND every row carries a non-empty `user_id`
    → split by user_id.
  - If `split_key="user_id"` AND any row is missing user_id → **fail
    loudly** with a message telling the caller to either re-mine triples
    with the new builder or pass `--split-key session_id` explicitly.
    No silent fallback (per `feedback_no_data_leakage.md`).
  - `split_key="session_id"` / `"row"` paths preserve current behavior for
    legacy triples.
- Split mechanics are otherwise unchanged: deterministic shuffle on
  `seed`, val_fraction of the keys go to val, all rows under those keys
  go to val, remaining rows go to train.

**CLI**: add `--split-key {user_id,session_id,row}` to
`scripts/train_bi_encoder.py`. Default `user_id`.

### 6.5.4 Δ3 — User-disjoint batch sampler with data-driven without-replacement neg sampling

**Pos/neg ratio K_data** — data-distribution-driven, fixed-for-the-run:
- At dataset construction, compute the empirical distribution of
  `len(row["neg"])` across all loaded triples.
- `K_data = min(args.n_negatives, floor(P05(neg_count)))` where `P05` is
  the 5th percentile of neg-counts. This is the largest ratio that ≥95% of
  rows can support without replacement.
- Rows with `len(row["neg"]) < K_data` are **DROPPED** (logged count).
  This eliminates the current `random.choice` upsampling-with-replacement
  path (issue E) which duplicated negs and inflated their gradient.
- Within each row, `__getitem__` picks the K_data negs via
  `random.sample(row["neg"], K_data)` — sampling **without replacement**
  inside the row.

Why "fixed by data distribution": every row contributes exactly 1 pos +
K_data distinct negs to the in-batch denominator. The denominator size is
predictable, the contrastive signal is uniform across rows, and no negative
is reused inside a row.

**New class** `UserDisjointBatchSampler(torch.utils.data.Sampler)`:
- Constructed from `row_user_ids: list[str]` + `batch_size: int` + `seed: int`.
- Yields **batches of indices** (it is a `BatchSampler`, not a `Sampler`):
  each yielded batch is a list of `batch_size` row indices whose
  `user_id` values are pairwise distinct.
- Within an epoch, every row index is yielded **at most once** (sampling
  without replacement). When the remaining row pool can no longer
  assemble a full batch of distinct users, the residual rows are emitted
  as a final short batch and the epoch ends.
- Deterministic via `seed` + epoch counter.
- Pattern mirrors sentence-transformers' `NoDuplicatesBatchSampler`
  but keyed on `user_id` instead of row-text identity.

Why distinct-user batches are the right common method: in-batch InfoNCE
assumes every other row's positive in the batch is a valid negative for
this row's query. That assumption breaks for two rows from the same user
(their gold tracks are both "what this user likes" → pushing them apart is
wrong, and shows up as label noise in the contrastive signal). Ensuring
pairwise-distinct users per batch is the cheapest standard remedy.

**False-positive mask (issue C)**: even with distinct users per batch, two
queries can share the same gold `track_id` (e.g., a popular track). The
in-batch loss treats that as positive for one query and negative for the
other simultaneously → contradiction. Fix: mask `scores[i, j]` to `-inf`
where `j` is the positive-column of another query whose `pos_tid` equals
this query's `pos_tid`. Cheap; requires `pos_tid` available on the batch
(already on the row via builder's `pos_tid` field).

**Loss / effective batch**: KEEP current `_info_nce_loss_in_batch` at the
existing effective batch (bs=16, accum=2 → effective 32). CachedMNRL /
GradCache for effective bs=256 is **out of scope for this amendment**;
defer to a follow-up. The signal lift from this amendment is hygiene
(user-disjoint split + within-batch user discipline + no false negatives
+ no upsampling), not denominator size.

### 6.5.5 Hyperparameter table delta

Only rows that change relative to §6 "Hyperparameters". All other rows
stay as specified.

| Param | §6 value | §6.5 new value | Rationale |
|---|---|---|---|
| split_key | implicit `session_id` (post-`ed18ee2`) | **explicit `user_id`** | User's all sessions live in one partition. |
| neg-per-row ratio K | `--n-negatives 15` with upsampling-with-replacement fallback | **K_data = floor(P05 of neg_count)**, rows below dropped | Fixed by data distribution; without replacement. |
| batch sampler | `DataLoader(shuffle=True)` | `UserDisjointBatchSampler(batch_size, user_ids, seed)` | Removes same-user false-negative collisions per batch. |
| in-batch false-positive mask | none | mask same-`pos_tid` positions to `-inf` | Removes popular-track collision contradiction. |
| `--val-fraction` default | 0.05 | **0.10** | 5% gives ~30 val users at typical mine size — too noisy (issue D). |
| effective batch size | 32 (bs=16 × accum=2) | unchanged at 32 | CachedMNRL deferred. |

### 6.5.6 Failure modes added to §10 Risk table

| Symptom | Likely cause | Mitigation |
|---|---|---|
| Val nDCG still > 0.05 above dev nDCG after fix | User-level memorization remains (e.g., per-user popular tracks dominate); or `user_id` is too coarse and many "users" are actually role-personae shared across data | Inspect per-user row-count distribution; drop top-1% over-represented users |
| `UserDisjointBatchSampler` exhausts distinct users mid-epoch (epoch shrinks) | Long-tail user count < batch_size | Allow a final ragged batch; if shrink > 5% of epoch, reduce `batch_size` until distinct-user supply is sufficient |
| K_data drops > 5% of rows | Miner produced highly variable neg counts per row | Re-mine with `pool_size` larger; or lower `--n-negatives` to match the data |
| Missing `user_id` in some triples after re-mining | Older session row in HF dataset lacks the field | Builder writes `user_id=None`; loader fails loud at `--split-key user_id` (see §6.5.3) |
| HF train/test split shares user_ids | HF curators didn't enforce user-disjointness across splits | nb 70 cell 7 asserts disjointness and refuses to score if violated — escalate to organizers |

### 6.5.7 Testing additions to §11

| Layer | What | File |
|---|---|---|
| Unit | `test_user_id_propagated_into_triples` (builder emits the field) | `tests/test_build_bi_encoder_training_data.py` |
| Unit | `test_user_disjoint_split_no_user_in_both_folds` | `tests/test_train_bi_encoder.py` |
| Unit | `test_user_disjoint_split_fails_loud_when_user_id_missing` | `tests/test_train_bi_encoder.py` |
| Unit | `test_user_disjoint_batch_sampler_unique_users_per_batch` | `tests/test_train_bi_encoder.py` |
| Unit | `test_user_disjoint_batch_sampler_indices_used_at_most_once` (without-replacement guarantee) | `tests/test_train_bi_encoder.py` |
| Unit | `test_k_data_computed_from_neg_count_distribution_p05` | `tests/test_train_bi_encoder.py` |
| Unit | `test_rows_below_k_data_dropped_with_log` | `tests/test_train_bi_encoder.py` |
| Unit | `test_in_batch_loss_masks_duplicate_pos_tid` (issue C) | `tests/test_train_bi_encoder.py` |
| Unit | `test_collate_uses_two_tokenizers_no_mutation` (issue G) | `tests/test_train_bi_encoder.py` |
| Smoke | 100-step training run with new sampler → assert val loss strictly decreases over first 100 opt-steps | nb 70 smoke cell |

### 6.5.8 Files affected

| File | Change scope |
|---|---|
| `scripts/build_bi_encoder_training_data.py` | +2 lines (read `user_id` in `_iter_conversation_turns`; emit in `build_triples_for_row`) |
| `scripts/train_bi_encoder.py` | ~120 lines: `TripleJsonlDataset` split_key generalization + K_data drop (~30), `UserDisjointBatchSampler` class (~40), in-batch false-positive mask (~15), two-tokenizer collate (~15), CLI plumbing (~10), default flag bumps (~5), issue-A loud-fail detection. |
| `tests/test_build_bi_encoder_training_data.py` | +1 test |
| `tests/test_train_bi_encoder.py` | +7 tests |
| `colab/70_train_bi_encoder.ipynb` | Cell 5: fix stale denom comment (issue H), pass `--split-key user_id`, pass `--val-fraction 0.10`. Cell 7: assert HF train↔test session+user disjointness. |

### 6.5.9 Out of scope (this amendment)

- CachedMNRL / GradCache for effective bs ≥ 256 — deferred to a separate
  amendment once the hygiene fixes here are validated by a clean val/dev gap.
- Cross-encoder (Stage B) and LightGBM (Stage C) batch composition — they
  use different loss surfaces; revisit if Stage A diff lifts val/dev gap
  but Stage B regresses.
- Curriculum HN mining (RocketQA / ANCE style iterative re-mining).
  Static mining stays per §6.
- Replacing the bi-encoder base model. BGE-M3 stays.

### 6.5.10 Concrete ML-hygiene issues being patched (A–H)

End-to-end review surfaced eight issues; this amendment patches all of them.

| # | Severity | Issue | Fix in this amendment |
|---|---|---|---|
| A | HIGH | `has_session` check at `train_bi_encoder.py:98` inspects only `all_rows[0]` — empty session_id on row 0 silently falls back to row-shuffle. | Replace with `all(r.get("user_id") for r in all_rows)`; raise loudly if mixed (no silent fallback). |
| B | HIGH | `DataLoader(shuffle=True)` allows same-user rows in one batch → false negatives in in-batch InfoNCE. | `UserDisjointBatchSampler` (Δ3). |
| C | MED | Same-track collision: two queries sharing a gold `track_id` → contradictory gradient. | Mask `scores[i, j] = -inf` where `pos_tid[i] == pos_tid[j]` and `j` is another query's positive column. |
| D | MED | `--val-fraction 0.05` → ~30 val users → noisy metric. | Default bumped to **0.10**. |
| E | LOW | Short neg-lists upsampled with `random.choice` → duplicate negs in same row, inflated gradient. | K_data fixed by P05 of neg distribution; rows below K_data are **dropped**, not upsampled. |
| F | LOW | Resume-from intentionally resets optimizer+scheduler. | Acceptable as-is; not changed. |
| G | LOW | `_collate_batch` mutates `tokenizer.truncation_side` per call. Fragile under `persistent_workers=True`. | Two separate tokenizer instances at training start (one left-truncate for queries, one right-truncate for docs); no mutation. |
| H | INFO | nb 70 cell 5 comment says "bs=8" but code uses bs=16; denom math stale. | Updated comment. |

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
