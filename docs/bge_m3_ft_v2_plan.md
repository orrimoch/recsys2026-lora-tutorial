<!-- STATUS (2026-06-01): Phase 0 COMPLETE + pushed on branch recall-union-lgbm.
Commits: 76ca787 (0.1 parity), 943990b + 4c70fa6 (0.2 [STATE]), da1c53d (0.3 clean negs),
3ed5d5e (0.4 best-ckpt), abf4539 (0.5 nb77 train + nb78 eval).
NEXT = Phase 1 Colab run: nb77 (smoke MAX_INPUT_ROWS=20000 first) -> nb78; gate = beat
+0.0387 additive recall@100 and 0.1637 dev nDCG@20. Then phases 2-5. -->

# Plan: BGE-M3-FT v2 — an improved learned retrieval + rerank model

## Context

`bge_m3_ft` (the fine-tuned bi-encoder, `OrRim123/recsys2026-bge-m3-music-v1-merged`) is the one validated lever against the new-artist wall: it is ADDITIVE to the union (+0.0387 recall@100 over the 0.5061 baseline) and ~70-77% of the golds it rescues are NEW-ARTIST tracks — the ~45% of golds that session channels (CF/same-artist/SASRec) structurally cannot reach. But this session's deep review found it is both under-SERVED (train/serve text-format mismatch; the empirical winner was `build_doc_text`, but the format of the black-box hub model is unknown) and under-BUILT (a fast Stage-A recipe with real bugs). Meanwhile the reranker (LGBM LambdaMART) has no real relevance feature — `ce_score`/`*_rank_inv` all degenerate to `1/wrrf_rank` (`build_lgbm_features.py:369-375`), so it fuses position priors, not relevance.

The goal: rebuild this as a proper learned recall + rerank pair that we fully control — an improved contrastive encoder (recall) plus the existing cross-encoder (rerank) — with train/serve parity guaranteed by construction and the new-artist wall as the explicit target. Built in disciplined phases so each lever is attributed and de-risked.

Key constraint discovered during planning: the `thought` field is UNSAFE to use in the query — it is richly populated at train (user intent + recommender rationale) but mostly EMPTY in Blind-A (music-turn thought 0% populated), so putting it in the retrieval query is hard leakage. The serve-safe intent signal is the existing StateTracker LM-extracted `user_state` (`mcrs/query_rewriters/state_tracker.py`), computed identically at train and serve.

Approved scope: phased + measure each; cross-encoder score as an LGBM feature (with an A/B vs CE-as-final-stage); one small non-generative semantic-ID probe.

## Approach (phased, with go/no-go gates)

Every phase is measured against the +0.0387 recall@100 additive baseline and the 0.1637 dev nDCG@20 baseline, using one parity-locked eval notebook. The new-artist subset recall (golds whose artist is absent from session history) is the primary signal. Promote to the next phase only on a clear win.

### Phase 0 — Foundations (text-only bge-m3 v1). Low risk, the must-haves.

Data prep + recipe fixes, then one clean retrain.

1. Parity by construction. Create `mcrs/retrieval_modules/track_text.py::format_catalog_track_text` (canonical track-text = the existing `id_to_metadata` format). Route `music_catalog.id_to_metadata`, `build_bi_encoder_training_data._format_history_music_turn`, and `embed_catalog.py` (+ `embed_catalog_multimodal.py`) through it. Add `--doc-format id_to_metadata` to `embed_catalog.py`. This makes the train positive text == served catalog text == served history text by construction, killing the recurring format-mismatch class.
2. Serve-safe intent. Add a `[STATE]` block (StateTracker `user_state`: mood/intent/energy/sonic_pref/era_pref) to `build_retrieval_query` (`crs_baseline.py`, `bge_m3_structured` branch) and `bge_m3_format.format_query_text`. Wire it at serve (`crs_baseline.batch_chat` already computes `extracted_states`). For train parity, create `scripts/precompute_train_user_state.py` to batch-extract `user_state` over the 121k train turns (cache like `state_tracker.py:205`), and have the builder emit the identical `[STATE]` block. These sonic/mood/era descriptors are content (not artist) — learnable intent→content signal for unseen artists.
3. Clean negatives. In `mcrs/retrieval_modules/hn_miner.py`: add `global_gold_ids` exclusion to `batch_mine_negatives` (never mine ANY query's gold as a negative — fixes cross-batch false negatives). In `build_bi_encoder_training_data.py`: default `--mining-strategy simans` (no percpos drop of the hard new-artist tail), build + pass `global_gold_ids`, and add a no-drop random-pad fallback so short-pool queries are never discarded (flags `--exclude-global-golds`, `--no-drop-short`).
4. Best checkpoint. In `scripts/train_bi_encoder.py`: save `output_dir/best` on improved full-catalog nDCG@20 and make `_merge_and_push` load `best` (not last epoch). Require `--val-full-catalog-every-n-steps 200`.
5. Recipe (text-only): base `BAAI/bge-m3`, LoRA r64/a128, temperature 0.02, in-batch (masked) + 15 hard negs, effective batch 128, lr 2e-5 cosine, 3 epochs, `--split-key user_id`. Distillation/multipositive OFF in v1.

### Phase 1 — Measure v1. Gate: beat +0.0387 additive AND raise new-artist subset recall.

### Phase 2 — Iterative re-mining (ANCE/RocketQA). Re-mine round-2 negatives with the merged v1 checkpoint (`build_bi_encoder_training_data.py` already accepts an arbitrary `--bge-m3-model`; add `--reuse-queries-from <round1.jsonl>` to keep query/state strings fixed). Retrain, measure. Gate: +Δ new-artist recall over v1.

### Phase 3 — Multimodal (the real wall-attacker: CLAP audio). New-artist tracks have no CF but DO have audio. Reuse `mcrs/training/multimodal_bi_encoder.py` (text + CLAP-512 + CF-128 + tag-emb + release-year + user_cf soft-prompt, backbone `bge-base-en-v1.5`), `scripts/precompute_multimodal_artifacts.py`, and the `--use-multimodal` train path. Ensure `[STATE]` flows through the multimodal builder and the multimodal catalog embed routes through `format_catalog_track_text`. Gate: +Δ over best text-only AND a modality ablation (zero-CLAP) showing audio carries the new-artist gain — else drop audio. Audit CLAP coverage of new-artist tracks first.

### Phase 4 — Cross-encoder rerank (the rerank half) + bi-encoder↔CE coupling.

1. Retrain the cross-encoder (`scripts/train_cross_encoder.py`, reuse) with listwise softmax LCE (`--loss softmax`, `--n-negatives 15`, `--max-grad-norm 25`, base `bge-reranker-v2-m3`) on the bug-fixed Stage-A top-100 → `recsys2026-mm-reranker-v2`.
2. Wire CE as an LGBM feature (approved): create `scripts/precompute_stage_a_teacher_scores.py` to score each session's fused top-100 with the CE and write `{session, tid, ce_score, ce_rank}`. `build_lgbm_features.py:369-375` already has the dead `ce_score`/`ce_rank` hooks — populate them, retrain the LGBM WITH the bge channel + ce_score. A/B vs CE-as-final-stage (LGBM→top-50→CE→top-20) in the eval nb.
3. Fix the teacher key (the missing piece): the CE builder's teacher parquet is keyed on bi-encoder mined-neg tids but its candidates are the top-100 → silent single-positive fallback. The new precompute must key `(pos_tid, tuple(sorted(top-100 negs)))` to match `build_cross_encoder_training_data.py`; add a loud coverage assertion mirroring `train_bi_encoder.py:302-338`.
4. RocketQAv2-style distillation back into the bi-encoder: enable `--use-distillation` AFTER fixing the MarginMSE scale bug in `train_bi_encoder.py:927-965` (student cosine margins live in [-2,2]; teacher logits ~[-10,10] are unreachable). Fix = a learnable student scale scalar, or switch to scale-free listwise-KL. Feed the re-keyed Stage-A teacher parquet. Gate hard — only ship if it moves recall; one loop max.

### Phase 5 — Semantic-ID probe (non-generative, time-boxed). `scripts/build_sid_quantizer.py` already emits per-track RQ-VAE codes (`{track_id, code_1..3}`). Add them as 3 categorical features to `build_lgbm_features.py` (and optionally as learned tokens in the CE, like `tag_embed`). One run; gate ≥ +0.002 nDCG@20 or close. NO generative SID (W1-W5 already failed here at composite 0.20).

## Critical files

Create:
- `mcrs/retrieval_modules/track_text.py` — canonical `format_catalog_track_text` (parity source of truth).
- `scripts/precompute_train_user_state.py` — batch StateTracker extraction over train turns.
- `scripts/precompute_stage_a_teacher_scores.py` — CE re-score of Stage-A top-100, correctly keyed.
- `colab/77_train_bi_encoder_v2.ipynb` — training nb (mirror `colab/70_train_bi_encoder.ipynb`; add parity-gate cell + new-artist additivity cell).
- `colab/78_e2e_stageA_stageB_rerank_ndcg.ipynb` — eval nb (mirror nb74 cells 4/10/11; the parity-locked measurement spine).
- `tests/test_track_text_parity.py`, `tests/test_state_query_parity.py`.

Modify:
- `mcrs/retrieval_modules/hn_miner.py` — global-gold exclusion + no-drop fallback.
- `scripts/build_bi_encoder_training_data.py` — simans default, global golds, `[STATE]` query, `--reuse-queries-from`.
- `mcrs/crs_baseline.py` + `mcrs/retrieval_modules/bge_m3_format.py` — `[STATE]` block, threaded train+serve.
- `scripts/train_bi_encoder.py` — best-ckpt-on-nDCG; MarginMSE scale fix.
- `scripts/build_cross_encoder_training_data.py` — teacher-coverage assertion + new teacher key.
- `scripts/embed_catalog.py` / `embed_catalog_multimodal.py` — route through `format_catalog_track_text`.
- `scripts/build_lgbm_features.py` — populate `ce_score` (Phase 4) + SID code features (Phase 5).

Reuse unchanged: `scripts/train_cross_encoder.py` (flip to `--loss softmax`), `mcrs/rerankers/multimodal_cross_encoder_rerank.py` (point at v2), `mcrs/training/multimodal_bi_encoder.py`, `mcrs/training/multimodal_cross_encoder.py`, `scripts/precompute_multimodal_artifacts.py`, nb74 cells 4/10/11 (eval spine).

Deferred (with rationale): base-model swap to Qwen3-Embedding-4B (real MTEB headroom, but the multimodal tower is 768-d CLS-pool, 121k pairs risk overfitting a 4B LoRA, and serving cost rises — revisit only if both encoders plateau).

## Verification

Eval notebook `colab/78_...` (parity-locked: one cell builds `queries/golds/struct_queries/ctx/fused` once; every measurement consumes those globals; asserts `len(struct_queries)==len(golds)` and train==serve `[STATE]`/track-text strings — hard-fail on mismatch). Cells + gates:
- C0 parity asserts pass.
- C1 Stage-A isolated: standalone recall@100 + union additivity + new-artist rescue. Gate: rescue ≥5% AND new-artist ≥50% (vs current +0.0387).
- C2 Stage-A fused→union→LGBM nDCG@20 vs 0.1637 (flat ⇒ retrain LGBM WITH the bge channel before judging — not a no-go).
- C3 Stage-B CE on the identical bug-fixed top-100: (a) LGBM, (b) LGBM+ce_score feature, (c) LGBM→top-50→CE. Gate: best ≥ baseline +0.005 nDCG@20.
- C4 ce_score feature ablation (with/without). Gate ≥ +0.003.
- C5 SID feature probe. Gate ≥ +0.002 or close.
- C6 composite read (`0.50·nDCG + 0.10·CatDiv + 0.10·LexDiv + 0.30·LLM-judge` via the nb75 Gemini-judge harness) — guards against an nDCG win that hurts the responder.
Unit tests green before any Colab run: `pytest tests/ -q` (the two new parity tests + existing suite). Final online check: ship the winning config through `run_inference_blindset.py` → Blind-A submission, compare composite vs the current 0.35.
