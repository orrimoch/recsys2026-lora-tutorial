# nDCG@20 Improvement — task checklist

Companion to `NDCG_Improvement_Plan.md` (read it for rationale + `file:line` anchors). Each task is
one measurable change with a ship gate. Ordered by the K3b → ColBERT → K2 cascade so shared signals
compound. Tick `[x]` when the gate passes; if a gate fails, revert and note it.

Conventions
- Proxy = dev final-turn nDCG@20 (`nb/phase3_dev_experiments.ipynb`), the Blind-A regime.
- Ship gate = measured lift over the current best on the proxy, sliced cold/warm, train==serve parity held.
- Each task is independently runnable in the named notebook; the order only maximises compounding.

---

## Phase 0 — baseline floor (run in parallel, low priority)
- [ ] T0.1 Retrain each model at larger data/epochs as the cheap floor (K3b `TRAIN_SUBSET`↑, ColBERT `K_NEGS`↑ / more sessions, K2 more train sessions). Gate: no regression vs current; record the number as the floor to beat. (Plan: "obvious lever" note.)

## Phase 1 — K3b cross-encoder (do first: it's the teacher + K2 feature source)
- [x] T1.1 BIGGEST: `CROSS_ENCODER_K` 100 → 200 (train==serve) — applied in `nb/phase2_ce_finetune.ipynb` config. COLAB GATE PENDING: proxy nDCG@20 up vs K=100; no warm regression. Est +0.01–0.03.
- [x] T1.2 `N_NEG` 15 → 30 — applied in the config (pairs with T1.1). COLAB GATE PENDING: lift over T1.1 alone.
- [x] T1.3 False-negative denoise — MECHANISM BUILT + TESTED (`ce_data.false_negative_drop_set`, `sample_negatives(neg_scores, fp_quantile)`, `build_ce_training_groups(teacher_score_fn, fp_quantile)`); wired behind `FALSE_NEG_DROP_QUANTILE` (default 0.0 = off, frozen-CE teacher). TO ENABLE on Colab: set quantile ~0.05–0.10. Gate: lift, and `DOES_NOT_MOVE_TOWARD_GOAL` slice not regressed (spec §4.3).
- [ ] T1.4 (A100) Full `TRAIN_SUBSET`=15,199 + `CROSS_ENCODER_K`=500. Gate: lift over T1.1–T1.3; cost acceptable. Est +0.015–0.04.
- [ ] T1.5 Freeze the winning K3b adapter as the teacher for Phase 2 and as the K2 frozen-CE source (record adapter id + depth).

## Phase 2 — ColBERT (uses the Phase 1 teacher)
- [x] T2.1 BIGGEST: distillation D3 — DATA LAYER BUILT + TESTED (`build_triples_from_pools(teacher_score_fn)` attaches `pos_score`/`neg_scores`; `triples_to_distillation_rows` -> `{query, documents, scores}`); loss switch wired behind `USE_DISTILLATION` (`losses.Distillation` + non-NO_DUPLICATES sampler). COLAB: VERIFY the pinned-pylate KD dataset/collator API, then enable + gate (G3 recall AND proxy nDCG@20 up). Teacher = K3b adapter.
- [x] T2.2 `D_LEN` 300 → 512 — applied in `nb/phase2_colbert_finetune.ipynb`; busts the PLAID index via the `d{D_LEN}` doc-sig. COLAB GATE PENDING: G3 recall up; index size acceptable.
- [ ] T2.3 Faceted / multi-aspect query A3 (`mcrs/retrieval/query.py` + notebook wiring). Gate: G3 recall up vs focused-only; keep query focused, just richer aspects. NOT STARTED.
- [x] T2.4 False-negative filter on hard negs via teacher — BUILT + TESTED (`build_triples_from_pools(fp_quantile)` reuses `ce_data.false_negative_drop_set`); wired behind `FALSE_NEG_DROP_QUANTILE` (default 0.0). COLAB: enable + gate (lift / no recall loss).
- [x] T2.5 `K_NEGS` 15 → 25 — applied in `nb/phase2_colbert_finetune.ipynb`. COLAB GATE PENDING: small G3 lift.

## Phase 3 — K2 LGBM (re-rank the richer pool; recompute CE feature at new depth)
- [ ] T3.1 BIGGEST-a (low risk): consensus/interaction features (`top5_bm25_and_dense`, `consensus_3plus`, `dense_cf_agree`) + per-turn calibration of `dense_cos` in `mcrs/rerank/features.py:build()`. Gate: proxy nDCG@20 up; check feature importances are non-dead.
- [ ] T3.2 BIGGEST-b: `lambdarank_truncation_level=20` in `mcrs/rerank/lgbm.py:fit()`. Gate: lift or neutral (keep if ≥0).
- [ ] T3.3 Recompute the frozen `ce_score` K2 feature at the T1.5 depth (`nb/phase2_rerank.ipynb` CE-stack block). Gate: lift from a better-aligned feature.
- [ ] T3.4 Popularity-percentile + recency features (`mcrs/rerank/features.py`). Gate: lift, and catalog-diversity not worse.
- [ ] T3.5 Per-segment cold/warm boosters (`mcrs/rerank/lgbm.py:fit/rerank`). Gate: cold-slice lift, warm not regressed.
- [ ] T3.6 GATED: goal-progress graded labels (`mcrs/rerank/train.py:build_rerank_groups`, `mcrs/rerank/lgbm.py:_xy/fit`). Pre-gate: confirm `goal_progress_assessments` present on serve/blind rows; mind the train/test shift (44/43 vs 77/10). Ship gate: overall lift AND `DOES_NOT_MOVE_TOWARD_GOAL` slice not regressed — else revert.

## Phase 4 — integrate + submit
- [ ] T4.1 Wire winners into `nb/phase3_blindA_submission.ipynb` (new K2 path / `K2_MODEL_PATH`, ColBERT checkpoint, K3b adapter). Run `BLIND=False` dev-confirm; read nDCG@20.
- [ ] T4.2 If dev-confirm beats the current Blind-A best, retrain winners on train+dev, then `BLIND=True` submit. Gate: blind layout checks pass (80 rows, keys==targets, non-empty).

---

Avoid (from the plan, do not spend slots on): raising K3b `MAX_LEN` past 2048; top-N negative capping in K2; re-encoding RRF order as a K2 feature; full-dialogue ColBERT query; swapping ColBERT's ANN backend; running dense+ColBERT both as query channels; trying to break the ~0.66@500 recall wall.
