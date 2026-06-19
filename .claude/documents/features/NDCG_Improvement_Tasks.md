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
- [x] T3.1 BIGGEST-a (low risk): interaction/consensus features (`n_channels_top10`, `consensus_3plus`, `top5_bm25_and_dense`) + per-turn min-max calibration of every injected score (`<score>_norm`) in `mcrs/rerank/features.py:build()`. Built + unit-tested (on by default). COLAB GATE PENDING: proxy nDCG@20 up; importances non-dead.
- [x] T3.2 BIGGEST-b: `lambdarank_truncation_level=20` (+ `eval_at=[20]`) in `mcrs/rerank/lgbm.py:fit()`. Applied (on by default). COLAB GATE PENDING: lift or neutral.
- [ ] T3.3 Recompute the frozen `ce_score` K2 feature at the T1.5 depth (`nb/phase2_rerank.ipynb` CE-stack block). NOT STARTED (minor notebook knob; the frozen-CE K is separate from K3b's re-score depth).
- [x] T3.4 Popularity-percentile + recency features (`mcrs/rerank/features.py`). Built + unit-tested (on by default). COLAB GATE PENDING: lift, catalog-diversity not worse.
- [ ] T3.5 Per-segment cold/warm boosters (`mcrs/rerank/lgbm.py:fit/rerank`). NOT STARTED — structural (changes the saved-model format + serve routing in phase3_blindA); do as a focused increment.
- [ ] T3.6 GATED: goal-progress graded labels (`mcrs/rerank/train.py:build_rerank_groups`, `mcrs/rerank/lgbm.py:_xy/fit`). Pre-gate: confirm `goal_progress_assessments` present on serve/blind rows; mind the train/test shift (44/43 vs 77/10). Ship gate: overall lift AND `DOES_NOT_MOVE_TOWARD_GOAL` slice not regressed — else revert.

## Review (Phases 1–3) — adversarial, 3 reviewers, one per phase

Verdict: SHIP. No CRITICAL/HIGH leak; every default-off lever proven an EXACT no-op until enabled.
Fix-first items applied + tested in this batch:
- K3b serve depth: `mcrs/run/k3b_serve.py` default `cross_encoder_k` 100 → 200 (was a latent train/serve skew vs T1.1).
- ColBERT KD teacher query (H2): the teacher CE now scores with the FULL query it serves with (threaded `teacher_query_builder=qb_full`), not ColBERT's focused query — else KD soft labels come from an unseen query dist.
- ColBERT KD robustness: seeded pre-shuffle of KD rows (decorrelate session order); raise if <50% of triples carry teacher scores (partial-cache guard); `dropped_fp` stat surfaced.
- K2 `recency` clamped to [0,1]; added edge-case tests (all-equal `_norm`, missing-date recency, future-year clamp).

KEY CAVEAT (Phase 1 H1): the K3b LoRA adapter is NOT chained in `phase3_blindA_submission.ipynb`
today (blind serve = fusion → K2, with the cross-encoder only as a frozen K2 feature at K=50). So the
K3b K=200 lever improves the K3b gate and its role as the ColBERT distillation teacher, but will NOT
move the leaderboard until the adapter is chained at serve with matching `cross_encoder_k=200` — and
chained-K3 previously regressed dev, so that wiring is its own gated decision (see T4 / Avoid list).

Colab pre-flight (cannot verify locally): K2 `eval_at` in both constructor + fit (lightgbm absent);
the PyLate KD dataset/collator API for the pinned pylate; enable + gate the off-by-default levers
(K3b/ColBERT false-neg denoise, ColBERT distillation) on dev nDCG@20 + the off-goal slice +
`dropped_few_neg`/`dropped_fp` on-vs-off.

## Phase 4 — integrate + submit

### Wiring audit — every change → blind serve (`phase3_blindA_submission.ipynb`)
- ColBERT `D_LEN` 300 → 512 (T2.2): FIXED in blindA + dev_experiments configs; serving the 512-trained
  checkpoint at 300 would re-truncate docs vs how it learned. doc-sig now `|d{D_LEN}|ef{EXPANSION_FIRST}`
  in all three nbs (finetune/blindA/dev_experiments) — identical format, so blindA REUSES the index the
  fine-tune built (no re-encode) and a future D_LEN/expansion change busts it.
- K2 new features (T3.1 interaction/`<score>_norm`, T3.4 popularity_percentile/recency): AUTOMATIC —
  blindA builds the SAME `FeatureBuilder(cat, labels, score_fns)` (labels incl. `colbert`, dense_cos +
  ce_score), so the new columns are produced at serve; the `rerank()` feature-spec guard enforces
  train==serve and rejects an old K2. Per-turn `_norm` calibration works because `k2.rerank(t, pool)`
  builds all of a turn's candidates together.
- K2 frozen-CE feature depth: blindA `CROSS_ENCODER_K=50` matches phase2_rerank's frozen-feature K
  (NOT K3b's 200 — different knob; K3b is not chained in blind serve).
- mcrs package changes (features/lgbm/ce_data/colbert_data/k3b_serve): picked up automatically via the
  Colab `git pull` — no per-notebook wiring needed.
- K3b K=200 / distillation / false-neg denoise: TRAINING-side — they produce a better K2-feature CE and a
  better ColBERT checkpoint; blindA just loads the resulting artifacts. K3b adapter still not chained (by design).

- [x] T4.1 Wire winners into `nb/phase3_blindA_submission.ipynb` — done (audit above): ColBERT D_LEN/sig aligned, K2 features automatic via the shared FeatureBuilder, K2/ColBERT artifact paths confirmed. COLAB: run `BLIND=False` dev-confirm; read nDCG@20.
- [ ] T4.2 If dev-confirm beats the current Blind-A best, retrain winners on train+dev, then `BLIND=True` submit. Gate: blind layout checks pass (80 rows, keys==targets, non-empty).

---

Avoid (from the plan, do not spend slots on): raising K3b `MAX_LEN` past 2048; top-N negative capping in K2; re-encoding RRF order as a K2 feature; full-dialogue ColBERT query; swapping ColBERT's ANN backend; running dense+ColBERT both as query channels; trying to break the ~0.66@500 recall wall.
