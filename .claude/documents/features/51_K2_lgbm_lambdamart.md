# K2 — LightGBM LambdaMART (primary reranker)

> Phase-2 primary reranker. Reorders the fused R7 `Candidate` pool so the gold lands at rank 1–3,
> consuming **K1's** `Candidate.features` (never recomputing them) and emitting an F2 `RankedList` for
> L1. A trained artifact (its own §6.1 train→save→eval notebook), but **CPU-only plain LightGBM — NO
> LoRA** (it is not a foundation model). Grounds in plan §9.1 + salvage `lgbm_rerank.py`. See `000_INDEX.md`.

## 1. Purpose
Learn a `lambdarank` ranking function over the fused candidate pool that pushes the gold track as high as possible (target rank 1–3), turning R7's recall into nDCG. K2 is the precision-stage multiplier on the `R7 → K1 → K2 → L1` spine: it does not retrieve and it does not compute features — it consumes the frozen per-candidate feature vector K1 fills (`Candidate.features`, ordered by `K1.feature_names`) and re-scores it. Its ceiling is R7's recall: a gold absent from the pool can never be reranked into the top-20.

## 2. Interface / contract
Lives in `mcrs/rerank/lgbm.py`. Implements the F2 `Reranker` Protocol.

```python
class LGBMReranker:                      # implements F2 Reranker
    def __init__(self, cfg: "RerankConfig", feature_names: list[str]): ...
    @classmethod
    def from_hub(cls, repo_id: str, revision: str, feature_names: list[str]) -> "LGBMReranker": ...

    # F2 Reranker surface — reorder one turn's K1-featurized pool, descending score
    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList: ...

    # offline training entry (notebook §6.1) — fit on grouped (session,turn) features
    def fit(self, train_groups: "GroupedDataset", val_groups: "GroupedDataset") -> "FitReport": ...

    feature_names: list[str]   # MUST equal K1.feature_names (train==serve), order-locked
```

**Contract (`K2 → L1`, from `000_INDEX.md` Phase-2):** `rerank(ctx, candidates) -> RankedList` reorders the **same** `Candidate` objects (provenance — `rrf_score`/`channel_ranks`/`channel_scores`/`features` — preserved; only order changes, no candidate added or dropped here). The output `RankedList(turn=ctx, items=...)` is L1's sole input. K2 reads features strictly from `Candidate.features` via the booster's stored `feature_names` (no recompute, no side fetch). K3 (later) may feed its cross-encoder score back as **one more K1 feature** (stacking) rather than replacing K2 — K2 is the stable base.

**Input invariant:** `set(c.features.keys()) ⊇ set(self.feature_names)` for every candidate (K1 produced them). Matrix row order = `candidates` order; column order = `feature_names`. Missing values arrive as NaN sentinels (K1 contract) — LightGBM handles them natively; K2 never imputes.

## 3. Dependencies
- **Modules:** F2 (`Candidate`, `RankedList`, `TurnContext`, `Reranker`, config), K1 (`Candidate.features` + `feature_names` — the schema K2 trains/predicts against), R7 (the fused pool), F3 (`ndcg_at_k`, `recall_at_k`, mean-hit-rank for the gate), P0 (cold/warm split, fusion-K). F1 supplies catalog id space (used only to assert ids ⊆ catalog at the boundary; K2 does no metadata lookup).
- **Libs:** `lightgbm` (CPU), `numpy`, `pandas` (group assembly), `huggingface_hub` (artifact load/save). No GPU, no API, no torch.
- **Config:** `rerank.lgbm.*` (§9), `rerank.eval.k`, `segment.cold_threshold`, `model_revisions.lgbm_reranker`, `seed`.
- **Data (train only):** the offline training parquet built by the K1 feature builder over Train sessions — one row per `(session, turn, candidate)` with the K1 features + a `label` (1 gold / 0 else) + `group` key. Dev is **never** trained on (held clean for final selection per §9.1).

## 4. Design & logic (plan §9.1)
**Learning-to-rank setup.**
- **Group:** exactly one group per `(session_id, turn_number)` — LightGBM `group` array = the per-turn candidate counts (must sum to the row count, in row order). Ranking is *within* a turn's pool; cross-turn order is meaningless.
- **Label:** `1` for the gold track of that turn, `0` for every other candidate. Exactly one positive per group **when the gold is in the pool** (see §4.3 for the not-in-pool case).
- **Negatives:** the other K−1 fused candidates of that turn — **hard, in-distribution negatives** (the realistic confusables R7 surfaced). No synthetic/random negatives: the whole point is to separate the gold from the look-alikes the retriever already promoted.
- **Objective:** `objective="lambdarank"`, `metric=["ndcg"]`, `eval_at=[20]` (and `[5,10,20]` logged). `label_gain` left default (binary labels); `lambdarank_truncation_level` ≈ fusion-K so pairwise λ's cover the whole pool.
- **Features:** the K1 vector, consumed verbatim by `feature_names`. K2 declares no features of its own. Categorical columns (e.g. goal-category one-hot is already numericized by K1, or passed as LightGBM `categorical_feature` if K1 emits codes) follow whatever K1 froze; K2 only records which indices are categorical in metadata so serve matches train.

**Scoring (serve).** `rerank()` stacks `candidates`' features into an `(N, F)` matrix in `feature_names` order, calls `booster.predict(X)` (averaging boosters if bagged — see §4.2), sorts descending (stable, deterministic tie-break on original RRF order then `track_id`), and returns a `RankedList` with the candidates in that order. Empty pool → empty `RankedList` (L1/responder degrade gracefully).

### 4.1 Anti-overfit discipline (the load-bearing rule, plan §9.1)
This is where K2 lives or dies — the prior project repeatedly saw internal val rise while dev fell. Hard rules:
- **Session-level CV:** folds are split by `session_id` so **a session never appears in two folds** (all of a session's turns move together). Turn-level or row-level splitting leaks in-session continuity across the fold boundary and inflates val. CV is used for hyperparameter choice and the gap diagnostic; the shipped model trains on all of Train.
- **Early stopping on a Train-internal holdout:** carve a session-disjoint slice **out of Train** as the early-stopping eval set (`callbacks=[early_stopping(stopping_rounds=N)]`, `eval_at=[20]`). **Dev is kept entirely clean** and used **only** for final model selection / the gate — never as the early-stopping or HPO signal (otherwise dev nDCG is no longer an honest estimate of Blind).
- **Regularization:** shallow trees (`num_leaves` ≈ 31–63, `max_depth` ≈ 5–7), `lambda_l1` + `lambda_l2` > 0, `feature_fraction` < 1 (column subsample), `bagging_fraction` < 1 + `bagging_freq` > 0 (row subsample), `min_child_samples` raised. Conservative `learning_rate` (≈ 0.03–0.05) with more `num_boost_round` bounded by early stopping.
- **Watch the train↔val nDCG gap:** log both per round. A large gap = overfit (tighten reg / fewer leaves / more subsample); both low = underfit (more rounds / leaves / features). The gate (§6) requires this gap be controlled, not just a high absolute number.
- **Prune dead features:** after a fit, inspect `feature_importance(gain)` and `(split)`. Drop features with ~0 gain or that are degenerate (e.g. a rank-inverse that collapses to `1/rrf_rank` and merely re-encodes the RRF order — a known prior failure). Pruning is a **K1 config change** (K1 owns the schema); K2 reports the audit, K1 removes the column, both retrain on the new frozen `feature_names`. No silent divergence.
- **No in-sample model leak:** any *model-derived* K1 feature (SASRec/CF/cross-encoder score) must already be **out-of-fold** when it reaches K2 (K1's hard requirement). K2 trusts that contract but adds a guard: if a single feature's gain dominates (> ~50%) *and* it is a model-derived score, flag it as a suspected leak in the FitReport rather than shipping.

### 4.2 Determinism & bagging
- Fully deterministic: fixed `seed`, `deterministic=True`, `force_row_wise=True`, single-thread or seeded-thread reproducibility recorded. Same data + config + seed → byte-identical booster.
- Optional multi-seed **bagging** (`n_bag` > 1): train `n_bag` boosters on the same Train with different seeds; serve averages their scores (variance reduction). Backward-compatible with `n_bag=1` (single `booster.txt`). Mirrors the salvage `_predict` averaging.

### 4.3 Gold-not-in-pool (the recall ceiling, R7 bounds K2)
When R7's fused pool for a turn does **not** contain the gold, K2 cannot recover it — no reorder of a pool that lacks the answer can score it. Handling:
- **Training:** such a turn's group has **zero positives** (all labels 0). A pure-0 group teaches LambdaMART nothing (no pairwise comparison) and can bias the model — **skip these groups from the training set** (don't fabricate a positive, don't keep an all-negative group). The skip count is logged; a high rate is a signal to fix recall (A1/R6/R7), not K2.
- **Eval:** these turns are **kept** in the dev eval set and **score 0** (nDCG@20 = 0, the gold's hit-rank is undefined → excluded from the mean-hit-rank-among-hits but counted as a miss in recall). This keeps the dev nDCG an honest end-to-end number bounded by R7's recall@K — K2's gate is measured on the same denominator the Blind leaderboard uses, so we never flatter K2 by hiding unreachable golds.
- **Consequence:** K2's reachable ceiling = R7 fused recall@(pool-K). The gate's "clearly beats RRF-only" is measured on the in-pool turns where K2 can actually act; the absolute dev nDCG@20 ≥ 0.45 is over **all** turns (misses included), so a recall shortfall shows up directly and routes back to R7/A1/R6.

## 5. Reuse
Port `salvage/mcrs/rerankers/lgbm_rerank.py` — **keep** the LightGBM load/predict/bagging mechanics (`_predict` score-averaging, text-format `booster.txt` + `metadata.json`, the local-dir/`from_hub`-style load) and the `lambdarank`/`ndcg@20` recipe. **Drop** its feature-computation half (`_compute_feature_matrix`, `_flatten_track_row`, `_load_track_meta`, `_build_pop_rank_pct`, `_load_cfbpr`, the per-candidate metadata/cfbpr/clap lookups) — that is now **K1's** job; the prior project's train/serve skew bugs (dead `pop_rank_pct=0.5`, dropped `same_album`, in-sample `sasrec_rank_inv` leak) all lived in that recompute path and are exactly what the K1/K2 split + frozen `feature_names` eliminates. `relevance_scorer.py` (`qwen_meta_cos`/`bm25_score`) likewise belongs to K1's feature recipes, not K2. **Port the model harness, rewrite the train loop against grouped K1 features.**

## 6. Eval & acceptance gate
**Gate (plan §9.1 DoD):** on **dev** (clean, held out from train + early-stop + HPO), measured by F3's official-parity metrics:
- **nDCG@20 ≥ 0.45** overall, and
- **mean hit-rank ≤ ~2.5** (mean rank of the gold among hits — i.e. when the gold is in the pool, K2 puts it near the top), and
- **clearly beats RRF-only** order (R7 baseline) on the same dev turns — not a wiggle; a clear margin reported overall + cold/warm, and
- **CV gap controlled** (train↔val nDCG gap small under session-level CV; §4.1), and
- **importances sane** (no single feature dominating via leak; dead features pruned; §4.1).

Reported in `reports/experiments.md` with: the config hash, K1 `feature_names` hash, RRF-only vs K2 nDCG@20 (overall/cold/warm), mean hit-rank, in-pool vs all-turns nDCG, skipped-group rate (gold-not-in-pool), CV fold scores + train↔val gap, top feature importances. On track to ≥ 0.55 after K3/§12 levers. If the gate fails because recall is low (high skip rate), the fix routes to R7/A1/R6 — **not** more K2 tuning.

## 7. Tests
- **Group construction:** exactly one positive per group **for in-pool turns**; the `group` counts sum to the row count and align to row order; an all-negative (gold-not-in-pool) group is **skipped** from train (test asserts it's dropped, not fabricated).
- **No label leak:** no feature column is a function of the `label`; a leaky fixture (label copied into a feature) is caught by an importance/sanity check; the model trained without the gold-derived column ranks identically on a fixture.
- **Feature consumption (no recompute):** `rerank` reads only `Candidate.features` — a candidate whose `.features` is mutated changes the score; mutating catalog/embeddings (which K2 doesn't hold) does not. `feature_names` order mismatch between booster metadata and K1 raises (no silent misalignment).
- **Reorder-only invariant:** `rerank` output is a permutation of the input candidates (same set, provenance preserved); length unchanged; empty pool → empty `RankedList`.
- **Beats-RRF:** on a fixture/dev slice, K2 nDCG@20 > RRF-only nDCG@20.
- **Determinism:** fixed seed + config → identical booster (hash) and identical `rerank` order; tie-break deterministic.
- **Gold-not-in-pool eval:** a turn with the gold absent scores nDCG@20 = 0 and is excluded from mean-hit-rank-among-hits but counted as a recall miss (matches F3 / Blind denominator).
- **CV split:** the session-level fold splitter never places a `session_id` in two folds (no session straddles a boundary).
- **Bagging:** `n_bag=1` and `n_bag>1` both load+predict; averaged scores reduce to the single-booster path when `n_bag=1`.

## 8. Failure modes & guards
- **Internal-val-up / dev-down (the canonical trap):** internal val rises while dev falls. Guard: **dev is sacred** (never in train/early-stop/HPO); select on dev only; watch the CV gap; the gate requires beating RRF on *dev*, not on val.
- **Session leak via wrong CV split:** row/turn-level folds leak in-session continuity → fake val. Guard: session-level splitter + its test.
- **Train/serve feature skew:** booster expects a `feature_names`/order that K1 no longer emits. Guard: K2 stores `feature_names` + categorical indices in `metadata.json`; `rerank` asserts they match K1's frozen schema; D1 records both hashes.
- **Recompute drift:** any temptation to compute a feature inside K2 reintroduces the salvage skew bugs. Guard: K2 has **no** catalog/embedding/cfbpr handles; the "feature consumption" test proves it reads only `Candidate.features`.
- **Gold-not-in-pool mishandling:** fabricating a positive or keeping all-negative groups biases LambdaMART. Guard: skip-from-train + the eval test; log skip rate.
- **Degenerate / leaky dominant feature:** a single feature carries the model (e.g. a model-derived score with in-sample leak, or a rank-inverse re-encoding RRF). Guard: importance audit in FitReport; flag > ~50% single-feature gain on a model-derived feature; prune via K1.
- **Overfit / underfit:** large train↔val gap (overfit) or both low (underfit). Guard: regularization knobs (§4.1) + the logged gap; gate rejects an uncontrolled gap.
- **Nondeterminism:** unseeded threading → unreproducible booster. Guard: `deterministic=True` + `force_row_wise=True` + fixed seed; determinism test.

## 9. Config knobs
`rerank.lgbm.objective` (default `lambdarank`), `rerank.lgbm.metric` (`ndcg`), `rerank.lgbm.eval_at` (default `[20]`), `rerank.lgbm.num_leaves`, `max_depth`, `learning_rate`, `num_boost_round`, `min_child_samples`, `lambda_l1`, `lambda_l2`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambdarank_truncation_level`; `rerank.lgbm.early_stopping_rounds`, `rerank.lgbm.cv_folds` (session-level), `rerank.lgbm.n_bag` (default 1); `rerank.lgbm.skip_gold_not_in_pool` (default true, train only); `rerank.eval.k` (=20), `segment.cold_threshold` (shared, for cold/warm reporting), `model_revisions.lgbm_reranker` (Hub repo + pinned revision), `seed`. Defaults + types from the F2 loader; values lock into `config/<exp_id>.yaml` (train==serve).

### 9.1 Training notebook (§6.1 — train→checkpoint→save→eval, CPU, NO LoRA)
A self-contained `nb/phase2_lgbm.ipynb` (its own notebook because K2 is a trained artifact), following the §6.1 skeleton **minus the foundation-model/LoRA parts** (plain LightGBM on CPU):
1. **Setup** — clone repo@`fresh-start`, `pip install -r requirements.txt`, auth to HF Hub, set seeds, read `config/<exp_id>.yaml`.
2. **Data** — load the K1 training parquet (one row per `(session,turn,candidate)` with K1 `feature_names` + `label` + `group`); build the LightGBM `group` array; **skip all-negative (gold-not-in-pool) groups**; carve a **session-disjoint** Train-internal early-stop holdout (Dev untouched).
3. **Train** — `lgb.train` with `lambdarank`/`ndcg@20`, the §4.1 regularization, `early_stopping` on the internal holdout, `eval_at=[20]`; optional `n_bag` seeds. (CPU "checkpoint" = periodic `save_model` of the best-iteration booster; the run is cheap/short enough to restart, but the booster is persisted before eval so a disconnect doesn't lose it.)
4. **Save** — push `booster.txt` (or `booster_0..N.txt` when bagged) + `metadata.json` (`feature_names`, categorical indices, `best_iteration`, `best_val_ndcg20`, `n_bag`, config hash, K1 `feature_names` hash, seed) to a versioned Hub repo `recsys2026-phase2-lgbm-<exp_id>`; record the revision into `model_revisions.lgbm_reranker`.
5. **Eval** — load the saved booster via `from_hub`, score the **clean dev** pool through `rerank`, compute F3 nDCG@20 / mean-hit-rank / recall, RRF-only vs K2 (overall + cold/warm), in-pool vs all-turns, skip rate, CV gap, importances; write the gate row to `reports/experiments.md`.

`nb/experiments.ipynb` hosts the HPO/feature-prune sweeps (session-level CV); promising settings promote into `config/<exp_id>.yaml` — never the canonical artifact.

## 10. Definition of Done & review checklist
- [ ] `LGBMReranker` implements F2 `Reranker`; `rerank` returns a `RankedList` that is a provenance-preserving permutation of the input pool.
- [ ] Trains `lambdarank`/`ndcg@20`, one group per `(session,turn)`, label 1=gold/0=else, negatives = the other fused candidates.
- [ ] Reads only `Candidate.features` by stored `feature_names` (no recompute); train==serve schema asserted; both hashes recorded by D1.
- [ ] Anti-overfit applied: session-level CV, Train-internal early stop (Dev clean), shallow trees + L1/L2 + subsample, train↔val gap logged + controlled, dead features pruned via K1.
- [ ] Gold-not-in-pool handled: all-negative groups skipped in train; scored 0 in dev eval (recall ceiling honored).
- [ ] Gate met on **clean dev**: nDCG@20 ≥ 0.45, mean hit-rank ≤ ~2.5, clearly beats RRF-only (overall + cold/warm), CV gap controlled, importances sane — logged to `reports/experiments.md`.
- [ ] All §7 tests green; deterministic given seed; bagging path covered.
- [ ] Artifact pushed to HF Hub with a pinned revision; §6.1 notebook reproduces it end-to-end.
- [ ] Code review approved; no catalog/embedding handles on K2; config locked in `config/<exp_id>.yaml`.

## 11. Build order & dependencies
**Built second in Phase 2, after K1** (K1 must have frozen `feature_names` + produce the per-candidate feature vectors / training parquet) and after R7 (the fused pool) and F3 (the gate metrics). Depends on: F1 (id space), F2 (`Candidate`/`RankedList`/`Reranker`/config), F3 (nDCG/recall/hit-rank), P0 (cold/warm, fusion-K), R7 (pool), K1 (features). **Blocks:** L1 (consumes K2's `RankedList`) and the first reranked submission; K3 stacks **on top of** K2 (feeds its score back as a K1 feature) rather than replacing it. On the critical path `R7 → K1 → K2 → L1 → S1 → D1`.
