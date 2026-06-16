# K1 — Rerank Feature Builder (causal, pure)

> Phase-2 shared contract. The **single** place rerank features are computed: fills
> `Candidate.features` for every (turn, candidate) from causal, pure functions. K2 (LightGBM) and K3
> (neural) consume these features and never recompute them. Leak-discipline lives here. See `000_INDEX.md`.

## 1. Purpose
Turn the fused `Candidate` pool (R7) + `TurnContext` into a per-candidate feature vector that lets the reranker push the gold to rank 1–3. Features are causal (≤t only), pure (deterministic functions of inputs), and leak-free (no in-sample model signal, no future turn, no gold).

## 2. Interface / contract
Lives in `mcrs/rerank/features.py`.

```python
class FeatureBuilder:
    def __init__(self, catalog: "Catalog", track_emb: "TrackEmbeddings",
                 user_emb: "UserEmbeddings", cfg: "FeatureConfig"): ...
    # fills Candidate.features in-place (and returns) for one turn's pool
    def build(self, ctx: TurnContext, candidates: list[Candidate]) -> list[Candidate]: ...
    feature_names: list[str]   # stable, ordered — the schema K2/K3 train against
```

**Wiring:** input = R7's `list[Candidate]` (with `rrf_score`/`channel_ranks`/`channel_scores`) + `TurnContext`; output = same `Candidate`s with `.features` populated. K2 trains/predicts on `features`; K3's cross-encoder score is fed *back* as one more feature (stacking) on a later pass. `feature_names` is the frozen, ordered schema (train==serve).

## 3. Dependencies
F2 (`Candidate`, `TurnContext`, config), F1 (`Catalog` metadata + `id_to_index`, `TrackEmbeddings`, `UserEmbeddings`), R7 (the pool + per-channel ranks/scores). Optional model-derived features (SASRec/cross-encoder) come from R6/K3 **only as OOF scores**. Libs: numpy. CPU, deterministic.

## 4. Design & logic — feature groups (all causal, all pure; plan §9.1)
- **Retrieval signals:** per-channel raw score + 1-indexed rank (from `Candidate.channel_scores`/`channel_ranks`), `rrf_score`, `n_channels_hit` (how many channels retrieved this candidate), best/mean rank across channels.
- **Content match (query↔track):** tag overlap, artist-mentioned flag, genre/era/mood match, title/lexical overlap vs the constructed `Query` (and `Query.structured` if R2 ran) — computed against catalog metadata.
- **Personalization (warm):** candidate↔history-pool cosine (using `TrackEmbeddings`), `user_emb·candidate` (CF), candidate-artist ∈ history-artists, candidate popularity percentile.
- **Context:** `turn_number`, history length, `is_cold` (segment), goal-category one-hot, query length.
- **Track priors:** log popularity, release recency.
- **Edge cases:** missing embedding/CF → feature = NaN/sentinel the GBDT handles natively (never impute a misleading 0); cold user → personalization features absent + `is_cold=1`.

### 4.1 Leak discipline (the load-bearing rule)
- **Pure & causal:** every feature is a function of `TurnContext` (≤t) + `Candidate` + static catalog/embeddings. No feature reads turn `>t`, the gold, or anything derived from the eval split.
- **No in-sample model leak:** any feature that is itself a *trained model's* score over the same rows the reranker trains on (e.g. SASRec rank, cross-encoder score) is computed **out-of-fold** (cross-fit: model trained on folds excluding the row being featurized) — otherwise the model memorizes in-session transitions and the feature looks great in-sample but reverses on dev. This is a hard requirement, not an option.
- **Train==serve:** `feature_names` order + every transform is identical in training-data prep and serve; locked in config.

## 5. Reuse
Port the feature logic in `salvage/mcrs/rerankers/lgbm_rerank.py` (`build_lgbm_features`) + `relevance_scorer.py` — **keep the feature recipes, rebuild the OOF/cross-fit harness** for any model-derived feature (the prior project shipped an in-sample-leaking `sasrec_rank_inv`; the recipe is fine, the in-sample fit was the bug). **Port + harden.**

## 6. Eval & acceptance gate
Not a metric module on its own; gate = **purity + no-leak tests pass** and **K2 trained on these features beats RRF-only on dev nDCG@20** (i.e. the features carry signal). Diagnostic: feature importances are sane (no single degenerate feature dominating via leak); train↔dev gap controlled when K2 uses them.

## 7. Tests
- **Purity/determinism:** `build` is a pure function — same inputs → same `features`; no global state.
- **Causality/no-leak:** featurizing turn `t` reads no turn `>t`, no gold; a leaky fixture (gold injected) does **not** change any feature.
- **OOF guard:** a model-derived feature computed in-fold vs out-of-fold differs on a fixture (proves cross-fit is actually applied); in-sample variant is rejected by a test.
- **Schema stability:** `feature_names` order stable across runs; missing-embedding → sentinel (not 0); cold-user vector has personalization features absent + `is_cold=1`.
- **Wiring:** consumes R7 `Candidate`s unchanged; output feeds a tiny K2 smoke-train.

## 8. Failure modes & guards
- **In-sample model-feature leak** → great val, dead dev. Guard: OOF/cross-fit mandatory + the OOF test.
- **Silent future/gold leak** → inflated offline nDCG. Guard: causal asserts + leaky-fixture test.
- **Impute-0 confusion** (0 looks like a real low score) → use NaN sentinels LightGBM handles.
- **Train/serve feature skew** → frozen `feature_names` + identical transforms + config lock; D1 records the feature-config hash.
- **Degenerate feature** (e.g. rank-inverse collapsing to 1/rrf_rank) → importance audit + prune dead features (plan §9.1 anti-overfit).

## 9. Config knobs
`rerank.features.groups[]` (which groups on), `rerank.features.history_pool` (mean/recency-weighted), `rerank.features.oof_folds` (cross-fit fold count for model features), `rerank.features.popularity_norm`, `segment.cold_threshold` (shared). Defaults + types from F2 loader.

## 10. Definition of Done & review checklist
- [ ] `FeatureBuilder.build` fills `Candidate.features`; `feature_names` frozen + ordered.
- [ ] Purity, causality/no-leak, OOF-guard, schema, wiring tests green.
- [ ] Any model-derived feature is OOF/cross-fit (in-sample variant fails a test).
- [ ] Missing-embedding → sentinel; cold path covered; importances sane on a K2 smoke-train.
- [ ] Code review approved; train==serve transforms locked in config.

## 11. Build order & dependencies
**Built first in Phase 2, after R7** (needs the fused `Candidate` pool + per-channel ranks/scores) and P0 (cold/warm threshold, feature hints). Depends on: F1, F2, R7, (optionally R6/K3 for OOF model features). **Blocks:** K2 (trains on `features`), K3 (stacking feeds back here). On the path `R7 → K1 → K2 → L1 → … → D1`.
