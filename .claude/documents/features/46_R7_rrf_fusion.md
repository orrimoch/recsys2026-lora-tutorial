# R7 — Weighted-RRF Fusion + Top-K Sizing

> Phase-1 integration hub. Fuses the per-channel ranked lists (R3/R4/R5/R6) into one ordered
> `Candidate` pool for reranking. Rank-based, training-free, robust. This is where the §7 recall
> gate (recall@20 ≥ 0.75, recall@200 ≥ 0.90) is met or missed. Grounded in the salvage
> `rrf.py` (math verified correct in review) + plan §7.3 / §7.3.1. See `000_INDEX.md`.

## 1. Purpose
Combine the canonical ranked id-lists from all kept retrieval channels into a single fused pool of `Candidate`s (with `rrf_score` + per-channel ranks/scores preserved for K1), sized to the top-K the reranker consumes. Maximize fused recall@K; inject no noise.

## 2. Interface / contract
Lives in `mcrs/retrieval/fusion.py`. It is itself a F2 `RetrievalChannel` (so the harness/D1 can call it like any retriever) but additionally emits `Candidate`s for the reranker.

```python
class RRFFusion:                      # implements F2 RetrievalChannel
    label = "rrf"
    def __init__(self, channels: list[RetrievalChannel], cfg: "FusionConfig"): ...
    # cfg.channel_specs[i] = {label, weight, topk_internal, query_key?, cold_weight?, warm_weight?}

    # primary: run channels + fuse → ordered Candidate pool (this feeds K1/K2)
    def fuse(self, queries: list[Query], ctxs: list[TurnContext], topk: int) -> list[list[Candidate]]: ...

    # F2 RetrievalChannel surface (ids only) — thin wrapper over fuse(...)
    def batch_text_to_item_retrieval(self, queries, topk, batch_context=None, user_ids=None) -> list[list[str]]: ...

    # cheap re-fusion for the weight sweep (no re-retrieval) — uses cached per-channel rankings
    @staticmethod
    def fuse_per_sub(per_sub: list[list[list[str]]], weights: list[float], k: int, topk: int) -> list[list[str]]: ...
```

**Fusion math (verified correct in `rrf.py`):** `score(d) = Σ_r w_r / (k + rank_r(d))`, rank 1-indexed, a doc absent from channel `r` contributes 0, dedup by canonical `track_id`, deterministic sort. `w_r ≡ 1` ⇒ vanilla RRF; `k≈60`.

**Wiring:** consumes `Query` (R1/R2) + each channel's `list[list[track_id]]`; produces `list[Candidate]` with `rrf_score`, `channel_ranks`, `channel_scores` filled (so K1 can use per-channel rank/score features). The `Candidate` pool is the sole input to K1/K2 (plan §9). `topk` here = the fusion-K from P0.

## 3. Dependencies
- **Modules:** F1 (`canonical_track_id`, catalog id set), F2 (`RetrievalChannel`, `Query`, `Candidate`, `TurnContext`, config), the channel modules R3/R4/R5/R6, F3 (`recall_at_k` for the gate), P0 (decided `fusion_k`, `topk_internal`, `weights`, `channel_keep_list`).
- **Libs:** numpy only. No model, no GPU, no API. CPU.
- **Config:** `retrieval.fusion.{k, weights, fusion_strategy, channel_quota?}`, `retrieval.channels[]` (label/weight/topk_internal/query_key), `retrieval.topk` (=fusion_K).

## 4. Design & logic
- **Per-channel queries:** mirror salvage `resolve_sub_queries` — a channel with `query_key` reads its query from `Query.per_channel[label]` (e.g. ColBERT's compact query); others use `Query.text`. Surface a VISIBLE log if a `query_key` is set but the override is missing (avoid silent fallback — the C2-guard pattern from `rrf.py`).
- **Pull depth (the §7.3.1 trap):** each channel is pulled to its `topk_internal` (P0-sized, **≥ fusion_K**, target 300–500) *before* fusion — NOT the salvage default of 60. A gold a channel only surfaces at rank 300 must be pulled at ≥300 or fusion never sees it.
- **Two-tier K:** `fuse(...)` returns the wide pool (K = fusion_K ≈ 300–500) for the GBDT (K2); K3's cross-encoder later re-scores only the GBDT top ~100–200. Invariant `topk_internal ≥ fusion_K ≥ cross_encoder_K`.
- **Weight sweep:** `fuse_per_sub` is a pure function over cached per-channel rankings, so R7 (and P0) sweep `w_r`/`k` on dev recall without re-running retrieval. Chosen weights lock into `config/<exp>.yaml` (train==serve).
- **Segment-aware weights (optional):** `cold_weight`/`warm_weight` per channel (salvage `fuse_per_sub_segmented`), keyed off `TurnContext.segment`; default both = `weight` (plain path). Enable only on a P0/ablation per-segment win.
- **Channel-quota variant (optional, off by default):** salvage `fuse_per_sub_quota` reserves top-q single-channel rescues into the window for orthogonal channels. Treat as a gated lever (plan §7.3 req #2 favors dropping noise over rescuing it); keep behind a flag, default off.

## 5. Reuse
Port `salvage/mcrs/retrieval_modules/rrf.py` — `RRF_MODEL` / `fuse_per_sub` / `fuse_per_sub_segmented` / `fuse_per_sub_quota` / `resolve_sub_queries`. **Math verified correct in review (no bug).** Changes on port: (1) raise/require `topk_internal ≥ fusion_K` (don't ship the 60 default); (2) emit F2 `Candidate`s (salvage returned bare id-lists) with ranks/scores for K1; (3) add the `⊆ catalog` assert at the fuse boundary; (4) type against F2. **Port + extend.**

## 6. Eval & acceptance gate
**Gate (plan §7 DoD):** on dev, fused **recall@20 ≥ 0.75** and **recall@200 ≥ 0.90** (F3 `recall_at_k`), reported overall + cold/warm. If recall@200 < 0.90, R7 is NOT done — the gap routes to A1/R6 (enrichment / extension channels) per P0, since no reranker recovers a gold absent from the pool. Secondary: fused recall ≥ the best single channel's recall@K at every K (fusion must not regress its strongest input), and chosen weights logged with the sweep that justified them.

## 7. Tests
- **Math:** `fuse_per_sub` matches `Σ w_r/(k+rank)` on a hand fixture; `w≡1` == vanilla RRF; ties broken deterministically.
- **Reductions:** `fuse_per_sub_segmented`/`_quota` reduce to `fuse_per_sub` when knobs off (byte-identical).
- **Id-space (plan §7.3 req #1):** every channel output asserted ⊆ canonical catalog id set before scoring; a non-canonical id fixture fails.
- **Pull depth:** a gold present only at channel-rank 250 appears in the fused pool iff `topk_internal ≥ 250` (guards the §7.3.1 trap).
- **Candidate emission:** `rrf_score`, `channel_ranks`, `channel_scores` populated correctly; dedup by canonical id.
- **Determinism:** fixed channel orderings + seed → identical fused order.
- **Recall non-regression:** fused recall@K ≥ max single-channel recall@K on a fixture.

## 8. Failure modes & guards
- **Shallow pull caps recall** → require `topk_internal ≥ fusion_K`; assert at init.
- **Noise channel injects false top ranks** (plan §7.3 req #2) → keep-list + weights from P0; drop/zero-weight channels with no unique recall; recall-non-regression test.
- **Mixed id spaces** → `⊆ catalog` assert + canonicalize at channel boundary (F1).
- **Silent per-channel-query fallback** (query_key set, override missing) → VISIBLE log + a test that the override actually arrives.
- **Weight overfit to dev** → sweep on dev recall but confirm on cold/warm splits; lock in config; don't chase < +0.5% wiggles.

## 9. Config knobs
`retrieval.fusion.k` (default 60), `retrieval.fusion.weights` (per-label; from P0 sweep), `retrieval.fusion.fusion_strategy` (`symmetric`|`segmented`|`channel_quota`, default symmetric), `retrieval.fusion.channel_quota`/`quota_labels`/`quota_window` (off by default), `retrieval.topk` (= fusion_K, from P0), and per-channel `retrieval.channels[].{label,type,weight,topk_internal,query_key,cold_weight,warm_weight}`.

## 10. Definition of Done & review checklist
- [ ] `RRFFusion` implements F2 `RetrievalChannel` + emits `Candidate`s with ranks/scores.
- [ ] Fused recall@20 ≥ 0.75 and recall@200 ≥ 0.90 on dev (or gap routed to A1/R6 with a logged plan).
- [ ] `topk_internal ≥ fusion_K` enforced; ⊆-catalog assert at fuse boundary; per-channel-query log guard.
- [ ] Weight sweep run via `fuse_per_sub` on cached rankings; chosen weights + recall logged to `reports/experiments.md`.
- [ ] All §7 tests green; segmented/quota variants proven to reduce to plain RRF when off.
- [ ] Code review approved; weights locked in `config/<exp>.yaml` (train==serve).

## 11. Build order & dependencies
**Built after the channels exist** (R3/R4 minimum; R5/R6 as they land) and after P0 sets `fusion_k`/`topk_internal`/weights/keep-list. Depends on: F1, F2, F3, P0, R1 (Query), R3/R4/R5/R6. **Blocks:** K1 (features read `Candidate` ranks/scores), K2/K3 (rerank the fused pool), and the first end-to-end submission (`…→ R7 → L1 → responder → D1`, plan §17). On the critical path.
