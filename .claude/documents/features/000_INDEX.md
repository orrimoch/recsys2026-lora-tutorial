# Feature Modules — Index & Build Order

> Decomposition of `.claude/RecSys_Challenge_Plan.md` into self-contained, individually-testable modules.
> Each module doc follows the standard 11-section template (below) and carries **its own eval gate**.
> Target: nDCG@20 ≥ 0.55 on Blind B (see plan §4). Branch `fresh-start`.

## Standard module-doc template (every `NN_*.md` follows this)
1. **Purpose** — one-line responsibility
2. **Interface / contract** — inputs, outputs, types, signature; the up/downstream modules it wires to
3. **Dependencies** — data, modules, models, external APIs, config
4. **Design & logic** — algorithm, key decisions, edge cases, causal/no-leak constraints
5. **Reuse** — prior/pristine code pointer (plan §6.3 asset map) + keep/port/rewrite call
6. **Eval & acceptance gate** — the module's own metric + threshold + how/where measured
7. **Tests** — unit, no-leak, wiring/integration, determinism
8. **Failure modes & guards**
9. **Config knobs** — YAML keys it reads
10. **Definition of Done & review checklist**
11. **Build order & dependencies** — what must exist first

## Canonical data contracts
Defined once in **`11_F2_interfaces_contracts_config.md`** and consumed by every module:
`TurnContext`, `Query`, `Candidate`, `RankedList`, `SubmissionRow`, plus the `RetrievalChannel` / `Reranker` / `Filter` / `Responder` interfaces and the config schema. No module re-invents these.

## Phase-1 shared contracts (pin these so channels/fusion wire without drift)
- **Query flow:** `R1` builds the F2 `Query` (`.text`, optional `.per_channel` overrides); `R2` (gated) fills `Query.structured`. Every channel consumes `Query` — it never re-derives the query from raw turns.
- **Channel contract:** every channel (R3/R4/R5/R6) implements F2 `RetrievalChannel.batch_text_to_item_retrieval(queries, topk, batch_context, user_ids) -> list[list[track_id]]`, returns **canonical** ids only (F1 `canonical_track_id`), and is registered by a unique `label`. No channel fuses; it only ranks.
- **Enriched-doc accessor (A1 → R3/R4):** A1 writes a cached enriched corpus keyed by `track_id`; the doc text is read via `Catalog.id_to_metadata(track_id, enriched: bool)` (F1 surface, enrichment layered by A1). BM25/dense docs come from here — channels never build doc text themselves.
- **Embedding accessor (F1 → R4/R5/R6):** dense/personalization channels read `TrackEmbeddings.matrix(modality)` / `UserEmbeddings.vector(user_id)`; row order is `Catalog.id_to_index`.
- **Fusion hand-off (channels → R7 → rerank):** `R7` consumes the per-channel `list[list[track_id]]` + per-channel `weight`/`topk_internal`/`query_key` (config) and emits `list[Candidate]` with `rrf_score` + `channel_ranks`/`channel_scores` populated; `topk_internal ≥ fusion_K` (values from P0). This `Candidate` list is the input to K1/K2.

## Phase-2 shared contracts (rerank wiring)
- **Feature contract (K1 → K2/K3):** `K1` fills `Candidate.features: dict[str,float]` (causal, pure functions of `TurnContext` + `Candidate` + catalog/embeddings) — the **single** place features are computed. K2/K3 read `Candidate.features`; they never recompute features. Any **model-derived** feature (e.g. a SASRec/cross-encoder score) must be **out-of-fold / cross-fit** to avoid in-sample leak.
- **Reranker contract (K2/K3 → L1):** a `Reranker.rerank(ctx, candidates) -> RankedList` (F2) reorders the fused pool; K3 re-scores only the top `cross_encoder_k` and may feed its score back as a K1 feature (stacking) rather than replacing K2. Output `RankedList` is L1's input.

## Module catalogue

| # | File | Module | Phase | Eval gate (own metric) | Status |
|---|---|---|---|---|---|
| F1 | `10_F1_data_access.md` | Data access & catalog/user DB | Foundation | join-coverage / integrity asserts pass | spec: **done** |
| F2 | `11_F2_interfaces_contracts_config.md` | Interfaces, data contracts & config | Foundation | schema round-trip tests pass | spec: **done** |
| F3 | `12_F3_eval_harness.md` | Evaluation harness (official-parity + per-module metrics) | Foundation | parity == `music-crs-evaluator` on fixture | spec: **done** |
| P0 | `20_P0_eda_recall_probe.md` | EDA & recall-ceiling probe | Phase 0 | recall-ceiling table delivered; design params justified | spec: **done** |
| A1 | `30_A1_catalog_assets.md` | Catalog enrichment/doc2query + embedding (offline, cached) | Assets | coverage % + enriched-doc recall lift | spec: **done** |
| R1 | `40_R1_query_construction.md` | Query construction (causal, rule-based) | Retrieval | downstream recall non-regression | spec: **done** |
| R2 | `41_R2_query_refinement_llm.md` | LLM query refinement (gated) | Retrieval | recall@K lift vs R1-only | spec: **done** |
| R3 | `42_R3_bm25_channel.md` | BM25 channel (enriched docs) | Retrieval | recall@{50,100,200,500} | spec: **done** |
| R4 | `43_R4_dense_text_channel.md` | Dense-text channel | Retrieval | recall@{50,100,200,500} | spec: **done** |
| R5 | `44_R5_embedding_personalization_channels.md` | Content-kNN (history) + CF + same-artist | Retrieval | recall@K (warm segment) | spec: **done** |
| R6 | `45_R6_extension_channels.md` | CLAP / related-artist / propose-ground / SASRec (gated) | Retrieval | unique-recall lift per channel | spec: **done** |
| R7 | `46_R7_rrf_fusion.md` | Weighted-RRF fusion + top-K sizing | Retrieval | fused recall@20 ≥ 0.75, recall@200 ≥ 0.90 | spec: **done** |
| R8 | `47_R8_colbert_late_interaction_channel.md` | ColBERT late-interaction channel (replaces R4 query-dense; enriched docs) | Retrieval (§12) | recall@K + replace-R4 ablation | spec: **done** — PARKED hot lever |
| K1 | `50_K1_rerank_feature_builder.md` | Rerank feature builder (causal, pure) | Rerank | feature-purity + no-leak tests | spec: **done** |
| K2 | `51_K2_lgbm_lambdamart.md` | LightGBM LambdaMART (primary reranker) | Rerank | nDCG@20 ≥ 0.45, hit-rank ≤ 2.5 | spec: **done** |
| K3 | `52_K3_neural_reranker.md` | Cross-encoder/ColBERT (LoRA) + GBDT stacking | Rerank | nDCG@20 lift vs K2; → 0.55 | spec: **done** |
| K3b | `53_K3b_ce_lora_finetune.md` | Cross-encoder LoRA fine-tune (bge-reranker-v2-m3) + OOF stacking | Rerank | dev nDCG@20 lift vs K2 (final-stage + OOF feature) | spec: **done** |
| L1 | `60_L1_filter_assembly.md` | Filtering & top-20 assembly | Filter | schema-valid + nDCG non-regression + diversity hold | spec: **done** |
| S1 | `70_S1_responder.md` | Responder (LLM, grounded) | Respond | proxy-judge ↑ + Distinct-2 ≥ 0.2558 | spec: **done** |
| D1 | `80_D1_inference_submission_harness.md` | Inference & submission harness (Blind A/B) | Delivery | end-to-end schema-valid; local == leaderboard | spec: **done** |

## Build-order dependency graph
```
F1 ─┬─► F3 ─► P0 ─► (sizes K, cold/warm, context caps for all of Phase 1+)
F2 ─┘           │
                ▼
        A1 ─► R3, R4        R1 ─► R2 ─┐
        R1 ─► R3,R4,R5,R6    R3,R4,R5,R6 ─► R7
                                        │
                                        ▼
                            K1 ─► K2 ─► K3
                                        │
                                        ▼
                                       L1 ─► S1
                                        │
                                        ▼
                                       D1  (orchestrates the whole spine over Blind A/B)
```
Critical path to a first valid submission: **F1+F2+F3 → P0 → R1+R3 → R7 → L1 → (trivial responder) → D1** (plan §17 day 3–4). Everything else layers on.

## Per-module lifecycle
Each module: **spec (this doc) → `writing-plans` implementation plan → TDD build → tests green → code review → eval gate met → status updated here**. A module is not "done" until its eval gate (col. above) is met and logged in `reports/experiments.md`.
