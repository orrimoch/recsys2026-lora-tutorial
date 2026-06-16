# K3 — Neural Reranker (cross-encoder / ColBERT, LoRA) + GBDT Stacking

> Phase-2, optional. Sharpens the top-1–3 by re-scoring only the GBDT's top `cross_encoder_k`
> candidates with a semantic neural model. **Preferred integration = stacking** (feed its score back
> as an OOF K1 feature), not replacing K2. Gated: ships only on a measured dev nDCG@20 lift. See `000_INDEX.md`.

## 1. Purpose
Add token-/pair-level semantic precision the GBDT can't get from scalar features — push the gold from rank ~3 to rank 1. Re-scores a narrow top slice (cost scales with K), never enlarges the recall pool (R7 owns recall).

## 2. Interface / contract
Lives in `mcrs/rerank/neural.py`. Implements F2 `Reranker`, and also exposes a scorer for stacking.

```python
class NeuralReranker:                       # implements F2 Reranker
    def __init__(self, model_revision: str, kind: str, cfg: "NeuralConfig"): ...  # kind: "cross_encoder"|"colbert"
    def rerank(self, ctx: TurnContext, candidates: list[Candidate]) -> RankedList: ...   # re-scores top cross_encoder_k
    # stacking surface: OOF score per candidate, fed back into K1.Candidate.features
    def score(self, ctx: TurnContext, candidates: list[Candidate]) -> dict[str, float]: ...  # track_id -> neural score
```

**Wiring:** input = K2's `RankedList` (already GBDT-ranked); K3 re-scores the **top `cross_encoder_k`** (≈100–200, from P0/§7.3.1, with `topk_internal ≥ fusion_K ≥ cross_encoder_k`). Two integration modes: (a) **stacking (default)** — `score()` produces an OOF feature consumed by K1 → K2 retrains with it (best-of-both, robust); (b) **final-stage re-score** — `rerank()` directly reorders the top slice. Output `RankedList` → L1.

## 3. Dependencies
F2 (`Reranker`, `Candidate`, `RankedList`, `TurnContext`), K1 (stacking feature), K2 (its ranked pool), R7, F3 (nDCG@20 + hit-rank), F1 (`id_to_metadata(enriched=True)` for the doc side). Models (open-weight): `BAAI/bge-reranker-v2-m3` (cross-encoder), ColBERT (pylate/late-interaction). GPU for scoring/training; only top-K pairs/turn. Salvage: `bge_reranker.py`, `multimodal_cross_encoder_rerank.py`, `colbert_late.py`, `scripts/{build_colbert_index,build_colbert_train_data,train_colbert}.py`.

## 4. Design & logic
- **Cross-encoder:** score `(dialogue-context query, enriched-track-doc)` pairs with `bge-reranker-v2-m3`; strongest single semantic reranker. Doc-side truncation per §8 (truncate the document, preserve the query/latest-intent side).
- **ColBERT late-interaction:** token-level MaxSim between query and enriched doc; middle ground, good for phrase-level intent. Needs a built index (`build_colbert_index`) + optional fine-tune (`train_colbert`).
- **LoRA fine-tune (option):** fine-tune the cross-encoder (or a ColBERT head) with LoRA on Train (its own §6.1 notebook → Hub adapter, session-disjoint val) to beat the off-the-shelf model cheaply. Gate behind an ablation win.
- **Two-tier K:** only re-score the GBDT top `cross_encoder_k`; the rest keep their K2 order below the slice. Never re-open the recall pool.
- **Stacking vs final-stage:** default **stacking** — K3's score becomes an **OOF** feature (cross-fit per K1 §4.1, else in-sample leak) the GBDT blends with all other signals; usually beats letting the neural model overwrite a well-calibrated GBDT order. Final-stage re-score is the simpler fallback when stacking's OOF plumbing isn't ready.

## 5. Reuse
Port `salvage/mcrs/rerankers/bge_reranker.py` + `multimodal_cross_encoder_rerank.py` (cross-encoder), `salvage/mcrs/retrieval_modules/colbert_late.py` + `scripts/{build_colbert_index,build_colbert_train_data,train_colbert}.py` (ColBERT). Prior ColBERT notebooks `salvage/notebooks/82_*`, `90_*` are recipe references only — **no carried-over numbers**. **Port behind gate.**

## 6. Eval & acceptance gate
Optional module: ships only if dev **nDCG@20 (and hit-rank) improves over K2-alone** (via F3), within the cross-encoder cost budget (§15). Report the lift per segment (cold/warm) and the latency/cost per turn. If no lift, it does not ship — K2 stands alone.

## 7. Tests
- F2 `Reranker` conformance; re-scoring only the top `cross_encoder_k`, rest order preserved; never adds an id outside the pool.
- Stacking OOF guard (K1 §4.1): the stacked score is cross-fit; in-sample variant fails the test.
- Doc-side truncation preserves the query side; deterministic given seed + pinned `model_revision`.
- nDCG@20 non-regression vs K2 on a fixture when enabled; cost within budget.

## 8. Failure modes & guards
- **In-sample stacking leak** → OOF/cross-fit mandatory (K1 §4.1) + test.
- **Thinking/length overrun** inflating latency/cost → cap output; budget assert per turn.
- **Train/serve skew** (model revision/prompt/truncation) → pin `model_revision`, identical pair-format train==serve; D1 records the hash.
- **Pool shrink** (cross-encoder budget squeezing recall) → K3 only re-scores within K2's top slice; R7's pool is never reduced by K3's K.
- **Over-trusting the neural order** → default to stacking; final-stage only on a clear win.

## 9. Config knobs
`rerank.neural.enabled` (default false), `.kind` (`cross_encoder`|`colbert`), `.model_revision`, `.cross_encoder_k` (from P0), `.mode` (`stacking`|`final`, default `stacking`), `.lora.{enabled,r,alpha,target_modules}`, `.max_doc_tokens`, `.batch_size`. Defaults/types from F2 loader.

## 10. Definition of Done & review checklist
> Status after the 2026-06-16 K3-vs-plan review (impl: `mcrs/rerank/{neural,cross_encoder}.py`).
- [x] Implements F2 `Reranker` + `score()` stacking surface; re-scores only top `cross_encoder_k`
      (final-stage `rerank()` done; `ChainReranker` composes K2→K3; 7 tests). `model_revision` carried.
- [x] Stacking score is OOF: **CLARIFIED — OOF not required for the off-the-shelf model.** A frozen
      pretrained cross-encoder never sees the gold labels, so its score is a fixed function of
      (query, doc); stacking it into K1 is leak-free, identical to the shipped `dense_cos` bi-encoder
      feature. OOF/cross-fit is mandatory ONLY if the cross-encoder is FINE-TUNED on Train (§4 LoRA) —
      gated to that path with its own harness + in-sample-fails-a-test. Doc-side truncation is
      **token-level** (`build_cross_encoder_score_fn` / `truncate_doc_tokens`, doc only) — query
      side preserved; tested.
- [x] Per-turn cost budget assert (§8): `NeuralReranker(max_pairs_per_turn=...)` raises if a turn
      would score more than the budget; tested.
- [~] Dev nDCG@20 lift over K2 measured; cost within budget; cold/warm reported: **wired** in
      `phase2_rerank` 6b (K2 vs K2+K3 overall + per-segment), pending a run. Ships only if K2+K3 > K2.
- [ ] If LoRA used: adapter on Hub by revision, session-disjoint val, train==serve pinned. (LoRA N/A yet.)
- [~] Code review approved: reviewed; cheap items applied (fail-loud scorer, edge tests, `model_revision`);
      OOF harness + per-turn cost budget assert remain open.

## 11. Build order & dependencies
**Built after K1 + K2** (needs the GBDT pool + the feature/stacking harness) and P0 (`cross_encoder_k`). Depends on: F1, F2, F3, K1, K2, R7. **Blocks:** nothing hard (optional precision lever feeding L1); its stacked feature loops back into K1/K2. Sits on the "push to 0.55" path (plan §17 day 8–10).
