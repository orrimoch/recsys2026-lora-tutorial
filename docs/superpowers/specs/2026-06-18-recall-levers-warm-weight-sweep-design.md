# Recall levers: warm-tuned RRF weights + ColBERT promotion (design)

Date: 2026-06-18
Branch: fresh-start
Status: approved design, pre-implementation

## Goal

Raise the Blind-A composite by attacking its binding constraint: retrieval recall. The composite is
approximately `0.5·nDCG@20 + 0.1·catalog_diversity` (inferred from the first submission:
nDCG@20=0.3221, catalog_diversity=0.031, composite=0.1641), so nDCG carries ~half the weight and the
reranker can only reorder what retrieval surfaces (fused recall wall ≈ 0.69@500). We tackle two recall
sub-levers together:

1. Re-tune RRF channel weights for the WARM (history-rich) regime — Blind-A is evaluated on the
   trailing turn of each session, which is warm-heavy, so the existing cold-skew weight plan is
   mis-aimed.
2. Promote the LoRA-fine-tuned ColBERT (`colbert_ft`) from a marginal channel toward primary
   retrieval (the planned replace-R4-dense ablation; colbert_ft now beats dense at every depth).

Both are expressed as RRF weight vectors, so they are swept jointly.

## Success criteria & gate

- Primary gate: final-turn dev nDCG@20 beats the baseline (Section "Substrate") by more than noise
  (≈ +0.005 on 1000 dev sessions). Measured slot-free on the dev final-turn proxy.
- Guardrail: `catalog_diversity` must not crater relative to baseline (we are not optimizing it here).
- Recall diagnostic (not a gate): fused recall@200/@500 on the final-turn dev set should rise, confirming
  the ceiling actually moved. If recall rises but nDCG does not, that is a reranker-limited signal to
  record, not a ship.

## Background / current infrastructure (reuse, do not rebuild)

- `mcrs/retrieval/fusion.py` — `RRFFusion(channels, weights=None, k=60, segment_weights=None)`; reads
  `ctx["segment"]` via `_weights_for(ctx)` and applies per-segment weight vectors. `fuse(..., per_channel_queries=...)`
  routes focused queries by each channel's `query_key`. `fuse_per_sub(...)` re-fuses cached per-channel lists.
- `mcrs/eval/weight_sweep.py` — `fused_recall(per_channel, weights, golds, ks, ...)` (0-weight channel = dropped)
  and `segment_weight_sweep(per_channel, golds, segments, candidates, ks, k)` returning best weights per segment.
- `mcrs/eval/probe.py` — `recall_ceiling(per_channel, golds, ks, segments, fusion_weights, fusion_k)` returns
  per-channel recall + by_segment + `unique_recall`, and fused recall by segment.
- `mcrs/eval/diagnostics.py` — `score_diagnostic(predictions, ground_truth, catalog_size, k_recall, k_ndcg, segment_of)`
  returns overall + by_segment + by_turn recall@k and nDCG@k.
- `mcrs/data/conversations.py` — `target_turns()` (final turn per session, with gold on dev) and `turns()`.
- `mcrs/retrieval/dense_channel.py` `DenseChannel` and `mcrs/retrieval/colbert_channel.py` `ColBERTChannel`
  share the `batch_text_to_item_retrieval` interface; ColBERT carries `query_key` for focused routing.
- `mcrs/rerank/neural.py` — `build_ce_score_lookup`, `make_ce_feature_fn(lookup, default=-1.0)` (keyed by
  (session_id, turn_number, tid)).

## Design

### Substrate (the measurement base)

- Dev final-turn proxy: build dev serve turns with `conv_serve.target_turns()` (one trailing turn per
  dev session; on dev that turn carries a gold, so it is scorable). This matches Blind-A's regime
  (final turn, warm-heavy). All experiments score nDCG@20 on this set.
- Baseline number: current uniform RRF weights + the current K2, dev final-turn nDCG@20. This is the
  reference every lever must beat. (Expectation: near the submitted 0.32, validating the proxy.)
- Per-channel cache: run each channel once over the final-turn queries with correct routing (colbert
  focused via `per_channel_queries`, others full), caching the per-channel ranked lists at a deep
  `topk_internal` (≈1000) plus per-turn gold and segment. This feeds Stage 1.

### Stage 1 — recall sweep (no K2; seconds)

- Feed the cached per-channel lists to `recall_ceiling` (per-channel + fused, by segment, with
  `unique_recall`) and `segment_weight_sweep`.
- Candidate weight vectors are generated data-drivenly from `unique_recall`: channels contributing
  unique golds are kept/up-weighted; low-unique channels are zeroed. The candidate set spans both
  levers — up-weight personalization (cf / same_artist / related_artist), up-weight colbert, zero dense
  (replace-R4-dense), and colbert-primary.
- Rank configs by WARM-segment recall@200/@500 (blind regime is warm-heavy).
- `recall_ceiling` per-channel output provides attribution: what colbert-promotion contributed vs what
  the reweighting contributed, even though both move at once.
- Output: top 1–3 weight configs (per-segment weight dicts).

### Stage 2 — nDCG confirm (K2 retrain for finalists only)

- A recall gain counts only if it survives reranking, so confirm the top 1–3 configs on final-turn dev
  nDCG@20.
- CE-reuse optimization (makes this tractable): the frozen-CE score for a given (turn, tid) is
  weight-independent, so score CE ONCE over the union of candidates across the finalists and reuse the
  `ce_lookup` for every config. Each finalist then costs only a ~5-min K2 fit on its fused pools, not a
  fresh ~60-min CE pass.
- For each finalist: set `RRFFusion(segment_weights=…)`, retrain K2 on the TRAIN fused pools with those
  weights (K2 training regime otherwise unchanged), run inference on dev final-turns, and read nDCG@20 +
  catalog_diversity via `score_diagnostic`.
- Pick the winner by the gate above.

### Deliverable & wiring

- The winning per-segment `segment_weights` dict, wired into the submission notebook's `RRFFusion`
  construction, ready for the next `BLIND=True` slot.
- A `DEV_FINAL_ONLY` toggle in the submission notebook so its dev-confirm path can measure the same
  final-turn regime (dev serve → `target_turns()`).

## Scope

In scope: dev final-turn proxy, per-channel cache, recall sweep, nDCG-confirm of finalists, winning
weights wired in.

Out of scope (deliberately, to keep attribution clean — one variable family at a time):
- Retraining K2 on final-turns-only (roadmap lever 4).
- Diversity / MMR re-rank (lever 5).
- Dropping or fine-tuning the CE feature (lever 6).
- Breaking the recall wall via new retrieval models or query reformulation (separate, higher effort).

## What is new vs reused

- New: one experiment notebook (extend `nb/phase1_colbert_probe.ipynb` or a sibling) running
  substrate → Stage 1 → Stage 2; a `DEV_FINAL_ONLY` toggle in the submission notebook; if a pure
  candidate-weight-generation helper is introduced, it gets a unit test.
- Reused: everything in "Background" above.

## Testing

- The reused infra (`weight_sweep`, `probe.recall_ceiling`, `diagnostics`, `fusion`, `target_turns`)
  already has unit tests.
- Any new pure helper (e.g. `unique_recall` → candidate weight vectors) gets a focused unit test before
  use (TDD).
- The experiment notebook is validated by the baseline-proxy sanity check (final-turn dev nDCG ≈ the
  submitted 0.32) before any lever is trusted.

## Risks

- Recall-best weights need not be nDCG-best — mitigated by confirming the top 1–3, not just the #1.
- Changing weights shifts the `rrf_score` feature K2 trained on — mitigated by the Stage-2 K2 retrain.
- 1000-session dev noise — mitigated by the +0.005 margin and reporting per-segment breakdowns.
- Proxy/Blind-A drift — the proxy is final-turn dev, not Blind-A; only a real submission confirms, but
  the proxy regime matches and is the best slot-free predictor available.
