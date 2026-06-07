# Tier-1 #3.1a — attributes-qwen3 as a second dense content channel

Date: 2026-06-07
Status: approved (design)

## Context

nDCG campaign, Tier 1 (cold-firable recall). The union has exactly one dense
content channel today: `dense_metadata_qwen3_instruct` (weight 0.7). The
architecture study verified the catalog's `attributes-qwen3` field is genuinely
orthogonal to `metadata-qwen3` (0.62 same-track cosine, 3.6% top-10 neighbor
overlap), so adding it as a second dense channel can surface new-artist / cold
golds the metadata field misses (the documented recall wall). Lyrics was already
tried and rejected (-0.0046 recall); metadata is already wired; attributes-instruct
is the one untried orthogonal field.

## Goal

Add `attributes-qwen3` (instruct) as a second, opt-in dense channel in
`wrrf_union_v1`, and measure whether it lifts cold/turn-1 recall@100 before
committing to any reranker retrain.

## Change (minimal, additive, opt-in)

`mcrs/retrieval_modules/__init__.py` `_wrrf_union_v1_specs`: add a gate

```python
if ec.get("use_attributes"):
    specs.append({"type": "dense_attributes_qwen3_instruct",
                  "corpus_types": corpus_types, "topk_internal": 100,
                  "weight": float(ec.get("w_attributes", 0.4))})
```

The factory key `dense_attributes_qwen3_instruct`, the `QWEN3_MUSIC_INSTRUCT`
constant, and `DENSE_PRECOMPUTED` multi-instance support already exist. No other
code changes. Scope decision: ADD alongside metadata (keep both); do NOT swap.

## Non-goals (this increment)

- No swap of metadata -> attributes.
- No lyrics channel (already rejected).
- No reranker retrain / nDCG / Blind submission yet (gated on the recall result).
- No weight sweep yet (only after turn-1 recall lifts).

## Parity

Serve (`crs_baseline`) and the feature builder (`build_lgbm_features`) both build
the union through this same factory + `extra_config`, so enabling `use_attributes`
flows to both with no separate wiring and no train/serve skew.

## Regression safety (must not break Tier 0)

- Change is gated `use_attributes` default OFF -> shipped config 194 is bit-identical;
  the Tier-0 SASRec-parity retrain is unaffected.
- Touches no Tier-0 file (SASRec dialog parity, eval_ndcg, carve, val guard, OOF concat).
- After the change: run the FULL test suite; pass = all prior tests still green and
  only the 4 known pre-existing unrelated failures remain.
- Tier-end QA re-runs the Tier-0 test files and confirms the opt-in default-off invariant.

## Tests (TDD)

- `use_attributes=True` appends one `dense_attributes_qwen3_instruct` sub at
  `w_attributes` (default 0.4); absent by default. (mirror `test_union_factory.py`
  / the clap_recall gating test)
- `recall_by_turn` helper in `mcrs/eval_ndcg.py`: per-turn + turn-1 recall@k
  (mirrors `ndcg_by_turn`), with unit tests.

## Validation gate (no retrain)

nb74 cell 4: build the union with `use_attributes=True` (w sweep candidates
0.3/0.4/0.7) and report recall@100 overall AND turn-1 vs the baseline union.
STOP and report. Proceed to rebuild features -> retrain -> turn-stratified nDCG ->
Blind ONLY if turn-1 recall lifts beyond inter-seed noise.

## Follow-up (separate increments, not now)

LLM structured-query extraction (#3.1b), goal/culture/profile query signals (#3.2),
two-tower (#3.3), cf-bpr channel (#3.4).
