# Tier-1 #3.2 — culture + profile signals in the retrieval query

Date: 2026-06-07
Status: approved (design)

## Context

Cold-firable recall (Tier 1). On single-turn / zero-history Blind queries the
session channels are dead, so the dense + BM25 content channels carry the load.
The retrieval query currently uses goal (mode `raw_with_goal`) but withholds
`preferred_musical_culture` (a short genre/scene field, e.g. "Western Alternative
Rock", "Alternative/Industrial Music Culture") and the user profile — taste
signals that can bias retrieval toward the right scene and surface new-artist
golds. These fields are legal at inference (present in Blind), not leakage.

## Goal

Add an opt-in retrieval-query mode that appends culture + profile, and measure
whether it lifts turn-1 recall@100 before any reranker retrain.

## Change (this increment = validation stage only)

`mcrs/crs_baseline.py` `build_retrieval_query`: add mode `raw_enriched` =
`raw_with_goal` plus, each only when present:

```
goal: <listener_goal>
culture: <preferred_musical_culture>
user: age=<age_group> country=<country_name> gender=<gender>
```

Also add `raw_enriched` to the allowed `query_preprocessing_mode` set in
`CRS_BASELINE.__init__` so it CAN be configured later (does not change the
default). No other production change in this increment.

## Non-goals (this increment)

- Do NOT change the shipped `query_preprocessing_mode` (stays `raw_with_goal`).
- Do NOT modify `build_lgbm_features` or retrain yet (gated on the recall result).
- Profile fields render `unknown` when missing (positionally stable), but the
  whole `user:`/`culture:` line is omitted if the field is empty.

## Parity (when we wire it, post-validation)

If recall lifts, wiring to serve requires the same three sites to agree (serve
`build_retrieval_query` mode, the `build_lgbm_features` training query, the nb74
harness) + a reranker retrain — same discipline as Tier-0 #2.

## Regression safety (must not break Tier 0 / 3.1a)

- New mode is inert unless selected; shipped config 194 (`raw_with_goal`) is
  bit-identical. `_wrrf_union_v1_specs` and all Tier-0 files untouched.
- Full test suite after the change: all prior tests green; only the 4 known
  pre-existing unrelated failures remain.
- Tier-end QA re-runs Tier-0 + 3.1a guard tests.

## Tests (TDD)

- `raw_enriched` appends culture + profile lines after the goal line; omits a
  line when its field is empty; falls back to `raw_with_goal` output when profile
  and culture are absent.
- `raw_enriched` is accepted by `CRS_BASELINE.__init__` mode validation.

## Validation gate (no retrain)

nb74: build `q_base` (mode `raw_with_goal`) and `q_enriched` (mode `raw_enriched`)
via `build_retrieval_query` for the dev set, retrieve with the same 3-channel
union, and report recall@100 overall + turn-1 for both. STOP and report. Proceed
to wire serve + feature-builder + retrain ONLY if turn-1 recall lifts beyond
inter-seed noise (~0.005).
