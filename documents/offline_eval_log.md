# Offline eval log — Tier-1 retrieval-only

Append-only. Populated by `scripts/offline_eval.py`. Tier-1 is fast,
deterministic-seeded, retrieval-only; use this as the Blind-A pre-filter.

Rule: a candidate config ships to Blind-A ONLY if its offline
composite_retrieval + 2*sigma ≥ prior champion's offline composite
AND nDCG@20 ≥ wRRF baseline. Sigma estimated from repeated-seed runs.

## Table

| tid | seed | holdout_size | n_pairs | nDCG@1 | nDCG@10 | nDCG@20 | CatDiv | composite_retrieval |
|---|---|---|---|---|---|---|---|---|
| 020-two-step-wrrf-lyrics-qwen15b-devset | 42 | 20 | 160 | 0.0187 | 0.0808 | 0.1163 | 0.0208 | 0.0602 |
| 021-two-step-wrrf-lyrics-qwen15b-blindsetA | 42 | 200 | 1600 | 0.015 | 0.087 | 0.1209 | 0.1523 | 0.0757 |
