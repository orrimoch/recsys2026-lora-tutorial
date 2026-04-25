# Offline eval log — Tier-1 retrieval-only

Append-only. Populated by `scripts/offline_eval.py`. Tier-1 is fast,
deterministic-seeded, retrieval-only; use this as the Blind-A pre-filter.

Rule: a candidate config ships to Blind-A ONLY if its offline
composite_retrieval + 2*sigma ≥ prior champion's offline composite
AND nDCG@20 ≥ wRRF baseline. Sigma estimated from repeated-seed runs.

`first_turn_only=yes` rows restrict gold to turn-1 music picks, mirroring
Blind-A's single-turn distribution. These rows are more predictive of
online Blind-A scores than all-turn rows (which have noisy multi-turn
chat history that BM25 drowns on but the Blind-A set doesn't).

## Table

| tid | seed | holdout_size | first_turn_only | n_pairs | nDCG@1 | nDCG@10 | nDCG@20 | CatDiv | composite_retrieval |
|---|---|---|---|---|---|---|---|---|---|
| 020-two-step-wrrf-lyrics-qwen15b-devset | 42 | 20 | no | 160 | 0.0187 | 0.0808 | 0.1163 | 0.0208 | 0.0602 |
| 021-two-step-wrrf-lyrics-qwen15b-blindsetA | 42 | 200 | no | 1600 | 0.015 | 0.087 | 0.1209 | 0.1523 | 0.0757 |
| 025-wrrf-cfbpr-qwen15b-blindsetA | 42 | 200 | no | 1600 | 0.0187 | 0.0936 | 0.1278 | 0.1517 | 0.0791 |
| 021-two-step-wrrf-lyrics-qwen15b-blindsetA | 42 | 200 | yes | 200 | 0.02 | 0.0578 | 0.069 | 0.0634 | 0.0408 |
| 025-wrrf-cfbpr-qwen15b-blindsetA | 42 | 200 | yes | 200 | 0.02 | 0.0574 | 0.0725 | 0.0635 | 0.0426 |
| 021-two-step-wrrf-lyrics-qwen15b-blindsetA | 42 | 200 | no | 1600 | 0.0138 | 0.0471 | 0.0591 | 0.242 | 0.0537 |
