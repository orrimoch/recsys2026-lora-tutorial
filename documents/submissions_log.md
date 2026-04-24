# Submissions log — tabular

One row per evaluation (dev or blind). `tag` column: `[dev-local]` for local dev evals, `[blindA]` / `[blindB]` for scored CodaBench submissions.

Schema per `recsys_challenge_plan.md` §4.3. Blind rows also carry the Gemini LLM term; dev rows have `LLM=—`.

## Notes

### ⚠️ Retrieval-only rows below are INFORMATIONAL, not experiments

Per user directive 2026-04-24: shippable experiments are either two-step (retrieval + LLM) or SID end-to-end. The `002`/`003`/`004`/`005`/`006`/`007`/`008`/`009`/`010`/`011`/`012` rows are retrieval-only inferences (no `predicted_response`), scored to **map the retrieval landscape** for picking the retrieval branch of a Wave-2 two-step stack. They are NOT candidates for blind submission (LexDiv=0 tanks composite).

**Best informational retrieval configs** (by nDCG@20, for picking Wave 2's retrieval branch):
- `010-wrrf-bm25-dense-lyrics-v1`: nDCG@20=0.0994 ← retrieval branch candidate
- `009-wrrf-bm25-dense-v1`: nDCG@20=0.0981
- `011-wrrf-bm25-dense-audio-clap-v1`: nDCG@20=0.0981
- `002-bm25-field-expansion` / `004-bm25-imputed`: nDCG@20=0.0971 (tied; 002 and 004 are equivalent)

### General


- `composite_retrieval` = `0.50·nDCG@20 + 0.10·CatDiv + 0.10·LexDiv` (the LLM-free portion — see plan §2.5.1).
- Rows with `(pre-existing)` in the date column were seeded from cached score files under `music-crs-evaluator/exp/scores/devset/` at Wave-0 bootstrap time.
- **B-floor anchor**: `llama1b_bm25_devset` — matches the official challenge baseline (nDCG@10=0.0627).
- **Random baseline** has a surprisingly high `composite_retrieval` (0.097) — driven entirely by catalog_diversity near 1.0 (random draws the whole catalog). This exposes how the composite formula over-rewards diversity at low-nDCG regimes. `composite_retrieval` alone is not a sufficient rank signal at the low end — always cross-check nDCG@20.
- **Popularity baseline** has the opposite failure — near-zero nDCG AND near-zero diversity (always recommends the same top-20). `composite_retrieval ≈ 0.001`.
- **Note on appends**: the auto-appender in `scripts/local_eval.py` appends at file end. Keep the table at the bottom of this file so appended rows land contiguously. The duplicate `llama1b_bm25_devset` row in the table below is the first such append (from the CLI smoke test) — left in place as per intentional log state.

## Table

| exp_id | tag | nDCG@1 | nDCG@10 | nDCG@20 | CatDiv | LexDiv | LLM | composite_retrieval | composite | date | status | git_sha |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | [dev-local] | 0.0000 | 0.0001 | 0.0001 | 0.9652 | 0.0000 | — | 0.0966 | — | 2026-04-24 (pre-existing) | — | — |
| popularity | [dev-local] | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.0000 | — | 0.0013 | — | 2026-04-24 (pre-existing) | — | — |
| llama1b_bm25_devset | [dev-local] | 0.0098 | 0.0627 | 0.0815 | 0.3795 | 0.2549 | — | 0.1042 | — | 2026-04-24 (pre-existing) | B-floor anchor | — |
| llama1b_bm25_devset | [dev-local] | 0.0098 | 0.0627 | 0.0815 | 0.3795 | 0.2549 | — | 0.1042 | — | 2026-04-24 | ITERATE | e63ac40 |
| 002-bm25-field-expansion | [dev-local] | 0.0096 | 0.0753 | 0.0971 | 0.4542 | 0.0000 | — | 0.0940 | — | 2026-04-24 | ITERATE | e63ac40 |
| 003-retrieval-only-baseline | [dev-local] | 0.0099 | 0.0626 | 0.0817 | 0.3797 | 0.0000 | — | 0.0788 | — | 2026-04-24 | ITERATE | e63ac40 |
| 004-bm25-imputed | [dev-local] | 0.0096 | 0.0752 | 0.0971 | 0.4543 | 0.0000 | — | 0.0940 | — | 2026-04-24 | ITERATE | e63ac40 |
| 005-dense-qwen3-metadata | [dev-local] | 0.0065 | 0.0186 | 0.0232 | 0.2870 | 0.0000 | — | 0.0403 | — | 2026-04-24 | ITERATE | e63ac40 |
| 006-dense-metadata-instruct | [dev-local] | 0.0194 | 0.0471 | 0.0575 | 0.2800 | 0.0000 | — | 0.0567 | — | 2026-04-24 | ITERATE | e63ac40 |
| 007-rrf-bm25-dense-v1 | [dev-local] | 0.0158 | 0.0677 | 0.0890 | 0.4177 | 0.0000 | — | 0.0863 | — | 2026-04-24 | ITERATE | e63ac40 |
| 008-bm25-then-dense-rerank | [dev-local] | 0.0219 | 0.0717 | 0.0908 | 0.4133 | 0.0000 | — | 0.0867 | — | 2026-04-24 | ITERATE | e63ac40 |
| 009-wrrf-bm25-dense-v1 | [dev-local] | 0.0145 | 0.0775 | 0.0981 | 0.4412 | 0.0000 | — | 0.0932 | — | 2026-04-24 | ITERATE | e63ac40 |
| 010-wrrf-bm25-dense-lyrics-v1 | [dev-local] | 0.0160 | 0.0782 | 0.0994 | 0.4420 | 0.0000 | — | 0.0939 | — | 2026-04-24 | ITERATE | e63ac40 |
| 011-wrrf-bm25-dense-audio-clap-v1 | [dev-local] | 0.0145 | 0.0774 | 0.0981 | 0.4413 | 0.0000 | — | 0.0932 | — | 2026-04-24 | ITERATE | e63ac40 |
| 012-bm25-then-lgbm-v1 | [dev-local] | 0.0114 | 0.0487 | 0.0664 | 0.4182 | 0.0000 | — | 0.0750 | — | 2026-04-24 | ITERATE | e63ac40 |
| 020-two-step-wrrf-lyrics-qwen15b-devset | [dev-local] | 0.0164 | 0.0784 | 0.0995 | 0.4418 | 0.3217 | — | 0.1261 | — | 2026-04-24 | PROMOTE to blind | 5ae994e |
| 021-two-step-wrrf-lyrics-qwen15b-blindsetA | [blindA] | — | — | 0.19 | 0.03 | 0.67 | 3.15 | — | 0.33 | 2026-04-24 18:00 | shipped, rank 9/9 | f74baed |
| 022-persona-qwen15b-blindsetA | [blindA] | — | — | 0.14 | 0.03 | 0.80 | 2.15 | — | 0.24 | 2026-04-24 18:22 | REJECTED — persona regressed LLM −1.00 on Qwen 1.5B | 53837f0 |
| 023-rerank-qwen3b-blindsetA | [blindA] | — | — | 0.07 | 0.03 | 0.61 | 2.20 | — | 0.19 | 2026-04-24 | REJECTED — BGE-reranker misaligned with preference-based ground truth | 775488d |
