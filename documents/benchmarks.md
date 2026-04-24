# Benchmarks

Our three reference points per `recsys_challenge_plan.md` §2.1.

Updated automatically by `scripts/run_experiment.py` after every scored dev run that beats B-champ, and manually after every scored blind run.

## B-floor — Official LLaMA-1B + BM25 devset baseline

Source: challenge repo (`music-crs-evaluator/readme.md:179-183`); matches the on-disk score file `music-crs-evaluator/exp/scores/devset/llama1b_bm25_devset.json`.

| Metric | Value |
|---|---|
| nDCG@1 | 0.0098 |
| nDCG@10 | 0.0627 |
| nDCG@20 | 0.0815 |
| catalog_diversity | 0.3795 |
| lexical_diversity | 0.2549 |
| LLM-judge (blind) | — |
| **composite_retrieval** | **0.1042** |

## B-champ — Current internal champion

First champion set 2026-04-24 — Wave 2's `020-two-step-wrrf-lyrics-qwen15b-devset` cleared the B-floor promotion gate by +0.0219 on composite_retrieval (σ=0.005).

| Rank | exp_id | composite | composite_retrieval | nDCG@20 | CatDiv | LexDiv | LLM (blind) | Defining mechanism | Config | Date shipped |
|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **020-two-step-wrrf-lyrics-qwen15b-devset** | — | **0.1261** | 0.0995 | 0.4418 | 0.3217 | — | **wRRF(BM25 + dense-metadata-qwen3 + dense-lyrics-qwen3) + Qwen2.5-1.5B-Instruct response** | `music-crs-baselines/config/020-two-step-wrrf-lyrics-qwen15b-devset.yaml` | 2026-04-24 |
| 🥈 | — | — | — | — | — | — | — | — | — | — |
| 🥉 | — | — | — | — | — | — | — | — | — | — |

## B-target — Public Blind-A leaderboard top-1

Source: last-known leaderboard snapshot (see `.claude/memory/project_blind_a_state.md` — update when new data available).

| Metric | Value (approximate) |
|---|---|
| nDCG@20 | ~0.21–0.23 |
| CatDiv | ~0.03 |
| LexDiv | ~0.75 |
| LLM-judge | ~3.85 |
| composite | ~0.40 |

Note: CatDiv appears low on blind (~0.03) because the blind set is only 80 rows — most teams recommend from a small slice of popular catalog. This is a blind-set artifact, not a signal that CatDiv doesn't matter on dev (where catalog is spread across 1000 sessions × 8 turns = 8000 recommendations).
