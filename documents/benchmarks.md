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

Updated 2026-04-24 from the live CodaBench leaderboard captured after Exp 021 shipped.

**#1 Chris Deotte (2026-04-24 04:14) — composite 0.57.**

| Metric | #1 value | #2 (greenwolf) | #3 (yonghyunk1m) | Our 021 (rank 9/9) |
|---|---|---|---|---|
| composite | 0.57 | 0.56 | 0.53 | **0.33** |
| nDCG@20 | 0.44 | 0.44 | 0.38 | **0.19** |
| CatDiv | 0.03 | 0.03 | 0.03 | 0.03 |
| LexDiv | 0.82 | 0.83 | 0.79 | **0.67** |
| LLM judge | 4.55 | 4.45 | 4.50 | **3.15** |

Gap decomposition (us → #1, 0.33 → 0.57, Δ = 0.24):
- nDCG@20: 52% of gap — **biggest lever, 0.25 raw headroom**
- LLM judge: 44% of gap — 1.40 raw headroom
- LexDiv: 6% of gap — 0.15 raw headroom
- CatDiv: 0% — everyone tied at 0.03 (blind-set artifact, 80 rows → narrow catalog)

Note: the leader this season has nDCG@20 = 0.44, substantially higher than prior-branch leaderboard's ~0.23. Retrieval-side work has new leverage; prior branch's "LLM is the only lever" intuition is OUTDATED.

Note: CatDiv appears low on blind (~0.03) because the blind set is only 80 rows — most teams recommend from a small slice of popular catalog. This is a blind-set artifact, not a signal that CatDiv doesn't matter on dev (where catalog is spread across 1000 sessions × 8 turns = 8000 recommendations).
