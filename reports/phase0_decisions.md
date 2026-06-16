# P0 — Decisions & Design Parameters (full dev probe, 8000 turns)

Source: `nb/phase0_eda.ipynb` run (Colab, dense-text incl.). Full recall-ceiling table on Drive
(`outputs/eda.md`, `recall_ceiling.csv`).

## Measured recall-ceiling (depth 500)
| channel | r@20 | r@100 | r@200 | r@500 | unique |
|---|---|---|---|---|---|
| same_artist | 0.216 | 0.379 | 0.400 | 0.403 | 0.052 |
| dense (BGE) | 0.167 | 0.302 | 0.354 | 0.419 | 0.059 |
| content_knn | 0.153 | 0.223 | 0.244 | 0.275 | 0.014 |
| bm25 | 0.109 | 0.198 | 0.228 | 0.274 | 0.011 |
| cf | 0.019 | 0.066 | 0.102 | 0.160 | 0.056 |
| **FUSED** | **0.280** | **0.424** | **0.468** | **0.532** | — |

## Design parameters (locked)
- **fusion_K / topk_internal = 500** — use the deepest pool; r@500 (0.532) is the nDCG@20 ceiling a reranker can reach.
- **fusion_k (smallest K with recall ≥ 0.90) = NONE** — gate NOT met. **Recall is the binding constraint** → A1/R6/dense are the priority (plan §7 DoD).
- **channel_keep_list = all 5** — each has nonzero unique-recall (dense .059, cf .056, same_artist .052 highest; bm25/content_knn marginal but cheap). Weight sweep: up-weight dense + same_artist; cf/bm25 lower.
- **cold_threshold = 1** → cold/warm = 2000/6000 (25% / 75%).
- **context_cap ≈ 400 tokens** (ctx p95 = 362; fits the 512 encoder with prefix headroom); recency policy keeps the latest utterance + goal.
- **gold_in_history_rate = 0.0** → the gold is NEVER an in-session replay. So the L1 history-rule is **recall-safe** (filtering history can't drop a gold); default `off` (no nDCG effect) or `on` for diversity — either is safe.
- Integrity: 8000/8000 golds resolve to catalog; train∩dev sessions = 0 (no leak).

## The ceiling math (why this matters for the 0.55 target)
For a single gold, **nDCG@20 ≤ recall@(pool K)**. With the current pool, a *perfect* reranker tops out at **~0.53** (r@500); a realistic strong reranker lands ~0.30–0.40. **To reach nDCG@20 = 0.55 we must lift fused recall above ~0.6 first** — i.e. break the wall. ~47% of golds are unreachable by ALL channels even at depth 500 (mostly novel/new-artist content not similar to history and not matching current query text).

## Implications / priorities
1. **Recall is the make-or-break lever** (not reranking). Top levers, in order:
   - **A1 — catalog enrichment / doc2query** (Gemini): close the conversational↔metadata vocabulary gap so BM25/dense reach novel golds. Biggest expected lift.
   - **R2 — LLM query refinement**: extract explicit intent (artist/genre/era/mood) the raw dialogue buries.
   - **Stronger / multimodal dense**: dense (BGE) already the 2nd-best + highest unique; try a stronger encoder, and content-kNN on the OTHER provided modalities (audio-laion_clap, lyrics/attributes-qwen3) — cheap, orthogonal, no encoder.
   - **R6 — extension channels**: related-artist (co-occurrence), CLAP audio, propose-ground.
2. **In parallel**: wire K1→K2 end-to-end to *capture the recall we have* → a real ~0.3 submission now (validates the pipeline, gives a leaderboard anchor).
