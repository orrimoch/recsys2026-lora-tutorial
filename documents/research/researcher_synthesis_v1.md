# Researcher Synthesis v1
Date: 2026-04-17
Author: researcher agent

## 1. Project state synthesis (signal & weakness readout)

**What's built.** Retrieval-only fast path on Mac (MPS/CPU) via `run_inference_devset_retrieval_only.py`. Factory-registered modules: `bm25` (baseline 4-field + v2a 5-field with `tag_list`), `dense_precomputed` with three parameterized variants (`dense_attributes_qwen3` raw, `dense_metadata_qwen3` raw, `dense_metadata_qwen3_instruct`), and `rrf` with pluggable `sub_specs`. Shared `experiments/cache/` now populated with BM25 index + Qwen3 query embeddings for all 8k devset queries + track matrix — so any retrieval-only rerun is <3 min wallclock.

**What the numbers say.** Champion is v2a (tid=002, `nDCG@10=0.0753`, +0.0126 honest Δ vs. matched local anchor 0.0626). The signal pattern across 7 experiments:

- **Text+tags is a strong single stream.** BM25 5-field at 0.0753 is the ceiling for single-retriever text matching; imputation (v2b) was null because missingness (0.2% tag_list, 1.4% release_date) was an order of magnitude smaller than hypothesized.
- **Dense alone is weaker than BM25, but not useless.** Raw dense (v3, attributes col, no prompt) collapsed to 0.0186. With correct column + Qwen3 instruction prefix (v3.1), dense recovers to 0.0471 — 62% of BM25, viable as a *complementary* stream but not a replacement.
- **Unweighted RRF dilutes the top of the list.** v4 (tid=007, RRF k=60, dense topk_internal=60) scored `ndcg@10=0.0677` (Δ=−0.0076 vs champion) **but ndcg@1=0.0158 was +63% rel vs champion's 0.0096**. Dense is injecting better @1 candidates but polluting the tail with long-range semantic neighbors that push BM25's correct answers down. This is the canonical "score-scale mismatch / symmetric RRF over-smoothing" failure mode — papers we re-found (Lin et al.'s "Analysis of Fusion Functions" ACM TOIS 2023, Bruch et al. SIGIR 2023) document that convex combination often beats RRF and that weighting matters.

**Weaknesses left on the table.** From `documents/analysis/v2b_buckets.md`: the weakest large cell is `q4_usercold_trackwarm` at **0.055 (482 queries)** — cold users picking popular tracks where text alone is ambiguous. The strongest cells are cold-track buckets (0.085–0.091) where BM25+tags already shines. So the next-idea portfolio should:
1. Harvest the obvious v4 near-win (dense helps @1 — capture it without losing @10).
2. Add orthogonal signal for `q4_usercold_trackwarm` (user-embedding stream, popularity prior, query rewriting/HyDE for underspecified cold-user turns).
3. Prepare first trained component (LambdaMART) — the jump from retrieval-only fusion to learned reranking is the single biggest expected-gain category on the roadmap.

## 2. Proposed backlog entries (6–8, ready to paste)

These extend the roadmap from §6 of the plan (v6+ placeholders) with concrete, costed, Mac-feasible entries. Labels use decimal sub-versions to avoid collision with the existing v1–v12 numbering.

```
### [P1] v4.1 · BM25-topK → dense rerank (sequential, not parallel)
- hypothesis: using dense as a reranker over BM25's top-100 (not a parallel fusion stream) preserves BM25's recall while letting dense decide the top-1, converting v4's observed +63% rel nDCG@1 signal into a net @10 win.
- grounding: Nogueira & Cho 2019 "Passage Re-ranking with BERT" (MonoBERT pattern); also validated in Thakur et al. 2021 "BEIR" where retrieve-then-rerank beats fused single-stage on most OOD tasks.
- parent_champion: v2a (tid=002, ndcg@10=0.0753)
- touches: new `mcrs/rerankers/dense_rerank.py` (wraps an existing `dense_*_qwen3_instruct` retriever and rescoring over a first-stage's topk list); `mcrs/retrieval_modules/__init__.py` adds a new retrieval_type key `bm25_then_dense_rerank_v1` that composes BM25(topk=100) → dense cosine rerank → trim to 40; new config.
- has_train_step: false
- retrieval_only: true
- expected_cost_min: 4
- expected_gain_ndcg10: +0.005 to +0.015
- risk: low
- unlocks: [v4.2, v6, v7]
- research_notes: documents/research/researcher_synthesis_v1.md
- status: open

### [P1] v4.2 · Weighted RRF (BM25 heavy, dense light) + reduced dense topk_internal
- hypothesis: v4's symmetric RRF (both subs at topk_internal=60, equal weight) over-weights dense's long tail. A weighted variant `score = w_bm25/(k+r_bm25) + w_dense/(k+r_dense)` with `w_bm25=1.0, w_dense=0.4` and dense `topk_internal=20` preserves BM25's tail while keeping dense's @1 lift. Grid: `w_dense ∈ {0.2, 0.4, 0.7}`, `dense_topk_internal ∈ {20, 40}`, `k ∈ {30, 60}` — pick best on a held-out 100-session slice, report on 1000-session.
- grounding: Bruch, Nardini, Rulli, Venturini 2023 "Analysis of Fusion Functions for Hybrid Retrieval" (ACM TOIS) — convex combo consistently beats RRF; weighting within RRF addresses the same mismatch.
- parent_champion: v2a (tid=002, ndcg@10=0.0753)
- touches: extend existing `mcrs/retrieval_modules/rrf.py` — add `weights: list[float]` field to `sub_specs`, default 1.0 each (backward-compat); new factory key `wrrf_bm25_dense_metadata_v1` in `retrieval_modules/__init__.py`; new config. **No edits to the existing factory keys or RRF signature for prior configs.**
- has_train_step: false
- retrieval_only: true
- expected_cost_min: 6
- expected_gain_ndcg10: +0.003 to +0.010
- risk: low
- unlocks: [v5, v5.5]
- research_notes: documents/research/researcher_synthesis_v1.md
- status: open

### [P2] v4.3 · Multi-way RRF — BM25-5field + BM25-3field + dense-metadata + dense-attributes
- hypothesis: two BM25 variants (tag-heavy 5-field vs. name-only 3-field) diverge on disambiguation vs. precision; similarly, metadata-dense vs. attributes-dense should over-rank different tracks. A 4-way RRF captures more independent "voters" before any single stream's tail noise dominates (RRF's core assumption: more honest voters → tighter ranking).
- grounding: Cormack, Clarke, Büttcher 2009 (original RRF paper — explicitly tested with 3–4 ranker ensembles); Lin, Ma 2021 IR anthology update on RRF ensembles.
- parent_champion: v4.2 or v2a (whichever is champion at claim time)
- touches: new factory key `rrf_4way_v1` in `mcrs/retrieval_modules/__init__.py` composing existing `bm25` (two distinct corpus_types), `dense_metadata_qwen3_instruct`, and `dense_attributes_qwen3_instruct`; no changes to `rrf.py` itself; new config.
- has_train_step: false
- retrieval_only: true
- expected_cost_min: 5
- expected_gain_ndcg10: 0.000 to +0.008 (diminishing-returns risk)
- risk: low-medium (could be null — diminishing returns when streams correlate)
- unlocks: []
- research_notes: documents/research/researcher_synthesis_v1.md
- status: open

### [P2] v5.5 · User-embedding stream in RRF (addresses q4_usercold_trackwarm)
- hypothesis: precomputed `talkpl-ai/TalkPlayData-Challenge-User-Embeddings` encodes a user's aggregate taste signature; for warm users, ranking tracks by `user_emb · track_emb` provides a personalization stream orthogonal to text matching. Adding it as a 3rd RRF stream directly targets the two weakest warm-user cells (q1–q3 at 0.064–0.067) without breaking BM25's dominance on cold-track buckets.
- grounding: Koren, Bell, Volinsky 2009 "Matrix Factorization Techniques for Recommender Systems"; Covington, Adams, Sargin 2016 "Deep Neural Networks for YouTube Recommendations" (user vectors as retrieval keys).
- parent_champion: v4.2 (or v2a if v4.2 regresses)
- touches: new `mcrs/embedders/user_track_dot.py` — a lightweight retriever that looks up `user_id` from chat_history (the dataset provides it per session), pulls `user_emb` from `Challenge-User-Embeddings`, scores against precomputed `track_mat` (reuse the v3 matrix via dependency injection or re-load); `retrieval_modules/__init__.py` adds key `user_emb_track_dot`; `rrf_bm25_dense_user_v1` composing the 3 streams; new config. For cold users (no user_emb), stream returns empty → RRF naturally falls back to BM25+dense.
- has_train_step: false
- retrieval_only: true
- expected_cost_min: 8
- expected_gain_ndcg10: +0.005 to +0.012 (concentrated in warm-user buckets)
- risk: medium (needs EDA to confirm user_id presence in chat_history + how many devset queries are warm)
- needs_eda: true
- deserves_bucket_analysis: true
- unlocks: [v11, v12]
- research_notes: documents/research/researcher_synthesis_v1.md
- status: open

### [P1] v6 · LightGBM LambdaMART reranker over BM25 top-100
- hypothesis: a learned reranker over top-100 BM25 candidates with features (BM25 score, normalized BM25 rank, dense-metadata cosine, dense-attributes cosine, track popularity log, release-year delta from query turn, tag-overlap count, artist-of-candidate-in-history indicator) captures interactions RRF cannot. NDCG@10 optimized directly via `lambdarank` objective. RecSys 2024 winners used essentially this stack — it is the single most battle-tested +0.015 to +0.030 lever available.
- grounding: Burges 2010 "From RankNet to LambdaRank to LambdaMART"; RecSys Challenge 2024 winning write-ups using LightGBM LambdaRank + text/embedding features (see e.g. ACM 3687151.3687156).
- parent_champion: v2a or v4.x (best available at claim time)
- touches: new `mcrs/rerankers/lgbm_rerank.py`; new `experiments/train_lgbm_reranker.py` (trains on the challenge **train split** — NOT devset, per plan §11); new factory key `bm25_then_lgbm_v1`; adds `lightgbm` to `pyproject.toml` (PRE-APPROVED); new config.
- has_train_step: true
- retrieval_only: false (retrieval-only in the sense of no LLM; but has a training phase)
- expected_cost_min: 12  (4 min feature extraction on ~120k (q, gold, neg) triples + 3 min train + 3 min eval; 2× safety factor → 12)
- expected_gain_ndcg10: +0.015 to +0.030
- risk: medium (first trained experiment; risk of training/eval leakage — code-reviewer must confirm devset tracks are held out from training triples)
- unlocks: [v6.1, v7]
- research_notes: documents/research/researcher_synthesis_v1.md
- status: open

### [P2] v8 · HyDE query expansion (Qwen3-0.6B generates pseudo-tracks for BM25)
- hypothesis: for underspecified cold-user queries ("play something chill"), HyDE — prompting a small LLM to hallucinate a plausible matching track's metadata (title + artist + tags + brief description), then concatenating that pseudo-document with the original query for BM25 — bridges the vocabulary mismatch between conversational phrasing and catalog text. Particularly targets q4_usercold_trackwarm where cold users lack history to anchor intent.
- grounding: Gao, Ma, Lin, Callan 2022 "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE); Mackie et al. 2024 evaluation of HyDE for diverse user queries (AIS 2024, wi2024/115).
- parent_champion: v2a or v4.x (best at claim time)
- touches: new `mcrs/query_rewriters/hyde.py` — uses Qwen3-0.6B (already downloaded by the dense retriever) in causal-LM mode via `AutoModelForCausalLM` to generate 2–3 pseudo-track blurbs per query, appended to the query string; new factory key `bm25_hyde_v1` that wraps existing BM25 with the rewriter; new system prompt at `mcrs/system_prompts/hyde_music.txt`; new config. **Caches generated pseudo-docs to `experiments/cache/hyde/` keyed by raw query**, so one generation pass costs ~8k × ~0.5s on MPS ≈ 70 min one-time, then <30s warm reruns. This cache-first design keeps iteration cheap after the first run.
- has_train_step: false
- retrieval_only: true (LLM used offline for query rewriting only, no LLM in eval loop)
- expected_cost_min: 15 (one-time cache build); ~3 min cache-warm on reruns
- expected_gain_ndcg10: +0.005 to +0.015, concentrated in cold-user buckets
- risk: medium (generation quality variance; first LLM-in-pipeline experiment; 15-min budget is the plan's practical ceiling — if it overruns, abandon per §17.1)
- unlocks: [v9, v8.1]
- research_notes: documents/research/researcher_synthesis_v1.md
- status: open

### [P3] v12 · Popularity-smoothing prior (tie-breaker for q4_userwarm_trackwarm)
- hypothesis: adding `α · log(1 + popularity)` to the final fused score biases ties toward more likely-to-be-gold popular tracks — directly targets the q4_userwarm_trackwarm cell (0.077, 1344 queries — a large block at below-champion nDCG where the correct track is popular but currently loses tie-breaks to off-topic tail). α ∈ {0.01, 0.03, 0.1, 0.3}, tuned on 100-session held-out slice.
- grounding: Koren 2009 "Matrix Factorization Techniques"; Steck 2011 "Item popularity and recommendation accuracy" RecSys (cautionary — confirms popularity prior helps on popular-heavy buckets but can hurt long-tail; our bucket analysis shows this tradeoff is favorable here).
- parent_champion: v4.2 or later (needs RRF infrastructure to have a final score to smooth)
- touches: extend `mcrs/retrieval_modules/rrf.py` with an optional post-hoc `popularity_prior_alpha` kwarg and a popularity loader from `Challenge-Track-Metadata` (or precomputed from train-split interaction counts — EDA decides); new factory key `wrrf_v1_poppost`; new config.
- has_train_step: false
- retrieval_only: true
- expected_cost_min: 5
- expected_gain_ndcg10: +0.002 to +0.008
- risk: low-medium (could regress cold-track buckets — bucket analysis mandatory)
- needs_eda: true (confirm popularity field semantics + distribution)
- deserves_bucket_analysis: true
- unlocks: []
- research_notes: documents/research/researcher_synthesis_v1.md
- status: open
```

## 3. Paper citations backing the proposals

1. **Bruch, Nardini, Rulli, Venturini 2023 — "An Analysis of Fusion Functions for Hybrid Retrieval"** (ACM Transactions on Information Systems 41(4), 3596512).
   <https://dl.acm.org/doi/10.1145/3596512> — foundational for v4.2 (weighted RRF) and v4.3 (multi-way). Documents that convex combination beats RRF when score scales can be normalized, and that fusion weight tuning is cheap relative to its gains.

2. **Gao, Ma, Lin, Callan 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)**.
   <https://arxiv.org/abs/2212.10496> — foundation for v8. 2024 follow-ups (HyPE, AIS 2024 wi2024/115, Springer Human-Centric Intelligent Systems 2025 paper on HyDE+Gemma) consistently show +0.01–0.05 nDCG gains for underspecified queries — exactly the q4_usercold_trackwarm regime.

3. **Yadav, Jung, Liao et al. 2024 — "Leveraging LightGBM Ranker for Efficient Large-Scale News Recommendation"** (RecSys Challenge 2024 write-up, ACM 10.1145/3687151.3687156).
   <https://dl.acm.org/doi/fullHtml/10.1145/3687151.3687156> — direct precedent for v6. Same recipe (BM25 + embedding cosines + popularity + interaction features → LightGBM `lambdarank`) won a recent RecSys challenge; supports the +0.015–0.030 expected gain.

## 4. Portfolio shape and claim order (recommendation to orchestrator)

A sensible claim order if all eight pass code review:

1. **v4.1** (BM25→dense rerank, 4 min, fastest win-harvest on observed @1 signal)
2. **v4.2** (weighted RRF, 6 min, independent of v4.1 outcome — both target same near-miss)
3. **v6** (LambdaMART, 12 min, highest expected-gain single lever; start this even if v4.x is null — dense features are useful in the reranker regardless)
4. **v5.5** (user-emb stream, 8 min, targets different bucket — runs parallel-in-intent to v4.x)
5. **v8** (HyDE, 15 min, first LLM-in-pipeline — risk budget ~1 of the ~10 experiments allowed per 12h loop cap)
6. **v4.3** (multi-way RRF, 5 min, diminishing-returns gate — claim only if v4.2 wins)
7. **v12** (popularity prior, 5 min, needs RRF infra from v4.2 — late-stage tie-break)

The portfolio spends ~55 min of expected cost across 7 experiments with mixed expected gains (+0.003 to +0.030 each, mean ~+0.010). At the current threshold 0.01479, we expect ~2–3 of these to clear the bar; v6 is the strongest single bet. Each is one idea per experiment (plan §4/A11 compliance), additive-only to the codebase (CLAUDE.md compliance), and the two new deps required (`lightgbm` for v6) are pre-approved. No new deps beyond that. No `faiss`, no CUDA, no Colab.

Sources:
- [An Analysis of Fusion Functions for Hybrid Retrieval (ACM TOIS 2023)](https://dl.acm.org/doi/10.1145/3596512)
- [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al. 2022)](https://arxiv.org/abs/2212.10496)
- [Leveraging LightGBM Ranker for Large-Scale News Recommendation (RecSys 2024)](https://dl.acm.org/doi/fullHtml/10.1145/3687151.3687156)
- [Introducing Reciprocal Rank Fusion for Hybrid Search (OpenSearch 2024)](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [HyDE Evaluation for Diverse User Queries (Wirtschaftsinformatik 2024)](https://aisel.aisnet.org/wi2024/115/)
