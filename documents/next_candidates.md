# Next experiment candidates — ranked queue

Regenerated (not patched) after every scored evaluation per `recsys_challenge_plan.md` §8.1. Top-5 here is the live on-deck set; bank below holds all viable candidates.

EV formula: `EV = expected_Δcomposite × p(success) / effort_hours`, then weighted by §8.3.3 (×1.5 if matches validated mechanism in `agent_memory.md`; calibration bias applied), then sorted with §8.4 cheap-first tie-break.

## On-deck top-5 (live — reorder after each new result)

| Rank | Task ID | Name | Axis | Expected Δ | Effort | Why now |
|---|---|---|---|---|---|---|
| 1 | C-1.1 | Persona prompt + AI-speak word-ban | response-prompt | +0.02–0.04 LLM (blind) | XS | Cheap-first; response-branch change; only observable on blind |
| 2 | R-2.4 | Rule-based query preprocessor (lowercase/stem/stopword/synonym) | query-rewrite | +0.005–0.015 nDCG | XS | Deterministic; observable on local retrieval |
| 3 | R-2.1 | CMQR multi-query rewrite (3 rewrites → RRF fuse) | query-rewrite | +0.01–0.03 nDCG | S | Highest retrieval EV-per-hour per paper lit |
| 4 | R-3.1 | BGE-reranker-v2-m3 cross-encoder top-40→20 | reranker | +0.005–0.015 nDCG | S | Fills empty rerankers/ module; OSS-first (FlagOpen/FlagEmbedding) |
| 5 | R-1.1 | BGE-M3 self-built multi-granularity retriever | retrieval | +0.02–0.05 nDCG | M | Promoted only after XS/S above; major retrieval upgrade |

## Candidate bank (sorted by EV, not cheap-first)

### Retrieval (Track R)
- R-0.1–0.6 — Reproduce existing configs (002 BM25+tags, 005 dense-Qwen3, 007 RRF, 009/010 wRRF, Bert) — prereq baseline catalog
- R-1.1 BGE-M3 self-built
- R-1.2 E5-mistral-7b (GPU/Colab)
- R-1.3 Stella-1.5B (local)
- R-1.4 cf-bpr user×track (warm users)
- R-1.5 CLAP audio-text via precomputed
- R-1.6 Popularity prior per conversation_goal
- R-2.1 CMQR multi-query
- R-2.2 HyDE-light (hypothetical track title)
- R-2.3 Doc2Query offline (Qwen-0.5B)
- R-2.4 Rule-based query preprocessor
- R-2.5 RA-Rec JSON state tracker (conditional)
- R-3.1 BGE-reranker-v2-m3
- R-3.2 Qwen-7B listwise
- R-3.3 FIRST first-token listwise
- R-3.4 ProRank SLM (Colab)
- R-3.5 Rank-R1 GRPO (Colab A100)
- R-4.1 LightGBM LambdaMART
- R-4.2 Cross-encoder fine-tune on train
- R-4.3 Query-rewriter RL
- R-5.1 RRF over all retrievers
- R-5.2 Weighted RRF grid search
- R-5.3 Learned stacker

### Response (Track C)
- C-0.1 Qwen3B + stock prompt anchor
- C-0.2 Multi-track prompt (top-3 not top-1)
- C-0.3 Custom [system, user] chat template
- C-1.1 Persona + word-ban
- C-1.2 Concrete citation directive
- C-1.3 Dynamic NN few-shot (MiniLM on CPU, Qwen on MPS)
- C-1.4 Semantic cache 50 curated pairs
- C-2.2 Chain-of-Verification
- C-2.3 Forced metadata grounding (outlines)
- C-2.4 JSON → prose template rotation
- C-2.5 Step-back prompting
- C-3.1 Qwen-7B response
- C-3.2 Qwen-14B response (Colab)
- (C-2.1 multi-candidate self-judge: DEFERRED — requires local judge)

### SID / End-to-end (Track G, Wave 4+)
- G-0 Paper sweep + note
- G-1.1 RQ-VAE cf-bpr
- G-1.2 RQ-VAE metadata-Qwen3
- G-1.3 LETTER
- G-2.1 SFT Qwen-3B on (query → SID)
- G-2.2 Constrained beam decoding
- G-2.3 LIGER hybrid
- G-3.1 KTO
- G-3.2 S-DPO
- G-3.3 Rank-GRPO

### Ensemble (Track E)
- E-1 Specialist registry
- E-2 Learned aggregator
- E-3 Prediction validator + packager (W0-5)
- E-4 Train+Dev retrain helper

## Update protocol (§8.1)

After every scored eval:
1. Filter out any candidates matching Dead-ends in `agent_memory.md`
2. Recompute EV with calibration bias
3. Apply validated-mechanism boost / disproven-mechanism demotion
4. Sort top-5 by §8.4 cheap-first within EV tiers
5. Add new candidates from failure-mode analysis (worst-5 rows)
6. Refill from `recsys_challenge_plan.md` §10c when bank < 10 entries
