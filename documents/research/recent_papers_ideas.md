# Recent Papers — Ideas & Source of Truth

**Purpose**: Reference for designing a new music CRS model for RecSys Challenge 2026.
**Task**: retrieve top-20 tracks from 50K catalog → LLM generates conversational response.
**Current dev nDCG@10**: ~0.08 (Llama-1B + BM25 baseline).
**PDFs**: all papers below are in this folder (`documents/research/<Name>_<arxiv_id>.pdf`).

---

## TL;DR — Decision Map

| Goal | Top candidate | Why |
|---|---|---|
| Cheap retrieval win | **CMQR** | Multi-query rewrite fused into existing RRF; +1–6 MRR; drop-in |
| Add a reranker stage | **ProRank** or **Rank-R1** | Fills our empty `rerankers/` module; SLM-friendly |
| Fine-tune Qwen for this task | **Rank-GRPO** | Exact task shape (LLM-CRS + RL); rank-level credit |
| End-to-end generative model | **Text2Tracks** + **LIGER** | Music domain analogue + hybrid fallback |
| Fix cold-start / missing metadata | **MARec** + **LLM-ESR** | Principled imputation via alignment |
| Conversational preference tracking | **RA-Rec** or **PEBOL** | Structured state carries across turns |
| Self-improving loop | **Self-Rewarding LMs** + **Judging-Judges** (calibration) | Free preference signal from Qwen-as-Judge |

---

## 1. Retrieval

### CMQR — Multi-Query Rewriting for Conversational Passage Retrieval *(SIGIR 2024)*
- **Link**: https://arxiv.org/abs/2406.18960
- **Core idea**: Generate *multiple* rewrites of the conversational query, fuse at retrieval time (both sparse + dense).
- **Method**: LLM emits N rewrites in one call → each queries BM25 + dense → fuse via RRF.
- **Result**: +1.06 to +6.31 MRR absolute on QReCC.
- **Our fit**: Plugs into our empty `query_rewriters/` module and existing RRF. Highest ROI per engineering hour.

### Mistral-SPLADE — LLMs for Learned Sparse Retrieval *(2024)*
- **Link**: https://arxiv.org/abs/2408.11119
- **Core idea**: Use a decoder-only LLM (Mistral) to produce SPLADE-style expanded sparse vectors → leads BEIR.
- **Our fit**: Natural upgrade from vanilla BM25; stays sparse/invertible (fast for 50K), composes with RRF.

### TalkPlay — Multimodal Music Rec with LLMs *(arXiv 2025, talkpl-ai)*
- **Link**: https://arxiv.org/abs/2502.13713
- **Core idea**: Tokenize tracks with audio + lyrics + tags + playlist co-occurrence; expand LLM vocab with track tokens.
- **Our fit**: Same dataset family as our challenge. Even without full token-gen, the multimodal BM25 fields (lyrics + tags + playlist co-occurrence) are directly transferable.

### ReFICR — Retrieval Potential of LLMs in CRS *(RecSys 2024)*
- **Link**: https://dl.acm.org/doi/10.1145/3640457.3688146
- **Core idea**: One lightweight LLM instruction-tuned on both retrieval and generation subtasks.
- **Our fit**: Could replace BM25+dense retrieval leg with a Qwen fine-tuned for retrieval instructions.

### RA-Rec — Prompt-Based Semi-Structured State Tracking *(SIGIR 2024)*
- **Link**: https://arxiv.org/abs/2406.00033
- **Core idea**: LLM maintains a JSON preference state across turns; state doubles as structured query expansion.
- **Our fit**: Directly tackles conversational context loss; JSON state feeds both BM25 (as keywords) and dense (as text).

### PEBOL — Bayesian Preference Elicitation w/ NLI *(RecSys 2024)*
- **Link**: https://arxiv.org/abs/2405.00981
- **Core idea**: Score NLI between user utterances and item descriptions → Thompson Sampling / UCB for next query.
- **Result**: MRR@10 0.27 vs 0.17 for monolithic LLM after 10 turns.
- **Our fit**: NLI score between chat history and track tags/lyrics = extra fusion signal for RRF.

---

## 2. Reranking *(our `rerankers/` module is currently empty)*

### ProRank — Prompt Warmup RL for SLM Rerankers *(2025, Mixedbread)*
- **Link**: https://arxiv.org/abs/2506.03487
- **Core idea**: 2-stage GRPO warmup → last-token logit-diff scoring; a 0.5B SLM beats 32B rerankers on BEIR.
- **Our fit**: Add a tiny (0.5B–1B) reranker between BM25 and generator with negligible latency. Highest ROI-per-GPU-hour on this list.

### FIRST — Single-Token Listwise Reranking *(EMNLP 2024)*
- **Link**: https://arxiv.org/abs/2406.15657
- **Code**: `castorini/first_mistral` on HF
- **Core idea**: Use only the first-token logit over candidate-ID sequence → 50% latency cut vs RankZephyr.
- **Our fit**: Critical if we run an LLM reranker over top-100 in Colab time budgets.

### RankZephyr — Open Listwise Reranker *(2023, foundational)*
- **Link**: https://arxiv.org/abs/2312.02724
- **Code**: RankLLM toolkit (castorini/rank_llm)
- **Core idea**: 7B open-source listwise reranker that matches/beats RankGPT-4 zero-shot.
- **Our fit**: Bolt on top of top-20 BM25+dense+RRF output. nDCG@10 is bottlenecked by retrieval ordering — listwise rerank is the biggest lever.

### Rank-R1 — GRPO-Trained LLM Reranker *(2025)*
- **Link**: https://arxiv.org/abs/2503.06034
- **Code**: https://github.com/ielab/llm-rankers
- **Core idea**: GRPO on Qwen2.5-7B with sparse relevance rewards (no reasoning supervision).
- **Result**: 18% of SFT data needed; 14B variant beats zero-shot GPT-4 on BRIGHT.
- **Our fit**: Exact reranker recipe for our Qwen-7B configs. Replaces stub `sequential_rerank` with a trained one.

---

## 3. Generative Recsys / Semantic IDs

### Text2Tracks — Prompt-Based Music Rec via Generative Retrieval *(Spotify 2025)*
- **Link**: https://arxiv.org/abs/2503.24193
- **Core idea**: Decoder emits track IDs; compares Semantic-ID (RQ-VAE over CF embeddings) vs title decoding.
- **Result**: SIDs from CF beat title decoding by +48% Hits@10 and closest baseline by +127%.
- **Our fit**: **Near-exact analogue of our task.** Plug our track embeddings into RQ-VAE, fine-tune Qwen to emit SID sequences constrained to catalog. Strongest blueprint.

### GRID — Generative Recsys with Semantic IDs: Practitioner's Handbook *(CIKM 2025, Snap)*
- **Link**: https://arxiv.org/abs/2507.22224
- **Code**: https://github.com/snap-research/GRID
- **Core idea**: Unified benchmark & framework; ablates RQ-VAE vs RQ-KMeans, codebook size, collision, beam search, T5 vs decoder-only.
- **Catalog range**: 10K–100K (our 50K sits right in sweet spot).
- **Our fit**: **If we commit to SIDs, start here.** Swap our track embeddings into GRID defaults → saves weeks.

### LIGER — Unifying Generative + Dense Retrieval *(Meta FAIR 2024)*
- **Link**: https://arxiv.org/abs/2411.18814
- **Code**: https://github.com/facebookresearch/liger
- **Core idea**: Generate K candidates → dense retriever expands cold-start coverage + reranks.
- **Result**: Hybrid beats both TIGER and dense SASRec.
- **Our fit**: Keeps BM25/dense as safety net while layering SID-gen on top. Fits ensemble-first mindset.

### LETTER — Learnable Item Tokenization *(CIKM 2024)*
- **Link**: https://arxiv.org/abs/2405.07314
- **Core idea**: RQ-VAE + contrastive collaborative alignment + diversity loss → SIDs reflect text semantics AND CF co-occurrence.
- **Our fit**: Drop-in upgrade from plain TIGER/Text2Tracks SIDs. Music has strong co-listen signal that pure-text RQ-VAE misses.

### LC-Rec — Align LLM w/ Collaborative Semantics *(ICDE 2024)*
- **Link**: https://arxiv.org/abs/2311.09049
- **Code**: https://github.com/RUCAIBox/LC-Rec
- **Core idea**: Sinkhorn-uniform VQ prevents codebook collapse; alignment tasks teach LLM to reason over SIDs.
- **Our fit**: Canonical recipe for converting LLaMA/Qwen into SID-emitting recommender. Reference implementation.

### Joint Search+Rec SIDs *(RecSys 2025, Spotify)*
- **Link**: https://arxiv.org/abs/2508.10478
- **Core idea**: Search-tuned vs rec-tuned embeddings produce conflicting SIDs; bi-encoder fine-tuned on both tasks gives Pareto-optimal SIDs.
- **Our fit**: Our task mixes search-like asks and rec-like asks — this tells us how to tokenize correctly the first time.

### Bridging Search & Recommendation in Generative Retrieval *(RecSys 2024, Spotify)*
- **Link**: https://arxiv.org/abs/2410.16823
- **Core idea**: Jointly train one generative retrieval model for both tasks → shared reps regularize popularity.
- **Our fit**: Design reference for treating CRS as unified search+rec generative model.

### IDGenRec — Textual ID Learning *(SIGIR 2024)*
- **Link**: https://arxiv.org/abs/2403.19021
- **Core idea**: Items become short natural-language textual IDs (not numeric codebooks).
- **Our fit**: Alternative if we want one LLM to both pick tracks AND write justifications. Lighter fine-tuning.

### TIGER *(NeurIPS 2023, foundational)*
- **Link**: https://arxiv.org/abs/2305.05065
- **Core idea**: RQ-VAE over content embeddings → T5 autoregressive SID decoding with beam search over catalog trie.
- **Our fit**: Read first to understand every paper above it. Every SID paper defines itself against TIGER.

---

## 4. Post-Training & RL Alignment

### Rank-GRPO — RL for LLM-CRS *(ICLR 2026, Netflix)*
- **Link**: https://arxiv.org/abs/2510.20150
- **Core idea**: 2-stage pipeline — "Remap-Reflect-Adjust" BC warmup → Rank-GRPO with *rank-position* credit assignment (not token/sequence), geometric-mean importance ratio for stability.
- **Result**: Beats vanilla GRPO/PPO on Reddit-v2 (Recall, NDCG).
- **Our fit**: **Most on-point paper in the whole list.** Exactly our setting: LLM-CRS, out-of-catalog items, ranking quality degrading toward end of list. Plugs directly on our qwen3b_bm25_* configs.

### Rec-R1 — RL with Fixed Black-Box Recommender *(2025)*
- **Link**: https://arxiv.org/abs/2503.24289
- **Core idea**: LLM rewriter/generator optimized against fixed BM25 via GRPO; rule-based NDCG/Recall rewards; no reward model.
- **Our fit**: Cleanest way to turn current BM25+Qwen into RL-trained query rewriter without GPT-4 distillation. Closes the loop.

### S-DPO — Softmax DPO for Recommendation *(NeurIPS 2024)*
- **Link**: https://arxiv.org/abs/2406.09215
- **Code**: https://github.com/chenyuxin1999/S-DPO
- **Core idea**: DPO with 1-positive-N-negatives via Plackett-Luce / softmax formulation → exactly recsys shape.
- **Result**: Beats DPO/BPR on three rec datasets; mines hard negatives implicitly.
- **Our fit**: Convert dev interactions into (query, positive, hard-BM25-negatives) tuples → post-train qwen3b_lora_bm25.

### KTO — Prospect-Theoretic Alignment *(ICML 2024)*
- **Link**: https://arxiv.org/abs/2402.01306
- **Core idea**: Binary desirable/undesirable signal per example — no paired preferences.
- **Our fit**: Every dialog has "GT tracks" (desirable) and "not GT" (undesirable). Cheap first post-training pass before investing in S-DPO/Rank-GRPO.

### OPO / DRPO — DPO with Differentiable NDCG *(2024)*
- **Links**: https://arxiv.org/abs/2410.04346 (OPO), https://arxiv.org/abs/2410.18127 (DRPO)
- **Core idea**: Replace DPO's binary loss with a differentiable NDCG surrogate over an ordered list — loss = the metric we're scored on.
- **Our fit**: Drop-in for S-DPO if we have ordinal relevance (we do from dev set).

### Self-Rewarding Language Models *(ICML 2024, Meta)*
- **Link**: https://arxiv.org/abs/2401.10020
- **Core idea**: LLM scores its own candidate responses → those scores become preference pairs for next DPO round. Iterative.
- **Our fit**: Free preference signal from Qwen-as-Judge. This is what `qwen7b_bm25_fewshot_selfjudge` configs were gesturing at.

### iLoRA — Instance-wise LoRA MoE *(NeurIPS 2024)*
- **Link**: https://arxiv.org/abs/2408.10159
- **Core idea**: Tiny MoE gate over several LoRA experts, routed by sequence representation (<1% extra params).
- **Result**: +11.4% HR vs basic LoRA.
- **Our fit**: Natural upgrade if single qwen3b_lora adapter plateaus. Experts can specialize by persona cluster.

---

## 5. Cold-Start & Missing Metadata

### MARec — Metadata Alignment for Cold-Start *(RecSys 2024)*
- **Link**: https://arxiv.org/abs/2404.13298
- **Core idea**: Closed-form MF/autoencoder augmentation aligns metadata (incl. LLM embeddings) with interaction matrix.
- **Result**: +8.4–53.8% over SOTA on cold-start; +46.8–105.5% extra with LLM embeddings.
- **Our fit**: The *principled* version of our artist→category→global-mean imputation. Plugs in as a cold-item specialist.

### LLM-ESR — LLM Enhancement for Long-Tailed Sequential Rec *(NeurIPS 2024 Spotlight)*
- **Link**: https://arxiv.org/abs/2405.20646
- **Core idea**: Dual-view — frozen LLM semantic embeddings + collaborative sequential signal + retrieval-augmented self-distillation.
- **Our fit**: Lowest-risk way to use our Challenge-Track-Embeddings without per-item LLM retraining. Freeze LLM, cache embeddings, fuse.

### LLM Prior for Cold-Start Items *(2024)*
- **Link**: https://arxiv.org/abs/2411.09065
- **Core idea**: LLM item-item similarity as Bayesian regularizer on any recommender's training objective.
- **Our fit**: Generic bolt-on for dense retrievers / final aggregator. Complements MARec (MARec aligns embeddings; this regularizes training).

---

## 6. LLM-as-Judge (caution / diagnostic)

### Judging the Judges — Position Bias Study *(IJCNLP 2025)*
- **Link**: https://arxiv.org/abs/2406.07791
- **Core idea**: Three metrics — repetition stability, position consistency, preference fairness. Quantifies how position bias scales.
- **Our fit**: **Run before trusting any Qwen-7B self-judge pipeline.** Apply swap-and-average calibration. Cheap insurance against wasted training.

---

## 7. Cross-Cutting Themes & Conclusions

1. **Generative wins are scale-dependent.** HSTU-style gains require billion-user scale. At 50K catalog we'd underperform a strong hybrid. → Favor hybrid designs (LIGER pattern) over pure generative replacement.
2. **SIDs need to encode CF signal, not just text.** LETTER > TIGER for our domain. Co-listens matter for music.
3. **LLM-as-Judge is biased.** Always calibrate (Judging-Judges) before building self-play loops.
4. **GRPO > DPO for list-shaped tasks.** DPO is pairwise; Rank-GRPO and Rank-R1 use rank-level signal.
5. **Ensemble-first still wins.** LIGER pattern (generative + dense hybrid), MoE-LoRA specialization (iLoRA), and modular aggregator stay relevant even with semantic IDs.

---

## 8. Suggested Build Order (EV-ranked for dev nDCG@10 = 0.08)

1. **CMQR multi-query rewrite** — ~1 day, fuses into existing RRF. Expected lift: +0.01–0.03.
2. **Listwise reranker stage** (RankZephyr via castorini/rank_llm, or ProRank for SLM). Fills the empty `rerankers/` module. Expected lift: +0.02–0.04.
3. **KTO fine-tune** on (query, GT-track / not-GT-track) — cheap, binary, LoRA-friendly.
4. **RA-Rec or PEBOL state tracking** — if dialog context is hurting us.
5. **Text2Tracks generative specialist** — larger swing; follow GRID defaults; layer as additional specialist in ensemble, not as replacement.
6. **Rank-GRPO post-training** — if reranker stage alone plateaus. Most aligned with our actual objective.

---

## Appendix: File Manifest

All PDFs under `documents/research/` (git-ignored; local only).

| Paper | File |
|---|---|
| Bridging-Search-Rec | `Bridging-Search-Rec_2410.16823.pdf` |
| CMQR | `CMQR_2406.18960.pdf` |
| DRPO | `DRPO_2410.18127.pdf` |
| FIRST | `FIRST_2406.15657.pdf` |
| GRID | `GRID_2507.22224.pdf` |
| IDGenRec | `IDGenRec_2403.19021.pdf` |
| iLoRA | `iLoRA_2408.10159.pdf` |
| Joint-SIDs | `Joint-SIDs_2508.10478.pdf` |
| Judging-Judges | `Judging-Judges_2406.07791.pdf` |
| KTO | `KTO_2402.01306.pdf` |
| LC-Rec | `LC-Rec_2311.09049.pdf` |
| LETTER | `LETTER_2405.07314.pdf` |
| LIGER | `LIGER_2411.18814.pdf` |
| LLM-ESR | `LLM-ESR_2405.20646.pdf` |
| LLM-Prior-ColdStart | `LLM-Prior-ColdStart_2411.09065.pdf` |
| MARec | `MARec_2404.13298.pdf` |
| Mistral-SPLADE | `Mistral-SPLADE_2408.11119.pdf` |
| OPO | `OPO_2410.04346.pdf` |
| PEBOL | `PEBOL_2405.00981.pdf` |
| ProRank | `ProRank_2506.03487.pdf` |
| RA-Rec | `RA-Rec_2406.00033.pdf` |
| Rank-GRPO | `Rank-GRPO_2510.20150.pdf` |
| Rank-R1 | `Rank-R1_2503.06034.pdf` |
| RankZephyr | `RankZephyr_2312.02724.pdf` |
| Rec-R1 | `Rec-R1_2503.24289.pdf` |
| S-DPO | `S-DPO_2406.09215.pdf` |
| Self-Rewarding-LMs | `Self-Rewarding-LMs_2401.10020.pdf` |
| TalkPlay | `TalkPlay_2502.13713.pdf` |
| TalkPlay-Tools | `TalkPlay-Tools_2510.01698.pdf` |
| Text2Tracks | `Text2Tracks_2503.24193.pdf` |
| TIGER | `TIGER_2305.05065.pdf` |
