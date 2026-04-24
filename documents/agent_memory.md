# Agent memory — cross-experiment learnings

Per `recsys_challenge_plan.md` §8.3. This file is the system's long-term memory: what works, what doesn't, how well our predictions hold up, what the Gemini judge actually rewards, and how local ↔ blind composites relate.

Read at session start alongside `benchmarks.md` + `next_candidates.md`. Update after every scored eval (local or blind).

Last updated: 2026-04-24 (Wave 0 seed).

---

## Focus discipline (applies to candidate selection)

- **Don't sweep every embedder / reranker / LLM.** Time is binding; response-side has bigger leverage than marginal dense upgrades. (User directive 2026-04-24.)
- **Default active retrievers** (pick from these first): BM25 (multi-field) — already in repo; precomputed `metadata-qwen3_embedding_0.6b` dense — free; BGE-M3 self-built (THE "new dense" — multi-granularity makes it stand in for 3-4 alternatives); `cf-bpr` user×track for warm users — free; CLAP audio-text — free.
- **Deferred — don't try unless BGE-M3 shows signal AND retrieval is the bottleneck**: E5-mistral-7B, Stella-1.5B, NV-Embed-v2, gte-Qwen2-7B. These are marginal at 47k catalog.
- **Three-question gate before adding a retrieval/rerank candidate**: (1) OSS plug-in ≤S effort? (2) mechanism differs meaningfully from what we already run? (3) prior signal suggests it moves the needle? If not all three → defer.

## Validated mechanisms (keep doing)

Rules we've confirmed work on this task + this data. Each entry: rule + supporting exp_ids + confidence (low | med | high).

- **wRRF(BM25 + dense-metadata-qwen3 + dense-lyrics-qwen3) beats all 3 branches standalone** — 020-devset scored composite_retrieval 0.1261 vs best standalone (010-wrrf-bm25-dense-lyrics-v1) at 0.0939. Confidence: high (one data point, but matches prior-branch wRRF pattern).
- **Qwen 2.5-1.5B on bf16+CUDA+SDPA produces coherent, non-truncated, non-apologetic English on dev** (LexDiv 0.32 on dev, 0.67 on blind). Confidence: med (sample-of-one on blind).
- **Colab A100 + batch 32 + shared-encoder singleton** completes the 8000-row devset in ~15-18 min end-to-end. Confidence: high.

---

## Dead ends (don't retry)

Distilled from prior-iteration Blind-A lessons (the prior v1–v23 runs on the other branch) and any new failures on this branch. Copy of `recsys_challenge_plan.md` §10e kept here so this file is self-contained.

- **MPS + float16 for Qwen batch gen** — NaN in `torch.multinomial`, degenerate `!!!!!` output. Always fp32 on MPS; fp16 only on CUDA. Confidence: high.
- **Stock `response_generation.txt` prompt** — "apologize on mismatch" directive → "I'm sorry" responses tanked LLM-judge. Confidence: high.
- **Stock `LLAMA_MODEL.batch_response_generation` chat template** — injects track as fake-assistant turn → turn-1 hallucinations ("I'm glad you enjoyed X"). Use custom `[system, user]` only. Confidence: high.
- **Qwen 7B + rigid few-shot** — overfits exemplar structure → homogeneous formulaic responses → LLM regression. Only use 7B with diverse-structure few-shot or zero-shot. Confidence: high.
- **Greedy query expansion without anti-repetition** — collapses into `smooth jazz trap r&b × 4`, wrecks BM25. Need `no_repeat_ngram_size=3 + repetition_penalty=1.2`. Confidence: high.
- **LoRA-on-train without v10 persona parity** — train-response style (chat-agent terse) is opposite of what the Gemini judge rewards ("well-read critic"). LoRA tanked LLM-judge by 1.45 raw points even with all other bugs fixed. Confidence: high. Prevention: any fine-tune must preserve the prompt-anchor; consider distilling from C-1.1 outputs, not raw train.
- **Manual word-replacement post-processing** — regex-replacing AI-speak words with synonyms in the Qwen output tanked LLM-judge by 0.25 vs the persona-prompt approach. The v5→v10 win was PERSONA, not the word ban. Confidence: high. Prevention: fix at prompt level, not output level.
- **Retrieval change that shifts submission[0] away from BM25's natural ordering** — prior iteration observed that the Gemini judge penalises non-BM25 orderings even when nDCG@20 improves. Decouple prompt-top-3 (BM25) from submission-20 (new mechanism) path. CAVEAT: this lesson comes from the prior branch and should be re-validated on fresh-model before being treated as load-bearing. Confidence: med pending re-validation.
- **Flat zip with filename other than `prediction.json`** — CodaBench rejects (server reads `/app/input/res/prediction.json`). Enforced by `scripts/validate_prediction.py`. Confidence: high.
- **MPS deadlock with two HF models** — e.g. sentence-transformers (MiniLM) + Qwen both claiming MPS → 85-min deadlock. Force secondary model to `device="cpu"` when Qwen is on MPS. Confidence: high.

---

## Open questions / hypotheses

Beliefs we haven't tested yet. Each has a "test by" exp ID.

- **H-1**: On fresh-model, does a persona prompt lift Gemini LLM-judge by ≥0.2? — test by: first shipped C-1.1 blind submission.
- **H-2**: Does the retrieval-→-LLM coupling (prior-branch dead end above) replicate on fresh-model? — test by: any shipped non-BM25-ordered retrieval stack.
- **H-3**: Is `composite_retrieval` (local) a reliable ordering signal for blind rankings? — test by: correlation after 3 paired (local, blind) points.

---

## Prediction calibration

Table tracking predicted Δcomposite vs actual. Reveals axis-level bias in our EV estimates.

| Axis | N predictions | Mean predicted Δ | Mean actual Δ | Bias (actual / predicted) | Notes |
|---|---|---|---|---|---|
| retrieval | 0 | — | — | — | — |
| query-rewrite | 0 | — | — | — | — |
| reranker | 0 | — | — | — | — |
| response-prompt | 0 | — | — | — | — |
| response-model | 0 | — | — | — | — |
| fusion | 0 | — | — | — | — |
| aggregator | 0 | — | — | — | — |
| train-obj | 0 | — | — | — | — |

### Noise floor σ (W0-8)

`σ` = seed-variance of `composite_retrieval` for stochastic configs. Measured at Wave 0:

- **BM25 + deterministic greedy generation**: σ ≈ 0.0 (BM25 is deterministic given fixed text; generation is deterministic with seed fixed; composite is fully reproducible). Verified at Wave 0 by inspection — same config → same score file bit-identical.
- **LLM-sampling configs** (nucleus / temp>0): σ to be measured per-config when first used.
- **Dense retrievers with stochastic batch ordering**: σ ≈ 0 (cosine sim is order-independent).
- **LoRA/SFT-trained models**: σ can be substantial across seeds; measure before using for shipping decisions.

**Current default σ (used for champion gating, per plan §2.3.2): 0.005.** This is a placeholder, used only to avoid shipping essentially-tied stacks. Revise the first time a stochastic config runs and actually produces per-seed variance.

### Known infrastructure quirks (2026-04-24)

- `scripts/local_eval.py::append_submissions_log_row` appends at file end, not inside the existing table. Keep `documents/submissions_log.md` with the markdown table at the BOTTOM (notes at top) so appended rows land contiguously. The file was reorganised to this shape at Wave-0 bootstrap.

---

## Judge behavior notes (Gemini-specific)

What Gemini appears to reward / penalise. Grown **only** from blind-response mining (J-A1): reading our own `(query, predicted_response, score)` triples archived in `documents/blind_responses_scored.md` after each blind submission. No local Gemini runs in current phase.

*(Seeded empty — first observations after Wave 3 blind submission scores.)*

### Hypothesis registry (J-A2)

Claimed rewards / penalties, pending blind validation:

- **H-4 (carried forward from prior branch)**: persona prompt + generic-AI-speak word-ban lifts Gemini LLM judge by ~+0.40 on a Qwen 1.5B–3B backbone. Test: exp 022 (same stack as 021, persona prompt swapped in). If LLM blind returns ≥ 3.45 → validated. If < 3.30 → prior-branch → fresh-model transfer is weaker than expected.
- **H-5**: stock `response_generation.txt` "apologize on mismatch" directive depresses LLM judge; removing it AND adding persona together is the efficient fix (not one without the other). Tested alongside H-4 in exp 022; disentangled by future ablation if needed.
- **H-6**: on this season's leaderboard (#1 nDCG@20 = 0.44 vs our 0.19), retrieval upgrades (cross-encoder rerank, LambdaMART) have higher composite EV than further LLM upgrades. Test: exp 024.

---

## Cross-correlations

"When axis X moves, axis Y tends to move in direction Z."

*(Seeded empty — populated after ≥3 scored experiments on disjoint axes.)*

---

## Local → Blind calibration (§2.5.5)

Paired `(composite_retrieval_local, composite_blind)` points — one per scored blind submission. After 3 points, compute offset + Pearson ρ; after 5, decide whether to trust local for shipping decisions.

| # | Date | exp_id | composite_retrieval_local | composite_blind | Δ | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-04-24 | 020 (dev) → 021 (blind), same stack | 0.1261 | 0.33 | +0.20 | Blind composite includes the 0.30-weighted LLM term (dev composite_retrieval doesn't). Dev composite_projected with LLM assumed 3.15 would be 0.1261 + 0.30·(3.15-1)/4 = 0.1261 + 0.1613 = 0.2874 — still low vs blind 0.33 (0.042 gap, explained by blind's 2× higher nDCG@20 and 2× higher LexDiv due to single-turn vs multi-turn). Rule: blind scores are generally HIGHER than composite_projected on the same stack; the dev split is harder. |

---

## Operational state

- **Machine wake-lock**: `caffeinate -dimsu` may be running in background (PID logged in `/tmp/caffeinate.log`). Check `pgrep -lf "caffeinate -dimsu"`; kill when no longer needed.
- **Venv**: `recsys26/bin/activate` at repo root. Python 3.10.16; torch 2.10; transformers 4.57.6.
- **Data**: all 6 HF datasets downloaded under `data/` — no fetch needed for inference.
- **Pre-existing score files**: `music-crs-evaluator/exp/scores/devset/` has `llama1b_bm25_devset.json`, `popularity.json`, `random.json`. Seeded into `benchmarks.md` and `submissions_log.md` as B-floor.
