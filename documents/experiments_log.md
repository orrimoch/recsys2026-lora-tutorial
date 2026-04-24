# Experiments log — narrative

Append-only. One entry per scored experiment (dev or blind). Schema per `recsys_challenge_plan.md` §4.3.

New entries go at the TOP (newest first) so the most recent work is visible without scrolling.

---

### Exp 021-two-step-wrrf-lyrics-qwen15b-blindsetA — First Blind-A ship, stock response prompt — 2026-04-24 18:00

- **Hypothesis**: ship Wave 2's dev-champion (020) stack (wRRF retrieval + Qwen 2.5-1.5B response) to Blind-A to (a) get the first Gemini LLM-judge data point on the fresh-model branch, (b) establish a local→blind calibration anchor, (c) enter the leaderboard so subsequent experiments can be ranked on real composite deltas.
- **Axis**: first blind — no delta against a prior blind baseline; all 4 blind metric axes are new data points.
- **Config**: `music-crs-baselines/config/021-two-step-wrrf-lyrics-qwen15b-blindsetA.yaml`
- **Code**: git sha `f74baed` (fresh-model branch). Identical retrieval+LM stack to 020; only the `test_dataset_name` differs. Response prompt: **stock `system_prompts/response_generation.txt`** (contains the "apologize on mismatch" directive — a known dead-end from prior branch).
- **Shipped result** (Blind-A, 80 rows, Gemini-scored):
  - **Composite 0.33**, rank **9 / 9** (CodaBench id 695336)
  - nDCG@20 = 0.19
  - CatDiv = 0.03 (blind-set artifact — 80 rows draw from a narrow catalog slice, affects everyone identically)
  - LexDiv = 0.67
  - LLM judge = **3.15** (of 5)
  - Gap to #1 (Chris Deotte, composite 0.57): **Δ +0.24 composite**; decomposes as +0.125 from nDCG@20 (+0.25 raw), +0.015 from LexDiv (+0.15 raw), +0.105 from LLM (+1.40 raw), 0 from CatDiv.
- **Local → Blind calibration point #1**:
  - Local (dev, 020): composite_retrieval 0.1261, nDCG@20 0.0995, LexDiv 0.3217
  - Blind (A, 021): composite 0.33, nDCG@20 0.19, LexDiv 0.67, LLM 3.15
  - Blind nDCG@20 is ~2× dev nDCG@20 (0.19 vs 0.0995) — this is a dataset artifact (80 shorter single-turn Blind-A queries vs 8000 multi-turn dev). Expected per prior-branch calibration.
  - Blind LexDiv is ~2× dev LexDiv — same single-turn vs multi-turn effect (multi-turn responses dilute bigram novelty).
- **Lessons**:
  1. **Retrieval is the biggest gap vs #1, not LLM judge.** nDCG@20 contributes 52% of the 0.24 composite gap; LLM judge contributes 44%; LexDiv 6%. Prior-branch intuition ("LLM dominates") is outdated — this season's leader has nDCG@20 = 0.44, more than 2× ours. Retrieval-side work has been under-invested relative to its new leverage.
  2. **LLM 3.15 on stock prompt + Qwen 1.5B is consistent with prior-branch priors.** Prior branch v3 (Qwen 3B + custom-prompt-no-persona) scored 2.70; v10 (3B + persona+word-ban) scored 3.25. Our 3.15 sits between them — Qwen 1.5B with a more neutral prompt is roughly equivalent to 3B with a basic custom prompt. The persona+word-ban axis hasn't been activated yet, leaving ~+0.40 LLM on the table (prior-branch calibrated).
  3. **Bigger model is NOT automatically the next move.** Prior branch v6 (Qwen 7B + rigid few-shot) regressed LLM by −0.45 vs v5 (3B). Model size only helps when the prompt permits variety. Rewrite prompt first, then revisit model size.
  4. **CatDiv 0.03 is noise.** Every top-9 team scored 0.03 — the 80-row Blind-A forces everyone into near-identical catalog slices. Stop optimising for CatDiv on blind; focus on the three moving axes.
- **Verdict**: shipped, **rank 9 / 9 — worst on leaderboard (of the 9 scored teams visible)**. Establishes the blind calibration anchor. Retained as the "stock-prompt blind baseline" for computing Δ on future prompt-engineering experiments.
- **Suggests next**:
  1. **Exp 022 — persona prompt + word-ban** (next experiment; prior branch +0.40 LLM = +0.03 composite). Near-free effort. Single axis change: same retrieval, same LM, different response prompt file. Ideal attribution test.
  2. **Exp 023 — Qwen 1.5B → 3B** with the persona prompt from 022. Only try AFTER 022 lands, to decouple prompt effect from model-size effect.
  3. **Exp 024 — retrieval lift via cross-encoder rerank** (BGE-reranker-v2-m3) on top of the wRRF-top-40 candidates → top-20 output. Targets the 52% nDCG@20 gap. Parallelisable with 022 since retrieval and prompt are independent axes.

### Exp 022-persona-qwen15b-blindsetA — Persona+word-ban prompt REGRESSED, H-4 falsified — 2026-04-24 18:22

- **Hypothesis (H-4)**: prior-branch persona+word-ban prompt that lifted Qwen 3B LLM judge by +0.40 (v5→v10) would similarly lift Qwen 1.5B by ~+0.40 on fresh-model. Predicted LLM 3.15 → 3.55, composite 0.33 → 0.36.
- **Axis**: response-prompt (single-axis change vs 021).
- **Shipped result** (Blind-A, 80 rows, Gemini):
  - composite **0.24** (**−0.09** vs 021)
  - nDCG@20 **0.14** (−0.05 vs 021 — unexpected; see lesson 4)
  - LexDiv **0.80** (+0.13 vs 021)
  - LLM **2.15** (**−1.00** vs 021 — catastrophic)
- **Lessons**:
  1. **H-4 FALSIFIED.** Prior-branch persona mechanism does NOT transfer to Qwen 1.5B. Swing was opposite of predicted (−1.00 vs +0.40).
  2. **Root cause hypothesis**: the persona prompt has 10+ explicit directives (persona line, 4-word ban, length rule, "never apologise", "never ask for recs", etc.). Qwen 1.5B has insufficient instruction-following capacity for long directive lists — the model obeys the stylistic axes (LexDiv +0.13 confirms vocabulary shifted) but at the cost of grounding/specificity, which Gemini's Personalization and Explanation Quality dimensions penalise hard. Prior branch used 3B, which has more capacity.
  3. **LexDiv is not a proxy for LLM quality.** LexDiv +0.13 came with LLM −1.00 — the two can move in OPPOSITE directions. Stop treating high LexDiv as "good-sign" for composite.
  4. **Retrieval nDCG@20 −0.05 on a response-only change is unexpected.** Most likely cause: bf16 encoder non-determinism across Colab A100 instances (different runtimes give slightly different cosine ties, shuffled top-K). On 80-row Blind-A this registers as ±0.05 noise. Treat as evaluation variance, not a pipeline bug.
- **Verdict**: **REJECTED**. Reverts to 021's stock prompt as the LLM baseline. Persona file retained under `system_prompts/response_generation_persona.txt` for possible future retry with Qwen 3B (where it might work as prior branch showed).
- **Suggests next**: pivot to retrieval (52% of gap). Exp 023 stacks (a) BGE-reranker-v2-m3 cross-encoder over wRRF top-40→20, (b) Qwen 3B, (c) `max_new_tokens` 64→192. Three orthogonal axes; each touches different metrics so partial attribution is possible post-score.

### Exp 023-rerank-qwen3b-blindsetA — Three-axis stack regressed; BGE-reranker is wrong tool for this task — 2026-04-24

- **Hypothesis**: stacked 3 orthogonal changes on 021 baseline — (1) BGE-reranker-v2-m3 over wRRF top-40→20, (2) Qwen 1.5B→3B, (3) `max_new_tokens` 64→192. Predicted composite 0.33 → 0.39–0.41.
- **Shipped result** (Blind-A): composite **0.19**, nDCG@20 **0.07** (−0.12 vs 021), LexDiv 0.61 (−0.06), LLM **2.20** (−0.95). Rank regression 9 → ~12+.
- **Post-mortem diagnosis** (via local probe + qualitative inspection of 023 JSON vs 022 JSON):
  - BGE-reranker-v2-m3 **code and model are correct.** Local probe: lofi-study query scored +2.4 vs lofi-match doc and −6.6 vs death-metal doc (correct directionality, no NaN). Not a code bug.
  - 023 top-20 tids had **10/20 median overlap with 022 top-20** on the same 80 queries. The 10 differing picks are semantically sensible (e.g., "Music That You Can Dance To" for a dance query) but NOT the tracks Blind-A's ground truth rewards.
  - **Root cause**: Blind-A ground truth rewards user preference / session-contextual picks (what the real user chose in their session continuation), NOT query-track semantic similarity. BM25+dense fusion happens to sit near that preference distribution; a generic cross-encoder pulls away from it.
  - This is the same lesson from prior-branch v20: "retrieval changes that shift submission ordering from BM25's natural ordering cost LLM-judge, even when nDCG@20 improves". Here BM25 also already optimizes the right ordering AND the reranker hurts both nDCG AND LLM (because reranker's bad top-1 is passed to the LM as `recommend_item`, poisoning the response).
- **Lessons**:
  1. **H-6 (retrieval has high EV) needs refinement.** Retrieval HAS headroom (leader nDCG@20 0.44 vs ours 0.19) but only on task-aware signals: collaborative-filtering, user demographics, popularity, goal-conditioned retrieval. Off-the-shelf semantic rerankers HURT because they're optimizing the wrong objective. **BGE-reranker shelved.**
  2. **Cascading failure mode**: bad reranker → bad top-1 → bad `recommend_item` → LM generates confident-but-wrong content → LLM judge tanks ALONG WITH nDCG. Retrieval and response axes are coupled via the top-1 citation path. Decouple: use different retrieval for prompt-top-1 vs the 20-list submission.
  3. **Three-axis stacking violated attribution discipline.** We can only say "the stack regressed"; can't isolate Qwen 3B or max_new_tokens contributions. Exp 024 reverts to single-axis discipline.
  4. **BGE-reranker is a zero-training dead-end for this task.** A fine-tuned reranker (trained on train-conversation (query, gold-music) pairs with `goal_progress_assessments` as graded relevance) might work, but that's R-4.2 (~1 day). LambdaMART with task-specific features is R-4.1 (also ~1 day). Neither is a quick fix.
- **Verdict**: **REJECTED.** Revert to 021 baseline. Reranker code retained (no bug) for potential future use with a task-aware reranker.
- **Suggests next**:
  1. **Exp 024** — isolate response-side: Qwen 3B + max_new=192, NO reranker. Tests if the LM upgrade + length fix alone lifts composite without retrieval noise.
  2. **Exp 025** (future) — if exp 024 doesn't lift LLM meaningfully, try Qwen 2.5-**7B** with stock prompt (prior branch regressed only with rigid few-shot; 7B + stock might work).
  3. **Exp 026** (future, L effort) — LambdaMART-LtR (R-4.1) for task-aware retrieval upgrade. Only worth ~1-day investment if exp 024/025 hit a retrieval-bound plateau.

### Exp 026-reward-rerank-qwen15b-blindsetA — H-9 falsified: train reward model ≠ Gemini — 2026-04-24

- **Hypothesis (H-9)**: a cross-encoder fine-tuned on train `goal_progress_assessments` (MOVES_TOWARD_GOAL vs DOES_NOT) would, when used as a response reranker picking best-of-K sampled responses, lift Gemini LLM judge by +0.1–0.3.
- **Mechanism**: Qwen 1.5B greedy → Qwen 1.5B samples K=3 responses at temperatures [0.3, 0.7, 1.0], MiniLM-L-6 cross-encoder (fine-tuned on ~95k train (context, response) pairs with BCE, 2 epochs on A100) scores each, ship argmax.
- **Reward model trained quality**: AP = **0.98** on held-out (session-level) val. Accuracy 93.2%, F1 0.93, Precision 0.94, Recall 0.92. Clean learned signal.
- **Shipped result** (Blind-A, 80 rows):
  - composite **0.27** (−0.06 vs 021)
  - nDCG@20 **0.14** (−0.05 — consistent with ±0.05 noise pattern confirmed in 022/024; retrieval stack unchanged)
  - CatDiv 0.03 (tied)
  - LexDiv **0.78** (+0.11 — K-sampling diversity worked as designed)
  - LLM **2.60** (−0.55 — big regression)
- **Lessons**:
  1. **H-9 FALSIFIED.** Near-perfect classifier of train-user preference (AP 0.98) produces responses that Gemini scores 0.55 WORSE than greedy Qwen 1.5B.
  2. **Train-user preference ≠ Gemini-judge preference.** Two independent confirmations now: prior-branch LoRA-on-train (−1.45 LLM), and this reward-model reranker (−0.55 LLM). Different mechanisms, same failure mode.
  3. **Gemini apparently rewards AI-speak-heavy greedy responses MORE than in-distribution user-satisfying ones.** Consistent with 022 (persona = too diverse/specific → failed), 024 (3B rambling → failed), and now 026 (train-user-aligned → failed). The Gemini scoring function favours the TEMPLATED response style; deviations in any direction hurt.
  4. **Blind-A ±0.05 nDCG@20 noise is now confirmed four times** (021=0.19, 022=0.14, 024=0.14, 026=0.14 on same wRRF). Retrieval noise floor is ±0.05 on 80-row set.
  5. **Path 2 (GRPO LoRA on the same reward signal) is dead.** If the reward model misaligns with Gemini, RL-amplifying its signal makes things worse, not better.
  6. **LexDiv +0.11 is a neutral gain on the surface** — sampling did add vocabulary diversity — but Gemini's LLM-judge weight dominates (0.30 × (2.6−1)/4 = 0.12) and the LLM drop buried any LexDiv composite benefit.
- **Verdict**: **REJECTED.** 021 remains champion (composite 0.33). Reward model shelved — not a Gemini alignment signal. All training-based generation-side directions using goal_progress_assessments as ground truth are now dead ends.
- **Suggests next**:
  1. **A1 LambdaMART retrieval reranker** — our Tier-A rank-3 candidate, now the clear next move. Retrieval-only (doesn't touch the response side where we keep losing), uses train gold as supervised labels, feature extractor already built (commit 48415fa). Target +0.03–0.08 nDCG@20 = composite 0.33 → 0.35–0.37.
  2. **Reward model is NOT wasted** — it becomes a feature INSIDE LambdaMART's feature vector (predicted "user would like this" score per candidate). That's a task-aware signal no leader seems to exploit.
  3. **Response-side: accept 021 greedy-stock as near-ceiling on Qwen 1.5B.** Stop trying to beat LLM judge via generation changes. Future ship-worthy LLM lifts must either (a) upgrade the retriever so the prompt has better metadata to cite, OR (b) distil from 021's own high-scoring responses (never from train data).

### Exp 027-wrrf-lgbm-qwen15b-blindsetA — A1 LambdaMART regressed; worst Blind-A ship; H-6 effectively falsified — 2026-04-24

- **Hypothesis (H-10, derived from H-6)**: LightGBM LambdaRank over 11 engineered features (wrrf_rank, cfbpr_score, pop_log, recency_years, tag_count, artist_in_query, goal_category/specificity, user_age_group/country/gender) trained on 2000 train sessions would lift Blind-A nDCG@20 by +0.03–0.08, composite 0.33 → 0.35–0.37.
- **Training signal**: val nDCG@20 = **0.6271** on train holdout. Feature gain split: cfbpr_score 41%, wrrf_rank 40%, user_country 8%, pop_log 5%, artist_in_query 3%, others 3%. Early-stopped at 6 rounds.
- **Pre-ship interpretation**: val 0.63 inflated by cf-bpr leakage (cf-bpr was trained on train user-track interactions; val users ≈ train users by session split). But wrrf_rank at 40% gain suggested genuine ranking signal existed alongside leakage. Predicted real Blind-A lift: +0.02–0.05 nDCG@20.
- **Shipped result** (Blind-A, 80 rows):
  - composite **0.23** (−0.10 vs 021) — worst Blind-A ship on fresh-model
  - nDCG@20 **0.11** (−0.08 — LambdaMART moved gold tracks OUT of top-20)
  - CatDiv 0.03 (tied)
  - LexDiv 0.77 (+0.10 — downstream effect of different top-1 tracks landing in prompts)
  - LLM **2.20** (−0.95 — cascading from bad top-1 → LM hallucinates about mis-picked track)
- **Lessons**:
  1. **H-10 falsified; H-6 effectively falsified.** Learned retrieval reranker with task-specific features (incl cf-bpr, user demographics, goal categoricals) FAILS on Blind-A despite 0.63 train-val nDCG@20. The trained signal doesn't transfer to the 80-row Blind-A distribution. cf-bpr leakage on train is a partial explanation but can't explain ALL the regression — wrrf_rank's 40% contribution presumably carried real signal yet still regressed.
  2. **Prior-branch v20 lesson reaffirmed on fresh-model**: any retrieval reshuffling away from wRRF's natural ordering costs both nDCG@20 AND LLM-judge (via the top-1 citation path). Three independent fresh-model confirmations now: BGE-reranker (023), reward-model (026 — response-side but similar cascade), LambdaMART (027).
  3. **Blind-A distribution specifics**: with only 80 rows and 58 users (25 warm / 33 cold for cf-bpr), the user signal is sparse. LGBM learned patterns on 14k train queries that don't transfer to this small + specific benchmark.
  4. **Inferred structural result**: wRRF + Qwen 1.5B + stock prompt is a LOCAL MAXIMUM for this specific Blind-A. Six consecutive experiments failed to beat it. Further gains require qualitatively different data (Blind-B when it releases), a different model entirely (LLaMA 70B, Mixtral, etc.), or accept the plateau.
  5. **cf-bpr as "honest signal on Blind-A" was too optimistic.** Even without the leakage path, cf-bpr's learned user-track preferences from train don't transfer to Blind-A gold. The Blind-A held-out gold is a different distribution from train gold.
- **Verdict**: **REJECTED.** Reverts to 021 champion. This is the 6th consecutive Blind-A ship to regress; we've mapped the ceiling.
- **Suggests next**:
  1. **STOP** further Blind-A submissions of zero-/light-training candidates. The plateau at composite 0.33 / rank 9 is real for this methodology.
  2. **Pivot to Blind-B infrastructure** — Blind-B releases 2026-06-15. Focus next 7 weeks on: (a) fine-tune BGE-reranker on train (R-4.2 — the only untried training-side retrieval), (b) LoRA distillation from 021 responses (E1 — the only training-side response approach that explicitly avoids the train-user/Gemini mismatch), (c) Qwen 2.5-7B / 14B on Colab A100 (larger model with surgical prompt rules, not exp 022's heavy persona).
  3. **If still experimenting on Blind-A**: accept the ±0.05 noise floor means single-slot experiments can't cleanly distinguish the ~±0.01 regressions we'd expect from any training-side tweak. Would need multi-seed submissions to average out noise, burning the submission budget.

### Exp template (copy when starting a new experiment)

```markdown
### Exp NNN-{track}-{change} — {one-line headline} — YYYY-MM-DD

- **Hypothesis**: what we expect to move and why (mechanism)
- **Axis**: retrieval | reranker | query-rewrite | response-prompt | response-model | train-obj | fusion | aggregator
- **Config**: path to YAML
- **Code**: path(s) to any new modules; git sha
- **Code origin**: {github.com/... | HF: ... | from scratch, reason=...} (per §7.4 open-source-first rule)
- **Smoke result** (5 rows): pass/fail + output sample
- **Full result** (split: dev | blindA):
  - nDCG@{1,10,20}, CatDiv, LexDiv, LLM-judge (blind only), composite_retrieval, composite_projected (local) / composite_blind
  - Δ vs B-floor: ...
  - Δ vs B-champ: ...
  - Δ vs B-target: ...
- **Lessons**: what we learned (positive OR negative)
- **Verdict**: new champion | shipped not champion | rejected | dead end | deferred
- **Suggests next**: 1-3 follow-up experiment IDs
```

### Exp 020-two-step-wrrf-lyrics-qwen15b-devset — First two-step stack: wRRF retrieval + Qwen 1.5B response — 2026-04-24

- **Hypothesis**: stacking Wave 1's best retrieval branch (wRRF of BM25 + dense-metadata-qwen3 + dense-lyrics-qwen3; nDCG@20=0.0994) with a Qwen 2.5-1.5B response generator should produce the first two-step submission that beats B-floor composite_retrieval (0.1042) — retrieval holds, LexDiv rises from 0 (retrieval-only) to ~0.25–0.35 (Qwen 1.5B's natural prose variety), giving a +0.02–0.03 composite lift.
- **Axis**: retrieval **+** response-model (first two-step — stacking two changes vs the retrieval-only Wave 1 anchor; acceptable per plan §9.3 "Wave 2 = strong first stack", but means future decoupling experiments needed to attribute gains).
- **Config**: `music-crs-baselines/config/020-two-step-wrrf-lyrics-qwen15b-devset.yaml`
- **Code**: git sha `5ae994e` (fresh-model branch)
- **Code origin**: existing `run_inference_devset.py` extended with `--device` and `--attn_implementation` CLI flags; retrieval via `wrrf_bm25_dense_lyrics_v1` factory key; dense retrievers share encoder + query cache (singleton added this wave).
- **Smoke result** (5 rows): smoke run writes 40 rows (= 5 sessions × 8 turns); sample row 0 response "I'm so glad you enjoyed Nirvana's 'Heart-Shaped Box'. This song is a perfect example of grunge rock from the 90s…" — schema-valid, non-empty, English, cites metadata. Passed.
- **Full result** (split: dev, mode: full; Colab A100, batch 16, SDPA):
  - nDCG@1=0.0164, nDCG@10=0.0784, **nDCG@20=0.0995**
  - catalog_diversity=0.4418, **lexical_diversity=0.3217**
  - **composite_retrieval=0.1261**
  - Δ vs B-floor: **+0.0219** composite_retrieval, +0.0180 nDCG@20, +0.0668 LexDiv
  - Δ vs B-champ: (no prior champion — this IS the first B-champ)
  - Δ vs B-target: −0.0569 composite_retrieval, −0.1105 nDCG@20 (leader nDCG@20 ≈ 0.21)
  - `pytest tests/test_wave2_integration.py -v` — 6/6 passed
  - local_eval decision: **PROMOTE to blind**
- **Lessons**:
  1. **Wave 1's wRRF retrieval branch holds under the two-step pipeline.** nDCG@20 = 0.0995 matches the Wave 1 standalone's 0.0994 — adding a response generator did not degrade retrieval. Validates the decoupling assumption.
  2. **Qwen 2.5-1.5B LexDiv = 0.3217 on dev.** 26% higher than the B-floor's Llama-1B 0.2549. This was the bulk of the +0.0219 composite gain (LexDiv weight 0.10 in composite_retrieval → 0.0067 of the 0.0219, with retrieval contributing the other 0.0090 via +0.0180 nDCG@20 × 0.50).
  3. **Decoupling hygiene debt:** this experiment stacks retrieval (vs B-floor) + response-LM (vs B-floor). The composite gain is attributable to the two axes combined but not to each individually. Future ablation (021 retrieval-only with Qwen 1.5B, or 022 wRRF + Llama 1B) would separate them. Deferred to iteration queue.
  4. **Colab-first workflow established.** Full 8000-row inference: ~18 min on A100 (M4 was projected ~7h; user forbade local two-step runs). Cell-4 of `colab/Run_Devset_Inference.ipynb` is the single edit point per experiment. Conventions captured in auto-memory `project_colab_inference_workflow.md`.
  5. **Speedups baked in this wave:** shared dense encoder singleton (−1.2 GB VRAM, one model load vs two), shared query cache (cold queries encoded once per process vs twice), auto CUDA/MPS/CPU detect with bf16 on accelerators, `--attn_implementation sdpa` default on Colab, `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- **Verdict**: **new B-champ (internal, local-only).** Wave 2 closed. First submission that's a candidate for Blind-A shipping (Wave 3).
- **Suggests next**:
  1. **Wave 3 / Exp 021 — Blind-A ship of this stack.** Same retrieval + response config on the Blind-A split via `run_inference_blindset.py`. Expected first Gemini LLM-judge score + nDCG@20 / LexDiv on blind. One submission slot.
  2. **Exp 022 — retrieval decoupling ablation (deferred).** wRRF + Llama 1B response (pinning LM to B-floor) on dev, to measure standalone retrieval contribution vs response.
  3. **Prompt engineering lift (Wave 3b / Exp 023).** Apply `feedback_prompt_engineering.md` conventions: persona + word-ban + structured context. Prior iteration saw +0.40 LLM-judge from this single change. Stack on top of the current winning retrieval.
