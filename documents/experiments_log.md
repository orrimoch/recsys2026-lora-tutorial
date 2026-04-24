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
