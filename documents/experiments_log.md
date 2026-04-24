# Experiments log — narrative

Append-only. One entry per scored experiment (dev or blind). Schema per `recsys_challenge_plan.md` §4.3.

New entries go at the TOP (newest first) so the most recent work is visible without scrolling.

---

## Entry template (copy when starting a new experiment)

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
