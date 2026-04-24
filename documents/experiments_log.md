# Experiments Log — RecSys 2026 Blind A

Full, detailed record of every experiment: every method, every model, every hyperparameter, every outcome, every lesson. Complement to:
- `documents/submissions_log.md` — tabular leaderboard scores only.
- `experiments/ledger.tsv` — older dev-set ML experiments.

**Purpose:** make the next "what to try" call defensible. Captures both mechanism and outcome so future experiments build on validated levers.

---

## 0. What works best so far (updated after every scored submission)

### Top-3 champions (by shipped composite)

Always kept current. A newly-scored experiment that beats #3 evicts the prior #3; beats #1 → cascades down.

| Place | Version | Composite | nDCG@20 | Cat | Lex | LLM | Rank | Key mechanism |
|---|---|---|---|---|---|---|---|---|
| 🥇 #1 | **v3** | **0.29** | 0.19 | 0.03 | 0.66 | 2.70 | 4 | BM25 4-field + Qwen3B + custom prompt (no apology, no hallucinated history) + top-3 tracks + custom `[system, user]` chat template |
| 🥈 #2 | v4 | 0.28 | 0.12 | 0.03 | 0.67 | 3.05 | 5 | v3 + `tag_list` BM25 + LLM query expansion + 3 few-shot exemplars. Retrieval regressed; LLM lifted. |
| 🥉 #3 | v2 | 0.23 | 0.19 | 0.03 | 0.72 | 1.80 | 5 | BM25 4-field + Qwen1.5B + stock prompt/template (apology directive + fake-assistant-turn = LLM cap) |

**Pending (not yet scored):** v5 — predicted composite 0.32; would become #1 if it holds.

### Champion by metric (Blind-A leaderboard)

| Metric | Best value | Experiment | Mechanism that unlocked it |
|---|---|---|---|
| nDCG@20 | **0.19** | v1 / v2 / v3 (all tied) | BM25Okapi over 4-field corpus (`track_name + artist_name + album_name + release_date`) with catalog `all_tracks`. Not yet beaten by any text-side change. |
| Cat Div | 0.03 | all experiments | Saturated across all teams on Blind-A (small 80-row set). No leverage. |
| Lex Div (Distinct-2) | **0.72** | v2 | Qwen 2.5-1.5B stock prompt with greedy decoding; ironically less templated than v3/v4. v3 (0.66) and v4 (0.67) regressed because structured few-shot recycles phrases. |
| LLM-judge (1–5) | **3.05** | v4 | Few-shot exemplars (3 hand-crafted) in response prompt + custom v3 chat template `[system, user]` + top-3 tracks embedded in system prompt. |
| **Composite** | **0.29** | **v3** | BM25 4-field + Qwen 2.5-3B-Instruct + custom prompt (`response_generation_v2.txt`) + top-3 tracks + chat template fix. |

### Validated mechanisms (things we know help)

1. **Swapping templated response for LLM response:** +0.06 composite (v1→v2).
2. **Custom prompt without apology directive:** ~+0.5 LLM-judge (v2→v3).
3. **Chat template `[system, user]` only (no fake assistant turn for recommendations):** eliminates turn-1 hallucinations; part of v2→v3 lift.
4. **Top-3 tracks in system prompt (vs top-1 in the fake assistant turn):** part of v2→v3 lift.
5. **Few-shot exemplars in response prompt:** +0.35 LLM-judge (v3→v4, isolated signal).
6. **`max_new_tokens = 192` (vs 96):** eliminates truncation; negligible direct composite effect but prevents cut-off penalty.

### Rejected / broken mechanisms

1. **MPS + float16 for Qwen generation:** NaN in `torch.multinomial` or `!!!!!` garbage. Use float32 on MPS.
2. **Stock `response_generation.txt` prompt:** has apology directive → "I'm sorry" responses tank LLM-judge.
3. **Stock `LLAMA_MODEL.batch_response_generation` chat template:** injects track as prior assistant turn → "I'm glad you enjoyed X" hallucinations on turn 1.
4. **LLM query expansion with `repetition_penalty=1.2`:** pushed Qwen into non-ASCII tokens (20/80 rows Chinese) and fabricated artist names. Retrieval regressed −0.07 nDCG.
5. **Greedy decoding without `no_repeat_ngram_size` for short-output tasks:** collapses into n-gram repetition (seen in v4 smoke).

### Open questions (not yet validated on Blind-A)

- Does `tag_list` in the BM25 corpus help or hurt in isolation? (Dev-ledger says +0.01 on dev; v4 bundled it with expansion so we can't tell. Queued as v7.)
- Does larger LLM (Qwen 7B) materially lift LLM-judge beyond few-shot alone? (On-deck v6.)
- Would a self-built dense retriever (BGE-large, full 47 k catalog index) lift nDCG? (Queued as v8.)
- Would multi-candidate + self-judge lift LLM-judge further? (Queued as v9.)

---

## Entry template

```
### Exp NNN — <short name> (vN) — YYYY-MM-DD HH:MM

**Hypothesis:** expected outcome + mechanism.

**Methods & Models**
- Retrieval: <method name, config>
- Reranking: <method name, or "none">
- Generation model: <HF model ID + params + precision + device>
- Decoding: <sampling strategy + hyperparameters>
- Prompting: <which prompt files, key prompt design choices>
- Preprocessing / chat template: <any template or message-shape customization>

**Artifacts**
- config: path
- script: path
- prompt files: paths
- output JSON: path
- expansions / debug files: paths

**Changes vs prior champion**
- <bullet list of exactly what changed, and what stayed the same>

**Shipped Result (CodaBench)**
- nDCG@20 | Cat Div | Lex Div | LLM | Composite | Rank

**Local diagnostics**
- any pre-submission checks (validation, qualitative inspection).

**What it taught us** — 1–3 bullets pointing to specific evidence.

**Verdict** — shipped / kept-for-ensemble / rejected / abandoned. Why.

**Suggests next** — concrete follow-ups this result opens up.
```

---

## Exp 001 — Naive BM25 retrieval, templated response (v1) — 2026-04-19 16:51

**Hypothesis:** verify end-to-end pipeline works. BM25 alone should be competitive on retrieval; a templated response will bottom out on lexical-diversity + LLM-judge.

**Methods & Models**
- Retrieval: `bm25s` (BM25Okapi) over 4-field corpus `[track_name, artist_name, album_name, release_date]`, catalog `all_tracks` (47,071 tracks), `topk=40` → dedup + popularity backfill → `topk=20`.
- Reranking: none.
- Generation model: none (`lm_type: null`).
- Decoding: n/a.
- Prompting: n/a. Response is a templated f-string: `"You might enjoy {track_name} by {artist_name} based on what you just told me."`
- Preprocessing / chat template: n/a.

**Artifacts**
- config: `music-crs-baselines/config/naive_bm25_blindset_A.yaml`
- script: `music-crs-baselines/run_inference_blindset_retrieval_only.py`
- output JSON: `music-crs-baselines/exp/inference/blindset_A/naive_bm25_blindset_A.json`

**Changes vs prior champion:** N/A — first submission.

**Shipped Result (CodaBench)**
| nDCG@20 | Cat Div | Lex Div | LLM | Composite | Rank |
|---|---|---|---|---|---|
| **0.19** | 0.03 | 0.46 | 1.3 | **0.17** | 6 |

**Local diagnostics:** validation passed; templated response still yielded Distinct-2 > 0 because the top-track name/artist varies by row.

**What it taught us**
1. BM25 4-field alone is already top-3 on retrieval (0.19 vs leader's 0.23). Retrieval is not the primary bottleneck.
2. Composite decomposition: LLM-judge term (2.55 raw gap normalized = ~0.19 composite gap) accounts for ~80% of the gap to #1.
3. Even minimal lexical variation (templated) gives Distinct-2 ≈ 0.46.

**Verdict:** shipped as baseline floor. Superseded.

**Suggests next:** add real LLM for response; retrieval tweaks are second-order for now.

---

## Exp 002 — BM25 + Qwen 1.5B stock prompt (v2) — 2026-04-19 17:44

**Hypothesis:** real LLM response should ≈ double the composite via Lex Div and LLM-judge. Qwen 2.5-1.5B chosen as smallest MPS-stable model.

**Methods & Models**
- Retrieval: BM25Okapi 4-field, same as v1. `topk=40` → dedup → `topk=20`.
- Reranking: none.
- Generation model: `Qwen/Qwen2.5-1.5B-Instruct`, **MPS float32** (float16 attempted first, produced NaN → switched), `attn_implementation: eager`.
- Decoding: greedy (`do_sample=False`), `max_new_tokens=96`.
- Prompting: stock `mcrs/system_prompts/roleplay.txt` + `response_generation.txt` + `personalization.txt`; user_profile + conversation_goal appended to system prompt.
- Preprocessing / chat template: **stock** `LLAMA_MODEL.batch_response_generation` — appends top-1 track metadata as a prior assistant turn before the generation marker.

**Artifacts**
- config: `music-crs-baselines/config/qwen1.5b_bm25_blindset_A.yaml`
- script: `music-crs-baselines/run_inference_blindset_full.py`
- output JSON: `music-crs-baselines/exp/inference/blindset_A/qwen1.5b_bm25_blindset_A.json`

**Changes vs prior champion:** templated response → LLM response. Everything retrieval-side identical.

**Shipped Result (CodaBench)**
| nDCG@20 | Cat Div | Lex Div | LLM | Composite | Rank | Δ vs v1 |
|---|---|---|---|---|---|---|
| 0.19 | 0.03 | **0.72** | 1.8 | **0.23** | 5 | +0.06 |

**Local diagnostics:**
- Initial float16 run on MPS produced degenerate `!!!!!` output → switched to float32.
- Inspected responses post-run; multiple instances of "I'm sorry, couldn't find a match" and "I'm glad you enjoyed X" detected.
- 9/80 responses cut off mid-sentence (hit max_new_tokens=96).

**What it taught us**
1. **MPS + float16 is broken** for Qwen batch generation (NaN / repeated punctuation tokens). Default to float32 on MPS.
2. Stock `response_generation.txt` has item #2: *"If the recommended track doesn't match the user's query, apologize and acknowledge the mismatch."* → the LLM dutifully apologizes; judge penalizes it.
3. Stock chat template adds the recommended track as an assistant message BEFORE the generation → the LLM reads this as "the assistant already recommended this, now follow up" → on turn 1 it produces "I'm glad you enjoyed X" hallucinations.
4. Lex Div 0.46 → 0.72 — LLM-generated text is clearly richer even with a weak prompt.

**Verdict:** shipped, rank 6→5. Superseded by v3.

**Suggests next:** custom prompt with no apology directive + no prior-listening hallucinations; custom chat template that doesn't inject recommendations as fake turns; longer `max_new_tokens`; optionally bigger model.

---

## Exp 003 — Custom prompt + top-3 tracks + Qwen 3B (v3) — 2026-04-19 18:07

**Hypothesis:** fixing the three v2 defects (truncation, apology, hallucinated prior listening) plus upgrading to Qwen 3B should close most of the LLM-judge gap without touching retrieval.

**Methods & Models**
- Retrieval: BM25Okapi 4-field (unchanged from v1/v2).
- Reranking: none.
- Generation model: `Qwen/Qwen2.5-3B-Instruct`, MPS float32, `attn_implementation: eager`.
- Decoding: greedy (`do_sample=False, temperature=1.0, top_p=1.0, top_k=0`), `max_new_tokens=192` (doubled from v2).
- Prompting: `roleplay.txt` + **new** `response_generation_v2.txt` (explicit bans on apologies, hallucinated reactions, fabricated facts, requires metadata-grounded reasoning).
- Preprocessing / chat template: **custom** `custom_batch_generate()` — chat is `[system, user]` only. No fake assistant turn. Recommendation (top-3 tracks with name/artist/album/year/tags) is embedded in the system prompt as a "Candidate tracks (ranked)" block.
- Personalization: `user_profile` + `conversation_goal` + demographic profile injected into the system prompt as "About this user" section.

**Artifacts**
- config: `music-crs-baselines/config/qwen3b_bm25_blindset_A.yaml`
- script: `music-crs-baselines/run_inference_blindset_full_v2.py`
- prompt: `mcrs/system_prompts/response_generation_v2.txt`
- output JSON: `music-crs-baselines/exp/inference/blindset_A/qwen3b_bm25_blindset_A.json`

**Changes vs v2**
- Model 1.5B → 3B.
- Prompt rewritten (no apology directive, metadata-grounded, no prior-listening claims).
- Chat template [system, user] only (no fake assistant turn).
- Top-1 track → top-3 tracks in system prompt.
- `max_new_tokens` 96 → 192.
- Retrieval: **unchanged**.

**Shipped Result (CodaBench)**
| nDCG@20 | Cat Div | Lex Div | LLM | Composite | Rank | Δ vs v2 |
|---|---|---|---|---|---|---|
| 0.19 | 0.03 | 0.66 | **2.7** | **0.29** | **4** | +0.06 |

**Local diagnostics:**
- 0 apologies (grep `(?i)sorry|apologi`).
- 0 hallucinated prior listening (grep `(?i)glad you enjoyed|glad you liked`).
- 0 truncations (all responses end on `.`, `!`, `?`, or `"`).
- Lex Div regressed slightly (0.72 → 0.66) — structured "recommend → why → runner-up → invite" response shape reuses phrases.

**What it taught us**
1. Prompt + chat-template fixes ALONE are worth ~+0.9 LLM-judge. Prompt engineering is top-tier leverage for this competition.
2. Structural response shape improves judge but costs Distinct-2 via phrase recycling. Net positive due to LLM-judge weight (0.075/pt vs Lex 0.10/pt).
3. Our LLM score (2.7) now matches / beats rank-3 swyoo (2.6). Remaining gap to leaderboard top is retrieval (swyoo nDCG 0.23 vs ours 0.19) and a further LLM lift.

**Verdict:** shipped, rank 5→4. **Current overall-composite champion.**

**Suggests next:** retrieval is the new bottleneck. Try (a) `tag_list` in BM25 corpus, (b) LLM query expansion, (c) RRF(BM25, dense), (d) cross-encoder rerank. On the LLM side, try few-shot exemplars, bigger model (Qwen 7B), multi-candidate + self-judge.

---

## Exp 004 — BM25+tags + LLM query expansion + few-shot response (v4) — 2026-04-19 18:41

**Hypothesis:** stacking three independent levers should compound: (a) `tag_list` in BM25 corpus (dev-ledger prior: +0.01 nDCG@10 on dev), (b) LLM query expansion to bridge BM25's vocab gap on short conversational queries, (c) few-shot exemplars in response prompt to further lift LLM-judge. Target composite 0.33–0.36.

**Methods & Models**
- Retrieval: BM25Okapi 5-field (added `tag_list`). Cache `track_name_artist_name_album_name_release_date_tag_list` was pre-built — no rebuild time. `topk=40` → dedup → `topk=20`.
- Query expansion (new, pre-retrieval): Qwen 2.5-3B generates dense keyword query (genres, moods, eras, synonyms). Prompt = `mcrs/system_prompts/query_expansion.txt` (with 3 in-prompt exemplars). Decoding: greedy + **`no_repeat_ngram_size=3`, `repetition_penalty=1.2`**, `max_new_tokens=80`, `max_input_len=1024`. Output sanitized (strip preambles, drop refusals, cap 40 tokens).
- Retrieval input: `expanded_query + "\n" + raw_user_query` so literal artist/song names survive.
- Reranking: none.
- Generation model: `Qwen/Qwen2.5-3B-Instruct`, MPS float32, greedy, `max_new_tokens=192`.
- Prompting: `roleplay.txt` + **new** `response_generation_v3.txt` = v2 content + **3 hand-crafted few-shot exemplars** (Friday-funk, melancholy-acoustic, 90s east-coast hip-hop).
- Preprocessing / chat template: same `[system, user]` only from v3.

**Artifacts**
- config: `music-crs-baselines/config/qwen3b_bm25tags_blindset_A.yaml`
- script: `music-crs-baselines/run_inference_blindset_full_v3.py`
- prompts: `mcrs/system_prompts/query_expansion.txt`, `response_generation_v3.txt`
- output JSON: `music-crs-baselines/exp/inference/blindset_A/qwen3b_bm25tags_blindset_A.json`
- expansions debug: `music-crs-baselines/exp/inference/blindset_A/qwen3b_bm25tags_blindset_A.expansions.json`

**Changes vs v3**
- BM25 corpus: 4 → 5 fields (added `tag_list`).
- Added LLM query expansion stage (pre-BM25).
- Response prompt: `_v2.txt` → `_v3.txt` (adds 3 few-shot exemplars).
- Retrieval input is now `expanded + raw`, not `raw` alone.
- Generation model, decoding, chat template: unchanged from v3.

**Shipped Result (CodaBench)**
| nDCG@20 | Cat Div | Lex Div | LLM | Composite | Rank | Δ vs v3 |
|---|---|---|---|---|---|---|
| **0.12** | 0.03 | 0.67 | **3.05** | 0.28 | 5 | **−0.01** |

**Decomposition:** nDCG@20 **−0.07 regression** dominated the +0.35 LLM-judge gain and tiny Lex lift. Net composite slightly negative.

**Local diagnostics:**
- Expansion forensics (post-regression) on 80 rows:
  - **20/80 expansions (25%) contain non-ASCII tokens** (Chinese characters like "动感舞曲", "夜晚冥想"). `repetition_penalty=1.2` pushed Qwen into rare-token space. BM25 index is English-only → dead matching tokens.
  - **17/80 expansions** padded to the 40-token max (filler).
  - **≥1 confirmed fabricated artist:** row 14 expansion says "laura paucciño" (doesn't exist) when user mentioned 'Más que ayer'.
  - **4/80 numeric gibberish** ("2ndo year 2k9 ... 2oth anniversary 2kn").
- Response quality good: 0 apologies, 0 hallucinated prior-listening, 0 truncations. Few-shot structural shape visible (recommend → why → runner-up → invite).

**What it taught us**
1. **Stacking multiple unvalidated changes in one experiment hides which one hurt.** With (tags + expansion + few-shot) all changing, we can't attribute the −0.07 nDCG to tags or expansion alone. Needed an ablation (v5 / v7). → **`feedback_experiment_ablation_discipline.md`** (memory).
2. **LLM query expansion with `repetition_penalty=1.2` is unsafe.** Pushes greedy decoding into rare Unicode / fabricated names. For English-only BM25 indices, the expansion introduces noise that actively hurts retrieval.
3. **Few-shot exemplars lifted LLM-judge +0.35** (isolated by comparing to v3 which shares all response-side mechanisms except few-shot). The LLM-side of v4 is safe to keep; the retrieval side is not.
4. **Net effect of a bad retrieval change is an order of magnitude larger than a good LLM change** in composite terms: −0.07 nDCG × 0.5 = −0.035 vs +0.35 LLM × 0.075 = +0.026. Retrieval must not regress.

**Verdict:** **rejected.** Retrieval regression wipes out LLM gain. Response-side mechanism (`response_generation_v3.txt` + few-shot + top-3 tracks) kept for v5.

**Suggests next (v5):** decouple → ship v3 retrieval (proven nDCG 0.19) + v4 response (proven LLM 3.05). Predicted composite 0.32 → rank 2.

---

## Exp 005 — v3 retrieval + v4 response (v5) — 2026-04-19 ~19:20 _(running / pending CodaBench)_

**Hypothesis:** decouple v4's retrieval vs response changes. Isolate the LLM-side win. Predicted composite `0.50·0.19 + 0.10·0.03 + 0.10·0.67 + 0.30·(3.05−1)/4 = 0.319 ≈ 0.32` → rank 2.

**Methods & Models**
- Retrieval: BM25Okapi **4-field** (rolled back from v4's 5-field), `topk=40` → dedup → `topk=20`.
- No query expansion (rolled back from v4).
- Reranking: none.
- Generation model: `Qwen/Qwen2.5-3B-Instruct`, MPS float32, greedy, `max_new_tokens=192`.
- Prompting: `roleplay.txt` + `response_generation_v3.txt` (kept from v4 — includes 3 few-shot exemplars).
- Chat template: `[system, user]` only (kept).

**Artifacts**
- config: `music-crs-baselines/config/qwen3b_bm25_fewshot_blindset_A.yaml`
- script: `music-crs-baselines/run_inference_blindset_full_v4.py` (thin wrapper over v3 script; loads `response_generation_v3.txt` instead of `_v2.txt`).
- prompt: `mcrs/system_prompts/response_generation_v3.txt`
- output JSON (when done): `music-crs-baselines/exp/inference/blindset_A/qwen3b_bm25_fewshot_blindset_A.json`

**Changes vs v4**
- Dropped `tag_list` from BM25 corpus (5 → 4 fields).
- Dropped LLM query expansion stage.
- Retrieval input is `raw_user_query` only, not `expanded + raw`.
- Response-side unchanged (prompt, few-shot, chat template, model).

**Shipped Result (CodaBench):** _pending._

**Local diagnostics (smoke 3 rows, pre-full-run):**
- Responses confident, metadata-grounded, few-shot influence visible (e.g. "For that smooth, soulful early 90s east coast vibe, try 'Hip Hop' by Dead Prez from their 2000 album *Let's Get Free*...").
- No apologies, no hallucinations, no truncations in smoke.

**What it taught us:** _pending CodaBench._

**Verdict:** _pending._

**Suggests next (v6):** Qwen 2.5-7B-Instruct + v5's retrieval + v5's response prompt. 48 GB M4 fits 7B in fp32. Predicted composite 0.34–0.36.

---

## Method effectiveness matrix

Cross-reference of every method we've touched to the experiments where it appeared, with validated impact. Populated post-scoring.

| Method | Used in | Validated Δ (measured) | Validated Δ (estimated/inferred) | Status |
|---|---|---|---|---|
| BM25Okapi 4-field | v1, v2, v3, v5 | baseline nDCG@20 = 0.19 | — | kept |
| BM25Okapi 5-field (add `tag_list`) | v4 | confounded (in a regression stack) | +0.005 to +0.015 (dev prior) | unvalidated isolated on Blind-A |
| `topk=40 → dedup → 20` backfill | v1–v5 | — | guarantees 20 distinct IDs | kept as safety |
| LLM query expansion (Qwen 3B + `rep_pen=1.2` + `no_repeat_ngram=3`) | v4 | net −0.07 nDCG in stack | known dead end for English BM25 | **rejected** |
| Templated response (f-string) | v1 | Lex 0.46, LLM 1.3 | LLM floor | superseded |
| LLM response — Qwen 2.5-1.5B-Instruct | v2 | LLM 1.8, Lex 0.72 | — | superseded |
| LLM response — Qwen 2.5-3B-Instruct | v3, v4, v5 | v3: LLM 2.7, v4: 3.05 | 3B is the sweet-spot for M4 float32 + speed | **current champion** |
| MPS float32 for Qwen generation | v2+ | stable generation | — | required |
| MPS float16 for Qwen generation | v2 (attempted) | NaN / `!!!!!` output | — | **dead end** |
| Greedy decoding (`do_sample=False`) | v2, v3, v4, v5 | stable, deterministic | — | kept |
| `max_new_tokens=96` | v2 | 9/80 truncated | floor | superseded |
| `max_new_tokens=192` | v3, v4, v5 | 0 truncations | — | kept |
| Stock `response_generation.txt` (has apology directive) | v2 | LLM 1.8 | cap | **dead end** |
| Custom `response_generation_v2.txt` (no apology, no hallucinated history) | v3 | LLM 2.7 (+0.9 vs v2) | — | kept for response-side |
| Custom `response_generation_v3.txt` (= v2 + 3 few-shot) | v4, v5 | LLM 3.05 in v4 (+0.35 vs v3) | — | **current champion (response-side)** |
| Stock chat template (track as fake assistant turn) | v2 | "I'm glad you enjoyed X" hallucinations | cap | **dead end** |
| Custom chat template `[system, user]` only | v3+ | 0 hallucinated prior-listening | — | kept |
| Top-1 track in prompt | v2 | narrow grounding | floor | superseded |
| Top-3 tracks in system prompt | v3+ | better grounding, room for runner-ups | — | kept |
| user_profile (raw blob) in prompt | v2–v5 | — | raw blob may under-leverage | TODO parse into slots (A1 in candidates) |
| conversation_goal in prompt | v2–v5 | — | raw injection | kept |
| `no_repeat_ngram_size=3` on query expansion | v4 | mitigated some degeneracy but not enough | post-hoc | insufficient on its own |
| `repetition_penalty=1.2` on query expansion | v4 | pushed into Chinese tokens | post-hoc | **dead end for English BM25** |

---

## Cross-cutting lessons (generalizable rules)

Consolidated from all shipped experiments:

- **One-axis per experiment.** v4 stacked (tags + expansion + few-shot) and regressed unattributably. Future: ship one change at a time OR plan an ablation run before shipping the stack. → `feedback_experiment_ablation_discipline.md`.
- **MPS + float16 is unsafe** for Qwen batch generation. Default to float32. → `project_blind_a_state.md` gotchas.
- **Prompt engineering is top-tier leverage on LLM-judge.** Two separate prompt rewrites were each worth ≥ +0.35 LLM-judge (v2→v3 and v3→v4). Always scrutinize prompts before algorithmic changes.
- **Chat template matters.** Stock template with fake assistant turns caused hallucinations. Custom `[system, user]` template with context in system prompt is strictly better.
- **Greedy decoding needs anti-repetition guards for short outputs.** v4 smoke showed `smooth jazz trap r&b` repeated 4× before `no_repeat_ngram_size=3` was added.
- **LLM query expansion is fragile.** `repetition_penalty` pushes into rare-token space (Chinese, fabricated names). Classical PRF (RM3) is probably safer.
- **Retrieval regressions are expensive in composite terms** due to the 0.50 weight. −0.07 nDCG@20 = −0.035 composite; takes a +0.47 LLM gain to compensate.
- **LLM gains are reliable per unit of prompt work.** Few-shot (+0.35), prompt rewrite (+0.9). Rarely regress.
- **M4 48 GB unified memory fits Qwen 7B in float32 (~30 GB)**, not 14B (~56 GB). 14B needs Colab or int4 quantization.

## Dead ends (never retry)

- MPS float16 for Qwen batch generation.
- Stock `response_generation.txt` as-is (apology directive).
- Stock `LLAMA_MODEL.batch_response_generation` chat template.
- LLM query expansion with `repetition_penalty > 1.0` and English-only BM25.
- CodaBench zip with any filename other than `prediction.json` at archive root.

## How to add a new entry

1. Copy the template at the top into a new `### Exp NNN — <name> (vN) — YYYY-MM-DD HH:MM` block.
2. Pre-register the Hypothesis and Methods & Models BEFORE running. Makes post-hoc learning cleaner.
3. After CodaBench scores land, fill Shipped Result, What it taught us, Verdict, Suggests next.
4. Update "0. What works best so far" champion tracker if this experiment beat a metric.
5. Append new rows to the "Method effectiveness matrix".
6. If a new lesson generalizes, add to "Cross-cutting lessons". If a dead end, add to "Dead ends".
7. Mirror the key facts into `project_blind_a_state.md` inline ledger + `submissions_log.md` table.
