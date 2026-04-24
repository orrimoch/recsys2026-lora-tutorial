# Research Summary — Music CRS Blind-A Challenge

Consolidated reference for data, metrics, empirical findings, and next-step directions.
Last updated 2026-04-22 after v23 leaderboard result.

- Working champion: **v10** (composite 0.34, rank 2, LLM-judge 3.25).
- Gap to #1 (el_presidente, 0.40): **0.06 composite**, ≈**78% of that gap is in the LLM-judge term**.

---

## 1. Data — all available features

Pulled from five HuggingFace datasets. See [`data_exploration.md`](./data_exploration.md) for the exhaustive schema dive; below is the operational summary.

### 1.1 Sessions (`talkpl-ai/TalkPlayData-Challenge-Dataset`)
- **Rows:** 15,199 train, 1,000 dev. Blind-A is a separate 80-row test set.
- **Structure:** each row = one multi-turn conversation. Typical 24 turns = 8 (user, music, assistant) triples.
- **Ground truth signals per turn:** `music` role content is the gold track_id; `goal_progress_assessments` gives a per-turn label (`MOVES_TOWARD_GOAL` / `DOES_NOT_MOVE_TOWARD_GOAL` / null), ~93% non-null coverage in train.
- **Blind-A is multi-turn** (up to 13 turns per row) — NOT single-turn as earlier memory incorrectly claimed. Discovered during v22 debugging (row 3 has 13 turns).

### 1.2 Track catalog (`talkpl-ai/TalkPlayData-Challenge-Track-Metadata`)
- **Rows:** 47,071 tracks (`all_tracks` split).
- **Fields used by v10:** `track_name`, `artist_name`, `album_name`, `release_date`, `tag_list`.
- **Fields unused by v10:** `popularity` (0–93, mean ~36), `duration`, `ISRC`, `artist_id`, `album_id`.
- **Tags:** ~3,000 unique, noisy casing, 1–40+ per track.

### 1.3 Track embeddings (`talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`)
| Modality | Dim | dtype | Norm | Notes |
|---|---|---|---|---|
| audio-laion_clap | 512 | f64 | L2-norm≈1.0 | pre-normalized ✓ |
| image-siglip2 | 768 | f64 | ≈9.6 | unnormalized, needs rescaling |
| cf-bpr | 128 | f64 | ≈0.03 | sparse collaborative-filter vector |
| attributes-qwen3 | 1024 | f64 | dense | Qwen-3-0.6B over track attributes |
| lyrics-qwen3 | 1024 | f64 | dense | Qwen-3 over lyrics text |
| metadata-qwen3 | 1024 | f64 | dense | Qwen-3 over metadata (title+artist+album+tags) |

Coverage: all 47,071 tracks embedded in `all_tracks` split; imputation (artist → category → global mean) used for rows missing a vector.

### 1.4 User metadata (`talkpl-ai/TalkPlayData-Challenge-User-Metadata`)
- **Rows:** 8,772 users.
- **Fields:** `user_id`, `age` (int), `age_group` ('10s' / '20s' / …), `gender`, `country_code`, `country_name`.
- **Skew:** 63% in 20s, 72% male, US/BR/PL = 39%.
- **CRITICAL lesson from v21/v22:** the actual field names are `age_group` / `country_name` (NOT `age` / `country`). Original LoRA dataset builder had wrong field names → empty demographics in 100% of training prompts.

### 1.5 User embeddings (`talkpl-ai/TalkPlayData-Challenge-User-Embeddings`)
- `cf-bpr` (128-dim), split into `train` (8,591) / `test_warm` (371) / `test_cold` (129).

### 1.6 Blind-A (`talkpl-ai/TalkPlayData-Challenge-Blind-A`)
- 80 sessions, **variable turn count (1–13)**, `goal_progress_assessments` mostly null.
- We get user profile + partial conversation history; must predict a top-20 track list + generate a final response.

---

## 2. Metrics — composite score and what we can measure

### 2.1 Official composite (from [`evaluation_framework.md`](./evaluation_framework.md))

```
Composite = 0.50 · nDCG@20
          + 0.10 · CatalogDiversity
          + 0.10 · LexicalDiversity
          + 0.30 · (LLM_Judge − 1) / 4      # 1–5 scale, renormalized to [0,1]
```

### 2.2 What each term measures

| Term | Weight | Formula | Local-computable? |
|---|---|---|---|
| **nDCG@20** | 0.50 | `Σ 1[pred_i ∈ gold] / log2(i+1) / IDCG@k` over top-20, macro-averaged over sessions × turns | ✓ dev only (`metrics_recsys.py`) |
| **Catalog Diversity** | 0.10 | \|unique recommended tracks\| / 47,071 | ✓ any split (`metrics_diversity.py`) |
| **Lexical Diversity** | 0.10 | Distinct-2: unique bigrams / total bigrams across ALL responses pooled | ✓ any split (`metrics_diversity.py`) |
| **LLM Judge** | 0.30 | **Gemini** (specific model undisclosed) rates responses 1–5 on Personalization + Explanation Quality | ❌ **server-only on CodaBench** |

### 2.3 Critical: the LLM-judge is Gemini

Per section 1 of `evaluation_framework.md`:
> *"Judge model family is **Gemini**. Evaluation prompt is not published."*

This is the single most important fact about our optimization target:
- It is an LLM judge (subjective, scale 1–5).
- It rates **Personalization** + **Explanation Quality** independently of the recommendation's ranking-correctness.
- The prompt is undisclosed, so we cannot replicate locally.
- **70% of the composite gap to #1 lives in this one term.**

### 2.4 What we can optimize locally (verifiable rewards)

The evaluator's local code (`music-crs-evaluator/metrics/`) only computes nDCG, Catalog Diversity, and Lexical Diversity. It does NOT run the LLM judge. So:

**Directly verifiable (use as GRPO/DPO reward):**
- **nDCG@k** against gold track (train/dev has gold; Blind-A does not).
- **Catalog Diversity** = unique-track count over submitted lists.
- **Lexical Diversity** = Distinct-2 over pooled responses.
- **`goal_progress_assessments`** per train-turn (93% coverage) — raw ordinal reward from the dataset creators.
- **Metadata-citation fidelity**: regex-check whether response contains the gold artist / album / release year strings; 100% deterministic against catalog.
- **Hallucination check**: extracted entity names ∈ catalog? binary per mention.
- **Structural rules**: length ∈ [70, 100] words, ends with punctuation, no apology phrases, no warm openers.

**Observable only via submission:**
- The Gemini LLM-judge score. Every leaderboard attempt costs one informational unit.

**Proxy-judge option:** run Qwen-72B / Llama-3.1-405B / Claude / GPT-4 locally or via API, prompt-engineered to approximate Gemini's rubric (Personalization + Explanation Quality). Usable as a REWARD signal, with the caveat that proxy-vs-real-judge divergence is the same failure mode that killed v22 LoRA.

---

## 3. Empirical findings — the experimental record

23 versions tried; only 3 ever improved composite (v1→v2→v3→v5→v10 linear progression to 0.34). Everything since has regressed. See `documents/experiments_log.md` for the full narrative.

### 3.1 The wins

| Version | Δ from prior | What changed | Key lesson |
|---|---|---|---|
| v2 (Qwen 1.5B stock) | baseline | stock prompt, fake-assistant-turn template | BM25 retrieval alone = 0.19 nDCG baseline |
| v3 (Qwen 3B + v2 prompt + top-3) | **+0.06 composite** | (a) removed stock "apologize on mismatch" rule → no "I'm sorry" in output, (b) chat template `[system, user]` only (not `[system, user, music, user]`), (c) top-3 tracks in prompt (not top-1) | **Removing apologies = +0.9 raw LLM-judge.** Biggest single lever ever observed. |
| v5 (v3 + 3 few-shot) | +0.01 | 3 curated critic-tone exemplars in prompt | few-shot alone = +0.15 LLM-judge |
| **v10** (v5 + persona + word-ban) | **+0.04 composite** | single prompt change: "well-read music critic" persona + explicit ban on absolutely / fantastic / truly / amazing | +0.40 LLM-judge — the source of v10's champion status |

### 3.2 The losses (10 consecutive regressions after v10)

| Version | Mechanism | Δ Composite | Δ LLM | Root cause |
|---|---|---|---|---|
| v6 | Qwen 3B → 7B | −0.07 | −0.45 | Bigger model + rigid few-shot = homogeneous, formulaic output |
| v11 | +more prompt rules | −0.06 | n/a | Model quotes raw tag strings verbatim |
| v15 | +`tag_list` field in BM25 | −0.03 | n/a | Tag vocab too noisy → worse top-20 candidates |
| v16 | BM25 RRF(4-field, 5-field) | −0.07 | n/a | Self-fusion dragged retrieval below both branches |
| v17 | Dynamic NN few-shot from train | (not shipped) | n/a | Train responses are 30 words, compressed v10 output from 81 → 50 |
| v18 | wRRF(BM25, dense-Qwen3) | −0.02 | **−0.40** | nDCG +0.02 ✓ but LLM −0.40: reshuffled top-3 is less citable |
| v19 | Dual retrieval (BM25 prompt + RRF sub) | −0.01 | −0.15 | Even with identical response, non-BM25 submission costs LLM points |
| v20 | v19 + force submission[0] = BM25 top-1 | −0.03 | −0.40 | Demoting RRF top-1 costs MORE than the natural ordering |
| **v21 (LoRA r1)** | LoRA Qwen 3B on train data | **−0.18** | **−1.65** | Multiple bugs (inference + prompt format) made result uninterpretable |
| **v22 (LoRA r2)** | LoRA + ALL bugs fixed | **−0.13** | **−1.45** | Even clean, LoRA produces 45-word responses (vs v10's 81) and tanks judge |
| **v23** | Regex-replace 4 banned words in v10 output | **−0.02** | **−0.25** | **Word-ban hypothesis falsified** — judge prefers Qwen's natural "Absolutely" over manual synonyms like "Without question" |

### 3.3 Distilled lessons (in priority order)

1. **The LLM-judge is the dominant composite lever** (0.045 of the 0.06 gap to #1).
2. **Retrieval changes that reshuffle the submission list cost LLM-judge points** — verified across v18/v19/v20. The judge apparently scores the whole submission set + response as a unit, not just top-1.
3. **Surface-level output edits (word substitution) do NOT reliably help** — v23 proved our word-ban theory was wrong. The v5→v10 +0.40 LLM came from the PERSONA part of the prompt, not the ban-list part.
4. **LoRA fine-tuning on train data fundamentally fails** — train responses are chat-agent style, judge wants critic style. Confirmed across v21/v22 regardless of bug-fixes.
5. **Bigger Qwen (7B) with rigid few-shot overfits the exemplar pattern** → homogeneous output → judge penalty.
6. **Removing apologies was a genuine ~+0.9 LLM win (v2→v3)**. This remains the single biggest causal lever we have empirical data for.
7. **v10 is a brittle local optimum** — the specific combination (Qwen 3B, greedy decode, v4 prompt, BM25 4-field) is mutually load-bearing; any single deep change regresses.

### 3.4 Dead ends — never retry these families

- Any retrieval change: BM25 field tweaks, RRF (self or with dense), alignment swaps.
- LoRA SFT on train data (distillation from v10 is the only LoRA path that wouldn't regress, but has no upside).
- Bigger base model (7B, 14B) with existing few-shot.
- Manual word substitution post-processing (v23 falsified).
- More prompt rules added to v10 (v11 regressed).

### 3.5 Bugs we fixed (for future-self reference)

- `build_retrieval_input` must format ALL conversation turns (Blind-A is multi-turn, not single-turn).
- `UserProfileDB.id_to_profile_str` uses columns `user_id, age_group, gender, country_name` — not `age, country, gender`.
- `conversation_goal` must be stringified with plain `str()` (dict repr), NOT unwrapped as `.listener_goal`.
- Track catalog must use `all_tracks` split only — NOT `test_tracks` or concatenation.
- BM25 corpus lines must end with `\n` per field (v10's `_stringify_metadata` behaviour).
- TRL 0.13 dropped `completion_only_loss=True` from `SFTConfig` — we built a custom pre-tokenize + pad collator to restore it.
- `enable_input_require_grads()` is required when combining `gradient_checkpointing=True` with PEFT; otherwise gradients don't flow through embeddings.

---

## 4. Derived: reward-signal design space

Building on section 2.4, here are concrete verifiable rewards we can use in a GRPO / DPO training loop:

| Reward component | Source | Verifiability | Coverage |
|---|---|---|---|
| `r_ndcg = nDCG@20(pred, gold)` | train/dev gold track per turn | deterministic | 100% of train/dev turns |
| `r_goal_prog` = categorical goal-progress-assessment mapped to {+1, −1, 0} | `goal_progress_assessments[turn]` | deterministic | 93% of train turns |
| `r_len` = `1.0 if 70 ≤ words(response) ≤ 100 else linear decay` | generated response | deterministic | 100% |
| `r_meta` = `fraction({gold_artist, gold_album, gold_year} mentioned in response)` | generated response vs track metadata | deterministic (string match) | 100% |
| `r_nohall` = `1 − (hallucinated_mentions / total_mentions)` extracted artist/track names ∈ catalog | generated response vs 47k catalog | deterministic | 100% |
| `r_no_warm_opener` = `1 − is_warm_sentence(response.split_sentences[0])` | generated response | deterministic | 100% |
| `r_lex_batch` = distinct-2 over batch of generations per query | batch of N responses per query | deterministic | 100% |
| `r_proxy_judge` = Qwen-72B / external-LLM rubric-score of response | proxy LLM call | **not verifiable** | 100% but costly |

**Multi-objective reward template:**
```
r = 0.35·r_ndcg
  + 0.15·r_meta
  + 0.10·r_len
  + 0.10·r_nohall
  + 0.10·r_no_warm_opener
  + 0.05·r_lex_batch
  + 0.15·r_proxy_judge      # optional; high risk
```

Sum ≠ 1 intentionally (one signal left out) — keep weights tunable.

**GRPO rollout shape:** N=4–8 sampled completions per prompt, group-baseline advantage = `(r_i − mean_group) / std_group`, policy gradient via `trl.GRPOTrainer` (released 2025).

---

## 5. Recommended next experiments (post-v23)

Ordered by estimated probability of a meaningful win:

| Priority | Candidate | Mechanism | Expected composite | Dev effort | Key risk |
|---|---|---|---|---|---|
| 🥇 | **External judge reconnaissance** | Read nlp4musa challenge paper / TalkPlayData-2 paper / find public judge rubric or Gemini evaluation prompt | unknown — could unlock everything | 1–3 h reading | may find nothing |
| 🥈 | **Accept rank 2** | Stop; 0.34 / rank 2 is a legitimate finish given 10+ failed attempts from here | 0.34 (status quo) | 0 h | zero |
| 🥉 | **GRPO with verifiable rewards** | Train Qwen 3B with `trl.GRPOTrainer`, multi-objective reward per section 4. Optimize explicitly on nDCG + len + meta + nohall. Optional proxy-judge channel. | +0.01 to +0.03 if rewards aligned; −0.05 to −0.15 if misaligned | 2–5 days (Colab A100) | same information problem — we're guessing at proxy judge |
| 4 | **Multi-candidate self-judge (no training)** | v10 greedy + 2 samples (T=0.6, 0.9); Qwen-7B picks best per row by rubric | +0.01 to +0.02 | 1 day (Colab) | Qwen-7B ≠ Gemini |
| 5 | **Different persona prompt** | Swap "well-read music critic" → "Pitchfork reviewer" / "jazz critic" / "liner-notes writer". The persona change was the only confirmed +0.40 lever; maybe a different persona hits even higher. | −0.05 to +0.03 | 30 min | high variance — v10's specific persona may be the sweet spot |
| 6 | **Longer response target** | Increase max_new_tokens from 192 to 320; re-prompt to produce 120–150-word reviews. If judge rewards richer explanations, this could be worth +0.05 LLM. | −0.05 to +0.03 | 30 min | longer responses often drift / hallucinate |

### 5.1 My honest recommendation

**Start with #1 (external reconnaissance) before any more code.** The reason: 10 consecutive losses with carefully-designed experiments means we don't understand what the judge rewards. More principled code won't fix the information gap. If the challenge organizers publish the Gemini rubric anywhere, that unlocks everything.

**If reconnaissance yields nothing, #2 (stop) is rational.** We've extracted the easy wins (v2→v10 progression). Further attempts are ~25-35% coin flips.

**If you want to run one more experiment regardless, #3 (GRPO) is the most principled** — it's the only remaining mechanism that doesn't reduce to "another blind guess". But realistic probability of winning is ~30%, and it costs 2–5 days of dev + Colab credits.

---

## 6. Quick-reference: what TO try vs NEVER retry

| Try | Never retry |
|---|---|
| Reading challenge/judge docs externally | Adding fields to BM25 corpus |
| Accepting rank 2 | RRF fusion (dense, tag, any combo) |
| GRPO with verifiable rewards + proxy judge | LoRA SFT on train data |
| Multi-candidate self-judge at inference | Qwen 7B with existing few-shot |
| New persona prompt variants | Manual word substitution post-processing |
| Longer-response prompt variants | Dual-retrieval + alignment swap |
| Adding specific citation directives to prompt | More prompt rules with citation directives |

---

## 7. Appendix — file index

Paths below are relative to this file (in `documents/`).

- `../CLAUDE.md` — agent rules + repo guide
- `./data_exploration.md` — exhaustive HF schema dive
- `./evaluation_framework.md` — official composite + metric definitions (confirms LLM-judge = Gemini)
- `./experiments_log.md` — narrative of every experiment
- `./submissions_log.md` — tabular leaderboard history
- `./submission_format.md` — CodaBench zip packaging rules
- `../music-crs-baselines/` — v10 champion code (run_inference_blindset_full_v6.py)
- `../music-crs-evaluator/` — local dev-set evaluator (nDCG + diversity only; NO LLM-judge)
