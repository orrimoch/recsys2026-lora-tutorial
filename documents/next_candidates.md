# Next experiment candidates — ranked queue (fresh-model branch)

Regenerated after exp 024 (2026-04-24). Mirrors auto-memory `project_next_experiment_candidates.md`. Rank by `(expected_composite_gain × probability_of_success) / effort_hours`, discounted for Blind-A ±0.05 noise and this branch's 3-falsification track record.

**Key calibration update:** the easy composite lifts (persona prompt, model size swap, off-the-shelf reranker) are mapped and plateau at ~0.33. Further gains require **task-specific training** (LambdaMART / cf-bpr / fine-tuned cross-encoder) OR **offline pre-validation** to survive the 80-row eval noise.

## Current state (anchor for Δcomposite claims)

| Exp | Stack | composite | nDCG@20 | LexDiv | LLM | verdict |
|---|---|---|---|---|---|---|
| **021** 🏆 | wRRF + Qwen 1.5B + stock prompt + max_new=64 | **0.33** | 0.19 | 0.67 | 3.15 | **champion, rank 9/9** |
| 022 | + persona prompt | 0.24 | 0.14 | 0.80 | 2.15 | rejected (H-4 falsified) |
| 023 | 022 + BGE-reranker + Qwen 3B + max_new=192 | 0.19 | 0.07 | 0.61 | 2.20 | rejected (reranker misaligned) |
| 024 | 021 + Qwen 3B + max_new=192 | 0.29 | 0.14 | 0.62 | 3.00 | rejected (H-7, H-8 falsified) |

**Gap to #1 (composite 0.57)**: 52% nDCG@20 (+0.25 headroom) + 44% LLM (+1.40) + 6% LexDiv. Retrieval is the bigger lever this season.

## Ranked queue — live on-deck

| Rank | Exp ID | Name | Axis | Expected Δcomposite | Effort | Why this rank |
|---|---|---|---|---|---|---|
| **1** | **B1** | **Offline eval harness (held-out train slice)** | methodology | indirect — unblocks everything | ~4 h | Blind-A ±0.05 noise means 3 of our 4 ships were partially in noise. Without offline pre-filter, every candidate below burns submissions unreliably. Build FIRST. |
| 2 | A2 | cf-bpr user×item retriever (warm-user dot; BM25 fallback cold) | retrieval | +0.005–0.015 | ~4–6 h | Cheapest task-aware retrieval. cf-bpr precomputed, directly encodes preference (the signal BGE-reranker lacked). |
| 3 | A1 | LambdaMART LTR on `goal_progress_assessments` | retrieval | +0.015–0.040 | ~1 day | Highest retrieval ceiling; uses a dataset signal nobody seems to be exploiting (93% coverage). 12+ engineered features (per-field BM25, dense cosines, popularity, user-demo, tag overlap, goal category). |
| 4 | D1 | Minimal prompt refinement on Qwen 1.5B (1 directive: "cite one concrete musical detail") | response-prompt | +0.004–0.011 | S | Cheap test of whether 1.5B has any prompt headroom above 021; 022's blunt persona failed but surgical nudge might not. |
| 5 | D2 | Qwen 3B + narrow prompt (2 bans: "fantastic", "perfectly captures") | response-model+prompt | +0.008–0.023 | S | Tests if surgical guardrails thread the needle between 022 (full persona failed) and 024 (stock + 3B failed). Prior-branch v10 (3B + persona) got LLM 3.25. |
| 6 | A3 | Cross-encoder fine-tune on train (query, gold) pairs | retrieval | +0.015–0.040 | ~1 day | Competes with A1. Pick A1 if you prefer features-based interpretability; A3 if you want end-to-end learned semantics. Skip if A1 succeeds. |
| 7 | C3 | Doc2Query offline expansion (Qwen-0.5B, 3 queries/track) | retrieval | +0.010–0.025 | M one-time | Reusable across Blind-A + Blind-B; classical IR trick. Addresses BM25 vocab gap on conversational queries. |
| 8 | B2 | Re-ship 021 twice to measure blind noise floor | calibration | 0 direct | 10 min + 2 slots | Only if offline harness delayed and we need to disambiguate an ambiguous blind result. |
| 9 | C1 | Audio-CLAP text→audio retrieval as 4th wRRF branch | retrieval | +0.005–0.010 | S | CLAP's text-audio alignment is imperfect; prior dev-branch exp 011 (CLAP-only) was comparable-to-BM25 not better. Low ceiling but diversifies. |
| 10 | C2 | Popularity prior conditioned on `conversation_goal.category` | retrieval | +0.003–0.005 | XS | Low standalone lift; bigger as a LambdaMART feature. Near-zero downside. |
| 11 | E1 | LoRA fine-tune Qwen 3B on distilled 021 outputs | training | +0.015–0.038 | L | Only after A1+A2+D1/D2 fail to close to 0.38. Prior-branch LoRA on raw train tanked LLM by −1.45; distilling from OUR 021 calibrated outputs avoids that, but still unproven. |
| 12 | E2 | End-to-end SID (Text2Tracks/LIGER): RQ-VAE over cf-bpr + Qwen SFT | training | track-leader | XL, multi-day | Only if A1+A3 both fail AND we have Blind-B runway (release 2026-06-15). |

## Submission discipline going forward

- **No Blind-A submission without offline pre-validation** once B1 exists. Gate: offline composite ≥ prior champion + 0.01, and offline nDCG ≥ wRRF floor.
- **One axis per Blind-A submission.** Stacking 022/023/024 was the methodology mistake. Change ONE thing vs the champion.
- **Budget awareness**: 3/week cap per plan §2.6. Two reserved for final retrain → ≤1 exploratory blind/week.
- **Noise floor**: |Δcomposite| < 0.03 is inconclusive.

## Falsified / shelved mechanisms (don't retry on 1.5B)

| Dead end | Evidence | Don't retry unless |
|---|---|---|
| Persona prompt + word-ban (full 10-directive) | exp 022: LLM −1.00 on Qwen 1.5B | Move to 3B AND trim to 2–3 rules |
| Qwen 3B + stock prompt + max_new=192 | exp 024: LLM −0.15 via AI-speak inflation | Add explicit filler-ban directives |
| BGE-reranker-v2-m3 off-the-shelf | exp 023: nDCG@20 −0.12 | Fine-tune on train data first |
| Stacking 2+ changes per blind submission | exps 022/023/024 all stacked + regressed | Offline pre-validation green on all components |

## Update protocol — when any result lands

1. Update `documents/submissions_log.md` tabular row.
2. Add narrative to `documents/experiments_log.md` (hypothesis, result, 3–4 lessons, verdict, suggests-next).
3. Update `documents/benchmarks.md` B-champ pointer if composite_retrieval margin ≥ σ.
4. Mark hypotheses validated/falsified in `documents/agent_memory.md` and auto-memory `project_fresh_model_state.md`.
5. Re-rank THIS file. Promote candidates that stack on validated mechanisms; strike those related to falsified ones.
6. Pick top unblocked candidate; commit + push new config + Colab notebook; ship.
