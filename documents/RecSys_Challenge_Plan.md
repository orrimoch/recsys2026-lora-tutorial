# RecSys 2026 — Run for #1 (Music CRS Challenge)

## 1. Context

The RecSys Challenge 2026 is the TalkPlay Music Conversational Recommendation task: per turn, output (a) up to 20 ranked `track_id`s and (b) a natural-language response. Sessions are 8 turns. Composite leaderboard formula per our wiki (`papers/syntheses/winning-architecture.md:14`):

```
score = 0.50 · nDCG@20  +  0.10 · CatDiv  +  0.10 · LexDiv  +  0.30 · (LLM_judge − 1) / 4
```

> **Caveat:** the official challenge readme (`documents/recsys_challenge_notes.md:181`) states *"exact aggregation weights also not published."* The above formula is our **wiki best-guess** based on dimensions the org has disclosed (Personalization + Explanation Quality on Blind, retrieval nDCG, diversity). Treat all "composite" metrics in this plan as a proxy; the actual leaderboard composite may weight differently.

`LLM_judge` = closed Google **Gemini** scoring `predicted_response`. CatDiv and LexDiv are saturated; **the headroom is in nDCG@20 (~0.50 weight) and Gemini-judge (~0.30 weight)**.

**Today: 2026-05-02. Timeline:**
- Apr 10 → **Jun 23: Blind-A active**. The earlier "8/8 used; exhausted" claim was a misread: per-team weekly cap is the operational limit (3/week per `scripts/validate_prediction.py:172`), NOT a hard team budget. Blind-A is a live Gemini-judge signal source we should USE — submit W4-W7 dry-runs against it for cheap calibration before Blind-B opens.
- **Jun 23: Blind-B releases.**
- Jun 30: Challenge ends. Jul 9: Final code submission. Jul 20: Paper.

**Current champion:** exp 021 (`wRRF[BM25 4-field + dense-metadata-Qwen3 + dense-lyrics-Qwen3]` + Qwen-2.5-1.5B-Instruct + stock prompt). Composite ~0.33, nDCG@20 ~0.19, Gemini judge ~3.15. Every Blind-A variant since (022–029) regressed because Qwen-1.5B exhibits **structural-directive collapse** — any prompt structure trades grounding for diversity, Gemini punishes by ~−0.9 to −1.05.

**Why this plan exists.** Prompt-engineering on Qwen-1.5B is exhausted. Remaining levers: (i) **agentic retrieval** to push nDCG@20 higher, and (ii) a **fine-tuned response LLM** (Qwen-3B/7B) trained with **verifiable reward** to push Gemini-judge higher without collapsing grounding. The user has Colab Pro+ (A100), no Gemini API access, and wants components evaluated separately so progress is monitorable.

**Divergence from the wiki we explicitly own:** `papers/syntheses/winning-architecture.md:31-32` argues for **joint** training (Search-R1 multi-turn rollouts) as the win condition. This plan **demotes joint training to a W8 stretch** because (a) the verifiable-reward subproblem is unsolved at the single-turn level (`papers/concepts/verifiable-rewards.md:67-70`), so multi-turn rollouts have no stable reward, and (b) component-wise evaluation is a hard user requirement. This divergence is intentional and reversed in W8 if foundations land.

---

## 2. North-Star Architecture

```
┌──────────────────────── per-turn pipeline ────────────────────────┐
│  [user_query, chat_history, user_profile, conversation_goal]      │
│                       │                                           │
│  ┌──────────────  Component A: Agentic Retrieval  ──────────────┐ │
│  │  A1. State Tracker (RA-Rec) — Qwen-2.5-1.5B (already loaded) │ │
│  │       JSON {mood,intent,energy,sonic,era,...}                │ │
│  │       Parse-fail policy: retry@0.3 → cached → drop session   │ │
│  │  A2. Multi-Query Rewriter (CMQR) — N=4 rewrites              │ │
│  │       Per-rewrite top-50, fuse all 12 streams via RRF k=60   │ │
│  │  A3. wRRF retrieval (existing, top-100 not top-20)           │ │
│  │  A4. Cold-start specialist (MARec, deferred to W7 polish)    │ │
│  │  A5. Listwise LLM Reranker — DEFAULT: ProRank-0.5B           │ │
│  │       Stretch upgrade: Rank-R1 (Qwen-7B GRPO) only if W3 EV  │ │
│  │       Outputs top-20 + per-item rationale via `<rerank>`     │ │
│  │  A6. Catalog-membership filter (hard guard)                  │ │
│  │       Drop hallucinated UUIDs, replace from retrieval pool   │ │
│  │  A7. Post-fusion dedupe (hard guard)                         │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                       │ top-20 + rationales                       │
│  ┌────────────  Component B: Fine-Tuned Responder  ─────────────┐ │
│  │  Qwen-2.5-7B + LoRA r=32                                     │ │
│  │  Training ladder: KTO → (S-DPO opt) → Rank-GRPO              │ │
│  │  Reward: see §6                                              │ │
│  │  Prompt: extends `response_generation_cot_user_state.txt`    │ │
│  │          with new <reranker_rationales> slot                 │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                       │                                           │
│  → predicted_track_ids (top-20, deduped, all in catalog),         │
│    predicted_response (JSON written with ensure_ascii=False)      │
└───────────────────────────────────────────────────────────────────┘
```

**Two design commitments:**

1. **Components evaluated separately** at every milestone. A is scored on dev nDCG@1/10/20 with a frozen baseline responder (exp 021 stock Qwen-1.5B). B is scored on dev *responder composite* with a frozen baseline retriever. End-to-end only at W7 + W8.
2. **No-regression gate at every B-stage:** dev nDCG@20 must not drop > 0.005 vs frozen-retriever baseline at gates B1, B2, B3 — even if local proxy reward goes up. Real defense against structural-directive collapse re-emerging on 7B (the dismissal in earlier drafts of this plan was too glib).

---

## 3. Inference & External-API Electives (consolidated)

**CodaBench evaluates `prediction.json`, it does not run our code.** We generate the JSON in Colab and upload it. Whatever runs on Colab to *produce* the JSON (vLLM Qwen, external API calls, multi-agent orchestration) is our choice. Constraints are practical:

| Constraint | Implication |
|---|---|
| Final code submission Jul 9 | Pipeline self-contained + reproducible. External-API path requires fallback + documented credential handling |
| **No Gemini API access** | Cannot use Gemini at inference or as inline judge — must use local Qwen via vLLM |
| Latency budget | 8000 turns × ~2s/turn batched on A100 ≈ 5 hr; budget 12 hr max per Blind run |
| Optional Anthropic API | Claude API as orchestrator/responder ≈ $50–150 per Blind run; documented as elective only |

**Plan defaults to a 100% local Qwen pipeline** (vLLM + LoRA adapters). External-API augmentation is an elective enabled only if local plateaus 1–2 weeks before deadline. We do not depend on it.

> *On "Claude Code at inference":* Claude Code is an interactive CLI, not a programmatic inference component. The supported path for Anthropic-grade reasoning at inference would be the Anthropic Python SDK directly. Treat as the documented elective.

### VRAM budget — A100-40GB

We need to host on the same Colab session:

| Model | VRAM (bf16) | Role |
|---|---|---|
| Qwen-2.5-1.5B (base, vLLM) | ~3 GB | A1 state tracker + A2 CMQR rewriter |
| ProRank-0.5B (cross-encoder) | ~1 GB | A5 reranker (default) |
| Qwen-2.5-7B + LoRA r=32 (vLLM) | ~16 GB + LoRA ~150 MB | B responder |
| KV cache + activation | ~8 GB | per-batch |
| Headroom | ~10 GB | safety margin |

**Total ≈ 28 GB on a 40 GB A100** — fits with ~10 GB headroom. If we escalate to Rank-R1 (7B reranker) in addition to 7B responder, that's ~32 GB → too tight. **Cut-path:** if we train Rank-R1, hot-swap by unloading the responder during reranker rollouts and vice versa, OR run reranker training in a separate Colab session (offline checkpoint, then ship adapter).

---

## 4. Phase Roadmap (8 weeks, gated, with explicit cut-paths)

| Week | Phase | Deliverable | Go/No-Go Gate |
|---|---|---|---|
| **W1 (May 1–7)** | Foundation + reward correlation study + state-tracker prototype (parallel) | Reward function validated; W1 anchor dataset built | Spearman(R_turn − R_judge, true-Gemini-anchored composite) ≥ 0.7 with **95% CI lower bound ≥ 0.5**. ALSO: R_rule correlates ≥ 0.4 with composite *independently* (verifies it can carry judge proxy weight). |
| **W2 (May 8–14)** | Component A V1 — State Tracker + CMQR | Dev nDCG@10 reported (top-100 retrieval; not top-20) | nDCG@10 ≥ champion +0.005, OR diagnosed why and continue |
| **W3 (May 15–21)** | Component A V2 — ProRank reranker (default) | Dev nDCG@10 with reranker | nDCG@10 ≥ champion +0.015. **Cut-path:** if ProRank lands gate, optionally launch Rank-R1 training on a side Colab session for W4–W6 (overlaps with B); ship whichever wins at W7 |
| **W4 (May 22–28)** | Component B Stage 1 — KTO warmup | Dev responder composite (R_retr + R_rule + R_judge) | Format compliance ≥ 95%; **dev nDCG@20 not regressed > 0.005** vs frozen retriever |
| **W5 (May 29–Jun 4)** | Component B Stage 2 — S-DPO **(conditional, see cut-path)** | Dev responder composite | Run S-DPO only if KTO format compliance < 70%; otherwise free this week for Rank-GRPO debug or earlier integration |
| **W6 (Jun 5–11)** | Component B Stage 3 — Rank-GRPO main loop | Trained Qwen-7B-LoRA adapter | +0.03 R_turn over B1; **dev nDCG@20 not regressed > 0.005**; 50-rollout qual review |
| **W7 (Jun 12–18)** | Integration + retrain on train+dev (~28 A100-hr explicitly budgeted) | First Blind-B submission | Dev composite ≥ exp-021 anchor; submission validates via `scripts/validate_prediction.py` |
| **W8 (Jun 19–25)** | Iterate Blind-B + (stretch) joint Search-R1 rollout | Champion ensemble | Final submission Jun 30. Joint training has hard cut-off Jun 22; revert to W7 champion if not converging |

**Explicit slack:** 5 days between W8 end (Jun 25) and challenge end (Jun 30) for unexpected debugging. Final code submission Jul 9 uses W8 champion.

**State-tracker prototype is moved to W1 parallel work** (no dependency on the reward study) so W2 has only one new lever (CMQR) to evaluate.

---

## 5. Component A — Agentic Retrieval

### A1. State Tracker (RA-Rec pattern)

**Paper:** `documents/research/RA-Rec_2406.00033.pdf`.

**Implementation:**
- New file: `music-crs-baselines/mcrs/query_rewriters/state_tracker.py` (fills empty stub).
- **Model: Qwen-2.5-1.5B** (already loaded for the existing exp 021 pipeline — no extra weight load needed). Upgrade to 3B only if parse-validity < 99%.
- Reuse the existing CoT prompt schema for the `<user_state>` block (keys: `mood, intent, energy, sonic_pref, era_pref, language_pref, familiarity_pref, recent_signals, avoid`).
- Cache extracted states keyed by `(session_id, turn_number)` under `cache/state/`.
- **Parse-failure policy (NEW):** on JSON parse failure: retry once at temp=0.3 → if still failing, use cached prior-turn state from same session → if no cached state, drop the session from the *training* rollout batch (do NOT zero-reward the responder for a retrieval-stack failure). At inference time on dev/Blind, fall back to `last_user` query (no state injection) and continue.

### A2. Multi-Query Rewriter (CMQR)

**Paper:** `documents/research/CMQR_2406.18960.pdf`.

**Implementation:**
- New file: `music-crs-baselines/mcrs/query_rewriters/cmqr.py`.
- Single Qwen-1.5B call emits N=4 rewrites; each injects 1–2 fields from the user_state JSON.
- Wire into `wrrf_bm25_dense_lyrics_v1`: each rewrite hits all 3 streams at **top-50 per stream** (NOT top-100 per stream, so RRF doesn't dominate at the depth tail). Fuse all 12 ranked lists via RRF k=60 → keep top-100 union for downstream A5.
- **Mandatory post-fusion dedupe** (hard guard, see §A7). Cache rewrites by `(session_id, turn_number)`.

### A3. Existing wRRF retrieval (re-use)

`mcrs/retrieval_modules/__init__.py` keys `wrrf_bm25_dense_metadata_v1`, `wrrf_bm25_dense_lyrics_v1`, `cf_bpr`. **Increase `retrieval_topk` to 100** in every config from W2 onward (so A5 has signal to rerank).

### A4. Cold-start specialist (MARec) — DEFERRED to W7 polish

**Paper:** `documents/research/MARec_2404.13298.pdf`.

Rationale: A1+A2+A5 are higher EV; defer MARec until W7 *if* B-stage is on-track. Implementation file: `music-crs-baselines/mcrs/embedders/marec.py`.

**Blind-B catalog drift fallback (NEW):** for any track in Blind-B not seen at training time (so no MARec row), inherit the existing imputation chain from `mcrs/retrieval_modules/dense_precomputed.py`: artist-mean → category-mean → global-mean. MARec must wrap this fallback, not replace it.

### A5. Listwise reranker — DEFAULT: ProRank (NEW)

**Default paper:** `documents/research/ProRank_2506.03487.pdf` — 0.5B SLM, last-token-logit-diff scoring, 2-stage GRPO warmup, beats 32B rerankers on BEIR. `recent_papers_ideas.md:64` flags it "highest ROI-per-GPU-hour."

**Stretch upgrade:** `documents/research/Rank-R1_2503.06034.pdf` — Qwen-2.5-7B + LoRA + GRPO. Only attempted if ProRank lands the W3 gate AND we have a side Colab session free.

**Why ProRank as default (was Rank-R1 in earlier draft):**

| | ProRank-0.5B | Rank-R1 (Qwen-7B) |
|---|---|---|
| Training compute (A100) | ~2 hr | **28–107 hr** (G=4 × 8k turns × 8 epochs at vLLM throughput) |
| Inference latency | ~50ms/turn | ~2s/turn |
| Quality (BEIR-reported) | matches/beats Rank-R1 | strong but marginal vs ProRank for reranker tasks |

The earlier draft's "12 A100-hr" estimate for Rank-R1 was off by ~10× (correct math: 600 output tokens / 400 tok/s effective × 256k rollouts ≈ 107 hr at G=4×8 epochs). ProRank avoids the budget bug entirely.

**Implementation:**
- New file: `music-crs-baselines/mcrs/rerankers/pro_rank.py`.
- Input: (query, user_state, top-100 candidate metadata).
- Training: GRPO with reward = nDCG@10 of reranked top-20 against gold. ProRank's prompt-warmup stage initializes the policy first (per paper §3).
- Output: ranked top-20 + per-item one-line rationale (3–5 words: e.g., "matches era-pref + low-energy mood"). Rationale slot piped to Component B's prompt via the new `<reranker_rationales>` block (see §6.5).

**Cut-path:** if ProRank training is unstable, fall back to BGE-reranker-v2-m3 (already in the codebase at `mcrs/rerankers/bge_reranker.py`) for a no-training reranker baseline. Ship that to Blind-B as a safety net, escalate to Rank-R1 only if a paper-faithful ProRank under-performs the BGE no-training baseline.

### A6. Catalog-membership hard guard (NEW — addresses P0)

**Why:** trained Qwen models can hallucinate UUIDs that aren't in the catalog. `recsys_challenge_notes.md:144` makes this an absolute submission requirement. `metrics_recsys.py:127` raises `ValueError` on duplicates.

**Implementation:**
- New helper in `mcrs/crs_baseline.py`: `_filter_catalog_membership(predicted_track_ids, valid_set, retrieval_pool)`.
- After A5 outputs top-20 IDs, drop any ID not in `MusicCatalogDB.id_to_metadata`. Backfill from the retrieval pool (top-100 from A3) in original wRRF order.
- Inside the Component B reward pipeline (training): zero `R_turn` if any output ID is invalid. This makes the guard part of the policy gradient.
- **Test:** `scripts/validate_prediction.py` already validates schema (existing file); extend it to also check catalog membership.

### A7. Post-fusion dedupe (NEW — addresses P1)

**Why:** CMQR fuses 12 lists; same track appears multiple times. Rank-R1/ProRank can re-emit duplicates. `metrics_recsys.py:127` raises `ValueError`.

**Implementation:** single helper applied at every fusion step:
```python
def dedupe_keep_first(ids: list[str]) -> list[str]:
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out
```
Mandate at the boundary of every list-producing module (CMQR fusion, A5 rerank output, final top-20 emission).

---

## 6. Component B — Fine-Tuned Responder + Verifiable Reward

> **The most important section. The verifiable-reward subproblem is unsolved (`papers/concepts/verifiable-rewards.md:67-70`). Treat correlation validation as a HARD gate before any RL training.**

### 6.1 Reward Composition

> **W1 empirical revision (2026-05-01) — Option B.** Original draft:
> `R_turn = 0.50·R_retr + 0.20·R_judge + 0.20·R_rule + 0.05·R_format + 0.05·R_user_prof`.
> The W1 gate (`data/reward_gate_results.json`) empirically falsified two assumptions:
>   1. R_rule does NOT predict GPA judge labels on gold train data
>      (Spearman 0.012, AUC 0.507 ≈ random). Demoted to style guardrail.
>   2. R_user_prof has mean 0.009 on gold data — gold assistants never reference
>      user country/age/gender. Term dropped (weight → 0).
> Mass shifted to R_retr (which Source B + Source C confirmed IS the leaderboard
> nDCG term, Spearman 1.0 + composite formula matches within 0.005). R_judge
> kept at low 0.10 weight with trust-gating. Implementation: `scripts/reward_fns.py`
> `W_RETR = 0.70 ... W_USER_PROF = 0.00`.

Per-turn scalar `R_turn ∈ [0, 1]` (revised):

```
R_turn = 0.70 · R_retr        # nDCG-shaped, free, dense — confirmed by W1 G3 + Source C
       + 0.10 · R_judge       # local distilled judge (trust-gated; no Gemini API)
       + 0.15 · R_rule        # style guardrail (NOT a judge proxy — W1 falsified)
       + 0.05 · R_format      # well-formed envelope
       + 0.00 · R_user_prof   # DROPPED — no signal on gold data (mean 0.009)
```

**R_div moved to session-level only** (was per-turn 0.05 in earlier draft, but Distinct-2 of a single response is undefined). New session reward:

```
R_session = mean(R_turn over 8 turns)
          + 0.10 · monotonicity(R_judge across turns)
          − 0.05 · var(R_turn)
          + 0.10 · clipped(LexDiv across 8 responses)        # NEW location for R_div
          + 0.05 · clipped(CatDiv across 8 × 20 tracks)
```

**Catalog-membership penalty (multiplicative, hard):**
```
R_turn := 0.0   if any predicted_track_id ∉ catalog
R_turn := 0.0   if format envelope fails to parse
```

**Why this weight allocation now matches the leaderboard ratio:** R_retr 0.50 = composite nDCG 0.50. R_judge 0.20 + R_rule 0.20 + R_user_prof 0.05 = 0.45 standing in for composite judge 0.30 (slack absorbs noise: `R_judge` is uncertain without Gemini calibration; R_rule + R_user_prof act as judge-correlated mechanical proxies). Diversity 0.15 (session) ≈ composite CatDiv+LexDiv 0.20.

#### R_retr — retrieval-accuracy term (free, dense)

Reuse the *exact* function from the leaderboard, `music-crs-evaluator/metrics/metrics_recsys.py:get_ndcg`. Don't proxy what we can compute exactly:

```python
def r_retr(predicted_track_ids: list[str], gold_track_id: str) -> float:
    g = [gold_track_id]
    return 0.5 * get_ndcg(g, predicted_track_ids, 20) \
         + 0.3 * get_ndcg(g, predicted_track_ids, 10) \
         + 0.2 * get_ndcg(g, predicted_track_ids, 1)
```

#### R_rule — response rule-based term (free, dense)

```python
import re
BAD = re.compile(r"\b(absolutely|fantastic|sorry|unfortunately|i (cannot|can't))\b", re.I)
WHY = re.compile(
    r"\b(because|since|features|leans|driven by|atmosphere|tempo|groove|"
    r"arrangement|released|from \d{4}|era|decade)\b", re.I
)  # `released`, `era`, `decade` added (P2 — release_date wiring)

def r_rule(response: str, top1_meta: dict, user_state: dict, history_text: str) -> float:
    score = 0.0
    score += 0.20 * (top1_meta["track_name"].lower()  in response.lower())
    score += 0.15 * (top1_meta["artist_name"].lower() in response.lower())
    score += 0.15 * bool(WHY.search(response))
    # English-only assumption documented; for multilingual responses, supplement
    # with lemma-based matching (P2 future). For now we accept underscoring of
    # non-English responses and rely on R_judge to capture them.
    score += 0.10 * any(v.lower() in response.lower()
                        for v in user_state.values()
                        if isinstance(v, str) and v not in {"unknown", ""})
    score += 0.10 * (60 <= len(response.split()) <= 110)
    score += 0.10 * (1 <= response.count(".") + response.count("?") <= 4)
    score += 0.10 * (not BAD.search(response))
    score += 0.10 * any(tok in response.lower()
                        for tok in history_text.lower().split() if len(tok) > 4)
    return min(score, 1.0)
```

#### R_user_prof — user-profile mention (NEW, addresses P1 #8)

`UserProfileDB` exposes `age_group, gender, country_name`. Personalization is driven by reflecting these in the response. Wiki `verifiable-rewards.md:38` lists this explicitly.

```python
def r_user_prof(response: str, user_profile: dict | None) -> float:
    if not user_profile: return 0.0
    score = 0.0
    country = user_profile.get("country_name", "").lower()
    age = user_profile.get("age_group", "").lower()
    if country and country in response.lower(): score += 0.5
    if age and any(t in response.lower() for t in [age, age.replace("-", " ")]): score += 0.5
    return min(score, 1.0)
```

#### R_judge — local distilled judge (no Gemini)

**Re-uses existing pipeline.** `scripts/build_reward_dataset.py` extracts ~60–80k binary-labeled `(context, response, GPA_label)` pairs from train. The cross-encoder reward model in `mcrs/response_rerankers/reward_reranker.py` consumes them. Train Qwen-2.5-0.5B + LoRA cross-encoder head; output scalar in [0,1] = P(MOVES_TOWARD_GOAL).

**Calibration anchors (P0 fix #4):** **NOT 9 historical Blind-A submissions** (statistically vacuous, Spearman SE ~ 0.30 at n=9). Instead use:
- **Positive anchors (~8k):** the 8000 *gold* `system_response` strings from train, paired with their actual context.
- **Negative anchors (~12k):** all responses from train turns where `goal_progress_assessment == DOES_NOT_MOVE_TOWARD_GOAL`.

This gives ~20k labeled points → isotonic regression / monotone calibration is statistically meaningful, Spearman CI is tight.

**Judging-Judges position-bias correction (P2 #22):** apply swap-and-average at calibration time *even for the local judge* — when scoring `(context, candidate_A, candidate_B)`, also score the swapped order and average. Costs nothing; halves position bias.

**Trust gating:** `R_judge` weight in `R_turn` is multiplied by `min(1, recent_spearman / 0.7)` — if calibration drifts, the term auto-degrades.

#### R_format — well-formed envelope

```python
ENVELOPE = re.compile(r"<user_state>(.*?)</user_state>\s*<response>(.*?)</response>", re.S)
def r_format(text: str) -> float:
    return 1.0 if ENVELOPE.search(text) else 0.0
```

### 6.2 Reward Validation (W1 — HARD GATE, fixed from earlier draft)

**The earlier draft's gate was circular** (correlated `R_turn` against itself). Fixed protocol:

1. **Build the anchor dataset (W1 day 1–2):**
   - Take all cached prior-experiment rollouts: every `predicted_response` from exp 020–029 stored at `music-crs-baselines/exp/inference/devset/*.json`.
   - For exp 021–029 (Blind-A submissions), look up the *actual* Gemini score logged from CodaBench (these scores are recorded in git history, e.g., commit `a81696b`).
   - For dev rollouts without Gemini scores, use the gold `system_response` as a proxy "perfect" response (R_judge_anchor = 1.0) and GPA-DOES_NOT_MOVE responses as anchor 0.0.
   - Total: ~9 Blind-anchored sessions (rare, high-trust) + ~8k dev-anchored rollouts (abundant, lower-trust).

2. **Construct the reference composite for each anchor row:**
   ```
   composite_ref = 0.50·nDCG@20 + 0.10·CatDiv + 0.10·LexDiv + 0.30·(judge_anchor − 1)/4
   ```
   where `judge_anchor` is the Gemini score on Blind rows or the GPA-derived score (1 or 5) on dev rows.

3. **Compute `R_turn_no_judge = R_turn − 0.20·R_judge`** for each rollout. Note we **exclude** R_judge from the LHS — this is the bug fix. We test whether the *mechanical* terms (R_retr + R_rule + R_format + R_user_prof) correlate with the Gemini-anchored composite.

4. **Required gates:**
   - Spearman(`R_turn_no_judge`, `composite_ref`) ≥ **0.70 with 95% CI lower bound ≥ 0.50** on the dev anchor set (~8k rows, CI tight).
   - Spearman(`R_rule`, `composite_ref`) ≥ **0.40** independently (verifies R_rule can carry the judge-proxy weight).
   - Sanity: `R_retr` correlation with `composite_ref` ≥ 0.6 (it should be strong since nDCG dominates the composite).
5. **If gates fail:** re-weight or drop misaligned terms (likely culprits: R_user_prof if profile mention is rare in gold, R_format if envelope isn't natural). Re-validate. Worst-case fallback: `0.7·R_retr + 0.3·R_rule` reward (no R_judge, no R_user_prof) — proven mechanical-only baseline.

**No GRPO training before this gate passes.**

### 6.3 Training Recipe — KTO → (S-DPO conditional) → Rank-GRPO

| Stage | Algorithm | Data | Compute (A100) | Gate |
|---|---|---|---|---|
| **B1 (W4)** | **KTO** (LoRA r=32) | ~60k binary GPA pairs, augmented with envelope (NEW) | ~3 hr | Format ≥ 95%; **dev nDCG@20 not regressed > 0.005** |
| **B2 (W5)** | **S-DPO** (Plackett-Luce) — *conditional* | ~30k preference sets | ~6 hr | Run only if KTO format < 70% |
| **B3 (W6)** | **Rank-GRPO** (LoRA r=32) | on-policy, G=2 (was G=4) | ~10 hr | +0.03 R_turn vs B1; nDCG@20 not regressed |

**LoRA rank standardized at r=32 across stages** (P2 #21).

**Why this order:**
- **KTO** — cheapest first pass; binary labels we already have.
- **S-DPO is now conditional** (P1 #6) — frees a week if KTO format is fine.
- **Rank-GRPO** with **G=2 not G=4** (cuts compute ~50% to ~10 A100-hr from previously claimed 12, which was already wrong). Backup: Rec-R1 if Rank-GRPO unstable.

**KTO data envelope augmentation (P1 #17):** the existing `build_reward_dataset.py` outputs `(text_a, text_b, label)` *without* the `<user_state>...<response>` envelope. **Wrap** every `text_b` row in the envelope using cached A1 state-tracker outputs (pre-computed for all train turns in W1) before training so B1 and B3 share format expectations.

### 6.3.1 TRL Operational Requirements (post-skill audit)

These requirements come from auditing W4–W6 against the `huggingface-llm-trainer` skill. The skill's failure-mode catalog (50%+ of training failures attributable to dataset format issues, eval-strategy hangs, ephemeral-environment data loss) is incorporated as hard constraints:

**Dataset format (P0 — must validate before any GPU run):**
- `scripts/build_trl_datasets.py` converts `data/reward_train.parquet` → three TRL-shape parquets under `data/trl/`:
  - `kto.parquet`     — columns: `prompt`, `completion`, `label` (bool). One-to-one map.
  - `dpo.parquet`     — columns: `prompt`, `chosen`, `rejected`. Random POS↔NEG pairing (vanilla DPO; S-DPO with N hard negs is built separately for W5).
  - `grpo_prompts.parquet` — columns: `prompt`. Deduped from text_a context.
- Inline schema validators raise `ValueError` on column / dtype drift before write.
- Pytest contract: `tests/test_build_trl_datasets.py` (27 tests; all pass).
- **Hard pre-flight check** before each W4–W6 Colab run: upload the relevant TRL parquet to a private HF dataset repo and run the official dataset inspector (`https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py`) — output must be `✓ READY` for the target trainer.

**Trainer config defaults (TRL ≥ 0.12.0):**
- Use `max_length` (not `max_seq_length`) — `SFTConfig(max_length=2048)`. Default 1024 is fine for most cases; bump only for the CoT envelope + reranker rationales prompt.
- **`eval_dataset` MUST be provided whenever `eval_strategy` is set.** Skill flags this as a silent-hang gotcha. Default config:
  ```python
  ds_split = ds.train_test_split(test_size=0.1, seed=42)
  KTOConfig(eval_strategy="steps", eval_steps=100)  # NEEDS eval_dataset=ds_split["test"]
  ```
  If we don't have eval data for a stage, set `eval_strategy="no"` explicitly. Never enable eval without providing the dataset.
- LoRA defaults: `LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear")`. Plan §6.3 standardizes r=32 across B1/B2/B3.

**Persistence (Colab is ephemeral):**
- Skill mandate "MUST push to Hub or ALL training results are lost" applies to Colab runtime resets too. Every B-stage config must set:
  ```python
  push_to_hub=True
  hub_model_id="orrimoch/recsys2026-{stage}-{date}"   # e.g., recsys2026-kto-2026-05-22
  hub_strategy="every_save"                            # ship checkpoints, not just final
  hub_private_repo=True                                # private until ready
  ```
- AND continue to checkpoint to Drive every 500 steps (plan §10) as a redundant safety.
- Pre-flight: `huggingface-cli login` cell at top of every training notebook; `secrets={"HF_TOKEN": "$HF_TOKEN"}` if we ever migrate to HF Jobs.

**Monitoring (Trackio mandatory):**
- Every B-stage trainer config sets `report_to="trackio"`. Trackio works in Colab — writes to a private HF Space we own (`orrimoch/recsys2026-trackio`).
- Per-run: `trackio.init(project="recsys2026", run_name="b1-kto-2026-05-22", group="b-stage")`. Group `b-stage` ties B1/B2/B3 runs into one comparable dashboard.
- Cost: zero (Trackio is free on HF). One-time setup: create the Space.

**OOM recovery ladder (skill's documented sequence):**
1. `per_device_train_batch_size=1` + `gradient_accumulation_steps=8` (target effective batch ≈ 128 by accumulating).
2. `gradient_checkpointing=True` (~30% slower, ~50% less VRAM).
3. Drop sequence length: `max_length=1024` → `512` (only if step 1+2 don't fit).
4. **Unsloth fallback** (`references/unsloth.md` in the installed skill): drop-in replacement for `model="Qwen/..."` → `FastLanguageModel.from_pretrained(...)`. ~60% less VRAM, ~2× faster. Worth trying on Rank-GRPO with G=4 if standard TRL OOMs at G=2.
5. Upgrade hardware: A100-40GB → A100-80GB (~$5–10/hr more) or split across 2× A100s via Accelerate.

**Cost / time estimation:** before each B-stage run, use `scripts/estimate_cost.py` from the installed skill (`uv run .claude/skills/huggingface-llm-trainer/scripts/estimate_cost.py --help`). Numbers don't apply 1:1 (skill is HF Jobs $/hr, we're Colab Pro+ flat-rate) but the *time* estimates do. Add 30% buffer per skill recommendation.

### 6.4 Data Construction (concrete quantities)

- **KTO data:** all `(context, gold_response, GPA_label)` rows from `build_reward_dataset.py`, envelope-augmented. ~60k rows.
- **S-DPO data per turn (if used):** 1 positive + 6 negatives:
  - 3 hard-track negs (BM25 top-2..4 not gold)
  - 2 response-only negs (drop track-name, banned phrase, mutate state echo)
  - 1 GPA-DOES_NOT_MOVE response
- **Rank-GRPO rollouts:** G=2, ~25k optimizer steps × 2 candidates × 32 turns/step ≈ 1.6M rollouts. All scored locally.

### 6.5 Rank-R1/ProRank → Component B prompt feed-through (P1 #10)

The reranker A5 emits per-track rationales (3–5 words each). These feed into Component B via a NEW prompt block extending `mcrs/system_prompts/response_generation_cot_user_state.txt`:

```
... existing stuff ...
<reranker_rationales>
1. {track_name_1} — {rationale_1, max 8 tokens}
2. {track_name_2} — {rationale_2}
...
5. {track_name_5} — {rationale_5}
</reranker_rationales>
```

Token budget: 5 rationales × 8 tokens = 40 tokens; plus newlines, ~50 tokens total. Negligible against the 2048-token context.

**Critical:** R_judge training data must include this slot (filled by the reranker). Build path: re-extract `build_reward_dataset.py` after A5 lands, with the rationale block included in `text_a`. Else B3's R_format will diverge from B1's training distribution.

### 6.6 Reward-Hacking Guardrails

| Risk | Guardrail |
|---|---|
| Mention-everything | R_rule mention bonus capped per length; R_judge down-ranks; format-length cap |
| Empty "why" clauses | `WHY` regex AND ≥3 distinct musical-detail tokens across session |
| User-state echo gaming | Lemma-level match only on values not already in user query (novelty required) |
| Format-envelope collapse | `R_format=0` is a HARD floor; +1.0 KL penalty if envelope breaks |
| **Lexical-diversity collapse (the 022/029 failure mode)** | Session-level distinct-2 ≥ 0.6 floor; otherwise zero R_div and −0.05 penalty. **Plus the no-regression gate: dev nDCG@20 must not drop > 0.005.** |
| Top-K diversity gaming | R_retr 0.50 dominates; CatDiv capped at 0.05 |
| Judge over-reliance | R_judge multiplied by `min(1, recent_spearman / 0.7)` — auto-degrades on drift |
| **Catalog hallucination (NEW, P0)** | Hard zero on R_turn if any predicted_track_id ∉ catalog. Post-hoc filter swaps invalid IDs from retrieval pool. |
| **Structural-directive collapse re-emerging on 7B** | Per-stage no-regression gate on dev nDCG@20; 50-rollout qual review at every B-stage gate; if collapse detected, revert to last good checkpoint + re-weight R_div lower. |

### 6.7 Stretch — Joint Search-R1 Rollout (W8 only)

Per `papers/syntheses/winning-architecture.md:31-32` — eventual best architecture trains components jointly. **Deferred** to W8 because (a) multi-turn rollouts need mature single-turn rewards (W4–W6 builds those), (b) Search-R1 reward shape for music-CRS is unsolved (`papers/concepts/verifiable-rewards.md:67`).

If W7 ships above-baseline to Blind-B, attempt joint training in W8. Trainer: `verl` or `open-r1` per `papers/repos/`. 8 A100-hr budget; **hard cut-off Jun 22**, revert to W7 champion if not converging.

---

## 7. Component-Wise Evaluation Plan

| Subject | Frozen Counter-Part | Metric | When |
|---|---|---|---|
| **A1+A2** (state tracker + CMQR) | Frozen Qwen-1.5B responder (exp 021), top-100 retrieval | dev nDCG@1/10/20 + parse-validity | W2 |
| **A5** (ProRank reranker) | Frozen Qwen-1.5B responder, wRRF top-100 | dev nDCG@1/10/20, no-regression vs A1+A2 | W3 |
| **A4** (MARec cold-start) | Same as A5, cold subset only | dev nDCG@10 cold | W7 polish (deferred) |
| **B1** (KTO) | Frozen wRRF (exp 021) | dev composite (R_retr + R_rule + R_judge); **dev nDCG@20 no-regression** | W4 |
| **B2** (S-DPO, conditional) | Frozen wRRF | dev composite; no-regression | W5 |
| **B3** (Rank-GRPO) | Frozen wRRF | dev composite + 50-rollout qual review; no-regression | W6 |
| **End-to-end** (A1+A2+A5+B3) | — | dev composite, then Blind-B | W7 |
| **Joint Search-R1** (stretch) | — | Blind-B v2 | W8 |

**Eval harness:**
- `music-crs-evaluator/evaluate_devset.py` for nDCG / CatDiv / LexDiv (re-use, no changes).
- `scripts/local_eval.py`, `scripts/offline_eval.py` (existing) — extend with no-regression checks.
- New: `scripts/responder_eval.py` for composite-style score using local R_judge in place of Gemini. **Must use `json.dump(..., ensure_ascii=False)`** when caching predictions.
- `scripts/validate_prediction.py` (existing) — extend with catalog-membership check (no IDs missing from `MusicCatalogDB`).

**Blind submission policy:** ≤1 Blind-B submission per week (5 submissions over W7–W8). Spend each on a distinctly-different config (W7 baseline integration, W8 best ensemble, W8 stretch joint). Diff one axis only — `feedback_experiment_ablation_discipline.md`.

---

## 8. Critical Files & Re-Use Map

### Re-use as-is

| File | Why |
|---|---|
| `music-crs-baselines/mcrs/retrieval_modules/__init__.py` | wRRF + cf-bpr, all retrievers |
| `music-crs-baselines/mcrs/lm_modules/vllm_model.py` | vLLM wrapper for Colab |
| `music-crs-baselines/mcrs/db_user/user_profile.py`, `mcrs/db_item/music_catalog.py` | data loading |
| `music-crs-baselines/mcrs/system_prompts/response_generation_cot_user_state.txt` | extends with `<reranker_rationales>` |
| `music-crs-baselines/mcrs/response_rerankers/reward_reranker.py` | local R_judge cross-encoder |
| `music-crs-baselines/mcrs/rerankers/bge_reranker.py` | A5 fallback safety-net |
| `music-crs-evaluator/metrics/metrics_recsys.py` (`get_ndcg`) | exact leaderboard function |
| `music-crs-evaluator/evaluate_devset.py` | dev eval, no changes |
| `scripts/build_reward_dataset.py` | extracts ~60–80k labeled GPA pairs |
| `scripts/build_trl_datasets.py` | converts the GPA parquet → TRL-validated KTO/DPO/GRPO formats (P0 fix from skill audit) |
| `tests/test_build_trl_datasets.py` | 27 pytest contracts for TRL schema (`prompt`/`chosen`/`rejected`/`completion`/`label` bool) |
| `.claude/skills/huggingface-llm-trainer/` | TRL/Unsloth/Trackio guidance + production scripts (SFT/DPO/GRPO templates, GGUF, cost estimator) |
| `.claude/skills/huggingface-trackio/`, `hf-cli/`, `huggingface-datasets/`, `huggingface-papers/` | supporting HF skills installed for W4–W6 |
| `scripts/local_eval.py`, `scripts/offline_eval.py`, `scripts/validate_prediction.py` | existing eval scaffolding |

### To create (all paths absolute under repo root)

| File | Component | LOC | Notes |
|---|---|---|---|
| `music-crs-baselines/mcrs/query_rewriters/state_tracker.py` | A1 | 150 | Includes parse-failure 3-step fallback |
| `music-crs-baselines/mcrs/query_rewriters/cmqr.py` | A2 | 200 | Dedupe at fusion |
| `music-crs-baselines/mcrs/embedders/marec.py` | A4 (W7) | 250 | Inherits artist→category→global imputation |
| `music-crs-baselines/mcrs/rerankers/pro_rank.py` | A5 default | 250 | Last-token-logit-diff |
| `music-crs-baselines/mcrs/rerankers/rank_r1.py` (stretch) | A5 stretch | 350 | Side-Colab training |
| `music-crs-baselines/mcrs/crs_baseline.py` (extension) | A6 + A7 | +50 | `_filter_catalog_membership`, `dedupe_keep_first` |
| `music-crs-baselines/mcrs/system_prompts/response_generation_cot_user_state.txt` | edit | +15 | Add `<reranker_rationales>` block |
| `music-crs-baselines/config/100-cmqr-state-qwen15b-devset.yaml` | A1+A2 | 30 | `track_split_types=["all_tracks"]`, `retrieval_topk=100` |
| `music-crs-baselines/config/110-prorank-rerank-devset.yaml` | A5 default | 30 | Same |
| `music-crs-baselines/config/200-responder-kto-qwen7b-devset.yaml` | B1 | 30 | LoRA r=32, envelope-augmented data |
| `music-crs-baselines/config/210-responder-sdpo-qwen7b-devset.yaml` | B2 (cond) | 30 | |
| `music-crs-baselines/config/220-responder-rgrpo-qwen7b-devset.yaml` | B3 | 30 | G=2, r=32 |
| `music-crs-baselines/config/300-final-blindset-B.yaml` | end-to-end | 30 | All hard guards on |
| `colab/01_reward_correlation_study.ipynb` | W1 gate | 250 | Anchor build + gate test |
| `colab/02_state_tracker_prototype.ipynb` | W1 parallel | 200 | |
| `colab/10_train_cmqr_dev.ipynb` | W2 | 200 | |
| `colab/20_train_prorank.ipynb` | W3 default | 250 | |
| `colab/21_train_rank_r1.ipynb` (stretch) | W3 stretch | 300 | Side Colab session |
| `colab/30_train_responder_kto.ipynb` | W4 | 250 | Envelope-augmented |
| `colab/31_train_responder_sdpo.ipynb` | W5 (cond) | 300 | |
| `colab/32_train_responder_rgrpo.ipynb` | W6 | 400 | |
| `colab/40_run_blindset_B.ipynb` | W7 | 250 | `ensure_ascii=False` everywhere; `validate_prediction.py` smoke |
| `colab/41_run_blindset_B_joint.ipynb` (stretch) | W8 | 300 | |
| `scripts/responder_eval.py` | composite eval | 200 | `ensure_ascii=False` |
| `scripts/reward_fns.py` | shared reward module | 300 | All 5 R_* terms, hard guards, dedupe helper |
| `scripts/validate_prediction.py` (extend) | catalog-membership check | +30 | |

### Wiki references

- `papers/syntheses/winning-architecture.md` — composite formula + thesis
- `papers/concepts/verifiable-rewards.md` — reward design rules + correlation gate
- `papers/concepts/agentic-rag.md` — Component A grounding
- `papers/concepts/rl-search-agents.md` — joint training (stretch)
- `papers/concepts/rlvr.md` + `papers/concepts/preference-optimization.md` — Component B algorithm choice
- `documents/research/recent_papers_ideas.md` — paper-by-paper TL;DR

---

## 9. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Reward correlation < 0.7 (W1 gate fails) | **High** | Fall back to `0.7·R_retr + 0.3·R_rule` mechanical reward; document and proceed |
| Catalog hallucination from trained 7B | **High** | A6 hard guard: zero R_turn + post-hoc filter from retrieval pool |
| `ensure_ascii=False` not enforced → JSON corruption | **High** | Mandate in every new file; CI smoke via `scripts/validate_prediction.py` |
| Local R_judge mis-calibrated (no Gemini) | High | Use ~20k anchors not 9; isotonic regression with monotonicity; trust-gating multiplier; no GRPO before W1 gate passes |
| Structural-directive collapse repeats on 7B | High | No-regression gate on dev nDCG@20 at every B-stage; 50-rollout qual review; revert checkpoint on detection |
| Rank-GRPO unstable | Med | Backup Rec-R1; G=2 not G=4 for budget |
| ProRank under-performs BGE | Med | A5 cut-path: ship BGE no-training reranker if ProRank doesn't beat it |
| VRAM blow-up on 40GB A100 | Med | Default to 1.5B+0.5B+7B (~28 GB); Rank-R1 stretch requires hot-swap or side-session |
| Colab session timeouts mid-GRPO | Med | Checkpoint every 500 steps to Drive; ~80 MB / 8s — negligible overhead |
| Blind-B budget over-spent | Med | ≤1/week + diff-one-axis; 5 slots planned |
| Final-code submission code not self-contained (Jul 9) | Med | Default-local pipeline only; external API stays elective and documented |
| Joint Search-R1 (W8 stretch) blows 5-day slack | Med | Hard cut-off Jun 22; revert to W7 champion |
| Catalog drift between dev and Blind-B | Low (was misnamed "low" before; now actually mitigated) | A4 fallback chain inherited everywhere; A6 catalog-membership filter as final line |
| Multilingual response under-rewarded by English-only R_rule | Low | Documented assumption; R_judge captures multilingual; future: lemma-based regex (P2) |
| **TRL dataset format drift** (skill: 50%+ of training failures) | **High** | `scripts/build_trl_datasets.py` emits validated KTO/DPO/GRPO parquets with inline schema checks; pytest contract at `tests/test_build_trl_datasets.py`; pre-flight via dataset_inspector before each W4–W6 GPU run |
| **Eval-strategy hang** (TRL trainer freezes if `eval_strategy=steps` but no `eval_dataset`) | High | All B-stage configs `train_test_split(test_size=0.1)` and pass `eval_dataset=ds_split["test"]`. CI: assert `eval_dataset` is set whenever `eval_strategy != "no"` |
| **Colab runtime ephemeral → trained adapter lost** (skill mandate) | High | Every B-stage trainer has `push_to_hub=True` + `hub_model_id="orrimoch/recsys2026-{stage}-{date}"` + `hub_strategy="every_save"`. Drive checkpointing every 500 steps as a redundant safety. Runtime-reset means re-train from last hub revision, not from scratch |
| **A100 OOM during Rank-GRPO with G=4 + Qwen-7B + LoRA** | Med | Skill ladder: (1) `per_device_train_batch_size=1`+`grad_accumulation_steps=8`; (2) `gradient_checkpointing=True`; (3) Unsloth fallback (`references/unsloth.md` in installed skill — ~60% less VRAM, ~2× faster); (4) drop `max_length=1024→512`; (5) A100-80GB upgrade |

---

## 10. Verification — End-to-End Test Plan

**Per-stage smoke tests (before each gate):**

1. **W1 reward smoke** — load ~8k dev rollouts + ~9 Blind-anchored rows; compute `R_turn_no_judge`, print Spearman vs reference composite. Pass if ≥0.7 AND CI lower bound ≥0.5. Independently confirm R_rule ≥0.4. Save anchor table to `data/reward_calibration_anchors.parquet`.
2. **W2 retrieval smoke** — run on 100 dev sessions; assert dev nDCG@10 differs from baseline by >|0.005|; assert post-fusion dedupe yields zero duplicates.
3. **W3 reranker smoke** — assert ProRank LoRA loads via vLLM, top-20 in <1s/turn batched-8; assert rationales emitted ≥95% of turns and ≤8 tokens each.
4. **W4–W6 responder smokes** — generate 50 dev responses; assert format compliance, no banned phrases, length-band, track-mention, **catalog-membership** all ≥95%; assert dev nDCG@20 not regressed > 0.005.
5. **W7 end-to-end smoke** — full pipeline on 100 dev sessions; produces valid `prediction.json` (`scripts/validate_prediction.py` passes including new catalog-membership check); composite via `evaluate_devset.py`.
6. **W7 Blind-B submission smoke** — verify zip layout (`prediction.json` at root, `ensure_ascii=False`); upload one slot.

**Final integration test (W7):**
```bash
# 1. Build dev predictions end-to-end (writes ensure_ascii=False)
python music-crs-baselines/run_inference_devset.py \
    --tid 300-final-devset --batch_size 16 --device cuda

# 2. Validate JSON contract + catalog membership (extended check)
python scripts/validate_prediction.py --tid 300-final-devset --check-catalog

# 3. Score on dev
cd music-crs-evaluator
python evaluate_devset.py --tid 300-final-devset
# expect: nDCG@20 ≥ 0.21 (champion 0.19 + ~+0.02 lift target)

# 4. Score with local responder composite
python ../scripts/responder_eval.py --tid 300-final-devset
# expect: R_turn ≥ 0.55 across 8000 dev turns

# 5. Blind-B run
python music-crs-baselines/run_inference_blindset.py --tid 300-final-blindset-B
zip -j submission.zip exp/inference/blindset_B/300-final-blindset-B/prediction.json
# upload submission.zip to CodaBench

# 6. Compare Blind-B leaderboard vs exp 021 anchor
```

**Pass criterion: Blind-B composite ≥ 0.40** (vs exp 021 anchor 0.33), with nDCG@20 ≥ 0.23 and Gemini-judge ≥ 3.0. Top-1 historically requires composite ≥ 0.50; we ladder there via W8 (joint training stretch + final ensemble).

---

## 11. Per-Gate Decision Tracker (was: Open Questions)

Each gate has explicit YES/NO/DEFER decisions to log in `documents/experiments_log.md`:

| Gate | Decision points | Tracked field |
|---|---|---|
| **W1 reward gate** | (a) Did Spearman ≥0.7 + CI ≥0.5 hold? (b) Does R_rule independently pass 0.4? (c) Use full R_turn vs fallback `0.7·R_retr + 0.3·R_rule`? | `reward_design_v1` |
| **W2 retrieval gate** | (a) Did CMQR lift nDCG@10 ≥0.005? (b) State-tracker parse rate ≥99%? (c) Upgrade A1 to Qwen-3B? | `agentic_retrieval_v1` |
| **W3 reranker gate** | (a) ProRank lift ≥0.015? (b) Launch Rank-R1 stretch on side Colab? (c) BGE cut-path activated? | `reranker_choice` |
| **W4 KTO gate** | (a) Format ≥95%? (b) nDCG@20 not regressed? (c) Run S-DPO (W5) or skip? | `kto_warmup` |
| **W5 (conditional)** | If W4 format <70%, ship S-DPO. Else: free week for B3 debug. | `sdpo_run` |
| **W6 GRPO gate** | (a) +0.03 R_turn? (b) Qual review of 50 rollouts pass? (c) nDCG@20 not regressed? | `rgrpo_main` |
| **W7 integration gate** | (a) Dev composite ≥ exp-021 anchor? (b) Blind-B submission valid (catalog + dedupe + ascii)? | `blindset_b_v1` |
| **W8 stretch gate** | (a) Joint Search-R1 converging by Jun 22? (b) Otherwise revert to W7 champion. | `joint_training` |
| **MARec retrofit** | At W7: do we have time + does cold subset show headroom? | `marec_retrofit` |
| **Anthropic API elective** | At W7: does pipeline plateau below 0.40 target? | `external_api` |
