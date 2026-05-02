# RecSys 2026 — Per-gate decision log

Per `RecSys_Challenge_Plan.md` §11. Each W1–W8 gate fires a YES / NO / DEFER
decision that determines whether the next phase proceeds, falls back to a
cut-path, or skips. This file records each decision with date, evidence
location, and follow-up action so the gate history is auditable.

**How to update**: when a gate fires (typically at the end of a phase's
GPU run), append a row to that gate's section below. Include:
- `date`: ISO `YYYY-MM-DD`.
- `decision`: YES / NO / DEFER.
- `evidence`: file path or run ID — `gate_result.json`, `experiments_log.md`
   row, leaderboard score, etc.
- `next`: 1-line description of the action triggered by this decision.

`scripts/run_w1_gate.py` and any future `run_*_gate.py` may append rows
programmatically; they MUST keep the table format intact.

---

## W1 — Reward correlation gate (`reward_design_v1`)

Plan §6.2 / §11. Decision points:
- (a) Spearman ≥ 0.7 with 95% CI lower-bound ≥ 0.5 between `R_turn − R_judge`
  and Gemini-anchored composite?
- (b) Does R_rule independently correlate ≥ 0.4 with composite (so it can
  carry judge proxy weight when R_judge is stub-zero)?
- (c) Use full R_turn or fallback weighting `0.7·R_retr + 0.3·R_rule`?

| Date | Decision | Evidence | Next |
|---|---|---|---|
| 2026-04-29 | DEFER | `data/reward_gate_results.json` (Spearman 0.012 vs gold; AUC 0.507 — random) | Switched to v1 weights (R_retr=0.70 dominant, R_judge=0.10 trust-gated). v3 weights re-introduced via Option B refactor. |

## W2 — Retrieval gate (`agentic_retrieval_v1`)

Plan §A1+A2 / §11. Decision points:
- (a) CMQR lift dev nDCG@10 ≥ 0.005 over wRRF champion?
- (b) State-tracker JSON parse rate ≥ 99% on ~8000-turn dev set?
- (c) Upgrade A1 from Qwen-1.5B to Qwen-3B?

| Date | Decision | Evidence | Next |
|---|---|---|---|
| _pending_ | _pending_ | _pending_ | _pending_ |

## W3 — Reranker gate (`reranker_choice`)

Plan §A5 / §11. Decision points:
- (a) ProRank lift dev nDCG@10 ≥ 0.015 over wRRF baseline?
- (b) Launch Rank-R1 stretch on side Colab session?
- (c) BGE cut-path activated (ProRank training unstable)?

| Date | Decision | Evidence | Next |
|---|---|---|---|
| _pending_ | _pending_ | _pending_ | _pending_ |

## W4 — KTO gate (`kto_warmup`)

Plan §6.3 row B1 / §11. Decision points:
- (a) Format compliance ≥ 95% strict?
- (b) Dev nDCG@20 not regressed > 0.005 vs frozen retriever?
- (c) Run S-DPO (W5) or skip?

| Date | Decision | Evidence | Next |
|---|---|---|---|
| _pending_ | _pending_ | _pending_ | Run `colab/30_train_responder_kto.ipynb` then append. |

## W5 — S-DPO gate (`sdpo_run`)

Plan §6.3 row B2 / §11. Conditional:
- If W4 format < 70% → ship S-DPO.
- Else → skip; free week for B3 debug or earlier integration.

| Date | Decision | Evidence | Next |
|---|---|---|---|
| _pending_ | _pending_ | _pending_ | Auto-fires when `colab/31_train_responder_sdpo.ipynb`'s `gate_result.json` lands. |

## W6 — Rank-GRPO gate (`rgrpo_main`)

Plan §6.3 row B3 / §11. Decision points:
- (a) +0.03 R_turn over B1 (W4)?
- (b) Qual review of 50 rollouts pass?
- (c) Dev nDCG@20 not regressed > 0.005?

**Note (deep-review)**: the +0.03 threshold was derived under v1 weights
(R_retr=0.70). Under v3 weights (R_retr=0.40 + real R_judge=0.30), this
threshold may be too easy. Re-derive from empirical R_turn distribution
after the pilot run before applying to the full W6.

| Date | Decision | Evidence | Next |
|---|---|---|---|
| _pending (pilot)_ | _pending_ | _pending_ | Run `colab/31p_pilot_grpo_with_judge.ipynb` first (~30 units; relaxed gate format ≥90%, Δ ≥+0.015). |
| _pending (full)_ | _pending_ | _pending_ | If pilot PASSES, run `colab/32_train_responder_grpo.ipynb`. |

## W7 — Integration gate (`blindset_b_v1`)

Plan §W7 / §11. Decision points:
- (a) Dev composite ≥ exp-021 anchor (~0.33)?
- (b) Blind-B submission validates (catalog + dedupe + `ensure_ascii=False`)?

| Date | Decision | Evidence | Next |
|---|---|---|---|
| _pending_ | _pending_ | _pending_ | After W6 pass: run `colab/33_train_responder_grpo_train_plus_dev.ipynb` then `colab/40_run_blindset_B.ipynb`. |

## W8 — Joint stretch gate (`joint_training`)

Plan §6.7 / §11. Decision points:
- (a) Joint Search-R1 converging by Jun 22 (hard cut-off)?
- (b) Otherwise revert to W7 champion.

| Date | Decision | Evidence | Next |
|---|---|---|---|
| _pending_ | DEFER | not started — no scaffolding for `colab/41_run_blindset_B_joint.ipynb` | Decide after W7 ships; keep as stretch. |

## MARec retrofit (`marec_retrofit`)

Plan §A4 / §11. Decision points (asked at W7 polish):
- (a) Time available?
- (b) Cold-subset show nDCG@10 headroom vs warm-subset?

| Date | Decision | Evidence | Next |
|---|---|---|---|
| 2026-05-02 | DEFER | plan §A4 explicit: "DEFERRED to W7 polish if B-stage is on-track" | Re-evaluate at W7 polish; if cold/warm gap > 0.005 nDCG@10 → build `mcrs/embedders/marec.py` (~250 LOC). |

## Anthropic API elective (`external_api`)

Plan §11. Decision points:
- Pipeline plateau below 0.40 composite target at W7?

| Date | Decision | Evidence | Next |
|---|---|---|---|
| _pending_ | _pending_ | not yet hit gate condition | Re-evaluate at W7 if Blind-A/B Gemini scores show plateau. |

## `<reranker_rationales>` in prompt (deep-review P0-2)

Plan §6.5 design vs deep-review P0-2 train/inference mismatch finding.

| Date | Decision | Evidence | Next |
|---|---|---|---|
| 2026-05-02 | DEFER | deep-review caught train/inference mismatch; `build_grpo_dataset --include-rationales` is opt-in OFF default | Decide post-Blind-A: if Gemini judge shows responses lack grounding, wire `pro_rank.generate_rationales` into `mcrs/crs_baseline.batch_chat`. |

## `user_profile` end-to-end pipe (deep-review P1-6)

Honesty fix ledger.

| Date | Decision | Evidence | Next |
|---|---|---|---|
| 2026-05-02 | DEFER (honest) | data path was unwired in Option B refactor; reverted W_USER_PROF 0.05→0.00 | Re-pipe and restore weight when ready. (Step 3 of the gap-analysis follow-up — IN PROGRESS.) |
