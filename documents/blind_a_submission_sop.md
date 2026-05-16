# Blind-A Submission Standard Operating Procedure

For each SID Blind-A submission, follow these steps in order.

## Pre-submission (do NOT skip)

1. **Run the dev gate** for the candidate config (notebook 64 for ensemble; manual run for pure-SID).
2. **Check dev nDCG@20** vs config 132 baseline (0.06 on dev). If dev nDCG@20 < 0.05, **do NOT submit** — it'll just regress on Blind-A. Iterate first.
3. **Run inference on Blind-A** (notebook 63 for W4 ensemble; notebook 66 for W5 winner + pure-SID).
4. **Run the precheck**:
   ```bash
   python scripts/precheck_prediction.py \
       --input music-crs-baselines/exp/inference/blindset_A/<tid>.json
   ```
   Must exit 0. If it errors (hallucinated IDs, missing fields, duplicates), **DO NOT submit** — fix first.

## Submission

5. **Zip + upload** to CodaBench (per `project_codabench_submission.md`).
6. **Wait for the score** (Gemini judging takes minutes-hours).

## Post-submission (record everything)

7. **Append the score** to the tracker the moment it's visible:
   ```bash
   python scripts/blind_a_score_tracker.py append \
       --tracker ~/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_blind_a_submissions.md \
       --config_id <tid> \
       --composite X.XXXX --ndcg X.XXXX --llm X.XXX --lex_div X.XXXX \
       --url https://www.codabench.org/competitions/.../submission/... \
       --notes "<one-line context: what changed vs prior submission>"
   ```

8. **Interpret the score** using `project_blind_a_axis_interpretation.md` (memory):
   - Which axis moved most vs the previous submission?
   - Was the movement explained by the config change?
   - If not, escalate (see escalation order in the interpretation guide).

9. **Decide next move** based on the score table:
   - PASS (≥ 0.24): one more submission to be sure → freeze.
   - PARITY (0.21-0.23): submit the alternate config (pure-SID if you just submitted ensemble; lower-weight if you just submitted weight winner).
   - REGRESSION (< 0.21): STOP — don't burn more quota. Diagnose (see escalation order).

## Daily quota discipline

CodaBench has a per-day submission limit (verify the current quota before each session). Sequence submissions deliberately:
- Day 1: W4 baseline (config 170, weight=0.5)
- Day 2: W5 weight winner (highest from sweep)
- Day 3: Pure-SID (config 171)
- Days 4+: only if you have something specifically informed by the prior scores

Don't submit the same config twice on the same day — wastes a slot.

## When to stop iterating

Stop and freeze when ANY of:
- Best score is ≥ 0.24 AND the marginal improvement from the last submission < 0.005 (returns are diminishing).
- 3 SID configs have been submitted and at least one is ≥ 0.22 (good enough to freeze; v2 sprint can improve).
- You've burned 5 submissions on SID configs and none beat baseline (write postmortem; pivot to v2 retraining instead).

## Related references

- `project_blind_a_axis_interpretation.md` (memory) — per-axis diagnosis + escalation order
- `scripts/precheck_prediction.py` — pre-submission strict validation
- `scripts/blind_a_score_tracker.py` — append-only CodaBench score log
- `project_codabench_submission.md` (memory) — CodaBench zip format requirements
