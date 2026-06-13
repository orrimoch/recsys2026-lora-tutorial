# Research Charter — RecSys 2026 Music CRS Autonomous Loop

Re-read this file at the top of EVERY iteration before proposing an experiment.

## Division of labor
Human = compute executor ONLY (runs notebooks, pastes raw output). Claude = the researcher:
all hypotheses, priorities, verdicts, conclusions, and RecSys domain judgment are Claude's.
Never defer a research decision to the human.

## Objective
Maximize the Blind composite:
composite = 0.50·nDCG@20 + 0.10·CatalogDiversity + 0.10·LexicalDiversity + 0.30·LLM_judge_normalized
(LLM normalized = (judge_mean - 1) / 4). Source of truth: scripts/local_eval.py.

## Current state (update as it changes)
Best = config 204: Blind composite ~0.44 (nDCG 0.24 / Cat 0.03 / Lex 0.79 / LLM 4.2),
as of 2026-06-11. The composite win came from the Gemini responder, not recall.

## Trusted Reward — the three-tier signal hierarchy
TIER 1 (inner-loop reward, every iteration): leak-free held-out local composite via
  local_eval.py + the turn-1 / nb74 cell-49 gate, plus the offline Gemini judge
  (scripts/gemini_judge_responses.py) for the LLM axis.
BANNED reward signals (NEVER use to decide PASS): LGBM bag/internal validation,
  training-split validation, any in-sample feature score, any metric where a model is
  scored on a split it trained on. (These are the documented traps: CLAP 0.58-internal /
  worst-dev; sasrec_rank_inv +0.0685-internal / -0.0243-dev sign flip.)
TIER 2 (scarce confirmation, NEVER the search signal): the CodaBench blindset. Touched
  rarely to confirm a Tier-1 winner translates. The submission budget lives here.
TIER 3 (meta-discipline): every blind submission logs (local Δ, blind Δ). When local gains
  stop translating to blind, that is a trigger to SWITCH LEVERS, not to grind harder.

## Lever scope & priority
Full pipeline: recall + reranker + responder. Pick the highest-ROI lever each iteration
from memory + benchmarks. Prior: responder near ceiling, recall = the wall, reranker
tapped out — but re-evaluate per the Tier-3 watch.

## Submission budget
Weekly blind cap (validate_prediction.check_budget, default weekly_cap=3). Confirm the real
CodaBench cap during setup and set it here: WEEKLY_CAP = <confirm>.

## Pre-registration rule
Before every run, write the loop_experiments_log.md entry (hypothesis + exact gate + honest
baseline + decision rule) and commit it. The verdict can ONLY be marked against criteria
already committed. INCONCLUSIVE (|Δ| < noise band) is a first-class outcome, never a nudge
to keep a config.

## Review discipline (mandatory, every iteration)
1. Tests always: any code/config change ships with tests (TDD), full pytest green.
2. Code-review agent on every diff BEFORE handoff — must check leak-safety (no model scored
   on a split it trained on), not just correctness.
3. RecSys-researcher agent on every verdict BEFORE banking to memory — must sign off that the
   conclusion is leak-free, justified within the noise band, and not a known trap.
Both reviews run as independent subagents. Address or explicitly rebut every finding.

## Stop / switch conditions
- 3 consecutive INCONCLUSIVE cycles on a lever → switch lever.
- Tier-3 local↔blind decoupling on a lever → switch lever.
- Weekly budget exhausted → local-only mode (no submissions) until it resets.
