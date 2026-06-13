# Autonomous Research Loop — Design Spec

**Date:** 2026-06-13
**Status:** Approved design, pending implementation plan
**Branch context:** `recall-union-lgbm`

## 1. Objective

Let Claude Code drive the RecSys 2026 Music CRS challenge as an autonomous research
agent — proposing experiments, preparing code/configs, critiquing results, recording
conclusions to memory, and submitting to the CodaBench leaderboard within a budget —
with the human acting **only** as the compute executor (auth + running notebooks) and
not as the decision-maker for each step.

### 1.1 Division of labor (hard boundary)

- **Human = compute executor ONLY.** Runs the notebooks (auth + GPU), pastes back raw
  output. Nothing else. The human does **not** form hypotheses, set priorities, adjudicate
  experiments, or make any research decision.
- **Claude = the researcher.** Owns every hypothesis, priority call, experimental design,
  verdict, conclusion, and the recommendation-systems domain expertise behind them
  (recall/rerank/responder tradeoffs, leak-safety, evaluation methodology, best practices).
  Claude brings the field knowledge; it does not defer scientific judgment to the human.

Any step that needs a decision is Claude's. Any step that needs a button pressed or a cell
run is the human's. If a setup step looks like it needs human *reasoning* (e.g. reverse-
engineering an API), Claude does the reasoning; the human only executes a prepared action and
pastes the raw result.

The optimization target is the Blind composite:

```
composite = 0.50·nDCG@20 + 0.10·CatalogDiversity + 0.10·LexicalDiversity + 0.30·LLM_judge
```

(weights per `scripts/local_eval.py::COMPOSITE_WEIGHTS`.)

## 2. The central risk this design exists to defeat

The project memory is a documented graveyard of an optimizer fooling itself on a
misleading signal: *"internal val is a TRAP"*, *"highest internal val ever → worst dev"*
(CLAP), the `sasrec_rank_inv` feature at **+0.0685 internal / −0.0243 dev** (sign flip).
Every betrayal shared one property: **a model scored on data it trained on (in-sample
leak).** Meanwhile leak-free held-out signals held up directionally (dev recall@100 after
the A2 `listener_goal` harness fix; the offline Gemini judge for the LLM axis).

An autonomous loop without structural defenses will confidently march off this same cliff.
The design's core is therefore an **anti-self-deception discipline**, not plumbing.

### 2.1 The three-tier signal hierarchy (the "Trusted Reward")

- **Tier 1 — Trusted local gate** (every iteration, the inner-loop reward):
  leak-free held-out composite via `local_eval.py` + the turn-1 / `nb74` cell-49 gate,
  plus the **offline Gemini judge** (`scripts/gemini_judge_responses.py`) for the LLM axis
  (well-aligned because the blind judge is also Gemini).
  **BANNED reward signals (named explicitly):** LGBM bag/internal validation,
  training-split validation, any in-sample feature score, any metric where a model is
  scored on a split it trained on.

- **Tier 2 — Blindset** (scarce, gated *confirmation*, never the search signal):
  treated as a precious held-out test set, touched rarely to confirm a Tier-1 winner
  translates. This is where the submission budget lives. Never optimized against
  directly (doing so = leaderboard-probing overfit of 80 sessions × 1 turn, ±0.05 noise).

- **Tier 3 — The meta-discipline (correlation watch):**
  every blind submission logs the pair *(local Δ, blind Δ)*. The loop tracks this
  correlation. When local gains stop translating to blind (exactly what the recall arc
  did — local moved, Blind gave +0.02, hit "the wall"), that is **not** a cue to grind
  harder; it is an automatic trigger to **switch levers**. The charter encodes this as a
  stop/switch rule.

### 2.2 Mandatory review gates (every iteration)

No iteration is "done" until two reviews pass — both run as dispatched subagents so they
have independent context, not the optimizer's:

- **Code-review agent** — runs on every code/config diff *before handoff*. Checks
  correctness and, critically, **leak-safety**: does any new feature/metric score a model on
  a split it trained on? (This is the gate that would have caught `sasrec_rank_inv` and CLAP.)
- **RecSys-researcher agent** — runs on every verdict *before it is banked to memory*.
  Adversarially reviews the experimental reasoning: is the gate leak-free and pre-registered?
  Is the PASS/FAIL/INCONCLUSIVE call justified by the data and the noise band? Is this a
  known trap from memory? Is a banned (in-sample) signal contaminating the conclusion?

**Tests are always written** (TDD) for any new code — the test *is* part of the deliverable,
not an afterthought. A task with code and no test is incomplete.

## 3. Architecture (Approach A — charter + pre-registered protocol over existing harness)

No new orchestration infrastructure. The loop runs through normal chat turns, leaning on
existing scripts (`run_experiment.py`, `local_eval.py`), the existing logs, and the
file-based memory system. The discipline lives in a charter document re-read every cycle.

### 3.1 Artifacts

| Artifact | Purpose | New? |
|---|---|---|
| `RESEARCH_CHARTER.md` | The constitution — re-read at the top of every iteration | **new** |
| `documents/experiments_log.md` | Pre-registration journal; one entry per experiment, written **before** the run | **new** |
| `documents/benchmarks.md` | Trusted-gate results table | exists |
| `documents/submissions_log.md` | Blind submission ledger + daily-budget tracker | exists |
| Memory files (`.../memory/*.md` + `MEMORY.md`) | Durable conclusions | exists |
| `RESULTS_JSON` emitter cell | Final cell of each eval notebook prints a fenced, parseable block for paste-back | **new (small notebook edits)** |

### 3.2 `RESEARCH_CHARTER.md` contents

- **Objective** — the composite formula above.
- **Current state** — pointer to memory (best = config 204 @ Blind 0.44 as of 2026-06-11).
- **Trusted Reward** — the §2.1 three-tier hierarchy, with banned signals listed by name.
- **Lever priority** — full-pipeline scope (recall + reranker + responder), data-driven
  per iteration, seeded with current knowledge: *responder near ceiling, recall = the wall,
  reranker tapped out.* Claude picks the highest-ROI lever each cycle from this prior + memory.
- **Submission budget** — daily quota `N` (see §6; value confirmed during setup).
- **Pre-registration rule** — verdict criteria committed to git before any number exists.
- **Stop / switch conditions** — e.g. 3 consecutive INCONCLUSIVE cycles on a lever → switch
  lever; local↔blind decoupling (Tier 3) → switch lever; budget exhausted → local-only mode.

## 4. The iteration protocol (six phases per cycle)

1. **HYPOTHESIZE** — read charter + `MEMORY.md` + `benchmarks.md`; pick the highest-ROI
   lever; state a *falsifiable* claim with an expected metric delta.
2. **PRE-REGISTER** — write the `experiments_log.md` entry **before running**: hypothesis,
   the exact Tier-1 gate, the honest baseline it must beat (with source), and the decision
   rule (PASS / FAIL / INCONCLUSIVE bands, including the ±noise band). Commit it.
3. **PREPARE** — make the config/code change; write tests (TDD) + run the full `pytest`
   suite green (no GPU); **dispatch the code-review agent on the diff (§2.2)** and address
   its findings; then assemble the run-package.
4. **HANDOFF** — give the human the run-package (§5); they run it in Colab and paste back
   the `RESULTS_JSON` block.
5. **JUDGE** — parse results; compare against the **pre-registered** gate (no post-hoc
   storytelling). If a submission occurred, log *(local Δ, blind Δ)* and update the Tier-3
   correlation watch.
6. **REVIEW** — **dispatch the RecSys-researcher agent on the verdict (§2.2)**; it must
   sign off that the conclusion is leak-free, justified by the data + noise band, and not a
   known trap *before* anything is banked.
7. **RECORD + DECIDE** — append to `benchmarks.md`; write/update memory per the memory
   protocol; keep-or-revert; if a config clears the gate and budget allows, submit (§6).

### 4.1 Pre-registration entry format (`experiments_log.md`)

```
## EXP-042 — <lever> — 2026-06-13
Hypothesis:   <falsifiable claim>
Lever:        recall | reranker | responder
Change:       <config NNN / code diff summary>
Pre-registered gate:  <metric> on <Tier-1 signal> must beat <baseline X> by <Y>
Baseline:     <honest number + source, e.g. "config 204 composite 0.44, benchmarks.md">
Decision rule: PASS → keep / promote to blind candidate; FAIL → revert;
               INCONCLUSIVE (|Δ| < noise band Z) → no change, count toward switch rule
Smoke:        pytest <paths> green before handoff
--- run ---
Result:       <pasted RESULTS_JSON>
Verdict:      PASS | FAIL | INCONCLUSIVE
Local→Blind:  local Δ <>, blind Δ <>   (only if submitted)
Memory:       <file written/updated>
```

The verdict can only be marked against criteria already committed in git. INCONCLUSIVE is a
first-class outcome — never a nudge to keep the config.

## 5. Handoff + paste-back formats

**Run-package** (Claude → human):

```
RUN PACKAGE — EXP-042
1. git pull  (branch: recall-union-lgbm)
2. Open colab/74_e2e_sasrec_union_lgbm_ndcg.ipynb
3. Set CONFIG = 205
4. Run cells 1 → 49
5. Paste back the fenced RESULTS_JSON block printed by cell 49.
Expected wall time: ~25 min · Pre-registered gate: composite > 0.44 (204 baseline)
```

**Paste-back** (human → Claude) — emitted deterministically by the new emitter cell:

```
RESULTS_JSON
{"exp":"042","config":205,"ndcg@20":0.31,"cat_div":0.03,
 "lex_div":0.79,"llm_judge":4.2,"composite":0.46,
 "n_sessions":80,"gate":"turn1_cell49"}
```

Claude parses the JSON directly — no ambiguity about which number is which.

## 6. Submission path + budget

User confirmed CodaBench exposes an API token. CodaBench is Codalab-v2, which has a
token-authenticated REST API for submissions. The exact submit / auth / score endpoint
paths are **not yet confirmed** (the `/api/docs/` page is a JS-rendered Swagger UI;
`/api/schema/` 404s). Implementation must confirm them by **one of**: inspecting the
browser network calls during a manual submission, or reading the `codalab/codabench`
GitHub source. The CodaBench daily submission quota `N` must also be confirmed during setup.

- **Primary path (token works):** Claude submits `prediction.json` autonomously,
  decrements the daily-budget ledger in `submissions_log.md`, polls for the score, and
  logs *(local Δ, blind Δ)*.
- **Fallback (token brittle/unavailable):** Claude prepares `prediction.json`, manages the
  budget + correlation watch, and hands the human a one-line "submit this now
  (budget: 2/N today)"; human uploads and pastes the score back.

Budget logic and the correlation watch are identical across both paths; only the final
upload click differs. Submission packaging follows the existing protocol
(`prediction.json` singular at zip root — see memory `project_codabench_submission`).

## 7. One-time setup prerequisites

1. Confirm/obtain the CodaBench API token + daily quota `N`; store the token as an env var
   / secret (never committed).
2. Add the `RESULTS_JSON` emitter cell to the key eval notebooks (at minimum `nb74`
   e2e dev gate, `nb80` blind Gemini responder).
3. Write `RESEARCH_CHARTER.md` and seed `documents/experiments_log.md`.
4. Confirm the local Tier-1 gate is runnable end-to-end (`local_eval.py` +
   `gemini_judge_responses.py`) and reproduces a known baseline (config 204) within noise.

## 8. Non-goals (YAGNI)

- **No custom Python orchestrator** (Approach B) while the GPU step is human-run — it would
  only automate parsing/logging, which the protocol + emitter cell already handle.
- **No `/loop` timer** — the human notebook run is the gating event; a timer would wake
  Claude with nothing to do. (Approach C, wrapping the matured protocol into a `/research-tick`
  skill, is a *future* convenience, not in scope here.)
- **No optimization against Tier-2 (blindset) or any banned in-sample signal.**
- **No new model architectures** introduced by the loop itself beyond what the existing
  scripts/notebooks support; the loop tunes/configures, it does not invent new training code
  unless a hypothesis explicitly calls for a small, reviewed change.

## 9. Open items to resolve in the implementation plan

- Confirm CodaBench submit/auth/score endpoint paths + quota `N`.
- Decide the exact `experiments_log.md` ↔ `benchmarks.md` split (pre-registration vs
  results table) to avoid duplication.
- Define the noise band `Z` per metric on the 80-session blindset (memory suggests ±0.05
  composite) and the dev-gate noise band.
- Confirm which notebooks need the emitter cell beyond nb74/nb80.
