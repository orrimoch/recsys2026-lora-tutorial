# Autonomous Research Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up an autonomous research loop where Claude proposes/prepares/critiques RecSys experiments and submits to CodaBench within a budget, with the human only running notebooks.

**Architecture:** Approach A — a `RESEARCH_CHARTER.md` constitution + a pre-registered iteration protocol over the *existing* harness. New code is small: a results-block emitter, a budget-gated submission preparer, and a discovery-gated CodaBench HTTP layer. Everything else reuses `validate_prediction.py` (`check_budget`, `package_zip`, `validate_schema`), `local_eval.py` (`compute_composite_projected` — the single composite source of truth), and the existing `documents/*.md` logs + file-based memory.

**Tech Stack:** Python 3.10, pytest, `requests`, OmegaConf, the repo's existing `scripts/` + `music-crs-evaluator` + Colab notebooks.

**Spec:** `docs/superpowers/specs/2026-06-13-autonomous-research-loop-design.md`

---

## Review discipline (applies to EVERY task and every loop iteration)

Per spec §2.2, two gates are mandatory and non-negotiable:

1. **Tests always.** Every task that adds code ships with tests (TDD: failing test first).
   A code task with no test is incomplete.
2. **Two review agents, dispatched as subagents** (independent context, via the `Agent` tool):
   - **Code-review agent** — after any code/config change, before it is committed/handed off.
     Use the `superpowers:requesting-code-review` skill or `subagent_type: "general-purpose"`
     with a code-review prompt. It MUST check leak-safety (no model scored on a split it
     trained on), not just correctness.
   - **RecSys-researcher agent** — after any experimental verdict, before it is banked to
     memory. `subagent_type: "general-purpose"`, prompted as a recommendation-systems
     researcher, given the charter §Trusted-Reward + the result. It must sign off that the
     conclusion is leak-free, justified within the noise band, and not a known trap.

   Address (or explicitly rebut, with reasoning) every finding before proceeding.

---

## File Structure

| File | Responsibility | New? |
|---|---|---|
| `scripts/emit_results.py` | Build + print the fenced `RESULTS_JSON` paste-back block; composite delegated to `local_eval` | **new** |
| `tests/test_emit_results.py` | Tests for the emitter | **new** |
| `scripts/loop_submit.py` | Budget-gated submission *preparer*: validate → budget-gate → package zip → handoff dict | **new** |
| `tests/test_loop_submit.py` | Tests for the preparer | **new** |
| `scripts/codabench_submit.py` | CodaBench HTTP layer (auth/upload/poll); discovery-gated live wiring + testable poll/parse | **new** |
| `tests/test_codabench_submit.py` | Tests for the endpoint-independent poll/parse logic | **new** |
| `RESEARCH_CHARTER.md` | The loop constitution (re-read every iteration) | **new** |
| `documents/experiments_log.md` | Pre-registration journal | **new** |
| `colab/74_e2e_sasrec_union_lgbm_ndcg.ipynb` | Add `RESULTS_JSON` emitter cell | modify |
| `colab/80_blindA_gemini_responder.ipynb` | Add `RESULTS_JSON` emitter cell | modify |

**Test import convention** (from `tests/test_blind_a_score_tracker.py`): each test file does
`sys.path.insert(0, str(REPO_ROOT))` where `REPO_ROOT = Path(__file__).resolve().parents[1]`,
then `from scripts.X import ...`. Run tests with `python -m pytest` from the repo root.

---

## Task 1: Results-block emitter (`scripts/emit_results.py`)

**Files:**
- Create: `scripts/emit_results.py`
- Test: `tests/test_emit_results.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_emit_results.py`:

```python
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.emit_results import build_results_json, format_results_block


def test_composite_matches_local_eval_with_llm():
    # retrieval = 0.5*0.30 + 0.1*0.03 + 0.1*0.79 = 0.232
    # llm_norm = (4.2-1)/4 = 0.8 ; +0.3*0.8 = 0.24 ; total = 0.472
    payload = build_results_json(
        "042", 205, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=4.2, n_sessions=80, gate="turn1_cell49",
    )
    assert payload["composite"] == 0.472
    assert payload["exp"] == "042"
    assert payload["config"] == 205
    assert payload["llm_judge"] == 4.2


def test_composite_retrieval_only_when_llm_none():
    payload = build_results_json(
        "043", 206, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=None, n_sessions=80, gate="turn1_cell49",
    )
    assert payload["composite"] == 0.232
    assert payload["llm_judge"] is None


def test_format_block_is_parseable():
    payload = build_results_json(
        "042", 205, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=4.2, n_sessions=80, gate="turn1_cell49",
    )
    block = format_results_block(payload)
    assert block.startswith("RESULTS_JSON\n")
    parsed = json.loads(block.split("\n", 1)[1])
    assert parsed["config"] == 205
    assert parsed["n_sessions"] == 80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_emit_results.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.emit_results'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/emit_results.py`:

```python
"""Standardized RESULTS_JSON emitter for eval notebooks (autonomous research loop).

The final cell of each eval notebook calls print_results_block(...) so the human can paste
back one fenced, machine-parseable block. Centralizing the format here (not inline in
notebooks) keeps it consistent and testable. The composite is computed by local_eval — the
single source of truth — so this module never reimplements the weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.local_eval import compute_composite_projected  # noqa: E402

REQUIRED_FIELDS = (
    "exp", "config", "ndcg@20", "cat_div", "lex_div",
    "llm_judge", "composite", "n_sessions", "gate",
)


def build_results_json(
    exp: str,
    config: int,
    *,
    ndcg: float,
    cat_div: float,
    lex_div: float,
    llm_judge: Optional[float],
    n_sessions: int,
    gate: str,
) -> dict[str, Any]:
    """Assemble the paste-back payload.

    llm_judge is the 1-5 judge mean, or None if the LLM axis was not evaluated this run
    (composite is then retrieval-only). Composite is delegated to local_eval.
    """
    scores = {
        "ndcg@20": float(ndcg),
        "catalog_diversity": float(cat_div),
        "lexical_diversity": float(lex_div),
    }
    composite = round(compute_composite_projected(scores, llm_judge), 4)
    return {
        "exp": exp,
        "config": int(config),
        "ndcg@20": round(float(ndcg), 4),
        "cat_div": round(float(cat_div), 4),
        "lex_div": round(float(lex_div), 4),
        "llm_judge": (round(float(llm_judge), 4) if llm_judge is not None else None),
        "composite": composite,
        "n_sessions": int(n_sessions),
        "gate": gate,
    }


def format_results_block(payload: dict[str, Any]) -> str:
    """Render the fenced block the human pastes back."""
    return "RESULTS_JSON\n" + json.dumps(payload)


def print_results_block(**kwargs: Any) -> dict[str, Any]:
    payload = build_results_json(**kwargs)
    print(format_results_block(payload))
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_emit_results.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/emit_results.py tests/test_emit_results.py
git commit -m "feat: RESULTS_JSON emitter for the autonomous research loop"
```

---

## Task 2: Add the emitter cell to the eval notebooks

**Files:**
- Modify: `colab/74_e2e_sasrec_union_lgbm_ndcg.ipynb` (append a final cell)
- Modify: `colab/80_blindA_gemini_responder.ipynb` (append a final cell)

This task uses the `NotebookEdit` tool (load it via ToolSearch: `select:NotebookEdit`).

- [ ] **Step 1: Identify the metric variables in nb74**

Read the cell in `colab/74_e2e_sasrec_union_lgbm_ndcg.ipynb` that prints the dev-gate
metrics (the cell-49 gate). Note the exact variable names that hold nDCG@20, catalog
diversity, lexical diversity, the LLM judge mean (if present), the config id, and the
predictions list. Do NOT assume — confirm by reading.

- [ ] **Step 2: Append the emitter cell to nb74**

Insert a new final code cell. Map the kwargs to the variables found in Step 1 (the names
below are placeholders for whatever nb74 actually uses):

```python
# --- RESULTS_JSON emitter (autonomous research loop) ---
import sys
sys.path.insert(0, "/content/recsys2026")  # adjust to the repo clone path on Colab
from scripts.emit_results import print_results_block

print_results_block(
    exp="EXP-XXX",                         # set per experiment from the run package
    config=CONFIG_ID,                      # <- the config id variable in nb74
    ndcg=scores["ndcg@20"],                # <- map to nb74's metrics dict/vars
    cat_div=scores["catalog_diversity"],
    lex_div=scores["lexical_diversity"],
    llm_judge=None,                        # nb74 is the retrieval gate; no LLM axis here
    n_sessions=N_SESSIONS,                 # <- nb74's session count variable
    gate="turn1_cell49",
)
```

- [ ] **Step 3: Append the emitter cell to nb80**

Repeat Step 1+2 for `colab/80_blindA_gemini_responder.ipynb`. nb80 *does* have the LLM axis
(Gemini responder), so pass the judge mean:

```python
# --- RESULTS_JSON emitter (autonomous research loop) ---
import sys
sys.path.insert(0, "/content/recsys2026")
from scripts.emit_results import print_results_block

print_results_block(
    exp="EXP-XXX",
    config=CONFIG_ID,
    ndcg=scores["ndcg@20"],
    cat_div=scores["catalog_diversity"],
    lex_div=scores["lexical_diversity"],
    llm_judge=llm_judge_mean,              # <- nb80's 1-5 Gemini-judge mean variable
    n_sessions=N_SESSIONS,
    gate="blindA",
)
```

- [ ] **Step 4: Verify (human-assisted handoff)**

This requires running on Colab (GPU/auth). Hand the human a run package to execute nb74
end-to-end on a known config (204) and paste back the `RESULTS_JSON` block. Confirm the
block parses and the composite matches `benchmarks.md` for config 204 within the noise band.
(If the human is unavailable, mark this step pending — it does not block Tasks 3-6.)

- [ ] **Step 5: Commit**

```bash
git add colab/74_e2e_sasrec_union_lgbm_ndcg.ipynb colab/80_blindA_gemini_responder.ipynb
git commit -m "feat: RESULTS_JSON emitter cells in nb74/nb80 eval notebooks"
```

---

## Task 3: Write `RESEARCH_CHARTER.md`

**Files:**
- Create: `RESEARCH_CHARTER.md` (repo root)

- [ ] **Step 1: Create the charter**

Create `RESEARCH_CHARTER.md` with this exact content:

```markdown
# Research Charter — RecSys 2026 Music CRS Autonomous Loop

Re-read this file at the top of EVERY iteration before proposing an experiment.

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
Before every run, write the experiments_log.md entry (hypothesis + exact gate + honest
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
```

- [ ] **Step 2: Commit**

```bash
git add RESEARCH_CHARTER.md
git commit -m "docs: research charter (constitution) for the autonomous loop"
```

---

## Task 4: Seed `documents/experiments_log.md`

**Files:**
- Create: `documents/experiments_log.md`

- [ ] **Step 1: Create the journal with the template + a worked example**

Create `documents/experiments_log.md`:

```markdown
# Experiments Log — Pre-Registration Journal

Each experiment is pre-registered BELOW (hypothesis + gate + decision rule) BEFORE the run.
After results, fill the `--- run ---` block. INCONCLUSIVE is a first-class verdict.
Template:

## EXP-NNN — <lever> — YYYY-MM-DD
Hypothesis:   <falsifiable claim>
Lever:        recall | reranker | responder
Change:       <config NNN / code diff summary>
Pre-registered gate:  <metric> on <Tier-1 signal> must beat <baseline> by <delta>
Baseline:     <honest number + source>
Decision rule: PASS -> keep/promote; FAIL -> revert;
               INCONCLUSIVE (|Δ| < <noise band>) -> no change, counts toward switch rule
Smoke:        pytest <paths> green before handoff
--- run ---
Result:       <pasted RESULTS_JSON>
Verdict:      PASS | FAIL | INCONCLUSIVE
Local→Blind:  local Δ <>, blind Δ <>   (only if submitted)
Memory:       <file written/updated>

---

## EXP-000 — example (do not run) — 2026-06-13
Hypothesis:   Gemini responder on config-203 track_ids (nDCG 0.30, no Q*) lifts composite to ~0.47.
Lever:        responder
Change:       config 205 = 203 recall track_ids + gemini_responder.py
Pre-registered gate:  blindA composite must beat 0.44 (config 204) by > 0.05 (noise band)
Baseline:     config 204 composite 0.44 (benchmarks.md / memory project_blind_a_state_responder_lever)
Decision rule: PASS -> promote to blind candidate; FAIL -> revert; INCONCLUSIVE (|Δ|<0.05) -> next hypothesis
Smoke:        n/a (example)
--- run ---
Result:       (not run)
Verdict:      (n/a)
```

- [ ] **Step 2: Commit**

```bash
git add documents/experiments_log.md
git commit -m "docs: seed experiments_log pre-registration journal"
```

---

## Task 5: Budget-gated submission preparer (`scripts/loop_submit.py`)

**Files:**
- Create: `scripts/loop_submit.py`
- Test: `tests/test_loop_submit.py`

- [ ] **Step 1: Confirm the reused interfaces**

Read `scripts/validate_prediction.py` lines 68-100 (`load_prediction`, `validate_schema`)
and 184-300 (`check_budget`, `package_zip`, `_default_submission_log`) to confirm their exact
signatures and return shapes. The code below assumes `validate_schema(preds, split)` returns
a list (empty = valid) and `check_budget(log, weekly_cap=3)` returns `(ok: bool, msg: str)`.
If they differ, adjust the wrapper accordingly.

- [ ] **Step 2: Write the failing test**

Create `tests/test_loop_submit.py`:

```python
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.loop_submit as ls


def _write_pred(tmp_path):
    pred = tmp_path / "prediction.json"
    pred.write_text(json.dumps(
        [{"session_id": "s", "turn_number": 1,
          "predicted_track_ids": [], "predicted_response": "x"}]
    ))
    return pred


def test_blocked_when_budget_exceeded(tmp_path, monkeypatch):
    pred = _write_pred(tmp_path)
    monkeypatch.setattr(ls, "load_prediction", lambda p: json.loads(Path(p).read_text()))
    monkeypatch.setattr(ls, "validate_schema", lambda preds, split: [])
    monkeypatch.setattr(ls, "check_budget", lambda log, weekly_cap=3: (False, "cap exceeded"))
    plan = ls.prepare_submission(pred, tmp_path / "out.zip")
    assert plan.ok is False
    assert "cap" in plan.reason


def test_blocked_on_schema_errors(tmp_path, monkeypatch):
    pred = _write_pred(tmp_path)
    monkeypatch.setattr(ls, "load_prediction", lambda p: json.loads(Path(p).read_text()))
    monkeypatch.setattr(ls, "validate_schema", lambda preds, split: ["missing field"])
    plan = ls.prepare_submission(pred, tmp_path / "out.zip")
    assert plan.ok is False
    assert "schema" in plan.reason


def test_packages_when_ok(tmp_path, monkeypatch):
    pred = _write_pred(tmp_path)
    monkeypatch.setattr(ls, "load_prediction", lambda p: json.loads(Path(p).read_text()))
    monkeypatch.setattr(ls, "validate_schema", lambda preds, split: [])
    monkeypatch.setattr(ls, "check_budget", lambda log, weekly_cap=3: (True, "0/3"))
    out = tmp_path / "out.zip"
    plan = ls.prepare_submission(pred, out)
    assert plan.ok is True
    assert Path(plan.zip_path).exists()
    with zipfile.ZipFile(plan.zip_path) as z:
        assert z.namelist() == ["prediction.json"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_loop_submit.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.loop_submit'`

- [ ] **Step 4: Write minimal implementation**

Create `scripts/loop_submit.py`:

```python
"""Budget-gated submission preparer for the autonomous research loop.

Reuses the existing validators/packager in validate_prediction.py. Given a prediction.json
it: (1) schema-validates, (2) checks the weekly blind-submission budget, (3) packages the
CodaBench zip, and (4) returns a SubmissionPlan. It does NOT upload — uploading is the job
of codabench_submit.py (or a human one-click). Keeping prepare/upload separate means the
budget gate is enforced regardless of how the final upload happens.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_prediction import (  # noqa: E402
    load_prediction,
    validate_schema,
    check_budget,
    package_zip,
    _default_submission_log,
)


@dataclass
class SubmissionPlan:
    ok: bool
    reason: str
    zip_path: Optional[str]
    budget_msg: str


def prepare_submission(
    prediction_path: Path,
    output_zip: Path,
    *,
    split: str = "blindA",
    weekly_cap: int = 3,
    submission_log: Optional[Path] = None,
) -> SubmissionPlan:
    log = Path(submission_log) if submission_log else _default_submission_log()

    preds = load_prediction(prediction_path)
    errors = validate_schema(preds, split)
    if errors:
        return SubmissionPlan(False, f"schema invalid: {list(errors)[:3]}", None, "")

    ok, budget_msg = check_budget(log, weekly_cap=weekly_cap)
    if not ok:
        return SubmissionPlan(False, budget_msg, None, budget_msg)

    zip_out = package_zip(prediction_path, output_zip)
    return SubmissionPlan(True, "ready", str(zip_out), budget_msg)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_loop_submit.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add scripts/loop_submit.py tests/test_loop_submit.py
git commit -m "feat: budget-gated submission preparer (loop_submit)"
```

---

## Task 6: CodaBench HTTP layer (`scripts/codabench_submit.py`)

**Files:**
- Create: `scripts/codabench_submit.py`
- Test: `tests/test_codabench_submit.py`

The exact CodaBench endpoints are unconfirmed, so this task TDD's the endpoint-INDEPENDENT
logic (polling + score parsing) and isolates the live HTTP behind injected callables. The
live URLs are filled in during the discovery sub-step.

- [ ] **Step 1: Write the failing test (endpoint-independent logic)**

Create `tests/test_codabench_submit.py`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest
from scripts.codabench_submit import poll_until_scored, parse_score


def test_poll_returns_when_finished():
    states = iter([
        {"status": "running"},
        {"status": "running"},
        {"status": "finished", "scores": {"composite": 0.46}},
    ])
    result = poll_until_scored(lambda sid: next(states), "sub1",
                               max_tries=5, sleep_fn=lambda s: None)
    assert result["status"] == "finished"


def test_poll_raises_on_failure():
    with pytest.raises(RuntimeError):
        poll_until_scored(lambda sid: {"status": "failed"}, "sub1",
                          max_tries=5, sleep_fn=lambda s: None)


def test_poll_times_out():
    with pytest.raises(TimeoutError):
        poll_until_scored(lambda sid: {"status": "running"}, "sub1",
                          max_tries=3, sleep_fn=lambda s: None)


def test_parse_score_extracts_composite():
    status = {"status": "finished",
              "scores": {"composite": 0.46, "ndcg@20": 0.30}}
    parsed = parse_score(status)
    assert parsed["composite"] == 0.46
    assert parsed["ndcg@20"] == 0.30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_codabench_submit.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.codabench_submit'`

- [ ] **Step 3: Write minimal implementation (logic + discovery-gated live wrapper)**

Create `scripts/codabench_submit.py`:

```python
"""CodaBench submission HTTP layer for the autonomous research loop.

Endpoint-independent logic (poll_until_scored, parse_score) is fully tested. The live HTTP
calls are isolated in `_default_status_getter` / `submit_zip`, whose exact endpoints are
filled in during the discovery step (see DISCOVERY below) — confirmed by inspecting the
browser network traffic during one manual submission, or reading the codalab/codabench source.

DISCOVERY (fill these in, then delete this notice):
  AUTH:    how the token is sent (header e.g. `Authorization: Token <t>` vs JWT)
  SUBMIT:  POST <url> with the zip (multipart? presigned-URL upload then POST metadata?)
           required fields: competition_id, phase_id, ...
  STATUS:  GET <url>/<submission_id> -> json containing a status + scores
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_FINISHED = {"finished", "scored", "done", "complete"}
_FAILED = {"failed", "error", "cancelled"}


def poll_until_scored(
    get_status: Callable[[str], dict],
    submission_id: str,
    *,
    max_tries: int = 30,
    wait_s: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    """Poll an injected get_status(submission_id) until finished/failed or tries exhausted."""
    for _ in range(max_tries):
        status = get_status(submission_id)
        state = str(status.get("status", "")).lower()
        if state in _FINISHED:
            return status
        if state in _FAILED:
            raise RuntimeError(f"submission {submission_id} failed: {status}")
        sleep_fn(wait_s)
    raise TimeoutError(f"submission {submission_id} not scored after {max_tries} tries")


def parse_score(status: dict) -> dict[str, Any]:
    """Extract the score dict from a finished status payload."""
    scores = status.get("scores") or {}
    return dict(scores)


# --- live HTTP (discovery-gated) ---------------------------------------------

def _token() -> str:
    tok = os.environ.get("CODABENCH_TOKEN")
    if not tok:
        raise RuntimeError("CODABENCH_TOKEN env var not set")
    return tok


def submit_zip(zip_path: Path, competition_id: str, phase_id: str) -> str:
    """Upload the packaged zip and return a submission_id. ENDPOINTS PENDING DISCOVERY."""
    raise NotImplementedError("fill in after the DISCOVERY step (see module docstring)")


def _default_status_getter(submission_id: str) -> dict:
    """GET submission status from CodaBench. ENDPOINTS PENDING DISCOVERY."""
    raise NotImplementedError("fill in after the DISCOVERY step (see module docstring)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_codabench_submit.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/codabench_submit.py tests/test_codabench_submit.py
git commit -m "feat: CodaBench poll/parse logic + discovery-gated live wrapper"
```

- [ ] **Step 6: DISCOVERY — confirm endpoints + wire live HTTP (human-assisted)**

With the human: do ONE manual CodaBench submission with browser DevTools → Network open.
Capture: the auth header format, the submit POST URL + payload (multipart vs presigned
upload), and the status GET URL + the JSON field holding status + scores. Alternatively read
`codalab/codabench` on GitHub (`src/apps/api/`). Then implement `submit_zip` and
`_default_status_getter` against the real endpoints using `requests`, confirm `CODABENCH_TOKEN`
auth works against the live competition, and update the module docstring (remove the DISCOVERY
notice). Commit:

```bash
git add scripts/codabench_submit.py
git commit -m "feat: wire live CodaBench submit/status endpoints (post-discovery)"
```

If the live API proves brittle, STOP here — the loop still works end-to-end via Task 5's
preparer + a human one-click upload (the spec's documented fallback).

---

## Task 7: One-time setup verification (human-assisted)

**Files:** none (verification only)

- [ ] **Step 1: Confirm the Tier-1 local gate reproduces a known baseline**

Hand the human a run package: run nb74 end-to-end on config 204, paste back `RESULTS_JSON`.
Confirm the composite matches `benchmarks.md`/memory for 204 (~0.44) within the noise band
(±0.05). This validates the trusted reward before the loop relies on it.

- [ ] **Step 2: Confirm the CodaBench token + cap**

Confirm `CODABENCH_TOKEN` is set in the environment Claude runs in (for auto-submit) and the
real weekly/total submission cap. Update `RESEARCH_CHARTER.md` `WEEKLY_CAP` with the confirmed
value. Commit:

```bash
git add RESEARCH_CHARTER.md
git commit -m "docs: confirm submission cap in charter"
```

---

## Task 8: First live iteration (dry-run of the 6-phase protocol)

**Files:** appends to `documents/experiments_log.md`, `documents/benchmarks.md`, memory

- [ ] **Step 1: Run one full protocol cycle on a cheap experiment**

Execute the protocol once, end to end, to validate the loop:
1. HYPOTHESIZE — read charter + memory + benchmarks; pick a cheap, high-confidence lever
   (e.g. the EXP-000 example: Gemini responder on config-203 track_ids).
2. PRE-REGISTER — write + commit the `experiments_log.md` entry (hypothesis, gate, baseline,
   decision rule) BEFORE running.
3. PREPARE — create the config; write/with tests; run `python -m pytest -q` (all green);
   **dispatch the code-review agent on the diff (leak-safety + correctness)** and address
   findings; assemble the run package.
4. HANDOFF — give the human the run package; receive the `RESULTS_JSON` block.
5. JUDGE — mark PASS/FAIL/INCONCLUSIVE strictly against the pre-registered gate.
6. REVIEW — **dispatch the RecSys-researcher agent on the verdict**; it must sign off
   (leak-free, justified within the noise band, not a known trap) before anything is banked.
7. RECORD — append to `benchmarks.md`, update memory, and (if PASS + budget allows) prepare
   the submission via `loop_submit.prepare_submission` and submit (auto or one-click).

- [ ] **Step 2: Retro-check the loop**

Confirm: the pre-registration was committed before results; both review agents ran and their
findings were addressed; the verdict used only Tier-1 signals (no banned in-sample val); the
budget gate fired correctly; memory was updated. Note any friction for the future
`/research-tick` skill (spec §8, out of scope here).

---

## Self-Review

- **Spec coverage:** §3.1 artifacts → Tasks 1,3,4,5,6 + notebook edits (Task 2). §2.1 trusted
  reward + banned signals → encoded in the charter (Task 3) and the judge step (Task 8).
  §4 protocol → Task 8. §5 handoff/paste-back formats → emitter (Task 1) + run-package (Task 8).
  §6 submission + budget → Tasks 5,6. §7 setup → Task 7. §9 open items (endpoints, quota N,
  noise band Z) → Task 6 DISCOVERY + Task 7 + charter. All covered.
- **Placeholder scan:** notebook variable names in Task 2 are explicitly flagged as
  "confirm by reading" (real notebooks vary); the CodaBench endpoints in Task 6 are an
  explicit DISCOVERY step, not a hidden TODO. No silent placeholders.
- **Type consistency:** `build_results_json`/`format_results_block`/`print_results_block`
  (Task 1), `prepare_submission`→`SubmissionPlan` (Task 5), `poll_until_scored`/`parse_score`
  (Task 6) are referenced consistently across tasks and tests.
```

