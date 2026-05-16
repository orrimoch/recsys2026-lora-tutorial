# Blind-A Iteration + Freeze (W6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize Blind-A composite via the 3 SID configs we've built (W4 baseline ensemble + W5 weight winner + pure-SID 171), pick the strongest, and freeze it as `sid-v1-frozen`. Blind-B is deferred — it doesn't open until 2026-06-23 and only matters once Blind-A is locked.

**Architecture:** Two small tooling additions: (1) a pre-submission validator that runs more checks than CodaBench's accept-anything-with-80-entries policy; (2) a `scripts/blind_a_score_tracker.py` that lets you append CodaBench scores to a markdown table memory file (so we can see all 3 SID submissions in one place). The rest is documentation: a per-axis score-interpretation guide, a freeze-decision rule, and a postmortem template for the "all 3 SID configs underperform" branch. Blind-B becomes one final small task at the end.

**Tech Stack:** Python 3.10, `pandas` (already a dep), `pyyaml`. No new pip deps.

**Spec reference:** `documents/specs/2026-05-15-sid-retrieval-design.md` §5.1 W5+W6 rows + §5.3 deliverables checklist + `recsys_challenge_notes.md:159-187` composite formula.

**Inputs from W4 + W5:**
- W4 first Blind-A submission: config 170 (ensemble, SID weight=0.5)
- W5 second + third Blind-A submissions: winning weight from sweep + pure-SID 171
- Three CodaBench score pages (composite + nDCG@20 + LLM + lex_div + CatDiv per submission)

**Hard prerequisite:** at least the W4 Blind-A submission (notebook 63) must have a CodaBench score back before W6 starts. W5 submissions can land during W6 — that's fine, they get appended to the tracker as they arrive.

**Output artifacts:**
- `scripts/precheck_prediction.py` — pre-submission validator (stricter than `validate_prediction.py`)
- `scripts/blind_a_score_tracker.py` — append-only CodaBench score tracker
- `~/.claude/projects/.../memory/project_blind_a_submissions.md` — single source of truth for all Blind-A scores
- `~/.claude/projects/.../memory/project_sid_retrieval_v1_result.md` — final consolidated result + freeze decision
- `documents/specs/2026-05-15-sid-retrieval-design.md` amended with "What Shipped (W6 freeze)" appendix
- Git tag `sid-v1-frozen` on the freeze commit
- (Stretch, Task 8) `music-crs-baselines/config/180-wrrf-sid-v5kto-blindsetB.yaml` + `colab/67_run_blindset_sid_B.ipynb` for Blind-B once it opens

---

## File structure

| Path | Type | Responsibility |
|---|---|---|
| `scripts/precheck_prediction.py` | new | Stricter pre-submission validation: 80-entry count + each record has `session_id` + `turn_number` + `predicted_track_ids` (non-empty list) + every track id is in the catalog (no hallucinated UUIDs) + no duplicate tracks per session × turn |
| `tests/test_precheck_prediction.py` | new | 5 TDD tests for the precheck script |
| `scripts/blind_a_score_tracker.py` | new | CLI to append a CodaBench result (`--composite`, `--ndcg`, `--llm`, `--lex_div`, `--cat_div`, `--config_id`, `--submission_url`) to the score tracker memory file |
| `tests/test_blind_a_score_tracker.py` | new | 3 TDD tests for the score tracker |
| `~/.claude/projects/.../memory/project_blind_a_submissions.md` | new | Markdown table (one row per CodaBench submission) maintained by the tracker script |
| `~/.claude/projects/.../memory/project_sid_retrieval_v1_result.md` | new | Final result memory; written ONCE at freeze time (Task 6) |
| `documents/specs/2026-05-15-sid-retrieval-design.md` | append | "What Shipped (W6 freeze)" appendix |
| `music-crs-baselines/config/180-wrrf-sid-v5kto-blindsetB.yaml` | new (Task 8) | Blind-B variant of freeze config — only `test_dataset_name` differs |
| `colab/67_run_blindset_sid_B.ipynb` | new (Task 8) | Blind-B submission, mirrors notebook 63 |

---

## Task 1: TDD `scripts/precheck_prediction.py` — stricter pre-submission validation

CodaBench accepts any zip with `prediction.json` at root + 80 entries. It silently rejects bad payloads (e.g., hallucinated track IDs not in the catalog). We want to catch those locally so we don't waste a daily quota slot on a malformed submission.

**Files:**
- Create: `scripts/precheck_prediction.py`
- Create: `tests/test_precheck_prediction.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_precheck_prediction.py`:

```python
"""Tests for scripts/precheck_prediction.py."""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.precheck_prediction import precheck


def test_precheck_passes_clean_prediction(tmp_path):
    """All 80 entries valid → ok=True, no errors."""
    catalog = {"t1", "t2", "t3"}
    records = [
        {"session_id": f"S{i}", "user_id": f"U{i}", "turn_number": 1,
         "predicted_track_ids": ["t1", "t2", "t3"]}
        for i in range(80)
    ]
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(records))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert result["ok"]
    assert result["errors"] == []


def test_precheck_flags_wrong_entry_count(tmp_path):
    """Less than 80 entries → ok=False, count error."""
    catalog = {"t1"}
    p = tmp_path / "pred.json"
    p.write_text(json.dumps([
        {"session_id": "S0", "user_id": "U0", "turn_number": 1, "predicted_track_ids": ["t1"]}
    ]))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert not result["ok"]
    assert any("expected 80 entries, got 1" in e for e in result["errors"])


def test_precheck_flags_missing_required_fields(tmp_path):
    """Record missing 'predicted_track_ids' → ok=False with field error."""
    catalog = {"t1"}
    records = [
        {"session_id": f"S{i}", "user_id": f"U{i}", "turn_number": 1,
         "predicted_track_ids": ["t1"]}
        for i in range(79)
    ]
    records.append({"session_id": "S79", "user_id": "U79", "turn_number": 1})  # missing predicted_track_ids
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(records))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert not result["ok"]
    assert any("predicted_track_ids" in e for e in result["errors"])


def test_precheck_flags_hallucinated_track_ids(tmp_path):
    """Track IDs not in catalog → ok=False with hallucination error."""
    catalog = {"t1", "t2"}
    records = [
        {"session_id": f"S{i}", "user_id": f"U{i}", "turn_number": 1,
         "predicted_track_ids": ["t1", "HALLUCINATED_TRACK", "t2"]}
        for i in range(80)
    ]
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(records))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert not result["ok"]
    assert any("HALLUCINATED_TRACK" in e for e in result["errors"])


def test_precheck_flags_duplicate_tracks_per_record(tmp_path):
    """Same track id appears twice in a record's predicted_track_ids → ok=False."""
    catalog = {"t1", "t2"}
    records = [
        {"session_id": f"S{i}", "user_id": f"U{i}", "turn_number": 1,
         "predicted_track_ids": ["t1", "t2", "t1"]}  # duplicate t1
        for i in range(80)
    ]
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(records))
    result = precheck(p, catalog=catalog, expected_n=80)
    assert not result["ok"]
    assert any("duplicate" in e.lower() for e in result["errors"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest tests/test_precheck_prediction.py -q
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'scripts.precheck_prediction'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/precheck_prediction.py`:

```python
"""Stricter pre-submission validator for Blind-A prediction.json.

CodaBench's accept-anything-with-80-entries policy silently rejects payloads
with hallucinated track IDs, missing required fields, or duplicate tracks
within a record. Run this BEFORE zipping + uploading to avoid burning a
daily submission slot on a malformed payload.

Usage:
    python scripts/precheck_prediction.py \\
        --input music-crs-baselines/exp/inference/blindset_A/<tid>.json \\
        --catalog talkpl-ai/TalkPlayData-Challenge-Track-Metadata
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_FIELDS = ["session_id", "user_id", "turn_number", "predicted_track_ids"]


def precheck(pred_path: Path, catalog: set[str], expected_n: int = 80) -> dict:
    """Validate a Blind-A prediction.json.

    Returns:
        {"ok": bool, "errors": list[str], "warnings": list[str], "n_records": int}
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        records = json.loads(pred_path.read_text())
    except Exception as e:
        return {"ok": False, "errors": [f"failed to parse JSON: {e}"],
                "warnings": [], "n_records": 0}

    if not isinstance(records, list):
        return {"ok": False, "errors": [f"expected list of records, got {type(records).__name__}"],
                "warnings": [], "n_records": 0}

    n = len(records)
    if n != expected_n:
        errors.append(f"expected {expected_n} entries, got {n}")

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            errors.append(f"record {i}: expected dict, got {type(rec).__name__}")
            continue
        # Required-field check
        for field in REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"record {i} (session={rec.get('session_id', '?')}): missing required field '{field}'")
        # Track-ID checks
        tracks = rec.get("predicted_track_ids", [])
        if not isinstance(tracks, list):
            errors.append(f"record {i}: predicted_track_ids must be a list, got {type(tracks).__name__}")
            continue
        if not tracks:
            errors.append(f"record {i}: predicted_track_ids is empty")
            continue
        # Catalog membership
        for tid in tracks:
            if tid not in catalog:
                errors.append(f"record {i}: hallucinated track id (not in catalog): {tid!r}")
                # Don't spam — one hallucination per record is enough signal
                break
        # Duplicate-within-record
        if len(tracks) != len(set(tracks)):
            dupes = [t for t in tracks if tracks.count(t) > 1]
            errors.append(f"record {i}: duplicate track ids: {set(dupes)}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_records": n,
    }


def _load_catalog(dataset: str) -> set[str]:
    """Load all valid track IDs from the TalkPlayData catalog."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split="all_tracks")
    return {row["track_id"] for row in ds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Path to prediction.json")
    p.add_argument("--catalog", type=str,
                   default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                   help="HF dataset for the track catalog (provides valid track IDs)")
    p.add_argument("--expected_n", type=int, default=80,
                   help="Expected entry count (80 for Blind-A)")
    args = p.parse_args()

    print(f"[precheck] loading catalog from {args.catalog}...", file=sys.stderr)
    catalog = _load_catalog(args.catalog)
    print(f"[precheck] catalog has {len(catalog)} tracks", file=sys.stderr)

    print(f"[precheck] validating {args.input}...", file=sys.stderr)
    result = precheck(args.input, catalog=catalog, expected_n=args.expected_n)
    print(json.dumps({"ok": result["ok"], "n_records": result["n_records"],
                      "n_errors": len(result["errors"])}, indent=2))
    if result["errors"]:
        print("\nERRORS:", file=sys.stderr)
        for e in result["errors"][:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(result["errors"]) > 20:
            print(f"  ... and {len(result['errors']) - 20} more", file=sys.stderr)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_precheck_prediction.py -q
```

Expected: 5 passing.

- [ ] **Step 5: Smoke-test the CLI**

```bash
python scripts/precheck_prediction.py --help
```

Expected: argparse help with `--input`, `--catalog`, `--expected_n` flags.

- [ ] **Step 6: Commit**

```bash
git add scripts/precheck_prediction.py tests/test_precheck_prediction.py
git commit -m "sid w6: precheck_prediction.py — stricter pre-submission validation (TDD, 5 tests)"
```

---

## Task 2: TDD `scripts/blind_a_score_tracker.py` — append-only CodaBench score tracker

Each Blind-A submission produces 4 numbers (composite + nDCG + LLM + lex_div, plus CatDiv we ignore at 0.03). Without a tracker we lose context across 3+ submissions ("what was the W4 baseline composite again?"). One tracker memory file = single source of truth.

**Files:**
- Create: `scripts/blind_a_score_tracker.py`
- Create: `tests/test_blind_a_score_tracker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_blind_a_score_tracker.py`:

```python
"""Tests for scripts/blind_a_score_tracker.py."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.blind_a_score_tracker import append_score, read_tracker


def test_append_score_creates_tracker_with_header(tmp_path):
    """First append on a fresh path creates the tracker with markdown table header."""
    tracker = tmp_path / "tracker.md"
    append_score(
        tracker_path=tracker,
        config_id="170-wrrf-sid-v5kto-blindsetA",
        composite=0.245, ndcg=0.082, llm=2.50, lex_div=0.77,
        submission_url="https://codabench.org/competitions/.../submission/...",
        notes="W4 first SID submission, weight=0.5",
    )
    assert tracker.exists()
    text = tracker.read_text()
    assert "| config_id " in text  # header row
    assert "170-wrrf-sid-v5kto-blindsetA" in text
    assert "0.245" in text


def test_append_score_appends_second_row_below_existing(tmp_path):
    """Second append adds a row, doesn't duplicate the header."""
    tracker = tmp_path / "tracker.md"
    append_score(tracker_path=tracker, config_id="A", composite=0.1, ndcg=0.05,
                 llm=2.0, lex_div=0.7, submission_url="url1")
    append_score(tracker_path=tracker, config_id="B", composite=0.2, ndcg=0.08,
                 llm=2.5, lex_div=0.75, submission_url="url2")
    text = tracker.read_text()
    assert text.count("| config_id ") == 1  # header appears once
    assert "config_id: A" not in text  # rows aren't keyed by 'config_id:'
    assert "| A |" in text
    assert "| B |" in text


def test_read_tracker_returns_all_rows(tmp_path):
    """read_tracker parses the markdown table back into a list of dicts."""
    tracker = tmp_path / "tracker.md"
    append_score(tracker_path=tracker, config_id="A", composite=0.1, ndcg=0.05,
                 llm=2.0, lex_div=0.7, submission_url="url1")
    append_score(tracker_path=tracker, config_id="B", composite=0.2, ndcg=0.08,
                 llm=2.5, lex_div=0.75, submission_url="url2")
    rows = read_tracker(tracker)
    assert len(rows) == 2
    assert rows[0]["config_id"] == "A"
    assert float(rows[0]["composite"]) == 0.1
    assert rows[1]["config_id"] == "B"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_blind_a_score_tracker.py -q
```

Expected: 3 failures with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `scripts/blind_a_score_tracker.py`:

```python
"""Append-only Blind-A submission score tracker.

Maintains a markdown table in a memory file so all CodaBench Blind-A
submissions are visible in one place. Run after each CodaBench score
becomes visible:

    python scripts/blind_a_score_tracker.py append \\
        --tracker ~/.claude/projects/.../memory/project_blind_a_submissions.md \\
        --config_id 170-wrrf-sid-v5kto-blindsetA \\
        --composite 0.245 --ndcg 0.082 --llm 2.50 --lex_div 0.77 \\
        --url https://codabench.org/competitions/.../submission/...

    python scripts/blind_a_score_tracker.py read \\
        --tracker ~/.claude/projects/.../memory/project_blind_a_submissions.md
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Optional


HEADER = (
    "| config_id | composite | ndcg@20 | llm | lex_div | submitted_at | url | notes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def append_score(
    tracker_path: Path,
    config_id: str,
    composite: float,
    ndcg: float,
    llm: float,
    lex_div: float,
    submission_url: str,
    notes: str = "",
    submitted_at: Optional[str] = None,
) -> None:
    """Append one CodaBench submission result to the markdown table."""
    if submitted_at is None:
        submitted_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if not tracker_path.exists():
        # First write: include the table header.
        preamble = (
            "---\n"
            "name: Blind-A submissions tracker\n"
            "description: Append-only log of all CodaBench Blind-A submissions for the SID retriever sprint. Maintained by scripts/blind_a_score_tracker.py.\n"
            "type: project\n"
            "---\n\n"
            "# Blind-A submissions\n\n"
            "Each row = one CodaBench submission. composite = 0.5*nDCG@20 + 0.1*CatDiv + 0.1*LexDiv + 0.3*LLM (CatDiv is saturated at ~0.03 across the leaderboard; only nDCG/LLM/LexDiv are interesting).\n\n"
        )
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_text(preamble + HEADER)

    row = (
        f"| {config_id} "
        f"| {composite:.4f} "
        f"| {ndcg:.4f} "
        f"| {llm:.4f} "
        f"| {lex_div:.4f} "
        f"| {submitted_at} "
        f"| {submission_url} "
        f"| {notes} |\n"
    )
    with tracker_path.open("a") as f:
        f.write(row)


def read_tracker(tracker_path: Path) -> list[dict]:
    """Parse the markdown table back into a list of dicts."""
    if not tracker_path.exists():
        return []
    text = tracker_path.read_text()
    rows: list[dict] = []
    in_table = False
    columns: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("| config_id "):
            columns = [c.strip() for c in line.strip("|").split("|")]
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != len(columns):
                continue
            rows.append(dict(zip(columns, cells)))
        elif in_table and not line.startswith("|"):
            # Table ended (blank line or non-table content)
            in_table = False
    return rows


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    append_p = sub.add_parser("append")
    append_p.add_argument("--tracker", type=Path, required=True)
    append_p.add_argument("--config_id", type=str, required=True)
    append_p.add_argument("--composite", type=float, required=True)
    append_p.add_argument("--ndcg", type=float, required=True)
    append_p.add_argument("--llm", type=float, required=True)
    append_p.add_argument("--lex_div", type=float, required=True)
    append_p.add_argument("--url", type=str, dest="submission_url", required=True)
    append_p.add_argument("--notes", type=str, default="")

    read_p = sub.add_parser("read")
    read_p.add_argument("--tracker", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "append":
        append_score(
            tracker_path=args.tracker,
            config_id=args.config_id,
            composite=args.composite, ndcg=args.ndcg,
            llm=args.llm, lex_div=args.lex_div,
            submission_url=args.submission_url,
            notes=args.notes,
        )
        print(f"appended row for {args.config_id} → {args.tracker}")
    elif args.cmd == "read":
        rows = read_tracker(args.tracker)
        print(f"{len(rows)} rows:")
        for r in rows:
            print(f"  {r['config_id']:50s} composite={r['composite']} ndcg={r['ndcg@20']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_blind_a_score_tracker.py -q
```

Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add scripts/blind_a_score_tracker.py tests/test_blind_a_score_tracker.py
git commit -m "sid w6: blind_a_score_tracker.py — markdown-table CodaBench score log (TDD, 3 tests)"
```

---

## Task 3: Per-axis score interpretation guide (memory file)

When the first Blind-A score lands, the composite alone doesn't tell us where SID is working vs failing. We need a quick guide that maps each axis movement to a concrete diagnosis + next action.

**Files:**
- Create: `~/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_blind_a_axis_interpretation.md`

- [ ] **Step 1: Write the interpretation guide**

Create the file with this content:

```markdown
---
name: Blind-A per-axis interpretation guide
description: How to read each axis of a CodaBench Blind-A composite. Maps axis movements to root-cause hypotheses + actions. Single page reference for during-iteration triage.
type: reference
---

## Composite formula

`composite = 0.5*nDCG@20 + 0.1*CatDiv + 0.1*LexDiv + 0.3*LLM`

Source: `recsys_challenge_notes.md:159-187`. CatDiv saturated at ~0.03 across the leaderboard per `project_cat_div_saturated.md` — don't chase it.

## Baseline (pre-SID champion config 132)

| axis | value |
|---|---|
| composite | 0.21 |
| nDCG@20 | 0.06 |
| CatDiv | 0.03 |
| LexDiv | 0.77 |
| LLM | 2.35 |

## Axis-by-axis interpretation

### nDCG@20 (50% of composite — the most leverage)

Measures retrieval quality: is the gold track in our top-20? SID's purpose is to lift this.

| Observation | Likely cause | Next action |
|---|---|---|
| nDCG@20 > 0.10 (≥ +0.04 vs 132's 0.06) | SID is finding gold tracks at higher ranks than BM25 + dense. **SID is working.** | Lock the winning weight. Move to LLM tuning. |
| nDCG@20 = 0.06-0.09 (modest improvement) | SID is helping marginally. Could be coarse W1 codebook (3K SIDs) capping recall. | Try W5 weight sweep results. If pure-SID (171) > ensemble (170), SID is the right signal but ensemble is diluting it. |
| nDCG@20 ≈ 0.06 (unchanged) | SID is contributing zero or canceling out BM25's hits. | Check `_decode_beams_to_tracks` warning logs for SID failures > 50%. Likely tokenizer/vocab issue. |
| nDCG@20 < 0.06 (regression) | SID is actively hurting — its predictions are pushing good BM25/dense candidates out of top-20. | Lower the SID weight (0.5 → 0.3 → 0.1) OR submit pure-current-wRRF (no SID) as fallback. |

### LLM (30% of composite — second most leverage)

Gemini-scored responder output. Driven by the v5-kto responder, not SID directly. **SID changes shouldn't move this much** — if they do, something's wrong upstream.

| Observation | Likely cause | Next action |
|---|---|---|
| LLM > 2.50 (vs 132's 2.35) | Responder is getting better top-1 inputs from SID-augmented retrieval. | Lock the config. |
| LLM = 2.30-2.45 | Roughly unchanged; SID's contribution is in retrieval not responder input quality. | Expected. |
| LLM < 2.20 (regression) | SID top-1 is a worse responder input than BM25 top-1. Could be SID is fetching musically weird matches (e.g., long-tail tracks the responder can't synthesize about). | Try config 171 (pure-SID) to confirm — if 171's LLM is also low, SID's top-1 is the issue. |

### LexDiv (10% — minor leverage but easy bonus)

Measures vocabulary diversity in the predicted_response. Mostly a function of the responder + prompt.

| Observation | Likely cause | Next action |
|---|---|---|
| LexDiv > 0.78 | Responder is more varied. SID indirectly helped by providing more diverse top-1s. | Free win. |
| LexDiv = 0.75-0.78 | Unchanged. | Expected. |
| LexDiv < 0.75 | Responder is more repetitive — likely SID is delivering the same kind of track repeatedly (e.g., collision-bucket flooding). | Verify per-bucket cap is firing (`cap_per_bucket=1` in SID_GENERATOR). |

### CatDiv (10% — saturated; ignore)

Per `project_cat_div_saturated.md` most teams score ~0.03 here. Don't optimize for it.

## Composite read across the 3 SID submissions

After all 3 SID Blind-A submissions land:

| Composite | What it means | Action |
|---|---|---|
| Best of 3 ≥ 0.24 (≥ +0.03 over 132) | SID is a real win. | Freeze best of 3 as v1. Proceed to Blind-B prep. |
| Best of 3 = 0.21-0.23 (parity to small lift) | SID is competitive but not decisively better. | Freeze best of 3 BUT also keep config 132 as fallback. Investigate why lift is modest (W1 SID coarseness most likely). |
| Best of 3 < 0.21 (regression) | SID is hurting in production. Don't ship. | Submit config 132 (or current leaderboard config) as the Blind-A entry. Write v1 postmortem (see Task 7). Plan v2 sprint. |

## Standard escalation order when scores miss

1. **Check the precheck** — was the prediction.json well-formed? (`scripts/precheck_prediction.py`)
2. **Check SID decode failure rate** in the training_log / inference log — > 50% means tokenizer/vocab mismatch.
3. **Check that batch_context is actually being forwarded** at inference — without it, SID receives bare queries (W4 reviewer caught this in commit `658f07b`).
4. **Try pure-SID 171** if the ensemble underperforms — isolates whether SID itself is bad or the fusion weight is wrong.
5. **If pure-SID is also bad**: W3 generator is the bottleneck. Retrain with Qwen-2.5-3B (v2 candidate).
6. **If pure-SID is fine but ensemble bad**: weight is wrong. Sweep finer-grained {0.1, 0.2, 0.3, 0.4}.

## Source of truth

All scores: `~/.claude/projects/.../memory/project_blind_a_submissions.md` (maintained by `scripts/blind_a_score_tracker.py`).
```

- [ ] **Step 2: Verify the file exists**

```bash
ls -la ~/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_blind_a_axis_interpretation.md
wc -l ~/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_blind_a_axis_interpretation.md
```

Expected: ~120 lines.

- [ ] **Step 3: Memory file lives outside repo — no git commit needed.**

---

## Task 4: Submission strategy + standard operating procedure (in-repo doc)

A short doc that captures the END-TO-END workflow for each Blind-A submission. Lives in the repo so the procedure is versioned alongside the configs.

**Files:**
- Create: `documents/blind_a_submission_sop.md`

- [ ] **Step 1: Write the SOP**

```markdown
# Blind-A Submission Standard Operating Procedure

For each SID Blind-A submission, follow these steps in order:

## Pre-submission (do NOT skip)

1. **Run the dev gate** for the candidate config (notebook 64 for ensemble; manual run for pure-SID).
2. **Check dev nDCG@20** vs config 132 baseline (0.06 on dev). If dev nDCG@20 < 0.05, **do NOT submit** — it'll just regress on Blind-A. Iterate first.
3. **Run inference on Blind-A** (notebook 63 for W4 ensemble; notebook 66 for W5 winner + pure-SID).
4. **Run the precheck**:
   ```bash
   python scripts/precheck_prediction.py \
       --input music-crs-baselines/exp/inference/blindset_A/<tid>.json
   ```
   Must exit 0. If it errors (hallucinated IDs, missing fields, duplicates), DO NOT submit — fix first.

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

8. **Interpret the score** using `project_blind_a_axis_interpretation.md`:
   - Which axis moved most vs the previous submission?
   - Was the movement explained by the config change?
   - If not, escalate (see escalation order in the interpretation guide).

9. **Decide next move** based on the score table:
   - PASS (≥ 0.24): one more submission to be sure → freeze.
   - PARITY (0.21-0.23): submit the alternate config (pure-SID if you just submitted ensemble; lower-weight if you just submitted weight winner).
   - REGRESSION (< 0.21): STOP — don't burn more quota. Diagnose (see escalation order).

## Daily quota discipline

CodaBench has a per-day submission limit (per the platform's terms — verify the current quota before each session). Sequence submissions deliberately:
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
```

- [ ] **Step 2: Commit**

```bash
git add documents/blind_a_submission_sop.md
git commit -m "sid w6: Blind-A submission SOP — pre-submit precheck + post-submit tracker workflow"
```

---

## Task 5: Update notebooks 63 + 66 to call the precheck before zipping

Wire the new precheck into the existing submission notebooks so it runs automatically.

**Files:**
- Modify: `colab/63_run_blindset_sid.ipynb`
- Modify: `colab/66_blind_a_w5_submissions.ipynb`

- [ ] **Step 1: Patch notebook 63 cell 4** (currently calls `validate_prediction.py`). Add the precheck call right before the existing validator.

Use this patch script:

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python <<'EOF'
import json
from pathlib import Path

for nb_path_str in [
    'colab/63_run_blindset_sid.ipynb',
    'colab/66_blind_a_w5_submissions.ipynb',
]:
    nb_path = Path(nb_path_str)
    nb = json.loads(nb_path.read_text())
    patched = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'validate_prediction.py' in src and 'precheck_prediction.py' not in src:
            # Insert the precheck call BEFORE the existing validator call
            new_src = src.replace(
                '!python scripts/validate_prediction.py',
                '!python scripts/precheck_prediction.py --input {pred_path}\n'
                '!python scripts/validate_prediction.py',
            )
            # The above only works if --input {pred_path} substitution is meaningful.
            # For 63: pred_path is in scope. For 66: it's tid + pred_path in a loop.
            # Fix for 66: iterate loop already has pred_path; insert before validate call.
            cell['source'] = new_src.splitlines(keepends=True)
            patched = True
            print(f'patched {nb_path_str} cell {i}')
            break
    if patched:
        nb_path.write_text(json.dumps(nb, indent=1))
    else:
        print(f'no patch applied to {nb_path_str} (already done or no match)')
EOF
```

- [ ] **Step 2: Verify both notebooks are still well-formed**

```bash
for nb in colab/63_run_blindset_sid.ipynb colab/66_blind_a_w5_submissions.ipynb; do
    python -c "import nbformat; nb = nbformat.read('$nb', as_version=4); print(f'$nb: {len(nb.cells)} cells')"
done
```

Expected: both notebooks load cleanly.

- [ ] **Step 3: Commit**

```bash
git add colab/63_run_blindset_sid.ipynb colab/66_blind_a_w5_submissions.ipynb
git commit -m "sid w6: wire precheck_prediction into notebooks 63 + 66 (fail-fast on bad payload)"
```

---

## Task 6: Write the consolidated `project_sid_retrieval_v1_result.md` (only after 3 Blind-A scores land)

This is the freeze deliverable. Run AFTER all 3 SID Blind-A submissions have CodaBench scores.

**Files:**
- Create: `~/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_sid_retrieval_v1_result.md`

- [ ] **Step 1: Read the latest scores from the tracker**

```bash
python scripts/blind_a_score_tracker.py read \
    --tracker ~/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_blind_a_submissions.md
```

Expected: prints all 3 (or however many) SID submissions with their scores.

- [ ] **Step 2: Pick the winner — best composite among the 3 SID submissions**

```bash
python <<'EOF'
import sys
sys.path.insert(0, '.')
from scripts.blind_a_score_tracker import read_tracker
from pathlib import Path

tracker = Path('/Users/orrimoch/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_blind_a_submissions.md')
rows = read_tracker(tracker)
sid_rows = [r for r in rows if 'sid' in r['config_id'].lower()]
if not sid_rows:
    print('NO SID submissions in tracker yet — run Tasks 1-5 + submit first')
    sys.exit(1)
winner = max(sid_rows, key=lambda r: float(r['composite']))
print(f"WINNER: {winner['config_id']}")
print(f"  composite: {winner['composite']}")
print(f"  nDCG@20:   {winner['ndcg@20']}")
print(f"  LLM:       {winner['llm']}")
print(f"  lex_div:   {winner['lex_div']}")

# Compare to pre-SID champion (132 hardcoded 0.21):
delta = float(winner['composite']) - 0.21
print(f"\\nΔ vs pre-SID champion 132: {delta:+.4f}")
if delta >= 0.03:
    print('VERDICT: SHIP — significant lift, freeze this config as v1')
elif delta >= 0:
    print('VERDICT: SHIP-WITH-CAVEAT — marginal lift, freeze but document the small win')
else:
    print('VERDICT: DO NOT SHIP — regression vs 132. Write postmortem (Task 7).')
EOF
```

- [ ] **Step 3: Write the result file** (template inlined; replace `<FILL>` with values from Step 2 + tracker)

Create `/Users/orrimoch/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_sid_retrieval_v1_result.md`:

```markdown
---
name: SID retriever v1 — FROZEN at W6 (2026-MM-DD)
description: Final result of the 6-week SID generative retriever sprint. Frozen config + Blind-A scores + per-week summary + lessons + v2 candidate workstreams.
type: project
---

**Status**: SHIPPED at W6 (2026-MM-DD). The Blind-A winner from `project_blind_a_submissions.md` has been promoted to `sid-v1-frozen` (git tag).

## Frozen config

- **Config ID**: `<FILL from winner>`
- **Hub repo (SID generator)**: `OrRim123/recsys2026-sid-generator-qwen15b-v1-merged`
- **Responder**: `OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged`
- **Git tag**: `sid-v1-frozen`

## Blind-A scores (all SID submissions + pre-SID champion)

| Config | composite | nDCG@20 | LLM | lex_div | Δ vs 132 |
|---|---|---|---|---|---|
| Pre-SID champion (132) | 0.21 | 0.06 | 2.35 | 0.77 | — |
| W4 first (170, weight=0.5) | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| W5 winner (170-w<N>) | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| W5 pure-SID (171) | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| **WINNER** | <BOLD copy of winning row> | | | | |

## Per-week summary

| Week | Component | Outcome |
|---|---|---|
| W1 | Quantizer (RQ-VAE + Sinkhorn) | 3,017 unique SIDs from 47K tracks |
| W2 | Training data (raw 121K + metadata 47K) | 168K pairs after session-level dedup + 1-turn-per-session val |
| W3 | Generator fine-tune | <FILL: val nDCG@20 at end of training> |
| W4 | Inference pipeline + dev gate | Dev Δ = <FILL>, CI lo = <FILL> |
| W5 | Weight sweep + multi-Blind-A | Best weight = <FILL>; sweep dev nDCG@20 range = <FILL> |
| W6 | Freeze | Winner Blind-A composite = <FILL> (vs 132's 0.21) |

## Lessons learned (apply to v2)

(Copy items 1-8 from `documents/specs/2026-05-15-sid-retrieval-design.md` Appendix W6 — see Task 8 of this plan.)

## V2 candidate workstreams (if v1 didn't move composite enough)

1. **Bigger SID generator**: Qwen-2.5-3B instead of 1.5B. Memory: 1.5B may have a format ceiling per W6 v5-kto lesson.
2. **Finer W1 quantizer**: codebook_size 512, latent_dim 128, more epochs → 8-10K unique SIDs.
3. **Doc2query fix + full coverage**: redo notebook 54 with the audit-fixed prompt + parser.
4. **Rank-GRPO post-training**: per `recent_papers_ideas.md:150`.

## How to reproduce v1

```bash
git checkout sid-v1-frozen
# Then run notebooks 60 → 61 → 62 → 63 (or 66 for sweep variant)
```
```

After writing, fill in every `<FILL>` from the tracker.

- [ ] **Step 4: NO git commit** (memory files live outside the repo).

---

## Task 7: Amend the spec with "What Shipped (W6 freeze)" appendix + create git tag

**Files:**
- Modify: `documents/specs/2026-05-15-sid-retrieval-design.md` (append)
- Git: create annotated tag `sid-v1-frozen`

- [ ] **Step 1: Append the W6 freeze appendix to the spec**

At the END of `documents/specs/2026-05-15-sid-retrieval-design.md`, append:

```markdown
---

## Appendix W6 — What Shipped (freeze 2026-MM-DD)

### Frozen config + result

- **Winner**: <FILL from project_sid_retrieval_v1_result.md>
- **Blind-A composite**: <FILL> (vs pre-SID champion 0.21)
- **Blind-A nDCG@20**: <FILL> (vs pre-SID champion 0.06)

### Deviations from spec v2.1

#### §2.4 gate 1 (relative MSE)
- **Planned**: RQ-VAE MSE ≤ 1.5× PCA-256 baseline.
- **Shipped**: absolute MSE ≤ 0.02. PCA-relative threshold was geometrically impossible.

#### §2.4 gate 3 (cluster purity)
- **Planned**: set-intersection of `tag_list` across 100 sampled buckets.
- **Shipped**: dominant-tag fraction ≥ 0.20.

#### §3.2 training data composition
- **Planned**: 290K pairs (raw 8K + metadata 47K + doc2query 235K).
- **Shipped**: 168K pairs (raw 121K + metadata 47K + doc2query 0). Doc2query disabled in v1 after audit found prompt + parser bugs.

#### §3.5 W3 gate
- **Planned**: paired-bootstrap CI excluding 0 vs Phase 0.
- **Shipped**: point-estimate gate (`mean_ndcg ≥ 0.12`). Session/turn keys between W2 val and Phase 0 didn't align; CI deferred to W4.

#### §4.1 train/inference parity (NOT in original spec — discovered during sprint)
- W3 trained on `format_query_for_sid_input` outputs; W4 inference initially passed bare role-prefixed transcripts. Fixed in commit `658f07b` via `batch_context` plumbing through the retriever interface.

#### §5.1 W5 parameterization (NOT in original spec — added during sprint)
- Factory now accepts `sid_hub_repo` + `sid_stream_weight` via new `extra_config` YAML field. Forward-compatible for v2 SID models.

### Lessons learned (8 items)

1. Session-level data leakage is easy to miss. W2's initial row-level stratified split put 8 turns of every session across train+val. Fixed in `dfee6da`.
2. Val structure must mirror the eval set. Blind-A is 80×1; W2's initial multi-turn val was wrong shape. Fixed in `d55a206`.
3. Training/inference format parity for sequence-to-sequence retrievers. Fixed in `658f07b`.
4. `load_best_model_at_end` works with PEFT + extended vocab — confirmed in W3 review.
5. `safe_serialization` was removed from `push_to_hub` in newer transformers.
6. OmegaConf-based YAML overrides via `extra_config` beat duplicating YAML files for parameter sweeps.
7. Per-bucket cap with spillover (spec §2.5) is essential when the W1 codebook is coarse.
8. Real prediction-file key name is `predicted_track_ids` (not `predicted_items` / `tracks`). Caught in W4 second-round review (`8a0c6fc`).

### Total commits

<FILL: from `git log --oneline --grep='sid w[1-6]' | wc -l`> commits on `fresh-model`, between W1 commit `8413f8a` and W6 freeze tag `sid-v1-frozen`.
```

- [ ] **Step 2: Verify spec appended correctly**

```bash
grep -c "^## Appendix W6" documents/specs/2026-05-15-sid-retrieval-design.md
grep -c "FILL" documents/specs/2026-05-15-sid-retrieval-design.md
```

Expected: appendix appears once, no `FILL` placeholders left.

- [ ] **Step 3: Commit the spec amendment**

```bash
git add documents/specs/2026-05-15-sid-retrieval-design.md
git commit -m "sid w6: amend spec with W6 freeze appendix (deviations + frozen config + lessons)"
```

- [ ] **Step 4: Create git tag**

```bash
git tag -a sid-v1-frozen -m "SID retriever v1 freeze — Blind-A composite <FILL>. See project_sid_retrieval_v1_result.md memory."
git push origin sid-v1-frozen fresh-model
```

---

## Task 8: (CONDITIONAL — only if v1 ships; stretch) Blind-B prep

ONLY if Task 6 verdict was SHIP or SHIP-WITH-CAVEAT. If postmortem, skip this task and instead write `project_sid_retrieval_v1_postmortem.md` (template inlined below).

**Files (success branch):**
- Create: `music-crs-baselines/config/180-wrrf-sid-v5kto-blindsetB.yaml`
- Create: `colab/67_run_blindset_sid_B.ipynb`

**Files (postmortem branch):**
- Create: `~/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_sid_retrieval_v1_postmortem.md`

### Success branch (Blind-B prep)

- [ ] **Step 1: Create Blind-B config by copying the winner + replacing test_dataset_name**

```bash
WINNER=$(python <<'EOF'
import sys
sys.path.insert(0, '.')
from scripts.blind_a_score_tracker import read_tracker
from pathlib import Path
rows = read_tracker(Path('/Users/orrimoch/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_blind_a_submissions.md'))
sid_rows = [r for r in rows if 'sid' in r['config_id'].lower()]
winner = max(sid_rows, key=lambda r: float(r['composite']))
print(winner['config_id'])
EOF
)
echo "Winner: $WINNER"
python -c "
from pathlib import Path
src = Path(f'music-crs-baselines/config/{\"$WINNER\"}.yaml').read_text()
out = src.replace(
    'test_dataset_name: \"talkpl-ai/TalkPlayData-Challenge-Blind-A\"',
    'test_dataset_name: \"talkpl-ai/TalkPlayData-Challenge-Blind-B\"',
)
Path('music-crs-baselines/config/180-wrrf-sid-v5kto-blindsetB.yaml').write_text(out)
print('wrote 180-wrrf-sid-v5kto-blindsetB.yaml')
"
```

- [ ] **Step 2: Verify the diff is small + commit**

```bash
diff music-crs-baselines/config/$WINNER.yaml music-crs-baselines/config/180-wrrf-sid-v5kto-blindsetB.yaml
git add music-crs-baselines/config/180-wrrf-sid-v5kto-blindsetB.yaml
git commit -m "sid w6: config 180 — Blind-B variant of frozen winner"
```

- [ ] **Step 3: Generate notebook 67 (mirrors notebook 63)** — use the same Python heredoc pattern as notebook 63 but with `--eval_dataset blindset_B` and config 180. Commit.

### Postmortem branch (v1 did NOT ship)

- [ ] **Step 1: Write the postmortem**

Create `~/.claude/projects/.../memory/project_sid_retrieval_v1_postmortem.md`:

```markdown
---
name: SID retriever v1 — POSTMORTEM
description: Why v1 didn't ship. Root causes + v2 plan.
type: project
---

**Status**: v1 DID NOT SHIP. Best Blind-A composite = <FILL> vs pre-SID champion 0.21.

## What we tried (per project_blind_a_submissions.md tracker)

<paste the tracker table>

## Root cause (best guess based on the score pattern)

<one of these:>
- **All 3 SID configs regressed**: SID generator is producing worse top-K than BM25+dense. Likely W1 SID coarseness (3K SIDs from 47K tracks).
- **Ensemble bad, pure-SID worse**: SID's contribution is net negative; weight should have been lower (or zero).
- **Ensemble bad, pure-SID OK**: ensemble fusion is the problem; the wRRF mixing isn't right.

## V2 plan

<scope per the v2 candidates in this plan's Task 6 template>

## What v1 leaves for v2 (still useful)

- All factory parameterization (`extra_config`, `sid_stream_weight`, `sid_hub_repo`) — reusable.
- `scripts/compare_blind_predictions.py`, `scripts/precheck_prediction.py`, `scripts/blind_a_score_tracker.py` — reusable.
- All 121 tests — green-light v2 from the start.
- W1 quantizer + W2 training data infrastructure — rebuild only the artifacts, not the code.
```

- [ ] **Step 2: No git tag** — leave HEAD untagged.

---

## Self-review checklist (per writing-plans skill)

**1. Spec coverage** (against §5.1 W5+W6 rows + §5.3):

- §5.1 W5 sweep — already shipped in W5 plan (`2026-05-16-sid-weight-sweep-w5.md`)
- §5.1 W6 freeze decision — Task 6 ✓
- §5.1 ≥ 2 Blind-A submissions — W4 + W5 already shipped 3 ✓
- §5.3 spec amendment — Task 7 ✓
- §5.3 memory file — Task 6 ✓
- §5.3 MEMORY.md update — covered by Task 6 file path
- Blind-B prep — Task 8 (stretch) ✓
- **Added beyond spec**: precheck (Task 1), score tracker (Task 2), interpretation guide (Task 3), SOP (Task 4) — these address gaps in the original spec around "how do we actually run the Blind-A iteration loop".

**2. Placeholder scan**: `<FILL>` and `<PASTE>` appear inside the TEMPLATE strings for memory files — they get filled at W6-execution time from real CodaBench scores. Not plan failures; they're explicitly marked as "fill at execution" placeholders.

**3. Type consistency**:
- Score tracker schema: `config_id, composite, ndcg@20, llm, lex_div, submitted_at, url, notes` — used identically in Tasks 2, 4, 6, 8
- Composite formula: `0.5*nDCG + 0.1*CatDiv + 0.1*LexDiv + 0.3*LLM` — used identically in Tasks 3, 6, 7
- Pre-SID champion baseline (composite 0.21, nDCG 0.06) — used in Tasks 3, 6, 7

**4. Sequencing**:
- Tasks 1-2 build tooling (precheck + tracker) — independent of Blind-A scores
- Task 3 writes the interpretation guide — also independent of scores
- Task 4 writes the SOP — references Tasks 1 + 2
- Task 5 wires tooling into existing notebooks
- **Task 6 requires real Blind-A scores in the tracker** — gates on user having uploaded all 3 SID submissions
- Task 7 finalizes (spec amend + tag) — depends on Task 6's winner being decided
- Task 8 forks: Blind-B prep (success) or postmortem (failure)

**Plan complete.**

---

## Estimated wallclock

| Component | Time |
|---|---|
| Task 1 (precheck script + tests) | ~15 min via subagent |
| Task 2 (tracker script + tests) | ~10 min via subagent |
| Task 3 (interpretation guide memory) | ~20 min writing |
| Task 4 (SOP doc) | ~15 min writing |
| Task 5 (wire precheck into notebooks) | ~5 min via subagent |
| Task 6 (result memory) | ~30 min writing AFTER scores land |
| Task 7 (spec amend + tag) | ~15 min |
| Task 8 (Blind-B prep OR postmortem) | ~20 min |
| **Total coordinator time** | **~2 hr** |
| User CodaBench upload time | ~5 min per submission × 3 |
| User wait for CodaBench scores | hours-days |

W6 is the lightest week in terms of code (~300 lines new) but the most important in terms of decision-quality. The tooling (precheck + tracker + interpretation guide) is what lets us iterate Blind-A submissions intelligently rather than just submitting blindly.
