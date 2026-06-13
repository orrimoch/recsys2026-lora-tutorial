"""Budget-gated submission preparer for the autonomous research loop.

Reuses the existing validators/packager in validate_prediction.py. Given a prediction.json
it: (1) schema-validates, (2) checks the weekly blind-submission budget, (3) packages the
CodaBench zip, and (4) returns a SubmissionPlan. It does NOT upload — uploading is the job
of codabench_submit.py (or a human one-click). Keeping prepare/upload separate means the
budget gate is enforced regardless of how the final upload happens.
"""
from __future__ import annotations

import json
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
)

_DEFAULT_LOG = REPO_ROOT / "documents" / "submissions_log.md"


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
    log = Path(submission_log) if submission_log else _DEFAULT_LOG

    try:
        preds = load_prediction(prediction_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return SubmissionPlan(False, f"could not load prediction: {exc}", None, "")
    errors = validate_schema(preds, split)
    if errors:
        return SubmissionPlan(False, f"schema invalid: {list(errors)[:3]}", None, "not checked (schema failed)")

    ok, budget_msg = check_budget(log, weekly_cap=weekly_cap)
    if not ok:
        return SubmissionPlan(False, budget_msg, None, budget_msg)

    zip_out = package_zip(prediction_path, output_zip)
    return SubmissionPlan(True, "ready", str(zip_out), budget_msg)
