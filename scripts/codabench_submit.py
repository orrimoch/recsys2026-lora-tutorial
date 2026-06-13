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
from typing import Any, Callable

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
    for i in range(max_tries):
        if i > 0:
            sleep_fn(wait_s)
        status = get_status(submission_id)
        state = str(status.get("status", "")).lower()
        if state in _FINISHED:
            return status
        if state in _FAILED:
            raise RuntimeError(f"submission {submission_id} failed: {status}")
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


def poll_submission(submission_id: str, **kwargs: Any) -> dict:
    """Poll the live CodaBench status endpoint. Wires _default_status_getter (PENDING DISCOVERY)."""
    return poll_until_scored(_default_status_getter, submission_id, **kwargs)
