"""Validator + packager for CodaBench prediction.json submissions (W0-5 / W0-9 / E-3)."""

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path


REQUIRED_FIELDS = (
    "session_id",
    "user_id",
    "turn_number",
    "predicted_track_ids",
    "predicted_response",
)

# Expected row counts per split (plan §2.6 / §4).
# Deep-review P1-8: blindA count is hardcoded; if the dataset host updates
# size, validation rejects the submission. Override via env
# `RECSYS_EXPECTED_BLINDA_ROWS=<n>` for ad-hoc cases. blindB is intentionally
# absent (open count > 0 enforced below).
import os as _os
EXPECTED_ROWS = {
    "dev": 8000,      # 1000 sessions x 8 turns
    "blindA": int(_os.environ.get("RECSYS_EXPECTED_BLINDA_ROWS", 80)),
}
EXPECTED_TURNS_PER_SESSION = {
    "dev": 8,
    "blindA": 8,
    "blindB": 8,
}

# Key sets for the dual-axis attribution check (§2.6).
RETRIEVAL_KEYS = {
    "retrieval_type",
    "corpus_types",
    "track_split_types",
    "rerank_type",
    "top_k",
    "embedding_model",
    "query_rewriter",
}
RESPONSE_KEYS = {
    "lm_type",
    "lm_model",
    "system_prompt",
    "prompt_template",
    "response_style",
    "temperature",
    "max_new_tokens",
}


# ---------------------------------------------------------------------------
# Loading + schema validation
# ---------------------------------------------------------------------------

def load_prediction(path):
    """Parse the prediction JSON file at ``path`` and return a list of row dicts."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {p}, got {type(data).__name__}")
    return data


def validate_schema(predictions, split):
    """Validate a parsed prediction list. Returns a list of error strings; empty means valid."""
    errors = []

    if split not in ("dev", "blindA", "blindB"):
        errors.append(f"unknown split: {split!r}")
        return errors

    if not isinstance(predictions, list):
        errors.append(f"predictions must be a list, got {type(predictions).__name__}")
        return errors

    if split in EXPECTED_ROWS:
        expected = EXPECTED_ROWS[split]
        if len(predictions) != expected:
            errors.append(
                f"row count mismatch for split={split}: expected {expected}, got {len(predictions)}"
            )
    else:
        # blindB: accept any size > 0 for now.
        if len(predictions) == 0:
            errors.append(f"empty prediction list for split={split}")

    # Per-row validation + per-session turn bookkeeping.
    sessions_to_turns = {}  # session_id -> set(turn_numbers)
    sessions_to_user = {}   # session_id -> user_id (for consistency)
    for idx, row in enumerate(predictions):
        if not isinstance(row, dict):
            errors.append(f"row[{idx}] is not a dict (got {type(row).__name__})")
            continue

        for field in REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"row[{idx}] missing required field: {field}")

        if any(f not in row for f in REQUIRED_FIELDS):
            continue

        sid = row["session_id"]
        uid = row["user_id"]
        turn = row["turn_number"]
        tids = row["predicted_track_ids"]
        resp = row["predicted_response"]

        if not isinstance(sid, str) or not sid:
            errors.append(f"row[{idx}] session_id must be a non-empty string")
        if not isinstance(uid, str) or not uid:
            errors.append(f"row[{idx}] user_id must be a non-empty string")
        if not isinstance(turn, int) or isinstance(turn, bool):
            errors.append(f"row[{idx}] turn_number must be int (got {type(turn).__name__})")

        if not isinstance(tids, list):
            errors.append(f"row[{idx}] predicted_track_ids must be a list")
        else:
            if len(tids) < 1 or len(tids) > 20:
                errors.append(
                    f"row[{idx}] predicted_track_ids length {len(tids)} out of range [1,20]"
                )
            if not all(isinstance(t, str) for t in tids):
                errors.append(f"row[{idx}] predicted_track_ids must all be strings")
            elif len(set(tids)) != len(tids):
                errors.append(f"row[{idx}] predicted_track_ids contains duplicates")

        if not isinstance(resp, str):
            errors.append(
                f"row[{idx}] predicted_response must be a string (got {type(resp).__name__})"
            )
        # Empty response permitted: Random/Popularity baselines submit "".

        if isinstance(sid, str) and isinstance(turn, int) and not isinstance(turn, bool):
            sessions_to_turns.setdefault(sid, set()).add(turn)
            prior_uid = sessions_to_user.get(sid)
            if prior_uid is not None and prior_uid != uid:
                errors.append(
                    f"row[{idx}] session_id={sid!r} has inconsistent user_id "
                    f"({prior_uid!r} vs {uid!r})"
                )
            else:
                sessions_to_user[sid] = uid

    # For splits with a fixed turn layout, every session must have turns 1..N.
    n_turns = EXPECTED_TURNS_PER_SESSION.get(split)
    if n_turns is not None and split in ("dev", "blindA"):
        expected_turns = set(range(1, n_turns + 1))
        for sid, turns in sessions_to_turns.items():
            missing = expected_turns - turns
            extra = turns - expected_turns
            if missing:
                errors.append(
                    f"session {sid!r} missing turns: {sorted(missing)}"
                )
            if extra:
                errors.append(
                    f"session {sid!r} has unexpected turns: {sorted(extra)}"
                )

    return errors


# ---------------------------------------------------------------------------
# Submission budget enforcement (§2.6 / W0-9)
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def check_budget(submission_log_path, weekly_cap=3):
    """Return (ok, message). Counts blindA/blindB rows in the last 7 days in the log."""
    log_path = Path(submission_log_path)
    if not log_path.exists():
        return True, "no log found"

    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        return True, f"could not read log ({exc}); skipping budget check"

    today = datetime.now().date()
    cutoff = today - timedelta(days=7)
    count = 0
    for line in text.splitlines():
        if "[blindA]" not in line and "[blindB]" not in line:
            continue
        # Parse the first YYYY-MM-DD occurrence on the line as the submission date.
        m = _DATE_RE.search(line)
        if not m:
            continue
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if cutoff <= d <= today:
            count += 1

    if count >= weekly_cap:
        return (
            False,
            f"weekly blind submission cap exceeded: {count} in last 7 days "
            f"(cap={weekly_cap})",
        )
    return True, f"{count}/{weekly_cap} blind submissions in last 7 days"


# ---------------------------------------------------------------------------
# Attribution warning (§2.6 / W0-9)
# ---------------------------------------------------------------------------

def _load_yaml_like(path):
    """Parse a YAML or JSON file into a dict. Falls back to a minimal key:value parser."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    # Try YAML first (pyyaml is installed in the venv); fall back to JSON, then minimal parser.
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    # Minimal fallback: "key: value" lines, ignoring comments and blanks.
    result = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        result[key.strip()] = val.strip()
    return result


def check_attribution_warning(config_path, last_blind_config_path):
    """Warn if BOTH retrieval-branch AND response-branch keys changed vs the previous blind."""
    if config_path is None or last_blind_config_path is None:
        return None
    cur_p = Path(config_path)
    prev_p = Path(last_blind_config_path)
    if not cur_p.exists() or not prev_p.exists():
        return None

    cur = _load_yaml_like(cur_p)
    prev = _load_yaml_like(prev_p)

    retrieval_changed = any(cur.get(k) != prev.get(k) for k in RETRIEVAL_KEYS)
    response_changed = any(cur.get(k) != prev.get(k) for k in RESPONSE_KEYS)

    if retrieval_changed and response_changed:
        return (
            "attribution warning: both retrieval-branch and response-branch keys "
            "changed vs previous blind submission; ΔLLM-term will not be attributable "
            "(plan §2.6)"
        )
    return None


# ---------------------------------------------------------------------------
# Packaging (§1.2 / E-3)
# ---------------------------------------------------------------------------

def package_zip(prediction_path, output_zip_path):
    """Zip the prediction file so the archive contains exactly one root entry: ``prediction.json``."""
    src = Path(prediction_path)
    out = Path(output_zip_path)
    if not src.exists():
        raise FileNotFoundError(f"prediction file not found: {src}")

    out.parent.mkdir(parents=True, exist_ok=True)

    # Re-serialise with ensure_ascii=False to satisfy plan §1.2 and guarantee the archive
    # holds a clean UTF-8 prediction.json regardless of the source file's encoding.
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_submission_log():
    return Path(__file__).resolve().parent.parent / "documents" / "submissions_log.md"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate a prediction.json file and optionally package it for CodaBench."
    )
    parser.add_argument("--input", required=True, help="Path to prediction.json")
    parser.add_argument(
        "--split",
        required=True,
        choices=("dev", "blindA", "blindB"),
        help="Which split this prediction targets",
    )
    parser.add_argument("--package", help="If set, write a CodaBench-ready zip to this path")
    parser.add_argument(
        "--check-budget",
        action="store_true",
        help="Refuse packaging if the weekly blind cap would be exceeded",
    )
    parser.add_argument(
        "--attribution-check-vs",
        help="Path to the previous blind config YAML/JSON to compare against",
    )
    parser.add_argument(
        "--config",
        help="Path to the current experiment config (required for --attribution-check-vs)",
    )
    parser.add_argument(
        "--submissions-log",
        default=str(_default_submission_log()),
        help="Path to documents/submissions_log.md",
    )
    parser.add_argument(
        "--weekly-cap",
        type=int,
        default=3,
        help="Weekly blind submission cap (default 3, plan §2.6)",
    )

    args = parser.parse_args(argv)

    failed = False

    # 1. Schema validation.
    try:
        predictions = load_prediction(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not load prediction: {exc}", file=sys.stderr)
        return 1

    errors = validate_schema(predictions, args.split)
    if errors:
        print(f"ERROR: schema validation failed ({len(errors)} issue(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        failed = True
    else:
        print(f"schema OK ({len(predictions)} rows, split={args.split})")

    # 2. Budget check (only meaningful for blind splits).
    if args.check_budget:
        ok, msg = check_budget(args.submissions_log, weekly_cap=args.weekly_cap)
        if ok:
            print(f"budget OK: {msg}")
        else:
            print(f"ERROR: budget check failed: {msg}", file=sys.stderr)
            failed = True

    # 3. Attribution warning (non-fatal).
    if args.attribution_check_vs and args.config:
        warn = check_attribution_warning(args.config, args.attribution_check_vs)
        if warn:
            print(f"WARNING: {warn}", file=sys.stderr)

    # 4. Packaging (skipped on failure).
    if args.package:
        if failed:
            print("ERROR: refusing to package due to earlier failures", file=sys.stderr)
            return 1
        out = package_zip(args.input, args.package)
        print(f"wrote {out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
