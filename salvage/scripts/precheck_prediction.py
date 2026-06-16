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


REQUIRED_FIELDS = ["session_id", "user_id", "turn_number", "predicted_track_ids",
                   "predicted_response"]

# Placeholder responses that must NEVER ship: the retrieval-only stub ("ok",
# crs_baseline.py / run_inference_devset_retrieval_only.py) and empty strings.
# A leaked stub scores ~1/5 on the LLM axis (the EXP-016 / config-209 regression).
STUB_RESPONSES = {"ok", "n/a", "na", "none", "null", "todo", "tbd"}
MIN_RESPONSE_LEN = 5


def precheck(pred_path: Path, catalog: set[str], expected_n: int = 80,
             min_response_len: int = MIN_RESPONSE_LEN,
             stub_responses: set[str] | None = None) -> dict:
    """Validate a Blind-A prediction.json.

    Returns:
        {"ok": bool, "errors": list[str], "warnings": list[str], "n_records": int}
    """
    errors: list[str] = []
    warnings: list[str] = []
    stub_responses = STUB_RESPONSES if stub_responses is None else stub_responses

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
        # predicted_response content: a leaked retrieval-only stub ("ok") or an
        # empty/too-short response craters the LLM axis (config-209 regression).
        resp = rec.get("predicted_response")
        if not isinstance(resp, str):
            errors.append(f"record {i} (session={rec.get('session_id', '?')}): "
                          f"predicted_response must be a string, got {type(resp).__name__}")
        else:
            stripped = resp.strip()
            if not stripped:
                errors.append(f"record {i} (session={rec.get('session_id', '?')}): "
                              f"predicted_response is empty")
            elif stripped.lower() in stub_responses:
                errors.append(f"record {i} (session={rec.get('session_id', '?')}): "
                              f"predicted_response is a placeholder stub ({stripped!r}) — "
                              f"the responder did not run/fill this row")
            elif len(stripped) < min_response_len:
                errors.append(f"record {i} (session={rec.get('session_id', '?')}): "
                              f"predicted_response too short ({len(stripped)} chars < {min_response_len}): {stripped!r}")

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
