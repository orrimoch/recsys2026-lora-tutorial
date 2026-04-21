"""Wrap a predictions JSON into the exact zip layout CodaBench expects.

The server unzips into `/app/input/res/` and reads `prediction.json` (singular,
at the zip root). Any other layout/filename = FileNotFoundError on the leader-
board side.

CLI:
    python make_prediction_zip.py \
        --input ./output/predictions.json \
        --output ./output/prediction.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile


REQUIRED_FIELDS = {"session_id", "user_id", "turn_number",
                   "predicted_track_ids", "predicted_response"}


def validate(data: list[dict]) -> None:
    if not isinstance(data, list):
        raise ValueError("predictions JSON must be a list")
    if len(data) != 80:
        raise ValueError(f"expected 80 rows for Blind-A, got {len(data)}")
    seen_keys = set()
    for i, row in enumerate(data):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"row {i}: missing fields {missing}")
        if not isinstance(row["turn_number"], int):
            raise ValueError(f"row {i}: turn_number must be int")
        tids = row["predicted_track_ids"]
        if len(tids) != 20:
            raise ValueError(f"row {i}: predicted_track_ids must have 20 entries (got {len(tids)})")
        if len(set(tids)) != 20:
            raise ValueError(f"row {i}: predicted_track_ids must be 20 DISTINCT track ids")
        if not isinstance(row["predicted_response"], str) or not row["predicted_response"].strip():
            raise ValueError(f"row {i}: predicted_response must be a non-empty string")
        key = (row["session_id"], row["turn_number"])
        if key in seen_keys:
            raise ValueError(f"duplicate (session_id, turn_number): {key}")
        seen_keys.add(key)
    print(f"[zip] validation passed: 80 rows, all required fields present", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to predictions JSON.")
    parser.add_argument("--output", required=True, help="Output zip path.")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    validate(data)

    # Re-write into a tempdir as `prediction.json`, then zip.
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    with tempfile.TemporaryDirectory() as td:
        json_path = os.path.join(td, "prediction.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname="prediction.json")
    print(f"[zip] wrote {out_path}", file=sys.stderr)
    with zipfile.ZipFile(out_path) as zf:
        for n in zf.namelist():
            print(f"  - {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
