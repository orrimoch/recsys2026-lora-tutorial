"""Tests for the prediction.json schema/shape validator (W0-5 / W0-9)."""

import json
import sys
import zipfile
from pathlib import Path

import pytest

# Make scripts/ importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_prediction as vp  # noqa: E402


# ---------------------------------------------------------------------------
# Local helpers (mirror conftest patterns for a blindA-sized fixture).
# ---------------------------------------------------------------------------

def _tid(prefix: str, i: int) -> str:
    base = f"{prefix}{i:04d}"
    pad = "0" * (32 - len(base))
    hexs = (base + pad)[:32]
    return f"{hexs[0:8]}-{hexs[8:12]}-{hexs[12:16]}-{hexs[16:20]}-{hexs[20:32]}"


def _make_session(session_id, user_id, n_turns=8, n_tracks=20, tid_prefix="a", response="ok"):
    rows = []
    for t in range(1, n_turns + 1):
        rows.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": t,
            "predicted_track_ids": [_tid(f"{tid_prefix}{t}", i) for i in range(n_tracks)],
            "predicted_response": response,
        })
    return rows


@pytest.fixture
def blinda_valid():
    """10 sessions x 8 turns = 80 rows — the exact blindA shape."""
    preds = []
    for s in range(10):
        preds.extend(_make_session(f"sess-{s:04d}", f"user-{s:04d}", tid_prefix=chr(ord("a") + s)))
    return preds


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_valid_prediction_passes(blinda_valid):
    errors = vp.validate_schema(blinda_valid, "blindA")
    assert errors == [], f"expected no errors, got: {errors}"


def test_duplicate_tids_rejected(sample_prediction_duplicate_tids):
    # 1 session x 8 turns = 8 rows — use blindB (size > 0 accepted).
    errors = vp.validate_schema(sample_prediction_duplicate_tids, "blindB")
    assert any("duplicate" in e.lower() for e in errors), f"errors: {errors}"


def test_missing_turn_rejected(sample_prediction_missing_turn):
    # 7 rows for one session (turn 5 missing). Exercise the per-session turn check
    # via the blindA branch (which also enforces turns 1..8).
    errors = vp.validate_schema(sample_prediction_missing_turn, "blindA")
    assert any("missing turns" in e for e in errors), f"errors: {errors}"


def test_non_ascii_in_response_accepted(sample_prediction_non_ascii):
    errors = vp.validate_schema(sample_prediction_non_ascii, "blindB")
    # blindB accepts any size > 0; non-ASCII in predicted_response must not raise.
    assert errors == [], f"expected no errors, got: {errors}"


def test_empty_response_allowed(blinda_valid):
    for row in blinda_valid:
        row["predicted_response"] = ""
    errors = vp.validate_schema(blinda_valid, "blindA")
    assert errors == [], f"Random/Popularity baselines must be allowed empty responses: {errors}"


def test_row_count_mismatch_dev(sample_prediction_valid):
    # 2 sessions x 8 = 16 rows; dev expects 8000.
    errors = vp.validate_schema(sample_prediction_valid, "dev")
    assert any("row count mismatch" in e for e in errors), f"errors: {errors}"


# ---------------------------------------------------------------------------
# Packaging tests (E-3)
# ---------------------------------------------------------------------------

def test_zip_package_contains_prediction_json_at_root(tmp_path, blinda_valid):
    src = tmp_path / "prediction.json"
    with src.open("w", encoding="utf-8") as f:
        json.dump(blinda_valid, f, ensure_ascii=False)

    out_zip = tmp_path / "submission.zip"
    result = vp.package_zip(src, out_zip)

    assert result == out_zip
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    assert names == ["prediction.json"], f"zip must contain exactly prediction.json at root, got {names}"


def test_zip_package_roundtrip_preserves_rows(tmp_path, blinda_valid):
    src = tmp_path / "prediction.json"
    with src.open("w", encoding="utf-8") as f:
        json.dump(blinda_valid, f, ensure_ascii=False)
    out_zip = tmp_path / "submission.zip"
    vp.package_zip(src, out_zip)

    with zipfile.ZipFile(out_zip) as zf:
        with zf.open("prediction.json") as f:
            loaded = json.loads(f.read().decode("utf-8"))
    assert len(loaded) == len(blinda_valid)


# ---------------------------------------------------------------------------
# Budget check tests (W0-9)
# ---------------------------------------------------------------------------

def _write_log(path, entries):
    """Write a fake submissions_log.md containing the given (tag, date) tuples."""
    lines = [
        "# Submissions log",
        "",
        "| exp_id | tag | date | note |",
        "|---|---|---|---|",
    ]
    for i, (tag, date) in enumerate(entries):
        lines.append(f"| exp-{i} | {tag} | {date} | stub |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_budget_check_under_cap(tmp_path):
    from datetime import datetime, timedelta
    today = datetime.now().date()
    log = tmp_path / "submissions_log.md"
    _write_log(log, [
        ("[blindA]", (today - timedelta(days=1)).strftime("%Y-%m-%d")),
        ("[blindA]", (today - timedelta(days=3)).strftime("%Y-%m-%d")),
    ])
    ok, msg = vp.check_budget(log, weekly_cap=3)
    assert ok is True, msg
    assert "2/3" in msg or "2 " in msg


def test_budget_check_over_cap(tmp_path):
    from datetime import datetime, timedelta
    today = datetime.now().date()
    log = tmp_path / "submissions_log.md"
    _write_log(log, [
        ("[blindA]", (today - timedelta(days=0)).strftime("%Y-%m-%d")),
        ("[blindA]", (today - timedelta(days=2)).strftime("%Y-%m-%d")),
        ("[blindB]", (today - timedelta(days=4)).strftime("%Y-%m-%d")),
    ])
    ok, msg = vp.check_budget(log, weekly_cap=3)
    assert ok is False, msg
    assert "exceed" in msg.lower() or "cap" in msg.lower()


def test_budget_check_missing_log_ok(tmp_path):
    log = tmp_path / "does_not_exist.md"
    ok, msg = vp.check_budget(log, weekly_cap=3)
    assert ok is True
    assert "no log" in msg.lower()


def test_budget_check_ignores_old_entries(tmp_path):
    from datetime import datetime, timedelta
    today = datetime.now().date()
    log = tmp_path / "submissions_log.md"
    _write_log(log, [
        ("[blindA]", (today - timedelta(days=20)).strftime("%Y-%m-%d")),
        ("[blindA]", (today - timedelta(days=30)).strftime("%Y-%m-%d")),
        ("[blindA]", (today - timedelta(days=40)).strftime("%Y-%m-%d")),
    ])
    ok, msg = vp.check_budget(log, weekly_cap=3)
    assert ok is True, msg


def test_budget_check_ignores_dev_local(tmp_path):
    from datetime import datetime, timedelta
    today = datetime.now().date()
    log = tmp_path / "submissions_log.md"
    _write_log(log, [
        ("[dev-local]", today.strftime("%Y-%m-%d")),
        ("[dev-local]", today.strftime("%Y-%m-%d")),
        ("[dev-local]", today.strftime("%Y-%m-%d")),
        ("[dev-local]", today.strftime("%Y-%m-%d")),
    ])
    ok, msg = vp.check_budget(log, weekly_cap=3)
    assert ok is True, msg


# ---------------------------------------------------------------------------
# Attribution warning tests (W0-9)
# ---------------------------------------------------------------------------

def _write_yaml(path, data):
    lines = []
    for k, v in data.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(repr(x) for x in v)}]")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_attribution_warning_triggers_on_dual_change(tmp_path):
    prev = tmp_path / "prev.yaml"
    cur = tmp_path / "cur.yaml"
    _write_yaml(prev, {
        "retrieval_type": "bm25",
        "corpus_types": "tracks",
        "lm_type": "llama1b",
        "system_prompt": "You are a DJ.",
        "temperature": 0.7,
    })
    _write_yaml(cur, {
        "retrieval_type": "bge-m3",        # retrieval axis changed
        "corpus_types": "tracks",
        "lm_type": "qwen7b",               # response axis changed
        "system_prompt": "You are a music expert.",
        "temperature": 0.3,
    })
    warn = vp.check_attribution_warning(cur, prev)
    assert warn is not None
    assert "attribution" in warn.lower()


def test_attribution_warning_none_on_single_axis_change(tmp_path):
    prev = tmp_path / "prev.yaml"
    cur = tmp_path / "cur.yaml"
    _write_yaml(prev, {
        "retrieval_type": "bm25",
        "corpus_types": "tracks",
        "lm_type": "llama1b",
        "system_prompt": "You are a DJ.",
        "temperature": 0.7,
    })
    _write_yaml(cur, {
        "retrieval_type": "bge-m3",        # only retrieval axis changed
        "corpus_types": "tracks",
        "lm_type": "llama1b",
        "system_prompt": "You are a DJ.",
        "temperature": 0.7,
    })
    warn = vp.check_attribution_warning(cur, prev)
    assert warn is None


def test_attribution_warning_none_when_identical(tmp_path):
    prev = tmp_path / "prev.yaml"
    cur = tmp_path / "cur.yaml"
    payload = {
        "retrieval_type": "bm25",
        "lm_type": "llama1b",
        "system_prompt": "You are a DJ.",
    }
    _write_yaml(prev, payload)
    _write_yaml(cur, payload)
    assert vp.check_attribution_warning(cur, prev) is None


def test_attribution_warning_missing_file_returns_none(tmp_path):
    prev = tmp_path / "does_not_exist.yaml"
    cur = tmp_path / "also_missing.yaml"
    assert vp.check_attribution_warning(cur, prev) is None
