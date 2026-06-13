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
