"""Tests for scripts/run_experiment.py — config validation, smoke dispatch, git sha, log skeleton."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_EXPERIMENT_PATH = REPO_ROOT / "scripts" / "run_experiment.py"


def _import_run_experiment():
    spec = importlib.util.spec_from_file_location("_run_experiment", str(RUN_EXPERIMENT_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_run_experiment"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rx():
    return _import_run_experiment()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_validation_rejects_bad_split_type(tmp_path, rx):
    """A config with track_split_types != ['all_tracks'] must hard-fail."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "lm_type: test\n"
        "retrieval_type: bm25\n"
        "track_split_types:\n  - test_tracks\n"
        "user_split_types:\n  - all_users\n"
    )
    with pytest.raises(SystemExit):
        rx.validate_config(bad)


def test_config_validation_accepts_all_tracks(tmp_path, rx):
    good = tmp_path / "good.yaml"
    good.write_text(
        "lm_type: test\n"
        "retrieval_type: bm25\n"
        "track_split_types:\n  - all_tracks\n"
        "user_split_types:\n  - all_users\n"
    )
    cfg = rx.validate_config(good)
    assert cfg["track_split_types"] == ["all_tracks"]


# ---------------------------------------------------------------------------
# Smoke-mode invocation
# ---------------------------------------------------------------------------

def test_smoke_mode_invocation_dev(rx):
    """Smoke on dev should dispatch run_inference_devset module and pass the tid."""
    cmd = rx.build_inference_cmd(split="dev", tid="009-wrrf-bm25-dense-v1", batch_size=16, smoke=True)
    # smoke mode -> python -c "<wrapper src>"
    assert cmd[0] == sys.executable
    assert cmd[1] == "-c"
    src = cmd[2]
    assert "run_inference_devset" in src
    assert "'tid': '009-wrrf-bm25-dense-v1'" in src
    assert "'batch_size': 16" in src


def test_smoke_mode_invocation_blindA(rx):
    cmd = rx.build_inference_cmd(split="blindA", tid="llama1b_bm25_blindset_A", batch_size=8, smoke=True)
    src = cmd[2]
    assert "run_inference_blindset" in src
    assert "'eval_dataset': 'blindset_A'" in src
    assert "'tid': 'llama1b_bm25_blindset_A'" in src


def test_full_mode_invocation_dev(rx):
    cmd = rx.build_inference_cmd(split="dev", tid="foo", batch_size=16, smoke=False)
    # Full: direct argv call
    assert cmd[0] == sys.executable
    assert cmd[1] == "run_inference_devset.py"
    assert "--tid" in cmd and "foo" in cmd


def test_full_mode_invocation_blindA(rx):
    cmd = rx.build_inference_cmd(split="blindA", tid="bar", batch_size=8, smoke=False)
    assert cmd[1] == "run_inference_blindset.py"
    assert "--eval_dataset" in cmd and "blindset_A" in cmd


# ---------------------------------------------------------------------------
# Git sha helper
# ---------------------------------------------------------------------------

def test_git_sha_fetch(rx):
    sha = rx.git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


# ---------------------------------------------------------------------------
# Narrative skeleton append
# ---------------------------------------------------------------------------

def test_experiment_log_skeleton_append(tmp_path, rx):
    log = tmp_path / "experiments_log.md"
    log.write_text(
        "# Experiments log — narrative\n\nAppend-only. New entries at top.\n"
    )
    original = log.read_text()

    cfg_path = tmp_path / "configs" / "012-my-exp.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text("track_split_types:\n  - all_tracks\n")

    rx.append_experiment_skeleton(
        log_path=log,
        tid="012-my-exp",
        axis="retrieval",
        config_path=cfg_path,
        sha="abcd123",
        smoke=True,
        split="dev",
        metrics=None,
    )

    after = log.read_text()
    assert len(after) > len(original)
    assert "012-my-exp" in after
    assert "- **Axis**: retrieval" in after
    assert "abcd123" in after
    assert "- **Lessons**:" in after


def test_experiment_log_skeleton_creates_file_if_missing(tmp_path, rx):
    log = tmp_path / "fresh_log.md"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("track_split_types:\n  - all_tracks\n")
    rx.append_experiment_skeleton(
        log_path=log,
        tid="t1",
        axis="unknown",
        config_path=cfg_path,
        sha="deadbee",
        smoke=False,
        split="blindA",
        metrics={"composite_retrieval": 0.123},
    )
    assert log.exists()
    contents = log.read_text()
    assert "t1" in contents
    assert "composite_retrieval: 0.123" in contents


# ---------------------------------------------------------------------------
# tid extraction
# ---------------------------------------------------------------------------

def test_tid_from_config(rx):
    assert rx.tid_from_config(Path("/x/y/009-wrrf-bm25-dense-v1.yaml")) == "009-wrrf-bm25-dense-v1"
    assert rx.tid_from_config(Path("llama1b_bm25_devset.yaml")) == "llama1b_bm25_devset"


# ---------------------------------------------------------------------------
# main() with --dry-run — end-to-end no-side-effect path
# ---------------------------------------------------------------------------

def test_dry_run_does_not_run_inference(tmp_path, rx, mocker, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "lm_type: test\n"
        "retrieval_type: bm25\n"
        "track_split_types:\n  - all_tracks\n"
        "user_split_types:\n  - all_users\n"
    )
    spy = mocker.patch.object(rx, "run_inference")
    append_spy = mocker.patch.object(rx, "append_experiment_skeleton")

    rc = rx.main([
        "--config", str(cfg),
        "--smoke",
        "--split", "dev",
        "--dry-run",
    ])
    assert rc == 0
    spy.assert_not_called()
    append_spy.assert_not_called()
    out = capsys.readouterr().out
    assert "dry-run" in out
