"""Tests for the EXP-016/config-209 regression hardening.

Three guards added after a Blind submission regressed (composite 0.50->0.45, the
whole drop in the LLM axis): a leaked retrieval-only "ok" stub shipped, and an
accidental ColBERT retrain overwrote the 0.50 model+index in place.

1. precheck_prediction: reject empty / "ok"-stub / too-short predicted_response.
2. gemini_responder.summarize_run: report + abort (--fail-on-fallback) on any fallback row.
3. {train_colbert,build_colbert_index}.existing_artifact_blocks: skip-if-exists unless --force.
"""
import json

import pytest

from scripts.precheck_prediction import precheck, STUB_RESPONSES
from scripts.gemini_responder import summarize_run


# --- shared helpers -------------------------------------------------------- #

CATALOG = {"t-0001", "t-0002", "t-0003"}


def _rec(response="Here is a great track for your mood.", tracks=("t-0001", "t-0002")):
    return {
        "session_id": "s1", "user_id": "u1", "turn_number": 1,
        "predicted_track_ids": list(tracks), "predicted_response": response,
    }


def _write(tmp_path, records):
    p = tmp_path / "prediction.json"
    p.write_text(json.dumps(records))
    return p


# --- 1) precheck predicted_response validation ----------------------------- #

def test_precheck_passes_valid_response(tmp_path):
    res = precheck(_write(tmp_path, [_rec()]), catalog=CATALOG, expected_n=1)
    assert res["ok"], res["errors"]


@pytest.mark.parametrize("bad", ["ok", "OK", " ok ", "n/a", "TBD"])
def test_precheck_rejects_stub_response(tmp_path, bad):
    res = precheck(_write(tmp_path, [_rec(response=bad)]), catalog=CATALOG, expected_n=1)
    assert not res["ok"]
    assert any("stub" in e for e in res["errors"]), res["errors"]


def test_precheck_rejects_empty_response(tmp_path):
    res = precheck(_write(tmp_path, [_rec(response="   ")]), catalog=CATALOG, expected_n=1)
    assert not res["ok"]
    assert any("empty" in e for e in res["errors"])


def test_precheck_rejects_too_short_response(tmp_path):
    res = precheck(_write(tmp_path, [_rec(response="hi")]), catalog=CATALOG, expected_n=1)
    assert not res["ok"]
    assert any("too short" in e for e in res["errors"])


def test_precheck_rejects_missing_response(tmp_path):
    rec = _rec()
    del rec["predicted_response"]
    res = precheck(_write(tmp_path, [rec]), catalog=CATALOG, expected_n=1)
    assert not res["ok"]
    assert any("predicted_response" in e for e in res["errors"])


def test_precheck_response_is_required_field():
    assert "predicted_response" in __import__(
        "scripts.precheck_prediction", fromlist=["REQUIRED_FIELDS"]).REQUIRED_FIELDS
    assert "ok" in STUB_RESPONSES


# --- 2) gemini_responder.summarize_run ------------------------------------- #

def test_summarize_clean_run_ok_no_messages():
    r = summarize_run(n_gen=80, n_reused=0, fallback_sids=[], total=80, fail_on_fallback=True)
    assert r["ok"] and r["exit_code"] == 0 and r["messages"] == []


def test_summarize_fallback_warns_but_passes_without_flag():
    r = summarize_run(n_gen=70, n_reused=0, fallback_sids=["s5", "s9"], total=80,
                      fail_on_fallback=False)
    assert r["ok"] and r["exit_code"] == 0
    assert any("FELL BACK" in m for m in r["messages"])


def test_summarize_fallback_aborts_with_flag():
    r = summarize_run(n_gen=70, n_reused=0, fallback_sids=["s5"], total=80,
                      fail_on_fallback=True)
    assert not r["ok"] and r["exit_code"] == 1
    assert any("s5" in m for m in r["messages"])


def test_summarize_all_fallback_noop_warning():
    r = summarize_run(n_gen=0, n_reused=0, fallback_sids=["s%d" % i for i in range(80)],
                      total=80, fail_on_fallback=False)
    assert any("no-op" in m for m in r["messages"])
    # preview caps at 10 session ids
    assert any("..." in m for m in r["messages"])


# --- 3) colbert overwrite guard -------------------------------------------- #

@pytest.mark.parametrize("script", ["train_colbert", "build_colbert_index"])
def test_existing_artifact_blocks(tmp_path, script):
    mod = __import__(f"scripts.{script}", fromlist=["existing_artifact_blocks"])
    fn = mod.existing_artifact_blocks
    d = tmp_path / "music-colbert-v1"
    d.mkdir()
    assert fn(str(d), force=False) is True          # exists + no force -> skip
    assert fn(str(d), force=True) is False           # exists + force  -> proceed (overwrite)
    assert fn(str(tmp_path / "missing"), force=False) is False  # absent -> proceed (build)
    assert fn("", force=False) is False              # empty path -> proceed
