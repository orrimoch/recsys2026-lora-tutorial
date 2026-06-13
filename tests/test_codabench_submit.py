import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest
from scripts.codabench_submit import poll_until_scored, parse_score, poll_submission


def test_poll_returns_when_finished():
    states = iter([
        {"status": "running"},
        {"status": "running"},
        {"status": "finished", "scores": {"composite": 0.46}},
    ])
    result = poll_until_scored(lambda sid: next(states), "sub1",
                               max_tries=5, sleep_fn=lambda s: None)
    assert result["status"] == "finished"
    assert result["scores"]["composite"] == 0.46


def test_poll_raises_on_failure():
    with pytest.raises(RuntimeError):
        poll_until_scored(lambda sid: {"status": "failed"}, "sub1",
                          max_tries=5, sleep_fn=lambda s: None)


def test_poll_times_out():
    with pytest.raises(TimeoutError):
        poll_until_scored(lambda sid: {"status": "running"}, "sub1",
                          max_tries=3, sleep_fn=lambda s: None)


def test_parse_score_extracts_composite():
    status = {"status": "finished",
              "scores": {"composite": 0.46, "ndcg@20": 0.30}}
    parsed = parse_score(status)
    assert parsed["composite"] == 0.46
    assert parsed["ndcg@20"] == 0.30


def test_poll_propagates_get_status_exception():
    def _boom(sid):
        raise ConnectionError("down")
    with pytest.raises(ConnectionError):
        poll_until_scored(_boom, "sub1", max_tries=5, sleep_fn=lambda s: None)
