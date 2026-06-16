"""Retry-with-backoff helper for flaky LLM calls (A1 enrichment, later S1/R2)."""
from __future__ import annotations

import pytest

from mcrs.enrich.retry import call_with_retry


def test_returns_first_success_without_sleeping():
    slept = []
    out = call_with_retry(lambda: "ok", sleep=slept.append)
    assert out == "ok" and slept == []


def test_retries_transient_then_succeeds():
    calls = {"n": 0}
    slept = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 overloaded")
        return "done"

    out = call_with_retry(flaky, max_attempts=5, base_delay=0.1, sleep=slept.append)
    assert out == "done" and calls["n"] == 3 and len(slept) == 2   # slept before each retry


def test_gives_up_after_max_attempts_and_raises_last():
    slept = []
    def always_fail():
        raise RuntimeError("503")
    with pytest.raises(RuntimeError, match="503"):
        call_with_retry(always_fail, max_attempts=4, sleep=slept.append)
    assert len(slept) == 3   # max_attempts-1 retries


def test_non_retryable_raises_immediately():
    slept = []
    def auth_error():
        raise PermissionError("401 unauthorized")
    with pytest.raises(PermissionError):
        call_with_retry(auth_error, is_retryable=lambda e: not isinstance(e, PermissionError),
                        sleep=slept.append)
    assert slept == []   # no retry/backoff for a non-retryable error


def test_backoff_is_bounded_by_max_delay():
    slept = []
    def always_fail():
        raise RuntimeError("503")
    with pytest.raises(RuntimeError):
        call_with_retry(always_fail, max_attempts=8, base_delay=1.0, max_delay=4.0,
                        sleep=slept.append, jitter=False)
    assert max(slept) <= 4.0 and slept == [1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0]
