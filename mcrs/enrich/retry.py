"""Retry-with-exponential-backoff for flaky external calls (LLM APIs).

Pure and injectable (sleep/rng) so it's deterministically testable. Used to wrap Gemini
generate_content against transient 429/503/timeout errors during A1 enrichment.
"""
from __future__ import annotations

import random as _random
import time as _time
from typing import Callable, Optional


def call_with_retry(
    fn: Callable,
    *,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    is_retryable: Optional[Callable[[Exception], bool]] = None,
    jitter: bool = True,
    sleep: Callable[[float], None] = _time.sleep,
    rng: Optional[_random.Random] = None,
):
    """Call fn(); on exception, back off (base*2**attempt, capped at max_delay) and retry.

    - is_retryable(e): if provided and returns False, the error is raised immediately (e.g. auth).
    - jitter: add rng.random() seconds to each delay (spreads concurrent retries).
    - sleep/rng injectable for testing.
    """
    rng = rng or _random.Random()
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - retry policy decides what to re-raise
            last_exc = e
            if is_retryable is not None and not is_retryable(e):
                raise
            if attempt == max_attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** attempt))
            if jitter:
                delay += rng.random()
            sleep(delay)
    assert last_exc is not None
    raise last_exc
