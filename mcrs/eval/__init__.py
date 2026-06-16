"""F3 — evaluation harness (official-parity scoring + diagnostics)."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_evaluator_on_path() -> None:
    """Put the pristine `music-crs-evaluator` on sys.path so its metric funcs import."""
    try:
        import metrics.metrics_recsys  # noqa: F401
        return
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        ev = root / "music-crs-evaluator"
        if str(ev) not in sys.path:
            sys.path.insert(0, str(ev))
