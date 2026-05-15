"""Tests for SID quantizer validation gate functions."""
import numpy as np


def test_validate_codebook_utilization_passes_when_above_threshold():
    """All 256 codes used by >=1 track -> 100% utilization -> passes 80% threshold."""
    from mcrs.sid.validation import validate_codebook_utilization

    assignments = list(range(256)) + list(range(256)) + [42] * (1000 - 512)
    passed, util = validate_codebook_utilization(
        assignments, codebook_size=256, threshold=0.80,
    )
    assert passed is True
    assert util == 1.0


def test_validate_codebook_utilization_fails_when_below_threshold():
    """Only 100/256 codes used -> 39% utilization -> fails 80% threshold."""
    from mcrs.sid.validation import validate_codebook_utilization

    assignments = list(range(100)) * 10
    passed, util = validate_codebook_utilization(
        assignments, codebook_size=256, threshold=0.80,
    )
    assert passed is False
    assert abs(util - 100/256) < 1e-9


def test_validate_codebook_utilization_at_exact_threshold_passes():
    """80% utilization (205/256) at threshold 0.80 passes (>=, not >)."""
    from mcrs.sid.validation import validate_codebook_utilization

    assignments = list(range(205))
    passed, _ = validate_codebook_utilization(
        assignments, codebook_size=256, threshold=0.80,
    )
    assert passed is True
