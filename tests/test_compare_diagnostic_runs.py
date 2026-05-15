"""Tests for the cross-run comparison primitives.

paired_bootstrap_ci  — confidence interval on the mean of paired (exp - baseline)
                       differences, used to gate "is this experiment really better"
                       decisions without burning Blind-A budget.

compute_migration_matrix — failure-mode confusion: how many baseline misses
                           became experiment hits (and the inverse — regressions).
"""
import math


# ---- paired_bootstrap_ci ----

def test_paired_bootstrap_ci_all_zero_diffs_returns_zero_ci():
    """If every paired diff is 0, the CI is exactly (0, 0, 0)."""
    from scripts.compare_diagnostic_runs import paired_bootstrap_ci

    mean, lo, hi = paired_bootstrap_ci(
        paired_diffs=[0.0] * 50,
        n_resamples=200,
        alpha=0.05,
        seed=42,
    )

    assert mean == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_paired_bootstrap_ci_is_deterministic_with_seed():
    """Same inputs + seed → same CI, twice."""
    from scripts.compare_diagnostic_runs import paired_bootstrap_ci

    diffs = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.3, 0.05, 0.0, 0.1] * 5
    a = paired_bootstrap_ci(diffs, n_resamples=500, alpha=0.05, seed=42)
    b = paired_bootstrap_ci(diffs, n_resamples=500, alpha=0.05, seed=42)

    assert a == b


def test_paired_bootstrap_ci_strictly_positive_diffs_excludes_zero():
    """Strongly positive shift → CI lies strictly above 0 (significant improvement)."""
    from scripts.compare_diagnostic_runs import paired_bootstrap_ci

    diffs = [0.5] * 100  # a constant +0.5 per turn — overwhelming signal
    mean, lo, hi = paired_bootstrap_ci(diffs, n_resamples=500, alpha=0.05, seed=42)

    assert math.isclose(mean, 0.5, abs_tol=1e-9)
    assert lo > 0.0
    assert hi >= mean


def test_paired_bootstrap_ci_returns_mean_matching_input_average():
    """The reported mean must equal the arithmetic mean of the input diffs (not a bootstrap mean)."""
    from scripts.compare_diagnostic_runs import paired_bootstrap_ci

    diffs = [0.1, 0.2, 0.3, 0.4]
    mean, _, _ = paired_bootstrap_ci(diffs, n_resamples=100, alpha=0.05, seed=7)

    assert math.isclose(mean, 0.25, abs_tol=1e-9)


def test_paired_bootstrap_ci_empty_input_returns_zeros():
    """Empty input is a no-op — return (0, 0, 0) instead of dividing by zero."""
    from scripts.compare_diagnostic_runs import paired_bootstrap_ci

    assert paired_bootstrap_ci([], n_resamples=50, alpha=0.05, seed=1) == (0.0, 0.0, 0.0)


# ---- compute_migration_matrix ----

def test_compute_migration_matrix_empty_inputs_returns_empty_dict():
    from scripts.compare_diagnostic_runs import compute_migration_matrix
    assert compute_migration_matrix([], []) == {}


def test_compute_migration_matrix_no_change_diagonal_only():
    """If every turn keeps the same category, only diagonal cells are populated."""
    from scripts.compare_diagnostic_runs import compute_migration_matrix

    cats = ["hit_in_top_k", "hit_in_top_k", "not_in_either", "not_in_bm25_only"]
    matrix = compute_migration_matrix(cats, cats)

    assert matrix["hit_in_top_k"]["hit_in_top_k"] == 2
    assert matrix["not_in_either"]["not_in_either"] == 1
    assert matrix["not_in_bm25_only"]["not_in_bm25_only"] == 1


def test_compute_migration_matrix_records_off_diagonal_improvements_and_regressions():
    """Mix of improvements + regressions populates off-diagonal cells correctly."""
    from scripts.compare_diagnostic_runs import compute_migration_matrix

    baseline = ["not_in_either", "not_in_either", "hit_in_top_k", "hit_in_top_k"]
    experiment = ["hit_in_top_k", "in_both_low_rank", "hit_in_top_k", "not_in_dense_only"]
    matrix = compute_migration_matrix(baseline, experiment)

    # 2 baseline misses: one became hit (improvement), one became low-rank
    assert matrix["not_in_either"]["hit_in_top_k"] == 1
    assert matrix["not_in_either"]["in_both_low_rank"] == 1
    # 2 baseline hits: one stayed hit, one regressed
    assert matrix["hit_in_top_k"]["hit_in_top_k"] == 1
    assert matrix["hit_in_top_k"]["not_in_dense_only"] == 1


def test_compute_migration_matrix_raises_on_length_mismatch():
    """Lists must be index-aligned — length mismatch is a bug, not silent truncation."""
    from scripts.compare_diagnostic_runs import compute_migration_matrix
    import pytest
    with pytest.raises(ValueError):
        compute_migration_matrix(["hit_in_top_k"], ["hit_in_top_k", "not_in_either"])
