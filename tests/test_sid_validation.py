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


def test_validate_cluster_purity_pure_buckets_pass():
    """When every bucket's tracks share a tag, purity = 100% -> passes."""
    from mcrs.sid.validation import validate_cluster_purity

    buckets = {
        f"bucket_{i}": [f"t{i}_{j}" for j in range(3)]
        for i in range(5)
    }
    tag_lookup = {
        f"t{i}_{j}": ["rock", f"genre{i}"]
        for i in range(5) for j in range(3)
    }
    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=5, threshold=0.60, seed=42,
    )
    assert passed is True
    assert purity == 1.0


def test_validate_cluster_purity_random_tags_fail():
    """Tracks within buckets have no shared tags, purity = 0% -> fails."""
    from mcrs.sid.validation import validate_cluster_purity

    buckets = {
        f"bucket_{i}": [f"t{i}_{j}" for j in range(3)]
        for i in range(5)
    }
    tag_lookup = {
        f"t{i}_{j}": [f"tag_{i}_{j}"]
        for i in range(5) for j in range(3)
    }
    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=5, threshold=0.60, seed=42,
    )
    assert passed is False
    assert purity < 0.60


def test_validate_cluster_purity_handles_singleton_buckets():
    """Singleton buckets count as pure."""
    from mcrs.sid.validation import validate_cluster_purity

    buckets = {f"bucket_{i}": [f"t{i}"] for i in range(5)}
    tag_lookup = {f"t{i}": [f"tag_{i}"] for i in range(5)}

    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=5, threshold=0.60, seed=42,
    )
    assert passed is True
    assert purity == 1.0


def test_validate_cluster_purity_lowercases_and_strips_tags():
    """Tag matching is case-insensitive and whitespace-stripped."""
    from mcrs.sid.validation import validate_cluster_purity

    buckets = {"b": ["t1", "t2"]}
    tag_lookup = {"t1": ["Rock"], "t2": [" rock "]}

    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=1, threshold=0.60, seed=42,
    )
    assert passed is True


def test_validate_cluster_purity_samples_subset_when_n_samples_lt_total():
    """When buckets > n_samples, only sample n_samples for evaluation."""
    from mcrs.sid.validation import validate_cluster_purity

    buckets = {f"b_{i}": [f"t{i}_a", f"t{i}_b"] for i in range(100)}
    tag_lookup = {tid: ["rock"] for i in range(100) for tid in [f"t{i}_a", f"t{i}_b"]}

    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=10, threshold=0.60, seed=42,
    )
    assert passed is True
    assert purity == 1.0


def test_compute_relative_mse_gate_passes_via_absolute_path_for_low_mse():
    """Gate 1: pass via the ABSOLUTE path when rqvae_mse <= absolute_threshold (0.02)."""
    from mcrs.sid.validation import compute_relative_mse_gate

    passed, mse = compute_relative_mse_gate(rqvae_mse=0.001, pca_mse=0.018, multiplier=1.5)
    assert passed is True
    assert mse == 0.001  # returns absolute mse for log inspection


def test_compute_relative_mse_gate_fails_when_both_paths_fail():
    """High absolute mse AND ratio > multiplier -> fail."""
    from mcrs.sid.validation import compute_relative_mse_gate

    passed, mse = compute_relative_mse_gate(rqvae_mse=0.10, pca_mse=0.02, multiplier=1.5)
    assert passed is False  # 0.10 > 0.02 absolute, AND 0.10/0.02=5 > 1.5
    assert mse == 0.10


def test_compute_relative_mse_gate_passes_via_absolute_path_when_better_than_pca():
    """If absolute is good, we pass without even checking the relative ratio."""
    from mcrs.sid.validation import compute_relative_mse_gate

    passed, mse = compute_relative_mse_gate(rqvae_mse=0.005, pca_mse=0.020, multiplier=1.5)
    assert passed is True  # absolute path: 0.005 <= 0.02
    assert mse == 0.005


def test_compute_relative_mse_gate_passes_via_relative_path_when_absolute_fails():
    """High absolute MSE but acceptable relative-to-PCA -> pass via fallback."""
    from mcrs.sid.validation import compute_relative_mse_gate

    # MSE 0.05 fails absolute (>0.02) but is within 1.5x of PCA 0.04 -> pass via relative path
    passed, mse = compute_relative_mse_gate(rqvae_mse=0.05, pca_mse=0.04, multiplier=1.5)
    assert passed is True
    assert mse == 0.05


def test_compute_relative_mse_gate_handles_zero_pca_baseline():
    """Zero pca_mse with low absolute MSE: passes via absolute path."""
    from mcrs.sid.validation import compute_relative_mse_gate

    passed, mse = compute_relative_mse_gate(rqvae_mse=0.001, pca_mse=0.0, multiplier=1.5)
    assert passed is True
    assert mse == 0.001
