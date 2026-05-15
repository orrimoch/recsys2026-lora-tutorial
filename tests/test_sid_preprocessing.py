"""Tests for SID quantizer preprocessing pure functions."""
import numpy as np
import pytest


def test_concat_modalities_l2_normalizes_each_modality_independently():
    """Each modality should be L2-normalized to unit norm before concatenation."""
    from mcrs.sid.preprocessing import concat_modalities

    text = np.array([3.0, 4.0])         # norm = 5
    cf = np.array([1.0, 0.0, 0.0])      # norm = 1 (already unit)
    audio = np.array([0.0, 0.0, 0.0, 1.0])  # norm = 1

    out = concat_modalities(text=text, cf=cf, audio=audio)

    expected = np.array([0.6, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(out, expected, atol=1e-7)


def test_concat_modalities_preserves_dimensionality():
    """Output dim equals sum of input dims."""
    from mcrs.sid.preprocessing import concat_modalities

    text = np.random.randn(1024).astype(np.float32)
    cf = np.random.randn(128).astype(np.float32)
    audio = np.random.randn(512).astype(np.float32)

    out = concat_modalities(text=text, cf=cf, audio=audio)
    assert out.shape == (1024 + 128 + 512,)


def test_concat_modalities_handles_zero_cf_imputation():
    """Cold-start CF rows are zero vectors; concat must not divide by zero."""
    from mcrs.sid.preprocessing import concat_modalities

    text = np.array([1.0, 0.0])
    cf_zero = np.zeros(3, dtype=np.float32)
    audio = np.array([1.0, 0.0])

    out = concat_modalities(text=text, cf=cf_zero, audio=audio)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[2:5], np.zeros(3))


def test_concat_modalities_returns_float32():
    """Output dtype should be float32 (memory + downstream-compat)."""
    from mcrs.sid.preprocessing import concat_modalities

    text = np.random.randn(4).astype(np.float64)
    cf = np.random.randn(2).astype(np.float64)
    audio = np.random.randn(3).astype(np.float64)

    out = concat_modalities(text=text, cf=cf, audio=audio)
    assert out.dtype == np.float32


def test_compute_collision_buckets_groups_tracks_by_sid():
    from mcrs.sid.preprocessing import compute_collision_buckets

    track_ids = ["t1", "t2", "t3", "t4"]
    sid_assignments = [
        (10, 20, 30),  # t1 unique
        (5, 5, 5),     # t2 collides with t3
        (5, 5, 5),     # t3
        (1, 2, 3),     # t4 unique
    ]
    popularity = {"t1": 50.0, "t2": 30.0, "t3": 80.0, "t4": 10.0}

    buckets = compute_collision_buckets(track_ids, sid_assignments, popularity)

    assert buckets[(10, 20, 30)] == ["t1"]
    assert buckets[(5, 5, 5)] == ["t3", "t2"]
    assert buckets[(1, 2, 3)] == ["t4"]


def test_compute_collision_buckets_sorts_by_popularity_descending():
    from mcrs.sid.preprocessing import compute_collision_buckets

    track_ids = ["a", "b", "c"]
    sid_assignments = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    popularity = {"a": 1.0, "b": 100.0, "c": 50.0}

    buckets = compute_collision_buckets(track_ids, sid_assignments, popularity)
    assert buckets[(0, 0, 0)] == ["b", "c", "a"]


def test_compute_collision_buckets_handles_missing_popularity_as_zero():
    from mcrs.sid.preprocessing import compute_collision_buckets

    track_ids = ["a", "b"]
    sid_assignments = [(0, 0, 0), (0, 0, 0)]
    popularity = {"a": 5.0}

    buckets = compute_collision_buckets(track_ids, sid_assignments, popularity)
    assert buckets[(0, 0, 0)] == ["a", "b"]


def test_compute_collision_buckets_empty_input_returns_empty_dict():
    from mcrs.sid.preprocessing import compute_collision_buckets
    assert compute_collision_buckets([], [], {}) == {}
