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
