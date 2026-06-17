"""
Tests for scripts/measure_enriched_tokens.py

Tests the pure `recommend_max_length` function (no network/parquet required),
importability/signature of `measure`, and error-path behaviour of `measure`
using tiny temp parquets (no tokenizer/network needed — errors raise before
tokenization).
"""
import inspect
import tempfile
import os
import pytest
import pandas as pd

from scripts.measure_enriched_tokens import recommend_max_length, measure


# ---------------------------------------------------------------------------
# recommend_max_length — pure, no I/O
# ---------------------------------------------------------------------------

class TestRecommendMaxLength:
    def test_fits_1536_warm_query_enriched_doc(self):
        # 866 + 450 + 4 = 1320, which is > 1024 but <= 1536 → recommend 1536
        assert recommend_max_length(866, 450) == 1536

    def test_fits_1024_raw_doc_case(self):
        # 866 + 69 + 4 = 939, which fits inside 1024 → recommend 1024
        assert recommend_max_length(866, 69) == 1024

    def test_fits_2048_large_doc(self):
        # 866 + 1200 + 4 = 2070, which is > 1536 but <= 2048 → recommend 2048
        assert recommend_max_length(866, 1200) == 2048

    def test_overflow_returns_largest(self):
        # 866 + 3000 + 4 = 3870, exceeds all candidates → return 2048 (largest)
        assert recommend_max_length(866, 3000) == 2048

    def test_exactly_at_boundary_1024(self):
        # query_p99=500, doc_p99=520, specials=4 → 500+520+4=1024, exactly fits 1024
        assert recommend_max_length(500, 520) == 1024

    def test_exactly_at_boundary_1536(self):
        # 500 + 1032 + 4 = 1536, exactly fits 1536
        assert recommend_max_length(500, 1032) == 1536

    def test_exactly_at_boundary_2048(self):
        # 500 + 1544 + 4 = 2048, exactly fits 2048
        assert recommend_max_length(500, 1544) == 2048

    def test_one_over_2048_returns_2048(self):
        # 500 + 1545 + 4 = 2049, overflow → return 2048
        assert recommend_max_length(500, 1545) == 2048

    def test_custom_specials(self):
        # With specials=2: 866 + 154 + 2 = 1022 <= 1024 → 1024
        assert recommend_max_length(866, 154, specials=2) == 1024

    def test_custom_candidates(self):
        # Custom candidates=(512, 1024): 10+10+4=24 fits 512
        assert recommend_max_length(10, 10, candidates=(512, 1024)) == 512

    def test_custom_candidates_overflow(self):
        # Custom candidates=(512, 1024): 600+600+4=1204 > 1024 → return 1024
        assert recommend_max_length(600, 600, candidates=(512, 1024)) == 1024


# ---------------------------------------------------------------------------
# measure — just verify importable & callable (signature check only)
# ---------------------------------------------------------------------------

class TestMeasureImportable:
    def test_measure_is_callable(self):
        assert callable(measure)

    def test_measure_signature(self):
        sig = inspect.signature(measure)
        params = list(sig.parameters.keys())
        assert "parquet_glob" in params
        assert "tokenizer_name" in params
        assert "sample" in params


# ---------------------------------------------------------------------------
# measure — error-path tests using tiny temp parquets (no network/tokenizer)
# ---------------------------------------------------------------------------

class TestMeasureErrorPaths:
    """These tests exercise error handling in measure() before tokenization.

    They write a tiny parquet to a temp file so no HuggingFace download is
    triggered; the function must raise the correct exception before reaching
    the tokenizer load.
    """

    def _write_temp_parquet(self, df: pd.DataFrame) -> str:
        """Write df to a uniquely-named temp parquet and return its path."""
        fd, path = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        df.to_parquet(path, index=False)
        return path

    def test_missing_enriched_doc_column_raises_key_error(self):
        """A parquet without an 'enriched_doc' column must raise KeyError."""
        df = pd.DataFrame({"other_column": ["foo", "bar", "baz"]})
        path = self._write_temp_parquet(df)
        try:
            with pytest.raises(KeyError, match="enriched_doc"):
                measure(path, tokenizer_name="BAAI/bge-reranker-v2-m3")
        finally:
            os.unlink(path)

    def test_all_null_enriched_doc_raises_value_error(self):
        """A parquet whose enriched_doc column is entirely null must raise ValueError."""
        df = pd.DataFrame({"enriched_doc": [None, None, None]})
        path = self._write_temp_parquet(df)
        try:
            with pytest.raises(ValueError, match="No non-null enriched_doc"):
                measure(path, tokenizer_name="BAAI/bge-reranker-v2-m3")
        finally:
            os.unlink(path)
