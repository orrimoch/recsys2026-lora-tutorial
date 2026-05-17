"""Tests for W3 v1 post-mortem analysis helpers (mcrs.sid.diagnostics).

The W3 v1 SID generator trained cleanly (eval loss plateaued at 1.48) but failed
the nDCG@20 gate at 0.0204. These helpers underpin three diagnostic checks:
  Q1 — per-position teacher-forced accuracy (which SID position is the model wrong on?)
  Q2 — generated-SID popularity distribution (is the model collapsing to popular SIDs?)
  Q3 — unconstrained validity rate (does the model know the SID space without the trie?)
"""
from collections import Counter

import pytest


# ---------------------------------------------------------------------------
# Q1 helpers: per-position top-k accuracy
# ---------------------------------------------------------------------------

class TestTopKAccuracy:
    def test_top1_exact_match(self):
        from mcrs.sid.diagnostics import topk_accuracy
        # Each row: ranked predictions (most likely first). Gold matches first.
        rankings = [[5, 1, 2], [9, 8, 7]]
        golds = [5, 9]
        assert topk_accuracy(rankings, golds, k=1) == 1.0

    def test_top1_no_match(self):
        from mcrs.sid.diagnostics import topk_accuracy
        rankings = [[1, 2, 3], [4, 5, 6]]
        golds = [9, 9]
        assert topk_accuracy(rankings, golds, k=1) == 0.0

    def test_top5_partial_match(self):
        from mcrs.sid.diagnostics import topk_accuracy
        # Gold is 4th in first row → hits at k=5 not k=3
        rankings = [[1, 2, 3, 7, 5], [9, 8, 7, 6, 5]]
        golds = [7, 100]
        assert topk_accuracy(rankings, golds, k=5) == 0.5
        assert topk_accuracy(rankings, golds, k=3) == 0.0

    def test_empty_input(self):
        from mcrs.sid.diagnostics import topk_accuracy
        assert topk_accuracy([], [], k=1) == 0.0


# ---------------------------------------------------------------------------
# Helpers for finding tokens at each SID level
# ---------------------------------------------------------------------------

class TestBuildLevelTokenIds:
    def test_groups_by_level(self):
        from mcrs.sid.diagnostics import build_level_token_ids
        sid_lookup = {
            (0, 0): 100, (0, 1): 101, (0, 2): 102,
            (1, 0): 200, (1, 1): 201,
            (2, 0): 300,
        }
        result = build_level_token_ids(sid_lookup)
        assert result[0] == [100, 101, 102]
        assert result[1] == [200, 201]
        assert result[2] == [300]


# ---------------------------------------------------------------------------
# Q2 helpers: popularity collapse statistics
# ---------------------------------------------------------------------------

class TestPopularityStats:
    def test_uniform_distribution(self):
        """3 distinct SIDs, each 1/3 of the mass — entropy = log2(3) ≈ 1.585, gini = 0."""
        from mcrs.sid.diagnostics import popularity_stats
        counter = Counter({"a": 10, "b": 10, "c": 10})
        s = popularity_stats(counter, top_ks=[1, 2, 3])
        assert s["n_unique"] == 3
        assert s["total"] == 30
        assert s["top_1_frequency"] == pytest.approx(10 / 30)
        assert s["top_k_coverage"][1] == pytest.approx(10 / 30)
        assert s["top_k_coverage"][3] == pytest.approx(1.0)
        assert s["entropy_bits"] == pytest.approx(1.5849625, abs=1e-4)
        assert s["gini"] == pytest.approx(0.0, abs=1e-9)

    def test_total_collapse(self):
        """All mass on one SID — entropy = 0, gini → 1 in the limit."""
        from mcrs.sid.diagnostics import popularity_stats
        counter = Counter({"a": 100})
        s = popularity_stats(counter, top_ks=[1, 10])
        assert s["n_unique"] == 1
        assert s["top_1_frequency"] == 1.0
        assert s["entropy_bits"] == pytest.approx(0.0)

    def test_top_k_caps_at_n_unique(self):
        from mcrs.sid.diagnostics import popularity_stats
        counter = Counter({"a": 5, "b": 5})
        s = popularity_stats(counter, top_ks=[1, 5, 100])
        # Even though we asked for top-100, only 2 SIDs exist; coverage saturates at 1.0
        assert s["top_k_coverage"][1] == 0.5
        assert s["top_k_coverage"][5] == 1.0
        assert s["top_k_coverage"][100] == 1.0


class TestShannonEntropyBits:
    def test_uniform_two_bins(self):
        from mcrs.sid.diagnostics import shannon_entropy_bits
        assert shannon_entropy_bits({"a": 1, "b": 1}) == pytest.approx(1.0)

    def test_one_bin(self):
        from mcrs.sid.diagnostics import shannon_entropy_bits
        assert shannon_entropy_bits({"a": 5}) == pytest.approx(0.0)

    def test_empty(self):
        from mcrs.sid.diagnostics import shannon_entropy_bits
        assert shannon_entropy_bits({}) == 0.0


class TestGini:
    def test_perfect_equality(self):
        from mcrs.sid.diagnostics import gini
        assert gini([10, 10, 10, 10]) == pytest.approx(0.0)

    def test_perfect_inequality(self):
        """One bin has all the mass — gini → (n-1)/n; with n=4 → 0.75."""
        from mcrs.sid.diagnostics import gini
        assert gini([0, 0, 0, 100]) == pytest.approx(0.75, abs=1e-6)

    def test_empty_returns_zero(self):
        from mcrs.sid.diagnostics import gini
        assert gini([]) == 0.0


# ---------------------------------------------------------------------------
# Q3 helpers: unconstrained-generation validity
# ---------------------------------------------------------------------------

class TestTripletValidity:
    def _setup(self):
        """3-track codebook with explicit token ids."""
        sid_lookup = {
            (0, 0): 100, (0, 1): 101,
            (1, 0): 200, (1, 1): 201,
            (2, 0): 300, (2, 1): 301,
        }
        inverse = {v: k for k, v in sid_lookup.items()}
        # Codebook contains only 2 valid triplets out of 8 possible.
        sid_to_tracks = {(0, 0, 0): ["t1"], (1, 1, 1): ["t2"]}
        return sid_lookup, inverse, sid_to_tracks

    def test_all_levels_valid_and_triplet_in_codebook(self):
        from mcrs.sid.diagnostics import is_triplet_valid
        _, inv, codebook = self._setup()
        result = is_triplet_valid([100, 200, 300], inv, codebook)
        assert result["level_in_range"] == [True, True, True]
        assert result["all_levels_in_range"] is True
        assert result["triplet_in_codebook"] is True

    def test_all_levels_valid_but_triplet_not_in_codebook(self):
        """Valid SID tokens at each level, but the triplet doesn't map to any track."""
        from mcrs.sid.diagnostics import is_triplet_valid
        _, inv, codebook = self._setup()
        # (0, 0, 1) is grammatically valid but not in codebook
        result = is_triplet_valid([100, 200, 301], inv, codebook)
        assert result["all_levels_in_range"] is True
        assert result["triplet_in_codebook"] is False

    def test_wrong_level_at_position_1(self):
        """Token at position 1 is a level-2 token instead of level-1."""
        from mcrs.sid.diagnostics import is_triplet_valid
        _, inv, codebook = self._setup()
        result = is_triplet_valid([100, 300, 300], inv, codebook)
        assert result["level_in_range"] == [True, False, True]
        assert result["all_levels_in_range"] is False
        assert result["triplet_in_codebook"] is False

    def test_non_sid_token(self):
        """Token id not in the SID vocabulary at all (e.g., a normal word piece)."""
        from mcrs.sid.diagnostics import is_triplet_valid
        _, inv, codebook = self._setup()
        result = is_triplet_valid([42, 200, 300], inv, codebook)
        assert result["level_in_range"][0] is False
        assert result["all_levels_in_range"] is False
        assert result["triplet_in_codebook"] is False


class TestValidityRates:
    """Aggregate is_triplet_valid results across many generations into rates."""

    def test_aggregates_per_position_and_triplet(self):
        from mcrs.sid.diagnostics import aggregate_validity
        # Two generations: one fully valid + in codebook, one wrong at level 1
        records = [
            {"level_in_range": [True, True, True],   "triplet_in_codebook": True},
            {"level_in_range": [True, False, True],  "triplet_in_codebook": False},
        ]
        rates = aggregate_validity(records)
        assert rates["level_0_in_range_rate"] == 1.0
        assert rates["level_1_in_range_rate"] == 0.5
        assert rates["level_2_in_range_rate"] == 1.0
        assert rates["all_levels_in_range_rate"] == 0.5
        assert rates["triplet_in_codebook_rate"] == 0.5

    def test_empty(self):
        from mcrs.sid.diagnostics import aggregate_validity
        rates = aggregate_validity([])
        assert rates["all_levels_in_range_rate"] == 0.0
