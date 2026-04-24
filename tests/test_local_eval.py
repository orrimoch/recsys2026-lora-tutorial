"""Smoke tests for scripts/local_eval.py (W0-7)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import local_eval  # noqa: E402


def test_compute_composite_retrieval():
    scores = {"ndcg@20": 0.1, "catalog_diversity": 0.4, "lexical_diversity": 0.3}
    # 0.5*0.1 + 0.1*0.4 + 0.1*0.3 = 0.05 + 0.04 + 0.03 = 0.12
    assert local_eval.compute_composite_retrieval(scores) == pytest.approx(0.12)


def test_compute_composite_projected_with_llm():
    scores = {"ndcg@20": 0.1, "catalog_diversity": 0.4, "lexical_diversity": 0.3}
    # 0.12 + 0.3 * (3 - 1) / 4 = 0.12 + 0.15 = 0.27
    assert local_eval.compute_composite_projected(scores, llm_last_known=3.0) == pytest.approx(0.27)


def test_compute_composite_projected_no_llm():
    scores = {"ndcg@20": 0.1, "catalog_diversity": 0.4, "lexical_diversity": 0.3}
    assert local_eval.compute_composite_projected(scores, llm_last_known=None) == pytest.approx(
        local_eval.compute_composite_retrieval(scores)
    )


def test_load_benchmarks_parses_b_floor():
    benchmarks = local_eval.load_benchmarks()
    b_floor = benchmarks["b_floor"]
    assert b_floor, "B-floor section should parse to a non-empty dict"
    # Benchmarks markdown keeps the metric names with canonical case "nDCG@10".
    assert b_floor.get("nDCG@10") == pytest.approx(0.0627)


def test_run_evaluator_reads_cached_scores():
    tid = "llama1b_bm25_devset"
    cached = REPO_ROOT / "music-crs-evaluator" / "exp" / "scores" / "devset" / f"{tid}.json"
    if not cached.exists():
        pytest.skip(f"Cached scores file not present at {cached}")
    result = local_eval.run_evaluator(tid, split="devset", force=False)
    assert isinstance(result, dict)
    assert "ndcg@10" in result
    assert "ndcg@20" in result
    assert "catalog_diversity" in result
    assert "lexical_diversity" in result
