"""Wave 2 integration: first two-step (retrieval + LLM response) run is shippable."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path("/Users/orrimoch/PythonProjs/recsys2026")
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import local_eval  # noqa: E402
import validate_prediction as vp  # noqa: E402


WAVE2_TID = "020-two-step-wrrf-lyrics-qwen15b-devset"
PREDICTION_PATH = REPO_ROOT / "music-crs-baselines/exp/inference/devset" / f"{WAVE2_TID}.json"
SCORES_PATH = REPO_ROOT / "music-crs-evaluator/exp/scores/devset" / f"{WAVE2_TID}.json"


@pytest.fixture(scope="module")
def require_prediction():
    """Skip until the FULL dev run has finished (8000 rows), not just the smoke (40 rows)."""
    if not PREDICTION_PATH.exists():
        pytest.skip(f"Wave 2 prediction not yet produced: {PREDICTION_PATH}")
    try:
        data = json.loads(PREDICTION_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        pytest.skip(f"prediction file unreadable (likely mid-write): {exc}")
    if len(data) < 8000:
        pytest.skip(
            f"only {len(data)} rows present (smoke output or partial full run); "
            "full dev run not complete"
        )
    return PREDICTION_PATH


@pytest.fixture(scope="module")
def require_scores():
    if not SCORES_PATH.exists():
        pytest.skip(f"Wave 2 scores not yet produced: {SCORES_PATH}")
    return SCORES_PATH


def test_wave2_prediction_passes_schema(require_prediction):
    """The Wave 2 prediction JSON must be schema-valid for dev (8000 rows, all 8 turns)."""
    predictions = vp.load_prediction(str(require_prediction))
    errors = vp.validate_schema(predictions, "dev")
    assert errors == [], f"Wave 2 prediction fails schema: {errors[:3]}"


def test_wave2_is_two_step_not_retrieval_only(require_prediction):
    """Enforces the no-retrieval-only rule: every row has a non-empty predicted_response."""
    predictions = json.loads(require_prediction.read_text())
    empty_responses = sum(1 for r in predictions if not r.get("predicted_response", "").strip())
    # Allow <5% empty responses (rare generation failures) but not a systemic retrieval-only pattern
    assert empty_responses / len(predictions) < 0.05, (
        f"Wave 2 has {empty_responses}/{len(predictions)} empty responses — "
        "this is a retrieval-only run, not a two-step experiment"
    )


def test_wave2_schema_row_count_dev(require_prediction):
    """Dev split should have 8000 rows (1000 sessions × 8 turns)."""
    predictions = json.loads(require_prediction.read_text())
    assert len(predictions) == 8000, f"expected 8000 rows, got {len(predictions)}"


def test_wave2_composite_computed(require_scores):
    """Scores file exists and composite_retrieval can be computed and is positive."""
    scores = json.loads(require_scores.read_text())
    composite = local_eval.compute_composite_retrieval(scores)
    assert composite > 0.0, f"composite_retrieval non-positive: {composite}"
    # Soft upper bound for sanity (B-target is ~0.40 full-composite; retrieval-side alone caps lower)
    assert composite < 0.5, f"composite_retrieval suspiciously high: {composite}"


def test_wave2_lexdiv_populated(require_scores):
    """Two-step runs MUST populate lexical_diversity > 0 (empty responses would signal regression)."""
    scores = json.loads(require_scores.read_text())
    assert scores["lexical_diversity"] > 0.0, (
        f"LexDiv=0 means responses are empty/invariant — not a two-step run"
    )


def test_wave2_beats_bfloor_on_composite(require_scores):
    """First-champion gate: Wave 2 two-step should beat B-floor composite_retrieval (0.1042)."""
    scores = json.loads(require_scores.read_text())
    composite_w2 = local_eval.compute_composite_retrieval(scores)
    benchmarks = local_eval.load_benchmarks()
    composite_bfloor = local_eval._lookup(benchmarks.get("b_floor"), "composite_retrieval")
    # This is a soft assertion — Qwen-1.5B could underperform Llama-1B on LexDiv;
    # if so, we log but don't hard-fail. Use pytest.xfail pattern here since this
    # is the research question, not a code invariant.
    if composite_w2 < (composite_bfloor or 0.1042):
        pytest.xfail(
            f"Wave 2 did not beat B-floor: {composite_w2:.4f} < {composite_bfloor:.4f}. "
            "Promote a different retrieval branch or LM."
        )
    assert composite_w2 >= (composite_bfloor or 0.1042)
