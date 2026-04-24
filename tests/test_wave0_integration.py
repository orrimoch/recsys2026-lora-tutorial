"""End-to-end integration for Wave 0: config → prediction → evaluator → composite → log."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path("/Users/orrimoch/PythonProjs/recsys2026")
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import local_eval  # noqa: E402
import validate_prediction as vp  # noqa: E402


@pytest.fixture(scope="module")
def cached_llama_scores_tid():
    tid = "llama1b_bm25_devset"
    scores_path = REPO_ROOT / "music-crs-evaluator/exp/scores/devset" / f"{tid}.json"
    if not scores_path.exists():
        pytest.skip(f"cached scores missing: {scores_path}")
    return tid


def test_evaluator_reads_cached_scores(cached_llama_scores_tid):
    """run_evaluator returns the cached score dict without re-running."""
    scores = local_eval.run_evaluator(cached_llama_scores_tid, split="devset", force=False)
    assert isinstance(scores, dict)
    for k in ["ndcg@1", "ndcg@10", "ndcg@20", "catalog_diversity", "lexical_diversity"]:
        assert k in scores, f"missing metric: {k}"


def test_composite_retrieval_b_floor_matches(cached_llama_scores_tid):
    """End-to-end: cached scores → composite_retrieval = 0.1042 (B-floor)."""
    scores = local_eval.run_evaluator(cached_llama_scores_tid, split="devset", force=False)
    composite = local_eval.compute_composite_retrieval(scores)
    assert abs(composite - 0.1042) < 0.001, f"expected ~0.1042, got {composite}"


def test_benchmarks_b_floor_parses():
    """load_benchmarks can extract the B-floor nDCG@10 from the markdown."""
    bm = local_eval.load_benchmarks()
    assert bm["b_floor"]
    # load_benchmarks preserves the markdown header capitalization
    val = bm["b_floor"].get("nDCG@10") or bm["b_floor"].get("ndcg@10")
    assert val is not None, f"nDCG@10 missing from b_floor: {bm['b_floor']}"
    assert abs(val - 0.0627) < 0.001


def test_prediction_validator_accepts_real_blind_a():
    """The on-disk blind-A prediction fixtures validate against our schema."""
    # Use the cached random baseline — smallest file, known valid schema (blindA layout).
    prediction_path = REPO_ROOT / "music-crs-baselines/exp/inference/devset/random.json" \
        if not (REPO_ROOT / "music-crs-evaluator/exp/inference/devset/random.json").exists() \
        else REPO_ROOT / "music-crs-evaluator/exp/inference/devset/random.json"
    if not prediction_path.exists():
        pytest.skip(f"no real prediction on disk: {prediction_path}")
    predictions = vp.load_prediction(str(prediction_path))
    errors = vp.validate_schema(predictions, "dev")
    assert errors == [], f"real prediction failed schema: {errors[:3]}"


def test_zip_packager_produces_codabench_layout(tmp_path):
    """E-3 integration: package_zip writes a single prediction.json at zip root."""
    sample = [{
        "session_id": "69137__2020-02-08",
        "user_id": "69137",
        "turn_number": i,
        "predicted_track_ids": [f"trk-{i:03d}-{j:03d}" for j in range(20)],
        "predicted_response": "test",
    } for i in range(1, 9)]
    pred_path = tmp_path / "prediction_in.json"
    pred_path.write_text(json.dumps(sample, ensure_ascii=False))
    out_zip = tmp_path / "out.zip"
    vp.package_zip(str(pred_path), str(out_zip))
    import zipfile
    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
    assert names == ["prediction.json"], f"zip layout wrong: {names}"


def test_submissions_log_append_row(tmp_path, monkeypatch):
    """append_submissions_log_row adds a line to a temp log."""
    log = tmp_path / "submissions_log.md"
    log.write_text("# log\n\n| a | b |\n|---|---|\n")
    monkeypatch.setattr(local_eval, "SUBMISSIONS_LOG_PATH", log)
    local_eval.append_submissions_log_row(
        exp_id="test_integration",
        scores={"ndcg@1": 0.01, "ndcg@10": 0.05, "ndcg@20": 0.07,
                "catalog_diversity": 0.3, "lexical_diversity": 0.2},
        composite_retrieval=0.075,
        composite_projected=None,
        llm_last_known=None,
        status="ITERATE",
        log_path=log,
    )
    content = log.read_text()
    assert "test_integration" in content
    assert "ITERATE" in content
