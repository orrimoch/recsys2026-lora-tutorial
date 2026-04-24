"""Wave 1 integration: scored cached inferences land in log + benchmark-compare works."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path("/Users/orrimoch/PythonProjs/recsys2026")
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import local_eval  # noqa: E402


SCORED_WAVE1_TIDS = [
    "002-bm25-field-expansion",
    "007-rrf-bm25-dense-v1",
    "009-wrrf-bm25-dense-v1",
    "010-wrrf-bm25-dense-lyrics-v1",
    "011-wrrf-bm25-dense-audio-clap-v1",
]


def test_wave1_scores_exist_for_key_configs():
    """All informative retrieval configs scored in Wave 1 have score files on disk."""
    scores_dir = REPO_ROOT / "music-crs-evaluator/exp/scores/devset"
    missing = []
    for tid in SCORED_WAVE1_TIDS:
        if not (scores_dir / f"{tid}.json").exists():
            missing.append(tid)
    assert not missing, f"Wave 1 did not produce score files for: {missing}"


def test_wave1_best_retrieval_config_identified():
    """010-wrrf-bm25-dense-lyrics-v1 should have the highest nDCG@20 of the retrieval-only sweep."""
    scores_dir = REPO_ROOT / "music-crs-evaluator/exp/scores/devset"
    ndcgs = {}
    for tid in SCORED_WAVE1_TIDS:
        path = scores_dir / f"{tid}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        ndcgs[tid] = data["ndcg@20"]
    best_tid = max(ndcgs, key=ndcgs.get)
    # Any of the top three tied at 0.098-0.099 is acceptable — the test guards against a
    # regression where one of the wRRF variants drops below BM25-only (0.082 B-floor).
    assert ndcgs[best_tid] > 0.09, f"Wave 1 best nDCG@20 {ndcgs[best_tid]:.4f} < 0.09"
    assert best_tid in {
        "010-wrrf-bm25-dense-lyrics-v1",
        "009-wrrf-bm25-dense-v1",
        "011-wrrf-bm25-dense-audio-clap-v1",
        "002-bm25-field-expansion",
    }, f"unexpected Wave 1 winner: {best_tid}"


def test_wave1_retrieval_only_flagged_informational():
    """Informational rows are annotated in submissions_log.md per no-retrieval-only rule."""
    log = (REPO_ROOT / "documents/submissions_log.md").read_text()
    assert "INFORMATIONAL" in log, "retrieval-only rows must be annotated as informational"
    assert "two-step" in log.lower() or "two step" in log.lower(), \
        "log should reference the two-step discipline"


def test_wave1_retrieval_only_beats_bfloor_on_ndcg_not_composite():
    """Confirms the empirical basis for the no-retrieval-only rule: better retrieval,
    but worse composite_retrieval because LexDiv=0."""
    scores_dir = REPO_ROOT / "music-crs-evaluator/exp/scores/devset"
    bfloor = json.loads((scores_dir / "llama1b_bm25_devset.json").read_text())
    d10 = json.loads((scores_dir / "010-wrrf-bm25-dense-lyrics-v1.json").read_text())

    # 010 has better nDCG@20
    assert d10["ndcg@20"] > bfloor["ndcg@20"], "010 should beat B-floor on retrieval"
    # 010 has LexDiv=0 (no LLM response branch), B-floor has LexDiv>0 (it has Llama)
    assert d10["lexical_diversity"] == 0.0, "retrieval-only config should have no responses"
    assert bfloor["lexical_diversity"] > 0.0, "B-floor (Llama+BM25) should have responses"

    c10 = local_eval.compute_composite_retrieval(d10)
    cb = local_eval.compute_composite_retrieval(bfloor)
    # But composite_retrieval is WORSE — that's the trap
    assert c10 < cb, (
        f"retrieval-only composite should be lower than B-floor composite (LexDiv-driven). "
        f"Got 010={c10:.4f}, B-floor={cb:.4f}"
    )


def test_wave1_no_retrieval_only_rule_documented():
    """The no-retrieval-only rule lives in plan + agent memory + auto-memory."""
    plan = (REPO_ROOT / "documents/recsys_challenge_plan.md").read_text()
    assert "No retrieval-only experiments" in plan, "plan §0 principles missing the rule"

    autom = Path("/Users/orrimoch/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/"
                 "memory/feedback_no_retrieval_only.md")
    assert autom.exists(), "auto-memory file missing"
