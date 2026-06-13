import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.emit_results import build_results_json, format_results_block, parse_results_block, print_results_block


def test_composite_matches_local_eval_with_llm():
    # retrieval = 0.5*0.30 + 0.1*0.03 + 0.1*0.79 = 0.232
    # llm_norm = (4.2-1)/4 = 0.8 ; +0.3*0.8 = 0.24 ; total = 0.472
    payload = build_results_json(
        "042", 205, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=4.2, n_sessions=80, gate="turn1_cell49",
    )
    assert payload["composite"] == 0.472
    assert payload["exp"] == "042"
    assert payload["config"] == 205
    assert payload["llm_judge"] == 4.2


def test_composite_retrieval_only_when_llm_none():
    payload = build_results_json(
        "043", 206, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=None, n_sessions=80, gate="turn1_cell49",
    )
    assert payload["composite"] == 0.232
    assert payload["llm_judge"] is None


def test_format_block_is_parseable():
    payload = build_results_json(
        "042", 205, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=4.2, n_sessions=80, gate="turn1_cell49",
    )
    block = format_results_block(payload)
    assert block.startswith("RESULTS_JSON\n")
    parsed = json.loads(block.split("\n", 1)[1])
    assert parsed["config"] == 205
    assert parsed["n_sessions"] == 80


def test_print_results_block_round_trips(capsys):
    payload = print_results_block(
        exp="044", config=207, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=4.2, n_sessions=80, gate="turn1_cell49",
    )
    out = capsys.readouterr().out.strip()
    assert out.startswith("RESULTS_JSON\n")
    parsed = json.loads(out.split("\n", 1)[1])
    assert parsed == payload
    assert payload["config"] == 207


def test_parse_results_block_round_trips():
    payload = build_results_json(
        "042", 205, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=4.2, n_sessions=80, gate="turn1_cell49",
    )
    block = format_results_block(payload)
    assert parse_results_block(block) == payload


def test_parse_results_block_tolerates_surrounding_prose():
    payload = build_results_json(
        "042", 205, ndcg=0.30, cat_div=0.03, lex_div=0.79,
        llm_judge=4.2, n_sessions=80, gate="turn1_cell49",
    )
    block = format_results_block(payload)
    wrapped = "some colab log line\n" + block + "\nDone.\n"
    assert parse_results_block(wrapped) == payload


def test_parse_results_block_raises_when_absent():
    import pytest
    with pytest.raises(ValueError):
        parse_results_block("no sentinel here\njust text")
