"""Tests for scripts/blind_a_score_tracker.py."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.blind_a_score_tracker import append_score, read_tracker


def test_append_score_creates_tracker_with_header(tmp_path):
    """First append on a fresh path creates the tracker with markdown table header."""
    tracker = tmp_path / "tracker.md"
    append_score(
        tracker_path=tracker,
        config_id="170-wrrf-sid-v5kto-blindsetA",
        composite=0.245, ndcg=0.082, llm=2.50, lex_div=0.77,
        submission_url="https://codabench.org/competitions/.../submission/...",
        notes="W4 first SID submission, weight=0.5",
    )
    assert tracker.exists()
    text = tracker.read_text()
    assert "| config_id " in text  # header row
    assert "170-wrrf-sid-v5kto-blindsetA" in text
    assert "0.245" in text  # composite value formatted in table


def test_append_score_appends_second_row_below_existing(tmp_path):
    """Second append adds a row, doesn't duplicate the header."""
    tracker = tmp_path / "tracker.md"
    append_score(tracker_path=tracker, config_id="A", composite=0.1, ndcg=0.05,
                 llm=2.0, lex_div=0.7, submission_url="url1")
    append_score(tracker_path=tracker, config_id="B", composite=0.2, ndcg=0.08,
                 llm=2.5, lex_div=0.75, submission_url="url2")
    text = tracker.read_text()
    assert text.count("| config_id ") == 1  # header appears once
    assert "| A |" in text
    assert "| B |" in text


def test_read_tracker_returns_all_rows(tmp_path):
    """read_tracker parses the markdown table back into a list of dicts."""
    tracker = tmp_path / "tracker.md"
    append_score(tracker_path=tracker, config_id="A", composite=0.1, ndcg=0.05,
                 llm=2.0, lex_div=0.7, submission_url="url1")
    append_score(tracker_path=tracker, config_id="B", composite=0.2, ndcg=0.08,
                 llm=2.5, lex_div=0.75, submission_url="url2")
    rows = read_tracker(tracker)
    assert len(rows) == 2
    assert rows[0]["config_id"] == "A"
    assert float(rows[0]["composite"]) == 0.1
    assert rows[1]["config_id"] == "B"


def test_append_score_handles_empty_existing_file(tmp_path):
    """Regression test: append_score must initialize header even if the file
    exists but is empty (e.g., user touch'ed it as a placeholder). Without this
    guard the tracker silently corrupts and read_tracker returns []."""
    tracker = tmp_path / "tracker.md"
    tracker.touch()  # exists, but empty — the failure mode the W6 review caught
    assert tracker.exists() and tracker.stat().st_size == 0

    append_score(tracker_path=tracker, config_id="A", composite=0.1, ndcg=0.05,
                 llm=2.0, lex_div=0.7, submission_url="url1")
    text = tracker.read_text()
    assert "| config_id " in text  # header written despite file existing
    assert "| A |" in text

    rows = read_tracker(tracker)
    assert len(rows) == 1
    assert rows[0]["config_id"] == "A"


def test_append_score_sanitizes_pipe_chars_in_notes(tmp_path):
    """Regression test: pipe characters in notes must be escaped, otherwise the
    markdown table parser sees extra columns and silently drops the row."""
    tracker = tmp_path / "tracker.md"
    append_score(tracker_path=tracker, config_id="A", composite=0.1, ndcg=0.05,
                 llm=2.0, lex_div=0.7, submission_url="url1",
                 notes="tried w=0.3 | regressed | also lex_div tanked")
    rows = read_tracker(tracker)
    assert len(rows) == 1
    # The escaped pipe should appear in the notes field
    assert "regressed" in rows[0]["notes"]
