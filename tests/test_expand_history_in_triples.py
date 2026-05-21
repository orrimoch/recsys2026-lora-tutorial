"""Tests for scripts/expand_history_in_triples.py — the one-shot post-pass
that retroactively expands music-turn IDs in the `[HISTORY]:` block of an
existing triples JSONL into the id_to_metadata format.

Why this exists: see docs/superpowers/specs/2026-05-18-ndcg-stretch-design.md
§6.5 Option 1 + the train/inference parity discussion in the conversation
history. Salvages the ~50K-triple full mine without paying for a re-mine.
"""
import json


def _make_row(query: str, **kwargs) -> dict:
    base = {
        "query": query,
        "pos": ["pos text"],
        "neg": [f"neg text {i}" for i in range(15)],
        "pos_tid": "pos_tid",
        "neg_tids": [f"neg_tid_{i}" for i in range(15)],
        "user_id": "u1",
        "session_id": "s1",
    }
    base.update(kwargs)
    return base


def test_expand_history_replaces_known_id_in_history_block(tmp_path):
    """The only field rewritten is `query`; only `A: <id>` slots where <id>
    matches an entry in metadata_dict get expanded."""
    from scripts.expand_history_in_triples import expand_query_history

    metadata_dict = {
        "tk_A": {"track_id": "tk_A",
                 "track_name": ["Hotel California"],
                 "artist_name": ["Eagles"],
                 "album_name": ["Hotel California"]}
    }
    corpus_types = ["track_name", "artist_name", "album_name"]
    in_query = (
        "[USER]: age=25 country=US gender=unknown\n"
        "[GOAL]: discover music\n"
        "[HISTORY]: U: play rock | A: tk_A | U: more guitar\n"
        "[QUERY]: 70s upbeat"
    )
    out = expand_query_history(in_query, metadata_dict, corpus_types)
    # The expanded text from id_to_metadata.
    assert "A: track_id: tk_A, track_name: hotel california, artist_name: eagles, album_name: hotel california" in out
    # Everything else preserved.
    assert "[USER]: age=25 country=US gender=unknown" in out
    assert "[QUERY]: 70s upbeat" in out
    assert "U: play rock" in out
    assert "U: more guitar" in out


def test_expand_history_leaves_unknown_ids_unchanged(tmp_path):
    """ID not in metadata_dict → raw ID kept (catalog-drift fallback)."""
    from scripts.expand_history_in_triples import expand_query_history

    metadata_dict = {
        "tk_KNOWN": {"track_id": "tk_KNOWN", "track_name": ["A"], "artist_name": ["x"], "album_name": ["m"]},
    }
    in_query = (
        "[USER]: age=20\n[GOAL]: \n"
        "[HISTORY]: U: q | A: tk_KNOWN | U: q2 | A: tk_UNKNOWN | U: q3\n"
        "[QUERY]: final"
    )
    out = expand_query_history(in_query, metadata_dict,
                                ["track_name", "artist_name", "album_name"])
    assert "A: track_id: tk_KNOWN," in out
    assert "A: tk_UNKNOWN" in out


def test_expand_history_does_not_touch_user_segments(tmp_path):
    """A user turn happening to contain a string that LOOKS like an ID must
    not be expanded. We only rewrite `A: <id>` slots that match metadata_dict."""
    from scripts.expand_history_in_triples import expand_query_history

    metadata_dict = {
        "tk_A": {"track_id": "tk_A", "track_name": ["Song"], "artist_name": ["x"], "album_name": ["m"]},
    }
    in_query = (
        "[USER]: \n[GOAL]: \n"
        "[HISTORY]: U: tk_A | A: tk_A\n"  # user-turn content == an ID-like string
        "[QUERY]: ok"
    )
    out = expand_query_history(in_query, metadata_dict,
                                ["track_name", "artist_name", "album_name"])
    # U: segment must NOT be expanded.
    assert "U: tk_A" in out
    # A: segment MUST be expanded.
    assert "A: track_id: tk_A," in out


def test_expand_history_is_idempotent(tmp_path):
    """Running the post-pass on an already-expanded file leaves it unchanged
    (no ID-shaped strings remain in A: slots after one pass)."""
    from scripts.expand_history_in_triples import expand_query_history

    metadata_dict = {
        "tk_A": {"track_id": "tk_A", "track_name": ["Song A"], "artist_name": ["x"], "album_name": ["m"]},
    }
    corpus_types = ["track_name", "artist_name", "album_name"]
    in_query = (
        "[USER]: \n[GOAL]: \n"
        "[HISTORY]: U: q | A: tk_A\n"
        "[QUERY]: ok"
    )
    once  = expand_query_history(in_query, metadata_dict, corpus_types)
    twice = expand_query_history(once,    metadata_dict, corpus_types)
    assert once == twice


def test_expand_history_no_history_block_returns_query_unchanged():
    """Defensive: queries without a [HISTORY]: block (shouldn't happen with
    bge_m3_structured mode but be safe) pass through unchanged."""
    from scripts.expand_history_in_triples import expand_query_history

    in_query = "just a single line"
    out = expand_query_history(in_query, {"tk_A": {"track_id": "tk_A", "track_name": [], "artist_name": [], "album_name": []}},
                                ["track_name", "artist_name", "album_name"])
    assert out == in_query


def test_rewrite_jsonl_writes_corrected_query_and_preserves_other_fields(tmp_path):
    """End-to-end: rewrite_jsonl reads input, expands queries, writes output
    preserving pos/neg/pos_tid/neg_tids/user_id/session_id verbatim."""
    from scripts.expand_history_in_triples import rewrite_jsonl

    metadata_dict = {
        "tk_A": {"track_id": "tk_A",
                 "track_name": ["Hotel California"],
                 "artist_name": ["Eagles"],
                 "album_name": ["Hotel California"]}
    }
    corpus_types = ["track_name", "artist_name", "album_name"]
    rows = [
        _make_row(query=(
            "[USER]: age=25\n[GOAL]: \n"
            "[HISTORY]: U: hi | A: tk_A\n[QUERY]: more"
        ), pos_tid="gold_1"),
        _make_row(query=(
            "[USER]: age=30\n[GOAL]: \n"
            "[HISTORY]: U: hey\n[QUERY]: discover"
        ), pos_tid="gold_2"),  # no music turn in history
    ]
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    with open(in_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = rewrite_jsonl(str(in_path), str(out_path),
                          metadata_dict, corpus_types)
    assert stats["rows_total"] == 2
    assert stats["rows_touched"] >= 1  # first row had a music turn
    assert stats["ids_expanded"] >= 1
    out_rows = [json.loads(line) for line in open(out_path)]
    assert len(out_rows) == 2
    # Row 0's history now has expanded text.
    assert "A: track_id: tk_A," in out_rows[0]["query"]
    # Row 0's pos_tid, neg_tids, user_id, session_id all preserved.
    assert out_rows[0]["pos_tid"] == "gold_1"
    assert out_rows[0]["neg_tids"] == rows[0]["neg_tids"]
    assert out_rows[0]["user_id"] == "u1"
    assert out_rows[0]["session_id"] == "s1"
    # Row 1 (no music turn) unchanged.
    assert out_rows[1]["query"] == rows[1]["query"]
