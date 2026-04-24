"""Shared pytest fixtures for RecSys2026 test suite."""
import pytest


def _tid(prefix: str, i: int) -> str:
    """Deterministic UUID-shaped track id (36 chars, hyphenated)."""
    base = f"{prefix}{i:04d}"
    pad = "0" * (32 - len(base))
    hexs = (base + pad)[:32]
    return f"{hexs[0:8]}-{hexs[8:12]}-{hexs[12:16]}-{hexs[16:20]}-{hexs[20:32]}"


def _make_session(session_id: str, user_id: str, n_turns: int = 8, n_tracks: int = 20,
                  tid_prefix: str = "a", response: str = "Here are some tracks you might enjoy."):
    turns = []
    for t in range(1, n_turns + 1):
        tids = [_tid(f"{tid_prefix}{t}", i) for i in range(n_tracks)]
        turns.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": t,
            "predicted_track_ids": tids,
            "predicted_response": response,
        })
    return turns


@pytest.fixture
def sample_prediction_valid():
    """Minimal valid prediction: 2 sessions x 8 turns x 20 distinct track ids."""
    preds = []
    preds.extend(_make_session("sess-0001", "user-0001", tid_prefix="a"))
    preds.extend(_make_session("sess-0002", "user-0002", tid_prefix="b"))
    return preds


@pytest.fixture
def sample_prediction_duplicate_tids():
    """Invalid: duplicate track ids within a turn's predicted_track_ids."""
    preds = _make_session("sess-dup", "user-dup", tid_prefix="d")
    dup_tid = preds[0]["predicted_track_ids"][0]
    preds[0]["predicted_track_ids"][5] = dup_tid
    return preds


@pytest.fixture
def sample_prediction_missing_turn():
    """Invalid: session is missing turn 5."""
    preds = _make_session("sess-gap", "user-gap", tid_prefix="g")
    return [p for p in preds if p["turn_number"] != 5]


@pytest.fixture
def sample_prediction_non_ascii():
    """Valid: predicted_response contains non-ASCII chars (accented, CJK, emoji-like)."""
    return _make_session(
        "sess-utf8",
        "user-utf8",
        tid_prefix="u",
        response="Voici quelques morceaux que vous pourriez aimer — 音楽をどうぞ. Cafe resume naive.",
    )


@pytest.fixture
def track_catalog_sample():
    """10-row dict mocking HF track-metadata schema."""
    rows = {
        "track_id": [_tid("c", i) for i in range(10)],
        "track_name": [f"Track {i}" for i in range(10)],
        "artist_id": [_tid("ar", i % 3) for i in range(10)],
        "artist_name": [f"Artist {i % 3}" for i in range(10)],
        "album_id": [_tid("al", i % 4) for i in range(10)],
        "album_name": [f"Album {i % 4}" for i in range(10)],
        "category": ["pop", "rock", "jazz", "pop", "rock", "jazz", "pop", "rock", "jazz", "pop"],
        "duration_ms": [180000 + i * 1000 for i in range(10)],
        "release_year": [2000 + i for i in range(10)],
        "popularity": [float(i) / 10.0 for i in range(10)],
    }
    return rows
