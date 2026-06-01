"""Train/serve parity: ONE canonical track-text format, used by the catalog
embedder, the [HISTORY] expander, and id_to_metadata — so a fine-tuned encoder
ranks the served catalog against the exact string format it trained on.

The recurring failure mode (build_doc_text pipe/5-field vs id_to_metadata
comma/3-field) is what capped bge_m3_ft. format_catalog_track_text is the single
source of truth.
"""
from mcrs.retrieval_modules.track_text import format_catalog_track_text
from mcrs.db_item.music_catalog import MusicCatalogDB

_MD = {
    "t1": {"track_id": "t1", "track_name": ["Forbidden Dance"],
           "artist_name": ["Aterciopelados"], "album_name": ["La Pipa De La Paz"]},
    "t2": {"track_id": "t2", "track_name": ["Song"], "artist_name": [],
           "album_name": ["Al"]},
}
_CT = ["track_name", "artist_name", "album_name"]


def test_canonical_format_comma_prefixed_lowercased():
    assert format_catalog_track_text("t1", _MD, _CT) == (
        "track_id: t1, track_name: forbidden dance, "
        "artist_name: aterciopelados, album_name: la pipa de la paz")


def test_empty_field_renders_empty_value():
    assert format_catalog_track_text("t2", _MD, _CT) == (
        "track_id: t2, track_name: song, artist_name: , album_name: al")


def test_missing_track_returns_bare_id():
    assert format_catalog_track_text("nope", _MD, _CT) == "nope"


def test_id_to_metadata_routes_through_canonical():
    db = MusicCatalogDB.__new__(MusicCatalogDB)  # skip __init__ (no dataset load)
    db.metadata_dict, db.corpus_types = _MD, _CT
    for tid in _MD:
        assert db.id_to_metadata(tid) == format_catalog_track_text(tid, _MD, _CT)


def test_history_expander_routes_through_canonical():
    import sys, importlib
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    m = importlib.import_module("build_bi_encoder_training_data")
    for tid in _MD:
        assert m._format_history_music_turn(tid, _MD, _CT) == \
            format_catalog_track_text(tid, _MD, _CT)
