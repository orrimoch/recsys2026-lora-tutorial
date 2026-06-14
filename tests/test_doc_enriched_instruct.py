"""EXP-016 wiring: the enriched-doc dense channel must pick the query-instruct
prefix BY MODEL FAMILY — e5 (asymmetric) needs the e5 prefix, NOT the Qwen3 one
(the original channel hardcoded Qwen3, which would mis-encode e5 queries)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "music-crs-baselines"))
from mcrs.retrieval_modules import (  # noqa: E402
    _doc_enriched_instruct, E5_MUSIC_INSTRUCT, QWEN3_MUSIC_INSTRUCT,
)


def test_disabled_returns_no_instruct():
    assert _doc_enriched_instruct("intfloat/multilingual-e5-large-instruct", False) == (None, "raw")
    assert _doc_enriched_instruct("BAAI/bge-m3", False) == (None, "raw")


def test_e5_model_gets_e5_prefix():
    instr, label = _doc_enriched_instruct("intfloat/multilingual-e5-large-instruct", True)
    assert instr == E5_MUSIC_INSTRUCT and label == "e5-instruct-music-v1"


def test_qwen3_model_gets_qwen3_prefix():
    instr, label = _doc_enriched_instruct("Qwen/Qwen3-Embedding-0.6B", True)
    assert instr == QWEN3_MUSIC_INSTRUCT and label == "instruct-music-v1"


def test_e5_detection_is_case_insensitive():
    assert _doc_enriched_instruct("intfloat/Multilingual-E5-Large-Instruct", True)[0] == E5_MUSIC_INSTRUCT
