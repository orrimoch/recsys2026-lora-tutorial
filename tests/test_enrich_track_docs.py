"""Track B — offline LLM track-document enrichment (plan: drastically improve
nDCG@20). The dense channel is weak (recall@100 0.179 < BM25 0.346) because a
0.6B embedder sees only thin metadata. This harness LLM-writes a rich per-track
description (genre/mood/era/instrumentation/similar-artists) to re-embed with a
stronger model. The LLM call is integration (Colab); these test the pure prompt
build + completion parse with no model/API.
"""
from scripts.enrich_track_docs import build_enrich_prompt, parse_enriched_doc

META = {
    "track_id": "uuid-1",
    "track_name": ["Heart-Shaped Box"],
    "artist_name": ["Nirvana"],
    "album_name": ["In Utero"],
    "tag_list": ["grunge", "90s", "alternative rock"],
    "release_date": "1993-08-30",
}


def test_prompt_includes_core_metadata():
    p = build_enrich_prompt(META)
    assert "Heart-Shaped Box" in p and "Nirvana" in p
    assert "grunge" in p  # tags carried so the LLM can ground genre/mood
    assert "1993" in p or "1993-08-30" in p


def test_prompt_handles_missing_fields():
    p = build_enrich_prompt({"track_id": "x", "track_name": ["Untitled"]})
    assert "Untitled" in p  # no crash on absent artist/tags/album/date


def test_parse_strips_preamble_and_collapses_to_paragraph():
    raw = ('Here is the description:\n\n"A raw, anguished 90s grunge anthem by '
           'Nirvana,\ndriven by dynamic loud-quiet guitars."\n')
    out = parse_enriched_doc(raw)
    assert out.startswith("A raw, anguished")
    assert "\n" not in out          # collapsed to a single paragraph
    assert not out.startswith('"')  # surrounding quotes stripped
    assert "Here is the description" not in out


def test_parse_empty_is_empty():
    assert parse_enriched_doc("") == ""
    assert parse_enriched_doc("   \n  ") == ""


def test_parse_caps_length():
    raw = "word " * 1000
    out = parse_enriched_doc(raw, max_chars=600)
    assert len(out) <= 600
