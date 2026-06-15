"""EXP-217: curated-tag ColBERT doc enrichment (doc-side, deterministic).

Raw tag_list is noisy folksonomy + long (mean +111 tok). These pure helpers curate it
to a short, clean, high-signal suffix so enriched docs stay ~50-60 tok (d_len modest) and
inject genre/mood vocabulary (not junk) into ColBERT's MaxSim surface. SHARED by the train-
data builder AND the index builder so the doc text is byte-identical (doc-side parity).
"""
from mcrs.retrieval_modules.colbert_late import (
    build_tag_vocab, curate_tags, enrich_doc_text, colbert_doc_text,
)


# ---- build_tag_vocab: catalog-frequency filter (the noise killer) ----
def test_build_tag_vocab_counts_normalized_frequencies():
    rows = [["Jazz", "chill"], ["jazz", "lo-fi"], ["JAZZ"]]
    vocab = build_tag_vocab(rows, min_freq=1)
    assert vocab["jazz"] == 3          # case-folded count
    assert vocab["chill"] == 1


def test_build_tag_vocab_drops_below_min_freq():
    rows = [["jazz", "goeiepoep"], ["jazz", "chill"], ["jazz", "chill"]]
    vocab = build_tag_vocab(rows, min_freq=2)
    assert "jazz" in vocab and "chill" in vocab
    assert "goeiepoep" not in vocab     # freq 1 < 2 -> folksonomy junk dropped


def test_build_tag_vocab_ignores_blank_tags():
    vocab = build_tag_vocab([["", "  ", "jazz"]], min_freq=1)
    assert set(vocab) == {"jazz"}


# ---- curate_tags: per-track dedup + top-K by catalog frequency ----
_VOCAB = {"jazz": 1000, "chill": 800, "ambient": 500, "lo-fi": 300, "downtempo": 100}


def test_curate_tags_keeps_only_vocab_tags_sorted_by_frequency():
    out = curate_tags(["lo-fi", "jazz", "goeiepoep", "chill"], _VOCAB, top_k=10)
    assert out == ["jazz", "chill", "lo-fi"]   # vocab-only, freq-desc; junk dropped


def test_curate_tags_dedups_case_variants():
    out = curate_tags(["Jazz", "jazz", "JAZZ", "chill"], _VOCAB, top_k=10)
    assert out == ["jazz", "chill"]


def test_curate_tags_caps_at_top_k():
    out = curate_tags(["jazz", "chill", "ambient", "lo-fi", "downtempo"], _VOCAB, top_k=2)
    assert out == ["jazz", "chill"]            # top-2 by frequency


def test_curate_tags_empty_when_nothing_in_vocab():
    assert curate_tags(["goeiepoep", "welle work"], _VOCAB, top_k=10) == []


# ---- enrich_doc_text: assemble the final doc ----
def test_enrich_doc_text_appends_tags_suffix():
    out = enrich_doc_text("Song X, Artist Y, Album Z", ["jazz", "chill"])
    assert out == "Song X, Artist Y, Album Z, tags: jazz, chill"


def test_enrich_doc_text_no_tags_returns_base_unchanged():
    base = "Song X, Artist Y, Album Z"
    assert enrich_doc_text(base, []) == base


# ---- colbert_doc_text: the per-track convenience SHARED by both builders ----
def test_colbert_doc_text_composes_stripped_base_and_curated_tags():
    meta = {"t1": {"tag_list": ["jazz", "goeiepoep", "chill"]}}
    vocab = {"jazz": 1000, "chill": 800}
    id2m = lambda tid: "track_id: t1, Song X, Artist Y, Album Z"
    out = colbert_doc_text("t1", id2m, meta, vocab, top_k=10)
    assert out == "Song X, Artist Y, Album Z, tags: jazz, chill"


def test_colbert_doc_text_missing_track_falls_back_to_bare_doc():
    id2m = lambda tid: "track_id: t9, Song, Artist, Album"
    out = colbert_doc_text("t9", id2m, {}, {"jazz": 5}, top_k=10)
    assert out == "Song, Artist, Album"     # no row -> no tags, just stripped base
