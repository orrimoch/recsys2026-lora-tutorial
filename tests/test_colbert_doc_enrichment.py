"""EXP-217: curated-tag ColBERT doc enrichment (doc-side, deterministic).

Raw tag_list is noisy folksonomy + long (mean +111 tok). These pure helpers curate it
to a short, clean, high-signal suffix so enriched docs stay ~50-60 tok (d_len modest) and
inject genre/mood vocabulary (not junk) into ColBERT's MaxSim surface. SHARED by the train-
data builder AND the index builder so the doc text is byte-identical (doc-side parity).
"""
from mcrs.retrieval_modules.colbert_late import (
    build_tag_vocab, curate_tags, enrich_doc_text, colbert_doc_text,
    make_colbert_doc_text_fn,
)


class _StubDB:
    """Duck-types the bits of MusicCatalogDB the factory needs."""
    def __init__(self, metadata_dict):
        self.metadata_dict = metadata_dict
    def id_to_metadata(self, tid):
        return f"track_id: {tid}, name {tid}"


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


def test_build_tag_vocab_counts_document_frequency_not_occurrences():
    # I2: a track listing a tag twice must count ONCE toward catalog frequency
    # (document frequency = "appears on N tracks", not raw occurrence count).
    vocab = build_tag_vocab([["jazz", "jazz", "Jazz"], ["jazz"]], min_freq=1)
    assert vocab["jazz"] == 2          # 2 tracks, not 4 occurrences


# ---- make_colbert_doc_text_fn: the SHARED factory (train + index + nb82 parity) ----
def test_make_doc_text_fn_two_constructions_are_byte_identical():
    db = _StubDB({"t1": {"tag_list": ["jazz", "chill", "goeiepoep"]},
                  "t2": {"tag_list": ["jazz", "pop"]}})
    fn1, r1 = make_colbert_doc_text_fn(db, enrich_tags=True, tag_min_freq=2, tag_top_k=10)
    fn2, r2 = make_colbert_doc_text_fn(db, enrich_tags=True, tag_min_freq=2, tag_top_k=10)
    assert fn1("t1") == fn2("t1")      # the train-vs-index parity invariant
    assert r1 == r2


def test_make_doc_text_fn_enriches_and_freq_filters():
    db = _StubDB({"t1": {"tag_list": ["jazz", "chill", "goeiepoep"]},
                  "t2": {"tag_list": ["jazz", "pop"]}})
    fn, recipe = make_colbert_doc_text_fn(db, enrich_tags=True, tag_min_freq=2, tag_top_k=10)
    # jazz appears on 2 tracks (kept @min_freq=2); chill/pop/goeiepoep freq 1 (dropped)
    assert fn("t1") == "name t1, tags: jazz"
    assert recipe["enrich"] is True and recipe["tag_min_freq"] == 2 and recipe["tag_top_k"] == 10


def test_make_doc_text_fn_disabled_returns_bare_stripped_doc():
    db = _StubDB({"t1": {"tag_list": ["jazz"]}})
    fn, recipe = make_colbert_doc_text_fn(db, enrich_tags=False)
    assert fn("t1") == "name t1"       # stripped base, no tags
    assert recipe["enrich"] is False


def test_colbert_doc_text_present_row_without_taglist_is_bare():
    # M2: present row but tag_list None -> no tags, just stripped base.
    out = colbert_doc_text("t1", lambda t: "track_id: t1, name", {"t1": {}}, {"jazz": 5}, 10)
    assert out == "name"


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
