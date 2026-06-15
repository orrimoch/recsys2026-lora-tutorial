"""Tests for per-channel query routing in the RRF union (Option B).

The union normally feeds one query string to every sub-retriever. ColBERT wants a
DIFFERENT (compact) query than BM25/dense. `resolve_sub_queries` lets a sub read its
query from a batch_context key (e.g. 'colbert_query', pre-built at the call site via
build_retrieval_query(mode="compact_colbert")), falling back to the default query when
the key is missing/blank. Channels without a query_key are unaffected (identity).
"""
from mcrs.retrieval_modules.rrf import resolve_sub_queries

_DEF = ["full query A", "full query B", "full query C"]


def test_no_query_key_returns_default_queries_unchanged():
    ctx = [{"colbert_query": "compact A"}, {"colbert_query": "compact B"}, {}]
    assert resolve_sub_queries(_DEF, ctx, None) is _DEF


def test_query_key_swaps_in_context_query_per_row():
    ctx = [{"colbert_query": "compact A"},
           {"colbert_query": "compact B"},
           {"colbert_query": "compact C"}]
    assert resolve_sub_queries(_DEF, ctx, "colbert_query") == [
        "compact A", "compact B", "compact C"]


def test_missing_or_blank_key_falls_back_to_default_per_row():
    ctx = [{"colbert_query": "compact A"},   # use context
           {"colbert_query": "   "},          # blank -> fall back
           {}]                                # missing -> fall back
    assert resolve_sub_queries(_DEF, ctx, "colbert_query") == [
        "compact A", "full query B", "full query C"]


def test_none_batch_context_returns_default():
    assert resolve_sub_queries(_DEF, None, "colbert_query") is _DEF


def test_short_batch_context_falls_back_for_missing_rows():
    ctx = [{"colbert_query": "compact A"}]  # only one row of context
    assert resolve_sub_queries(_DEF, ctx, "colbert_query") == [
        "compact A", "full query B", "full query C"]


def test_only_colbert_spec_carries_query_key_when_compact_enabled():
    # Lock the wiring: with compact routing ON, ONLY colbert_index gets a query_key;
    # every other channel keeps query_key absent/None so it uses the shared query.
    from mcrs.retrieval_modules import _wrrf_union_v1_specs
    specs = _wrrf_union_v1_specs({
        "use_colbert": True, "colbert_compact_query": True, "use_sasrec": True})
    by_type = {s["type"]: s for s in specs}
    assert by_type["colbert_index"]["query_key"] == "colbert_query"
    for t, s in by_type.items():
        if t != "colbert_index":
            assert s.get("query_key") is None, f"{t} must not set query_key"


def test_colbert_query_key_is_none_when_compact_disabled():
    from mcrs.retrieval_modules import _wrrf_union_v1_specs
    specs = {s["type"]: s for s in _wrrf_union_v1_specs({"use_colbert": True})}
    assert specs["colbert_index"].get("query_key") is None
