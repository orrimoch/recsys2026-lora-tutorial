"""K3 — cross-encoder wrapper: token-level DOC truncation (preserve the query side)."""
from __future__ import annotations

from mcrs.rerank.cross_encoder import doc_token_budget, truncate_doc_tokens

# fake tokenizer: encode = whitespace split, decode = rejoin (pure, no model needed)
_enc = lambda s: s.split()
_dec = lambda toks: " ".join(toks)


def test_truncate_doc_caps_to_max_tokens():
    assert truncate_doc_tokens(_enc, _dec, "a b c d e f", 3) == "a b c"


def test_truncate_doc_passes_short_docs_through_unchanged():
    assert truncate_doc_tokens(_enc, _dec, "a b", 5) == "a b"
    assert truncate_doc_tokens(_enc, _dec, "", 5) == ""


def test_doc_token_budget_preserves_query_within_max_length():
    assert doc_token_budget(query_tokens=50, max_length=512, max_doc_tokens=480) == 458   # 512-50-4
    assert doc_token_budget(query_tokens=10, max_length=512, max_doc_tokens=480) == 480   # capped by max_doc
    assert doc_token_budget(query_tokens=600, max_length=512, max_doc_tokens=480) == 8    # floor for huge query
