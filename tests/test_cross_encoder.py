"""K3 — cross-encoder wrapper: token-level DOC truncation (preserve the query side)."""
from __future__ import annotations

from mcrs.rerank.cross_encoder import truncate_doc_tokens

# fake tokenizer: encode = whitespace split, decode = rejoin (pure, no model needed)
_enc = lambda s: s.split()
_dec = lambda toks: " ".join(toks)


def test_truncate_doc_caps_to_max_tokens():
    assert truncate_doc_tokens(_enc, _dec, "a b c d e f", 3) == "a b c"


def test_truncate_doc_passes_short_docs_through_unchanged():
    assert truncate_doc_tokens(_enc, _dec, "a b", 5) == "a b"
    assert truncate_doc_tokens(_enc, _dec, "", 5) == ""
