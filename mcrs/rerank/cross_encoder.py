"""K3 — cross-encoder score_fn factory (shared by notebook + serve, so train==serve).

`build_cross_encoder_score_fn` returns a `score_fn(pairs) -> list[float]` for NeuralReranker.
Doc-side TOKEN truncation (preserve the query side, plan §8 / K3 spec §4): each doc is capped to
`max_doc_tokens` using the model's own tokenizer BEFORE forming the pair, so the query is never
truncated and train/serve truncate identically. `truncate_doc_tokens` is pure (injected
encode/decode) for testing; the factory wires the real tokenizer and lazy-imports sentence-transformers.
"""
from __future__ import annotations

from typing import Callable, Optional


def truncate_doc_tokens(encode: Callable[[str], list], decode: Callable[[list], str],
                        doc: str, max_tokens: int) -> str:
    """Cap `doc` to its first `max_tokens` tokens (no-op if already short). Query side untouched."""
    ids = encode(doc)
    if len(ids) <= max_tokens:
        return doc
    return decode(ids[:max_tokens])


def build_cross_encoder_score_fn(model_name: str, device: str = "cuda", max_length: int = 512,
                                 max_doc_tokens: int = 480, batch_size: int = 64,
                                 revision: Optional[str] = None):
    """Load a CrossEncoder and return score_fn(pairs)->list[float] with doc-side token truncation."""
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name, max_length=max_length, device=device, revision=revision)
    tok = ce.tokenizer
    enc = lambda d: tok.encode(d, add_special_tokens=False)

    def score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        capped = [(q, truncate_doc_tokens(enc, tok.decode, d, max_doc_tokens)) for q, d in pairs]
        return [float(s) for s in ce.predict(capped, batch_size=batch_size, show_progress_bar=False)]

    return score_fn
