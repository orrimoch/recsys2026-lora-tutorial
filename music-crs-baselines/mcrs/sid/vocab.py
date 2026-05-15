"""Tokenizer + model vocabulary extension for SID generator (W3).

W3 fine-tunes Qwen2.5-1.5B-Instruct to emit 3 SID tokens per query. To do that
we expand the tokenizer with 768 new special tokens (3 levels × 256 codes),
resize the model embedding matrix, and (if tied) untangle embed_tokens from
lm_head so PEFT's modules_to_save can save them as separate trainable matrices.

All functions are pure (no side effects beyond mutating their input objects in
place, which the caller passes ownership of). Each is independently TDD-able.
"""
from __future__ import annotations

from typing import Optional


def make_sid_token_strings(num_levels: int = 3, codebook_size: int = 256) -> list[str]:
    """Return all SID token strings in level-major order.

    Format: <SID_L{level}_C{code}> for level in [0, num_levels) and code in
    [0, codebook_size). Total = num_levels * codebook_size strings.
    """
    return [
        f"<SID_L{level}_C{code}>"
        for level in range(num_levels)
        for code in range(codebook_size)
    ]


def add_sid_tokens_to_tokenizer(
    tokenizer,
    num_levels: int = 3,
    codebook_size: int = 256,
) -> tuple[object, int]:
    """Add SID special tokens to tokenizer; return (tokenizer, n_added).

    Idempotent — tokens already in the vocab are skipped, so a second call adds 0.
    """
    desired = make_sid_token_strings(num_levels, codebook_size)
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in desired if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return tokenizer, len(to_add)
