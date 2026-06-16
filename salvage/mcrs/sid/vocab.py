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


def untie_embeddings_if_tied(model) -> None:
    """If model has tied input/output embeddings (Qwen2.5-1.5B does), untie them
    by allocating a fresh nn.Linear for lm_head and copying the embedding weights.

    Required before applying LoRA with modules_to_save=["embed_tokens", "lm_head"]
    — otherwise PEFT may save two copies of the same tied weight, which the merger
    can't reconcile back into a single tied matrix.
    """
    import torch
    from torch import nn

    if not getattr(model.config, "tie_word_embeddings", False):
        return
    embed_weight = model.get_input_embeddings().weight.data.clone()
    hidden_size = model.config.hidden_size
    vocab_size = embed_weight.shape[0]
    # Create a fresh linear layer with the same weight values, then attach it.
    new_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    new_lm_head.weight.data = embed_weight
    # Match dtype + device of original lm_head before swapping.
    orig_lm_head = model.get_output_embeddings()
    new_lm_head = new_lm_head.to(
        dtype=orig_lm_head.weight.dtype, device=orig_lm_head.weight.device,
    )
    model.set_output_embeddings(new_lm_head)
    model.config.tie_word_embeddings = False


def extend_model_vocab(model, target_vocab_size: int) -> None:
    """Resize model's input + output embedding matrices to target_vocab_size.

    No-op if model is already at target size. New rows are randomly initialized
    by HF's resize_token_embeddings (which uses the model's init scheme).
    """
    current = model.get_input_embeddings().weight.shape[0]
    if current == target_vocab_size:
        return
    model.resize_token_embeddings(target_vocab_size)


def build_sid_to_token_id_lookup(
    tokenizer, num_levels: int = 3, codebook_size: int = 256,
) -> dict[tuple[int, int], int]:
    """Return {(level, code): token_id} for every SID token in the tokenizer.

    Caller must have already run add_sid_tokens_to_tokenizer on the tokenizer.
    """
    lookup: dict[tuple[int, int], int] = {}
    for level in range(num_levels):
        for code in range(codebook_size):
            tok_str = f"<SID_L{level}_C{code}>"
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"SID token {tok_str} encoded to {len(ids)} ids; "
                    f"add_sid_tokens_to_tokenizer must run first"
                )
            lookup[(level, code)] = ids[0]
    return lookup


def encode_sid_to_token_ids(
    code_1: int, code_2: int, code_3: int,
    *, lookup: dict[tuple[int, int], int],
) -> list[int]:
    """Convert a SID triplet into the 3 token ids the model emits."""
    return [lookup[(0, code_1)], lookup[(1, code_2)], lookup[(2, code_3)]]


def decode_token_ids_to_sid(
    token_ids: list[int],
    *, inverse: dict[int, tuple[int, int]],
) -> tuple[int, int, int]:
    """Convert 3 token ids back to (code_1, code_2, code_3).

    Raises ValueError if any id is not a SID token, or if levels are out of order.
    """
    if len(token_ids) != 3:
        raise ValueError(f"Expected 3 token ids, got {len(token_ids)}")
    codes = [None, None, None]
    for expected_level, tok_id in enumerate(token_ids):
        if tok_id not in inverse:
            raise ValueError(f"Token id {tok_id} is not a SID token")
        level, code = inverse[tok_id]
        if level != expected_level:
            raise ValueError(
                f"Position {expected_level} has SID level {level} (expected {expected_level})"
            )
        codes[expected_level] = code
    return (codes[0], codes[1], codes[2])
