"""Tests for SID tokenizer/model vocab extension utilities."""
import pytest


def test_make_sid_token_strings_default_shape():
    """Default num_levels=3, codebook_size=256 produces 768 unique strings."""
    from mcrs.sid.vocab import make_sid_token_strings
    toks = make_sid_token_strings(num_levels=3, codebook_size=256)
    assert len(toks) == 768
    assert len(set(toks)) == 768


def test_make_sid_token_strings_format():
    """Strings follow the <SID_L{level}_C{code}> pattern, level then code ordering."""
    from mcrs.sid.vocab import make_sid_token_strings
    toks = make_sid_token_strings(num_levels=2, codebook_size=3)
    assert toks == [
        "<SID_L0_C0>", "<SID_L0_C1>", "<SID_L0_C2>",
        "<SID_L1_C0>", "<SID_L1_C1>", "<SID_L1_C2>",
    ]


def test_add_sid_tokens_to_tokenizer_grows_vocab():
    """Adding 768 tokens to a fresh Qwen tokenizer grows vocab by exactly 768."""
    from transformers import AutoTokenizer
    from mcrs.sid.vocab import add_sid_tokens_to_tokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    base_vocab = len(tok)
    new_tok, n_added = add_sid_tokens_to_tokenizer(tok, num_levels=3, codebook_size=256)
    assert n_added == 768
    assert len(new_tok) == base_vocab + 768


def test_add_sid_tokens_idempotent():
    """Calling add_sid_tokens_to_tokenizer twice does NOT double-add."""
    from transformers import AutoTokenizer
    from mcrs.sid.vocab import add_sid_tokens_to_tokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok, n1 = add_sid_tokens_to_tokenizer(tok, num_levels=3, codebook_size=256)
    tok, n2 = add_sid_tokens_to_tokenizer(tok, num_levels=3, codebook_size=256)
    assert n1 == 768
    assert n2 == 0


def test_added_tokens_are_recognized_as_single_ids():
    """Each <SID_L*_C*> token encodes to exactly one token id (atomic)."""
    from transformers import AutoTokenizer
    from mcrs.sid.vocab import add_sid_tokens_to_tokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok, _ = add_sid_tokens_to_tokenizer(tok, num_levels=3, codebook_size=256)
    for tok_str in ["<SID_L0_C0>", "<SID_L1_C100>", "<SID_L2_C255>"]:
        ids = tok.encode(tok_str, add_special_tokens=False)
        assert len(ids) == 1, f"{tok_str} encoded to {ids} (expected length 1)"
