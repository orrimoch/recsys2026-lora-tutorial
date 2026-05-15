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


def test_untie_embeddings_creates_separate_matrices():
    """After untying, embed_tokens.weight and lm_head.weight are not the same Tensor."""
    import torch
    from transformers import AutoModelForCausalLM
    from mcrs.sid.vocab import untie_embeddings_if_tied
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float32
    )
    assert model.config.tie_word_embeddings is True  # sanity
    untie_embeddings_if_tied(model)
    assert model.config.tie_word_embeddings is False
    # Different storage = different .data_ptr()
    assert (
        model.get_input_embeddings().weight.data_ptr()
        != model.get_output_embeddings().weight.data_ptr()
    )


def test_untie_embeddings_preserves_values():
    """Untying copies the tied weight to lm_head — values must be IDENTICAL after untie."""
    import torch
    from transformers import AutoModelForCausalLM
    from mcrs.sid.vocab import untie_embeddings_if_tied
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float32
    )
    pre = model.get_input_embeddings().weight.data.clone()
    untie_embeddings_if_tied(model)
    assert torch.allclose(model.get_input_embeddings().weight, pre)
    assert torch.allclose(model.get_output_embeddings().weight, pre)


def test_extend_model_vocab_grows_to_target():
    """After extending, embed_tokens has exactly len(tokenizer) rows."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mcrs.sid.vocab import (
        add_sid_tokens_to_tokenizer, untie_embeddings_if_tied, extend_model_vocab,
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok, _ = add_sid_tokens_to_tokenizer(tok, 3, 256)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float32
    )
    untie_embeddings_if_tied(model)
    extend_model_vocab(model, len(tok))
    assert model.get_input_embeddings().weight.shape[0] == len(tok)
    assert model.get_output_embeddings().weight.shape[0] == len(tok)


def test_extend_model_vocab_idempotent():
    """Calling extend_model_vocab with a target equal to current size is a no-op."""
    import torch
    from transformers import AutoModelForCausalLM
    from mcrs.sid.vocab import untie_embeddings_if_tied, extend_model_vocab
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float32
    )
    untie_embeddings_if_tied(model)
    n_before = model.get_input_embeddings().weight.shape[0]
    extend_model_vocab(model, n_before)
    assert model.get_input_embeddings().weight.shape[0] == n_before


def test_build_sid_to_token_id_lookup_returns_full_768():
    """Lookup contains every (level, code) tuple in the 768-token grid."""
    from transformers import AutoTokenizer
    from mcrs.sid.vocab import add_sid_tokens_to_tokenizer, build_sid_to_token_id_lookup
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok, _ = add_sid_tokens_to_tokenizer(tok, 3, 256)
    lookup = build_sid_to_token_id_lookup(tok, 3, 256)
    assert len(lookup) == 768
    assert (0, 0) in lookup
    assert (2, 255) in lookup


def test_build_sid_to_token_id_lookup_returns_unique_ids():
    """Every (level, code) maps to a distinct token id."""
    from transformers import AutoTokenizer
    from mcrs.sid.vocab import add_sid_tokens_to_tokenizer, build_sid_to_token_id_lookup
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok, _ = add_sid_tokens_to_tokenizer(tok, 3, 256)
    lookup = build_sid_to_token_id_lookup(tok, 3, 256)
    assert len(set(lookup.values())) == 768


def test_encode_decode_sid_round_trip():
    """encode → decode is the identity for a sample of valid (c1, c2, c3)."""
    from transformers import AutoTokenizer
    from mcrs.sid.vocab import (
        add_sid_tokens_to_tokenizer, build_sid_to_token_id_lookup,
        encode_sid_to_token_ids, decode_token_ids_to_sid,
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok, _ = add_sid_tokens_to_tokenizer(tok, 3, 256)
    lookup = build_sid_to_token_id_lookup(tok, 3, 256)
    inverse = {v: k for k, v in lookup.items()}
    for triple in [(0, 0, 0), (5, 100, 200), (255, 255, 255)]:
        ids = encode_sid_to_token_ids(*triple, lookup=lookup)
        assert len(ids) == 3
        assert decode_token_ids_to_sid(ids, inverse=inverse) == triple


def test_decode_token_ids_to_sid_raises_on_non_sid_id():
    """Passing a non-SID token id raises a clear ValueError (debugging aid)."""
    import pytest
    from transformers import AutoTokenizer
    from mcrs.sid.vocab import (
        add_sid_tokens_to_tokenizer, build_sid_to_token_id_lookup,
        decode_token_ids_to_sid,
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok, _ = add_sid_tokens_to_tokenizer(tok, 3, 256)
    lookup = build_sid_to_token_id_lookup(tok, 3, 256)
    inverse = {v: k for k, v in lookup.items()}
    bos_id = tok.bos_token_id or 1
    with pytest.raises(ValueError, match="not a SID token"):
        decode_token_ids_to_sid([bos_id, bos_id, bos_id], inverse=inverse)
