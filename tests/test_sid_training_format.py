"""Tests for SID generator training-data formatter (W3)."""
import pytest


@pytest.fixture(scope="module")
def tok_and_lookup():
    """Shared fixture: extended tokenizer + SID lookup. Avoids 768-token re-add per test."""
    from transformers import AutoTokenizer
    from mcrs.sid.vocab import add_sid_tokens_to_tokenizer, build_sid_to_token_id_lookup
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok, _ = add_sid_tokens_to_tokenizer(tok, 3, 256)
    lookup = build_sid_to_token_id_lookup(tok, 3, 256)
    return tok, lookup


def test_format_example_returns_input_ids_attention_mask_labels(tok_and_lookup):
    """Function returns three lists of equal length."""
    from mcrs.sid.training_format import format_example_for_training
    tok, lookup = tok_and_lookup
    out = format_example_for_training(
        query="play me something dreamy",
        code_1=10, code_2=20, code_3=30,
        tokenizer=tok, sid_lookup=lookup, max_prompt_len=256,
    )
    assert set(out.keys()) == {"input_ids", "attention_mask", "labels"}
    n = len(out["input_ids"])
    assert len(out["attention_mask"]) == n
    assert len(out["labels"]) == n


def test_format_example_labels_only_sid_positions_unmasked(tok_and_lookup):
    """All labels are -100 except the LAST 3, which are the gold SID token ids."""
    from mcrs.sid.training_format import format_example_for_training
    from mcrs.sid.vocab import encode_sid_to_token_ids
    tok, lookup = tok_and_lookup
    c1, c2, c3 = 10, 20, 30
    out = format_example_for_training(
        query="hello", code_1=c1, code_2=c2, code_3=c3,
        tokenizer=tok, sid_lookup=lookup, max_prompt_len=256,
    )
    expected_sid_ids = encode_sid_to_token_ids(c1, c2, c3, lookup=lookup)
    # Last 3 label positions = SID ids; everything before = -100
    assert out["labels"][-3:] == expected_sid_ids
    for lab in out["labels"][:-3]:
        assert lab == -100, f"Non-SID label position has value {lab}, expected -100"


def test_format_example_input_ids_end_with_sid_tokens(tok_and_lookup):
    """The 3 SID tokens are appended to the END of input_ids."""
    from mcrs.sid.training_format import format_example_for_training
    from mcrs.sid.vocab import encode_sid_to_token_ids
    tok, lookup = tok_and_lookup
    c1, c2, c3 = 5, 100, 200
    out = format_example_for_training(
        query="hello world",
        code_1=c1, code_2=c2, code_3=c3,
        tokenizer=tok, sid_lookup=lookup, max_prompt_len=256,
    )
    expected = encode_sid_to_token_ids(c1, c2, c3, lookup=lookup)
    assert out["input_ids"][-3:] == expected


def test_format_example_truncates_prompt_from_front(tok_and_lookup):
    """If the prompt exceeds max_prompt_len, the FRONT is truncated (preserves recent context)."""
    from mcrs.sid.training_format import format_example_for_training
    tok, lookup = tok_and_lookup
    long_prefix = "old context " * 100  # ~1000 chars, ~250 tokens
    query = long_prefix + " ||LATEST|| more recent stuff"
    out = format_example_for_training(
        query=query, code_1=0, code_2=0, code_3=0,
        tokenizer=tok, sid_lookup=lookup, max_prompt_len=64,
    )
    # The most-recent text must be in the kept tokens (decoded back).
    decoded = tok.decode(out["input_ids"][:-3])  # drop the 3 SID tokens
    assert "LATEST" in decoded
    # Front should be truncated — early "old context" repetitions should be missing.
    assert decoded.count("old context") < 100


def test_format_example_attention_mask_all_ones(tok_and_lookup):
    """attention_mask is all 1s for a single example (padding happens in collate)."""
    from mcrs.sid.training_format import format_example_for_training
    tok, lookup = tok_and_lookup
    out = format_example_for_training(
        query="x", code_1=0, code_2=0, code_3=0,
        tokenizer=tok, sid_lookup=lookup, max_prompt_len=64,
    )
    assert all(m == 1 for m in out["attention_mask"])


def test_collate_training_batch_left_pads_to_longest():
    """Batch padded so all rows have len == max(len in batch); padding on the LEFT."""
    from mcrs.sid.training_format import collate_training_batch
    examples = [
        {"input_ids": [1, 2, 3, 100, 101, 102],
         "attention_mask": [1, 1, 1, 1, 1, 1],
         "labels": [-100, -100, -100, 100, 101, 102]},
        {"input_ids": [4, 5, 200, 201, 202],
         "attention_mask": [1, 1, 1, 1, 1],
         "labels": [-100, -100, 200, 201, 202]},
    ]
    out = collate_training_batch(examples, pad_token_id=0)
    # Both should be length 6 (longest); shorter row padded on the LEFT.
    assert out["input_ids"].shape == (2, 6)
    # Row 0: unchanged.
    assert out["input_ids"][0].tolist() == [1, 2, 3, 100, 101, 102]
    # Row 1: padded with one 0 on the left.
    assert out["input_ids"][1].tolist() == [0, 4, 5, 200, 201, 202]
    # attention_mask matches.
    assert out["attention_mask"][1].tolist() == [0, 1, 1, 1, 1, 1]
    # labels padding uses -100 (don't compute loss on padding positions).
    assert out["labels"][1].tolist() == [-100, -100, -100, 200, 201, 202]
