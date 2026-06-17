# tests/test_cross_encoder_lora.py
import inspect
from mcrs.rerank import cross_encoder as ce

def test_signature_defaults_pin_2048_and_lora_and_dtype():
    sig = inspect.signature(ce.build_cross_encoder_score_fn)
    assert sig.parameters["max_length"].default == 2048
    assert "lora_adapter" in sig.parameters and sig.parameters["lora_adapter"].default is None
    assert sig.parameters["dtype"].default == "bf16"

def test_doc_token_budget_unchanged_helper():
    # budget math still preserves the query at the new ceiling
    assert ce.doc_token_budget(query_tokens=200, max_length=2048, max_doc_tokens=1100) == 1100
    assert ce.doc_token_budget(query_tokens=1200, max_length=2048, max_doc_tokens=1100) == 844


# ── new edge-case / property / failure-mode tests ──────────────────────────

def test_doc_token_budget_floors_at_8_when_query_exceeds_max_length():
    """When query_tokens > max_length, available room is negative; result is floored at 8."""
    # max_length=512, query_tokens=600, margin=4 → 512-600-4 = -92 → max(8, ...) = 8
    assert ce.doc_token_budget(query_tokens=600, max_length=512, max_doc_tokens=1100) == 8


def test_doc_token_budget_exactly_at_boundary_floors_at_8():
    """When query_tokens == max_length the room is -margin (< 8), still floored at 8."""
    assert ce.doc_token_budget(query_tokens=512, max_length=512, max_doc_tokens=1100) == 8


def test_doc_token_budget_small_max_doc_tokens_caps_below_available_room():
    """A small max_doc_tokens caps the result even when there is plenty of room."""
    # room = 2048 - 10 - 4 = 2034; but max_doc_tokens=50 caps it
    result = ce.doc_token_budget(query_tokens=10, max_length=2048, max_doc_tokens=50)
    assert result == 50


def test_doc_token_budget_margin_subtracted():
    """The margin parameter is subtracted from the available room."""
    # With default margin=4
    room_default = ce.doc_token_budget(query_tokens=100, max_length=2048, max_doc_tokens=5000)
    # With margin=10
    room_large_margin = ce.doc_token_budget(query_tokens=100, max_length=2048, max_doc_tokens=5000, margin=10)
    assert room_default - room_large_margin == 6  # 10 - 4 = 6 fewer tokens


def test_truncate_doc_tokens_no_op_when_short():
    """truncate_doc_tokens returns the doc unchanged when it is within max_tokens."""
    encode = str.split      # "a b c" -> ["a", "b", "c"]
    decode = " ".join
    doc = "a b c d e"       # 5 tokens
    result = ce.truncate_doc_tokens(encode, decode, doc, max_tokens=10)
    assert result == doc


def test_truncate_doc_tokens_no_op_exact_boundary():
    """truncate_doc_tokens is a no-op when len(ids) == max_tokens (equal, not just less)."""
    encode = str.split
    decode = " ".join
    doc = "one two three"   # 3 tokens
    result = ce.truncate_doc_tokens(encode, decode, doc, max_tokens=3)
    assert result == doc


def test_truncate_doc_tokens_truncates_via_injected_decode():
    """truncate_doc_tokens slices via the injected encode/decode when the doc exceeds max_tokens."""
    encode = str.split
    decode = " ".join
    doc = "the quick brown fox jumps over the lazy dog"  # 9 tokens
    result = ce.truncate_doc_tokens(encode, decode, doc, max_tokens=4)
    assert result == "the quick brown fox"


def test_truncate_doc_tokens_to_one_token():
    """truncate_doc_tokens truncates to a single token when max_tokens=1."""
    encode = str.split
    decode = " ".join
    doc = "alpha beta gamma"
    result = ce.truncate_doc_tokens(encode, decode, doc, max_tokens=1)
    assert result == "alpha"


def test_invalid_dtype_raises_value_error():
    """build_cross_encoder_score_fn with dtype='int8' raises ValueError before any model load.

    In cross_encoder.py the dtype validation (line 44-45) is placed AFTER the lazy imports of
    torch and sentence_transformers (lines 40-41). This means the ValueError can only be reached
    if both packages are installed. When sentence_transformers is absent the function raises
    ModuleNotFoundError before the dtype check — in that case we skip the assertion.
    """
    import pytest
    import sys

    # Check that both lazy imports required before the dtype check are present.
    def _importable(name: str) -> bool:
        if name in sys.modules:
            return True
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    if not _importable("torch") or not _importable("sentence_transformers"):
        pytest.skip(
            "torch or sentence_transformers not installed — dtype check unreachable "
            "without them (ModuleNotFoundError fires first; this is a source-level ordering "
            "issue: the dtype guard in build_cross_encoder_score_fn comes after the imports, "
            "not before)"
        )

    with pytest.raises(ValueError, match="dtype must be one of"):
        ce.build_cross_encoder_score_fn("dummy-model-name", dtype="int8")
