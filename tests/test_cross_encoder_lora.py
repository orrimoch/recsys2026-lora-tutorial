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
