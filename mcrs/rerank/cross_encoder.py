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


def doc_token_budget(query_tokens: int, max_length: int, max_doc_tokens: int,
                     margin: int = 4) -> int:
    """Tokens the DOC may use so (query + doc + specials) fits `max_length` — preserves the query.
    Capped by `max_doc_tokens`, floored at 8 so a very long query still leaves a usable doc."""
    return max(8, min(max_doc_tokens, max_length - query_tokens - margin))


def build_cross_encoder_score_fn(model_name: str, device: str = "cuda", max_length: int = 2048,
                                 max_doc_tokens: int = 1100, batch_size: int = 64,
                                 revision: Optional[str] = None, dtype: str = "bf16",
                                 lora_adapter: Optional[str] = None,
                                 lora_revision: Optional[str] = None):
    """Load a CrossEncoder (+ optional PEFT LoRA adapter) and return score_fn(pairs)->list[float].

    `max_length`/`max_doc_tokens`/`dtype` MUST match the values used at fine-tune time (train==serve).
    `lora_adapter` may be a local dir or a HF Hub repo id; `lora_revision` pins the Hub revision (spec §4.7).
    """
    import torch
    from sentence_transformers import CrossEncoder

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype not in dtype_map:
        raise ValueError(f"dtype must be one of {sorted(dtype_map)}, got {dtype!r}")

    ce = CrossEncoder(model_name, max_length=max_length, device=device, revision=revision)
    if lora_adapter:
        from peft import PeftModel
        ce.model = PeftModel.from_pretrained(ce.model, lora_adapter, revision=lora_revision)
        ce.model = ce.model.merge_and_unload()        # fold LoRA into base for fast inference
    if str(device).startswith("cuda"):
        ce.model = ce.model.to(dtype=dtype_map[dtype])
    tok = ce.tokenizer
    # cap the counting-encode at max_length so a very long doc doesn't trip the tokenizer's
    # ">model_max_length" warning; we slice to the per-pair budget below anyway.
    enc = lambda s: tok.encode(s, add_special_tokens=False, truncation=True, max_length=max_length)

    def score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        capped = []
        for q, d in pairs:
            budget = doc_token_budget(len(enc(q)), max_length, max_doc_tokens)
            capped.append((q, truncate_doc_tokens(enc, tok.decode, d, budget)))  # doc fits, query kept
        return [float(s) for s in ce.predict(capped, batch_size=batch_size, show_progress_bar=False)]

    return score_fn
