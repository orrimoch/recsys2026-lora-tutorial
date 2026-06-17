"""K3 — cross-encoder score_fn factory (shared by notebook + serve, so train==serve).

`build_cross_encoder_score_fn` returns a `score_fn(pairs) -> list[float]` for NeuralReranker.
Doc-side TOKEN truncation (preserve the query side, plan §8 / K3 spec §4): each doc is capped to
`max_doc_tokens` using the model's own tokenizer BEFORE forming the pair, so the query is never
truncated and train/serve truncate identically. `truncate_doc_tokens` is pure (injected
encode/decode) for testing; the factory wires the real tokenizer.

Loading is via `transformers` `AutoModelForSequenceClassification` (+ optional PEFT adapter),
NOT sentence-transformers `CrossEncoder`: on some runtimes ST's CrossEncoder wrapper makes
`PeftModel.from_pretrained(ce.model, …).merge_and_unload()` raise `'…ForSequenceClassification'
object has no attribute 'merge_and_unload'`. AutoModel + PeftModel merges cleanly, and scoring via
a direct `model(**feats).logits` forward exactly mirrors the training collate (pair tokenization),
keeping train==serve. The merge is best-effort (kept as an unmerged PeftModel if unavailable);
an explicit assert prevents silently serving the un-fine-tuned base when an adapter fails to load.
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


def _resolve_dtype(dtype: str, bf16_supported: bool) -> str:
    """Resolve `dtype` to a concrete precision. `auto` picks bf16 only where the GPU supports it
    (Ampere+/L4/A100), else fp16 — so the same code runs on a Turing T4 ("G4"), which has no bf16."""
    valid = ("auto", "bf16", "fp16", "fp32")
    if dtype not in valid:
        raise ValueError(f"dtype must be one of {valid}, got {dtype!r}")
    if dtype == "auto":
        return "bf16" if bf16_supported else "fp16"
    return dtype


def build_cross_encoder_score_fn(model_name: str, device: str = "cuda", max_length: int = 2048,
                                 max_doc_tokens: int = 1100, batch_size: int = 64,
                                 revision: Optional[str] = None, dtype: str = "auto",
                                 lora_adapter: Optional[str] = None,
                                 lora_revision: Optional[str] = None):
    """Load a CrossEncoder (+ optional PEFT LoRA adapter) and return score_fn(pairs)->list[float].

    `max_length`/`max_doc_tokens`/`dtype` MUST match the values used at fine-tune time (train==serve).
    `dtype="auto"` picks bf16 where supported else fp16 (works on a T4/G4, which has no bf16).
    `lora_adapter` may be a local dir or a HF Hub repo id; `lora_revision` pins the Hub revision (spec §4.7).
    """
    if dtype not in ("auto", "bf16", "fp16", "fp32"):   # validate BEFORE lazy imports
        raise ValueError(f"dtype must be one of ('auto', 'bf16', 'fp16', 'fp32'), got {dtype!r}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved = _resolve_dtype(dtype, torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    on_cuda = str(device).startswith("cuda")

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, revision=revision)
    if lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_adapter, revision=lora_revision)
        # Guard: never silently serve the un-fine-tuned base if the adapter didn't attach.
        assert hasattr(model, "peft_config"), (
            f"LoRA adapter {lora_adapter!r} did not load onto {model_name!r} — "
            "would serve the base model. Check the adapter path/revision.")
        # merge_and_unload is a speed optimisation; if the runtime lacks it the unmerged
        # PeftModel scores identically (just a bit slower) — so it's best-effort, never fatal.
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model = model.to(device=device, dtype=dtype_map[resolved]) if on_cuda else model.to(device)
    model.eval()

    # cap the counting-encode at max_length so a very long doc doesn't trip the tokenizer's
    # ">model_max_length" warning; we slice to the per-pair budget below anyway.
    enc = lambda s: tok.encode(s, add_special_tokens=False, truncation=True, max_length=max_length)
    use_amp = on_cuda and resolved in ("bf16", "fp16")

    def score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        capped = []
        for q, d in pairs:
            budget = doc_token_budget(len(enc(q)), max_length, max_doc_tokens)
            capped.append((q, truncate_doc_tokens(enc, tok.decode, d, budget)))  # doc fits, query kept
        out: list[float] = []
        for i in range(0, len(capped), batch_size):
            chunk = capped[i:i + batch_size]
            feats = tok([q for q, _ in chunk], [d for _, d in chunk], padding=True,
                        truncation=True, max_length=max_length, return_tensors="pt")
            feats = {k: v.to(device) for k, v in feats.items()}
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=dtype_map[resolved], enabled=use_amp):
                    logits = model(**feats).logits.squeeze(-1)
            out.extend(logits.float().cpu().tolist())
        return out

    return score_fn
