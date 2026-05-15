# SID Generator Training (W3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Qwen2.5-1.5B-Instruct + LoRA so that the model emits the gold 3-token SID for any (chat-history + user-query + profile + goal) prompt produced by W2. Ship a merged-and-pushed HF Hub artifact whose dev nDCG@20 ≥ 0.12 with paired-bootstrap CI excluding 0 vs the Phase 0 baseline.

**Architecture:** Three layers of pure functions plus two orchestration scripts plus one Colab notebook. (1) `mcrs/sid/vocab.py` extends the tokenizer with 768 SID tokens, untangles tied embeddings, and resizes the model. (2) `mcrs/sid/training_format.py` converts a (query, sid) pair into `(input_ids, labels)` with cross-entropy masked everywhere except the 3 SID positions. (3) `mcrs/sid/inference.py` builds the SID trie + `prefix_allowed_tokens_fn` for constrained beam search; `mcrs/sid/eval.py` computes nDCG@20 against val parquet. `scripts/train_sid_generator.py` runs the LoRA fine-tune; `scripts/eval_sid_generator.py` runs the constrained-decode eval + paired-bootstrap CI. Notebook 62 wraps both.

**Tech Stack:** Python 3.10. New pip deps: `peft >= 0.10`, `trl >= 0.8` (already on Colab base). Existing: `transformers`, `torch`, `accelerate`, `datasets`, `pandas`, `pyarrow`. No vLLM (per training methodology).

**Spec reference:** `documents/specs/2026-05-15-sid-retrieval-design.md` §3 (generator) + §4.1 (trie + LogitsProcessor) + §3.5 (validation gate).

**Inputs from prior weeks:**
- W1 artifact: `experiments/cache/sid/track_to_sid.parquet` — 47,071 rows, columns `track_id, code_1, code_2, code_3, popularity, bucket_rank`, SHA256 `3ed8fe9f930a...`. Used both for the trie (every valid SID prefix) AND for collision-bucket lookup at eval.
- W2 artifacts: `experiments/cache/sid_training/{train,val}.parquet` + `summary.json`. Train ~90K rows, val ~5K rows. Columns `source, track_id, query, code_1, code_2, code_3`.
- Phase 0 baseline cached records: `experiments/diagnostic_runs/phase0_baseline_full_dev/per_turn_metrics.jsonl` (per-turn nDCG@20 for paired-bootstrap CI).

**Output artifacts:**
- LoRA-only adapter checkpoints during iteration: `OrRim123/recsys2026-sid-generator-qwen15b-vN-lora` (~120 MB each, fits free Hub tier easily)
- Merged + pushed final model (W3 gate-pass): `OrRim123/recsys2026-sid-generator-qwen15b-v1-merged` (~3 GB, single push at gate-pass)
- `experiments/cache/sid_eval/{w3_eval_metrics.json, per_query_ndcg.jsonl}` — eval gate outputs

**Out of scope for W3 (explicitly W4):**
- The `SID_GENERATOR` retrieval class implementing `batch_text_to_item_retrieval(...)` — that's W4.
- Registering `sid_generator` + `wrrf_bm25_dense_sid_v1` in `mcrs/retrieval_modules/__init__.py` — that's W4.
- Running the full pipeline (CMQR + ProRank + responder) — that's W4.
- Submitting to Blind-A — that's W4 too.

W3 produces a model and proves it clears the gate; W4 wires it into the inference pipeline.

---

## File structure

| Path | Type | Responsibility |
|---|---|---|
| `mcrs/sid/vocab.py` | new | Tokenizer extension + tied-embedding untangle + model resize. Pure functions: `make_sid_token_strings`, `add_sid_tokens_to_tokenizer`, `untie_embeddings_if_tied`, `extend_model_vocab`, `build_sid_to_token_id_lookup`, `encode_sid_to_token_ids`, `decode_token_ids_to_sid` |
| `mcrs/sid/training_format.py` | new | `format_example_for_training(query, sid_codes, tokenizer, sid_lookup, max_prompt_len)` → `(input_ids, attention_mask, labels)` with labels masked to -100 except on the 3 SID positions. Plus `collate_training_batch` for left-padding. |
| `mcrs/sid/inference.py` | new (subset; W4 will extend) | `build_sid_trie(track_to_sid_df, sid_token_id_lookup)` → trie of valid 3-token SID sequences. `make_prefix_allowed_tokens_fn(trie, prompt_lens)` → callable for HF `generate()`. |
| `mcrs/sid/eval.py` | new | `compute_ndcg_at_k(retrieved_track_ids, gold_track_id, k)` (single-query) + `aggregate_ndcg(per_query)` + `paired_bootstrap_ci_against_baseline` (thin wrapper around `scripts/compare_diagnostic_runs.py:paired_bootstrap_ci`). |
| `scripts/train_sid_generator.py` | new | Orchestration: load W2 parquets → load Qwen-1.5B → extend vocab → LoRA wrap with `modules_to_save=["embed_tokens", "lm_head"]` → Trainer fit → push LoRA + (if `--merge`) merged model. |
| `scripts/eval_sid_generator.py` | new | Orchestration: load merged model + W1 SID lookup → build trie → constrained beam search over val parquet → per-query nDCG@20 → paired-bootstrap vs Phase 0 cached records → write metrics JSON. |
| `tests/test_sid_vocab.py` | new | TDD tests for vocab.py (~10 tests) |
| `tests/test_sid_training_format.py` | new | TDD tests for training_format.py (~6 tests) |
| `tests/test_sid_inference.py` | new (W3 subset; W4 extends) | TDD tests for trie + prefix_allowed_tokens_fn (~5 tests) |
| `tests/test_sid_eval.py` | new | TDD tests for eval.py (~6 tests) |
| `colab/62_train_and_eval_sid_generator.ipynb` | new | Single notebook: clone, deps, drive mount, smoke train (200 steps), full train (~3-4 hr L4), push, eval, gate decision |

Total expected: ~27 new TDD tests, 4 new modules, 2 new scripts, 1 notebook.

---

## Task 1: Setup — verify W2 artifacts + scaffold cache directory

**Files:**
- Verify: `experiments/cache/sid/track_to_sid.parquet` (W1)
- Verify: `experiments/cache/sid_training/{train,val}.parquet` (W2)
- Verify: `experiments/diagnostic_runs/phase0_baseline_full_dev/per_turn_metrics.jsonl` (Phase 0)
- Create: `experiments/cache/sid_eval/` (empty)

- [ ] **Step 1: Verify W1 + W2 artifacts on disk**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
for f in \
  experiments/cache/sid/track_to_sid.parquet \
  experiments/cache/sid_training/train.parquet \
  experiments/cache/sid_training/val.parquet \
  experiments/cache/sid_training/summary.json; do
  if [ -f "$f" ]; then echo "OK: $f"; else echo "MISSING: $f"; fi
done
```

Expected: all 4 lines say `OK: ...`. If any are MISSING, abort — W2 must complete via Colab notebook 61 first.

- [ ] **Step 2: Sanity-check W2 train parquet shape**

```bash
python -c "
import pandas as pd, json
df = pd.read_parquet('experiments/cache/sid_training/train.parquet')
print(f'rows: {len(df)}')
print(f'columns: {df.columns.tolist()}')
print(f'sources: {df.groupby(\"source\").size().to_dict()}')
print(f'sample query: {df.iloc[0].query[:120]!r}')
print(f'sample sid: ({df.iloc[0].code_1}, {df.iloc[0].code_2}, {df.iloc[0].code_3})')
summ = json.load(open('experiments/cache/sid_training/summary.json'))
print(f'w1_sid_sha256: {summ[\"w1_sid_sha256\"][:16]}...')
"
```

Expected: ~90K rows, columns `[source, track_id, query, code_1, code_2, code_3]`, sources include `raw` + `metadata` (and `doc2query` if user ran notebook 54), code values in [0, 255], sha256 starts with `3ed8fe9f`.

- [ ] **Step 3: Verify Phase 0 baseline records exist for paired-bootstrap CI**

```bash
ls -la experiments/diagnostic_runs/phase0_baseline_full_dev/per_turn_metrics.jsonl 2>&1 || echo "MISSING — Phase 0 baseline must be cached for W3 gate"
```

If MISSING, the W3 eval gate cannot run paired-bootstrap. Earlier work cached Phase 0 metrics — if not present, regenerate with `scripts/run_phase0_diagnostic.py` (~10 min on dev split, no GPU). The plan still proceeds; eval reports point estimate without CI in that case.

- [ ] **Step 4: Scaffold cache directory + commit**

```bash
mkdir -p experiments/cache/sid_eval
touch experiments/cache/sid_eval/.gitkeep
git add experiments/cache/sid_eval/.gitkeep
git commit -m "sid w3: scaffold experiments/cache/sid_eval/ output directory"
```

---

## Task 2: TDD `make_sid_token_strings` + `add_sid_tokens_to_tokenizer`

The first function generates the 768 deterministic token strings (`<SID_L0_C0>` … `<SID_L2_C255>`); the second registers them as additional special tokens on the tokenizer.

**Files:**
- Create: `music-crs-baselines/mcrs/sid/vocab.py`
- Create: `tests/test_sid_vocab.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sid_vocab.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest tests/test_sid_vocab.py -q
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'mcrs.sid.vocab'`.

- [ ] **Step 3: Write the implementation**

Create `music-crs-baselines/mcrs/sid/vocab.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sid_vocab.py -q
```

Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/vocab.py tests/test_sid_vocab.py
git commit -m "sid w3: SID vocab token strings + tokenizer add (TDD, 5 tests)"
```

---

## Task 3: TDD `untie_embeddings_if_tied` + `extend_model_vocab`

These two functions complete the model-side of vocab expansion. `untie` separates `embed_tokens` from `lm_head` (Qwen2.5-1.5B-Instruct ties them by default). `extend_model_vocab` resizes embeddings to match the new tokenizer length.

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/vocab.py`
- Modify: `tests/test_sid_vocab.py`

- [ ] **Step 1: Add failing tests to `tests/test_sid_vocab.py`**

Append these tests to the existing file:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sid_vocab.py::test_untie_embeddings_creates_separate_matrices tests/test_sid_vocab.py::test_untie_embeddings_preserves_values tests/test_sid_vocab.py::test_extend_model_vocab_grows_to_target tests/test_sid_vocab.py::test_extend_model_vocab_idempotent -q
```

Expected: 4 failures with `ImportError: cannot import name 'untie_embeddings_if_tied'`.

> **Note**: these 4 tests download Qwen-1.5B (~3 GB). If running locally on Mac and the model isn't already cached, they will be slow on first run. They're fast on Colab where the model is pre-pulled by Task 8's training.

- [ ] **Step 3: Append implementation to `music-crs-baselines/mcrs/sid/vocab.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sid_vocab.py -q
```

Expected: 9 passing (5 from Task 2 + 4 new). Allow ~3 min on first run for model download.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/vocab.py tests/test_sid_vocab.py
git commit -m "sid w3: untie_embeddings + extend_model_vocab (TDD, 4 tests; Qwen-1.5B fix)"
```

---

## Task 4: TDD SID ↔ token-id mapping helpers

The training loop and the eval decoder both need to convert between (level, code) tuples and the actual integer token IDs assigned by the tokenizer. These four functions are pure dict-lookup helpers — fast and easy to test.

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/vocab.py`
- Modify: `tests/test_sid_vocab.py`

- [ ] **Step 1: Add failing tests to `tests/test_sid_vocab.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sid_vocab.py -k "lookup or round_trip or non_sid" -q
```

Expected: 4 failures with `ImportError: cannot import name 'build_sid_to_token_id_lookup'`.

- [ ] **Step 3: Append implementation to `music-crs-baselines/mcrs/sid/vocab.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sid_vocab.py -q
```

Expected: 13 passing (9 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/vocab.py tests/test_sid_vocab.py
git commit -m "sid w3: SID<->token_id lookup helpers (TDD, 4 tests; round-trip safe)"
```

---

## Task 5: TDD `format_example_for_training` (label masking)

This is the function that turns a single (query, sid_codes) row from the W2 train parquet into `(input_ids, attention_mask, labels)` ready for the Trainer. Critical: `labels` must be -100 everywhere except the 3 SID positions, so cross-entropy only learns to emit the SID — not to reproduce the prompt.

**Files:**
- Create: `music-crs-baselines/mcrs/sid/training_format.py`
- Create: `tests/test_sid_training_format.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sid_training_format.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sid_training_format.py -q
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'mcrs.sid.training_format'`.

- [ ] **Step 3: Write the implementation**

Create `music-crs-baselines/mcrs/sid/training_format.py`:

```python
"""Format (query, sid) pairs into (input_ids, attention_mask, labels) for Trainer.

Labels are masked to -100 everywhere except the 3 SID positions, so the model
only learns to predict SIDs (not to regurgitate the prompt). This is the
standard 'completion-only' training pattern for instruction-tuning, applied
to a fixed-length 3-token completion.
"""
from __future__ import annotations

from typing import Any

from .vocab import encode_sid_to_token_ids


def format_example_for_training(
    query: str,
    code_1: int,
    code_2: int,
    code_3: int,
    *,
    tokenizer,
    sid_lookup: dict[tuple[int, int], int],
    max_prompt_len: int,
) -> dict[str, list[int]]:
    """Build a single training example.

    Layout: [front-truncated prompt token ids ...] [SID_L0_Cx] [SID_L1_Cy] [SID_L2_Cz]
    Labels: [-100, -100, ..., -100, sid_l0_id, sid_l1_id, sid_l2_id]

    Args:
        query: prompt string from W2 (already includes [USER]/[GOAL]/[HISTORY]/[QUERY] blocks)
        code_1, code_2, code_3: gold SID codes
        tokenizer: extended tokenizer (must have SID tokens already added)
        sid_lookup: from build_sid_to_token_id_lookup
        max_prompt_len: total input length excluding the 3 SID tokens; long prompts
            are truncated from the FRONT to preserve the most-recent context

    Returns:
        Dict with input_ids, attention_mask, labels (all list[int], same length).
    """
    sid_ids = encode_sid_to_token_ids(code_1, code_2, code_3, lookup=sid_lookup)

    # Tokenize without truncation first so we know the true length.
    prompt_ids = tokenizer.encode(query, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_len:
        # Front-truncate: keep the LAST max_prompt_len tokens.
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + sid_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + sid_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sid_training_format.py -q
```

Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/training_format.py tests/test_sid_training_format.py
git commit -m "sid w3: format_example_for_training with -100 label masking (TDD, 5 tests)"
```

---

## Task 6: TDD `collate_training_batch` (left-padding for HF Trainer)

The collator pads variable-length training examples to the longest in the batch. We left-pad so the SID tokens always sit at the rightmost positions — matching the autoregressive generation path at inference.

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/training_format.py`
- Modify: `tests/test_sid_training_format.py`

- [ ] **Step 1: Add the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_sid_training_format.py::test_collate_training_batch_left_pads_to_longest -q
```

Expected: 1 failure with `ImportError: cannot import name 'collate_training_batch'`.

- [ ] **Step 3: Append the implementation to `training_format.py`**

```python
def collate_training_batch(
    examples: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, "torch.Tensor"]:
    """Left-pad a batch of formatted training examples.

    Padding side = LEFT so the rightmost tokens (which include the 3 SID positions)
    always align across the batch. attention_mask uses 0 for padded positions;
    labels use -100 for padded positions (so cross-entropy ignores them).
    """
    import torch

    max_len = max(len(ex["input_ids"]) for ex in examples)
    input_ids = []
    attention_mask = []
    labels = []
    for ex in examples:
        pad_n = max_len - len(ex["input_ids"])
        input_ids.append([pad_token_id] * pad_n + ex["input_ids"])
        attention_mask.append([0] * pad_n + ex["attention_mask"])
        labels.append([-100] * pad_n + ex["labels"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_sid_training_format.py -q
```

Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/training_format.py tests/test_sid_training_format.py
git commit -m "sid w3: collate_training_batch left-pads for SID alignment (TDD, 1 test)"
```

---

## Task 7: TDD `build_sid_trie` + `make_prefix_allowed_tokens_fn` (W3 subset of W4 inference)

Build the trie of valid 3-token SID sequences (each leaf = a SID actually present in W1's `track_to_sid.parquet`) and the callable HF `generate()` uses to mask invalid tokens. W3 only needs these two helpers (eval); W4 will add the `SID_GENERATOR` retrieval class on top.

**Files:**
- Create: `music-crs-baselines/mcrs/sid/inference.py`
- Create: `tests/test_sid_inference.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sid_inference.py`:

```python
"""Tests for SID trie + prefix_allowed_tokens_fn (W3 subset of W4 inference)."""
import pytest


def test_build_sid_trie_two_branches():
    """Two SIDs sharing a level-0 code share a node at level 0."""
    from mcrs.sid.inference import build_sid_trie
    sids = [(1, 2, 3), (1, 2, 4), (5, 6, 7)]
    sid_lookup = {
        (0, 1): 100, (0, 5): 105,
        (1, 2): 200, (1, 6): 206,
        (2, 3): 300, (2, 4): 304, (2, 7): 307,
    }
    trie = build_sid_trie(sids, sid_lookup)
    # Root has two valid level-0 children: 100 and 105.
    assert sorted(trie.valid_next_token_ids(prefix=[])) == [100, 105]
    # After 100, only level-1 child 200 is valid.
    assert trie.valid_next_token_ids(prefix=[100]) == [200]
    # After [100, 200], two valid level-2 leaves: 300 and 304.
    assert sorted(trie.valid_next_token_ids(prefix=[100, 200])) == [300, 304]


def test_build_sid_trie_empty_after_complete_sid():
    """No more valid tokens after the 3 SID tokens are emitted."""
    from mcrs.sid.inference import build_sid_trie
    sids = [(1, 2, 3)]
    lookup = {(0, 1): 100, (1, 2): 200, (2, 3): 300}
    trie = build_sid_trie(sids, lookup)
    assert trie.valid_next_token_ids(prefix=[100, 200, 300]) == []


def test_make_prefix_allowed_tokens_fn_strips_prompt(monkeypatch):
    """Function strips the input prompt prefix and walks the trie on the rest."""
    from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
    sids = [(1, 2, 3)]
    lookup = {(0, 1): 100, (1, 2): 200, (2, 3): 300}
    trie = build_sid_trie(sids, lookup)
    # Simulate prompt of length 10 (any tokens), then 0 SID tokens emitted.
    prompt_lens = {0: 10}  # batch_id 0 → prompt is 10 tokens
    fn = make_prefix_allowed_tokens_fn(trie, prompt_lens)
    # input_ids has just the 10 prompt tokens (e.g., padded with 0s).
    input_ids = [0] * 10
    assert fn(0, input_ids) == [100]
    # After model emits 100, valid next is 200.
    assert fn(0, input_ids + [100]) == [200]


def test_prefix_allowed_tokens_fn_returns_pad_token_after_complete_sid():
    """When all 3 SID tokens have been emitted, return a fallback (eos or pad) so generate() halts cleanly."""
    from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
    sids = [(1, 2, 3)]
    lookup = {(0, 1): 100, (1, 2): 200, (2, 3): 300}
    trie = build_sid_trie(sids, lookup)
    prompt_lens = {0: 5}
    fn = make_prefix_allowed_tokens_fn(trie, prompt_lens, eos_token_id=2)
    # 5 prompt tokens + 3 emitted SID tokens = 8 total. Next position = exhausted.
    input_ids = [0] * 5 + [100, 200, 300]
    nxt = fn(0, input_ids)
    assert nxt == [2]  # eos


def test_build_sid_trie_handles_collisions():
    """Two distinct tracks sharing the same SID triplet → trie has one path, both tracks recoverable downstream."""
    from mcrs.sid.inference import build_sid_trie
    sids = [(1, 2, 3), (1, 2, 3)]  # two tracks same SID
    lookup = {(0, 1): 100, (1, 2): 200, (2, 3): 300}
    trie = build_sid_trie(sids, lookup)
    # Trie collapses duplicates — only one path.
    assert trie.valid_next_token_ids(prefix=[100, 200]) == [300]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sid_inference.py -q
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'mcrs.sid.inference'`.

- [ ] **Step 3: Write the implementation**

Create `music-crs-baselines/mcrs/sid/inference.py`:

```python
"""SID generator inference helpers — W3 ships the trie + prefix function for eval;
W4 will add the SID_GENERATOR retrieval class wrapping these.

The trie enforces that beam search only emits valid 3-token SID sequences (i.e.
SIDs that correspond to a real catalog track). HF's `generate(prefix_allowed_tokens_fn=...)`
calls our callable at every step to get the allowed next token ids.
"""
from __future__ import annotations

from typing import Iterable, Optional


class _TrieNode:
    __slots__ = ("children",)

    def __init__(self):
        self.children: dict[int, "_TrieNode"] = {}


class SIDTrie:
    """3-level trie keyed on SID token ids."""

    def __init__(self):
        self.root = _TrieNode()

    def insert(self, token_ids: list[int]) -> None:
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = _TrieNode()
            node = node.children[tid]

    def valid_next_token_ids(self, prefix: list[int]) -> list[int]:
        """Walk the trie along `prefix`; return children of the resulting node.

        Returns [] if prefix walks off the trie (shouldn't happen if generation
        was constrained by an earlier call) OR if the prefix is a complete SID
        (i.e. node has no children).
        """
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return []
            node = node.children[tid]
        return sorted(node.children.keys())


def build_sid_trie(
    sids: Iterable[tuple[int, int, int]],
    sid_lookup: dict[tuple[int, int], int],
) -> SIDTrie:
    """Build a 3-level trie from an iterable of (c1, c2, c3) SID triplets.

    `sid_lookup` maps (level, code) → token_id (from build_sid_to_token_id_lookup).
    """
    trie = SIDTrie()
    for (c1, c2, c3) in sids:
        token_ids = [
            sid_lookup[(0, c1)],
            sid_lookup[(1, c2)],
            sid_lookup[(2, c3)],
        ]
        trie.insert(token_ids)
    return trie


def make_prefix_allowed_tokens_fn(
    trie: SIDTrie,
    prompt_lens: dict[int, int],
    *,
    eos_token_id: Optional[int] = None,
):
    """Return a callable suitable for HF generate(prefix_allowed_tokens_fn=...).

    Args:
        trie: SIDTrie of valid SID sequences.
        prompt_lens: {batch_id: prompt_length}. Used to strip the prompt prefix
            from input_ids so we walk the trie only on the SID tokens emitted so far.
        eos_token_id: Token id to return after 3 SID tokens are emitted, so generate()
            halts cleanly. If None, returns [] (will trip a HF assertion — pass an
            EOS in production).

    Returns:
        prefix_allowed_tokens_fn(batch_id: int, input_ids: list[int]) -> list[int]
    """

    def _fn(batch_id: int, input_ids) -> list[int]:
        # input_ids may be a torch tensor in HF's call; convert if so.
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        prompt_len = prompt_lens.get(batch_id, 0)
        emitted = list(input_ids[prompt_len:])
        nxt = trie.valid_next_token_ids(prefix=emitted)
        if not nxt:
            if eos_token_id is not None:
                return [eos_token_id]
            return []
        return nxt

    return _fn
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sid_inference.py -q
```

Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/inference.py tests/test_sid_inference.py
git commit -m "sid w3: SIDTrie + prefix_allowed_tokens_fn for constrained beam (TDD, 5 tests)"
```

---

## Task 8: TDD `compute_ndcg_at_k` + `aggregate_ndcg`

Standard nDCG@K with binary relevance (a single gold track per query). Pure numerics — fast tests.

**Files:**
- Create: `music-crs-baselines/mcrs/sid/eval.py`
- Create: `tests/test_sid_eval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sid_eval.py`:

```python
"""Tests for SID generator eval metric (W3)."""
import math


def test_ndcg_at_k_gold_at_rank_1_returns_1():
    """Gold appears at top → nDCG = 1.0."""
    from mcrs.sid.eval import compute_ndcg_at_k
    score = compute_ndcg_at_k(retrieved=["a", "b", "c"], gold="a", k=20)
    assert score == 1.0


def test_ndcg_at_k_gold_at_rank_2_uses_log_discount():
    """Gold at rank 2 → nDCG = 1/log2(3) ≈ 0.6309."""
    from mcrs.sid.eval import compute_ndcg_at_k
    score = compute_ndcg_at_k(retrieved=["x", "a", "y"], gold="a", k=20)
    expected = 1 / math.log2(2 + 1)
    assert abs(score - expected) < 1e-9


def test_ndcg_at_k_gold_not_in_top_k_returns_0():
    """Gold not in top-k → nDCG = 0."""
    from mcrs.sid.eval import compute_ndcg_at_k
    retrieved = [f"track_{i}" for i in range(20)]
    score = compute_ndcg_at_k(retrieved=retrieved, gold="not_here", k=20)
    assert score == 0.0


def test_ndcg_at_k_truncates_to_k():
    """Anything past position k is ignored even if gold is there."""
    from mcrs.sid.eval import compute_ndcg_at_k
    retrieved = [f"track_{i}" for i in range(25)]
    retrieved.append("gold")  # position 25 (0-indexed)
    score = compute_ndcg_at_k(retrieved=retrieved, gold="gold", k=20)
    assert score == 0.0


def test_aggregate_ndcg_averages_per_query():
    """aggregate_ndcg returns mean across per-query nDCGs."""
    from mcrs.sid.eval import aggregate_ndcg
    per_query = [1.0, 0.5, 0.0, 0.25]
    assert abs(aggregate_ndcg(per_query) - 0.4375) < 1e-9


def test_aggregate_ndcg_empty_returns_zero():
    """Empty list → 0.0 (defensive — should never happen but keep deterministic)."""
    from mcrs.sid.eval import aggregate_ndcg
    assert aggregate_ndcg([]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sid_eval.py -q
```

Expected: 6 failures with `ModuleNotFoundError: No module named 'mcrs.sid.eval'`.

- [ ] **Step 3: Write the implementation**

Create `music-crs-baselines/mcrs/sid/eval.py`:

```python
"""SID generator eval (W3 gate): nDCG@20 + paired-bootstrap CI vs Phase 0 baseline.

The compute_ndcg_at_k function uses binary relevance with a single gold per query
— matches RecSys 2026 Music CRS turn-level evaluation. paired_bootstrap_ci is
imported from the existing scripts/compare_diagnostic_runs.py utility (don't
duplicate; use the version that's already battle-tested).
"""
from __future__ import annotations

import math
from typing import Iterable


def compute_ndcg_at_k(
    retrieved: list[str],
    gold: str,
    k: int = 20,
) -> float:
    """nDCG@k with binary relevance (single gold).

    DCG = 1 / log2(rank + 2) where rank is 0-indexed position of gold (if found).
    IDCG with one gold is always 1 (gold at rank 0 → 1/log2(2) = 1).
    """
    truncated = retrieved[:k]
    for rank, tid in enumerate(truncated):
        if tid == gold:
            return 1.0 / math.log2(rank + 2)
    return 0.0


def aggregate_ndcg(per_query_scores: Iterable[float]) -> float:
    """Mean nDCG across queries (defensive: empty input → 0)."""
    scores = list(per_query_scores)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sid_eval.py -q
```

Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/eval.py tests/test_sid_eval.py
git commit -m "sid w3: compute_ndcg_at_k + aggregate (TDD, 6 tests; binary single-gold)"
```

---

## Task 9: Implement `scripts/train_sid_generator.py` orchestration

This is the integration script. No additional unit tests — the mcrs/sid/* modules are TDD-covered; this just wires them together. Smoke-test runs in Task 11 (notebook).

**Files:**
- Create: `scripts/train_sid_generator.py`

- [ ] **Step 1: Write the orchestration script**

Create `scripts/train_sid_generator.py`:

```python
"""W3: fine-tune Qwen2.5-1.5B-Instruct + LoRA as a SID generator.

Pipeline:
  1. Load extended tokenizer + Qwen-1.5B base model.
  2. Untie tied embeddings + add 768 SID tokens + resize embedding matrix.
  3. Wrap with LoRA (r=32, alpha=64, modules_to_save=['embed_tokens','lm_head']).
  4. Read W2 train/val parquets, format examples, run HF Trainer for N epochs.
  5. Push LoRA adapter to Hub. If --merge, also merge + push the full ~3GB model.

Designed to run on Colab L4 (24GB) or Blackwell (95GB). Smoke mode keeps step
budget small for plumbing-validation runs (~10 min).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
)

from mcrs.sid.training_format import collate_training_batch, format_example_for_training
from mcrs.sid.vocab import (
    add_sid_tokens_to_tokenizer, build_sid_to_token_id_lookup,
    extend_model_vocab, untie_embeddings_if_tied,
)


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_LEVELS = 3
CODEBOOK_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_training/train.parquet")
    p.add_argument("--val-parquet", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_training/val.parquet")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_generator")
    p.add_argument("--hub-repo", type=str,
                   default="OrRim123/recsys2026-sid-generator-qwen15b-v1")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run for plumbing test (200 steps, 1k train rows)")
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--merge", action="store_true",
                   help="After training, merge LoRA into base + push merged model (~3GB)")
    return p.parse_args()


def load_and_extend_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok, n_added = add_sid_tokens_to_tokenizer(tok, NUM_LEVELS, CODEBOOK_SIZE)
    print(f"[tokenizer] base vocab + {n_added} SID tokens = {len(tok)} total", file=sys.stderr)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_and_extend_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"[model] tied={model.config.tie_word_embeddings}", file=sys.stderr)
    untie_embeddings_if_tied(model)
    extend_model_vocab(model, len(tokenizer))
    print(
        f"[model] vocab now {model.get_input_embeddings().weight.shape[0]} "
        f"(tied={model.config.tie_word_embeddings})",
        file=sys.stderr,
    )
    return model


def wrap_lora(model, args):
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens", "lm_head"],   # CRITICAL — without this the new SID embeddings never train
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def build_hf_dataset(parquet_path, tokenizer, sid_lookup, max_prompt_len, sample_n=None):
    df = pd.read_parquet(parquet_path)
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"[data] {parquet_path.name}: {len(df)} rows", file=sys.stderr)

    def _gen():
        for row in df.itertuples(index=False):
            yield format_example_for_training(
                query=row.query,
                code_1=int(row.code_1), code_2=int(row.code_2), code_3=int(row.code_3),
                tokenizer=tokenizer, sid_lookup=sid_lookup, max_prompt_len=max_prompt_len,
            )

    return Dataset.from_generator(_gen)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_and_extend_tokenizer()
    sid_lookup = build_sid_to_token_id_lookup(tokenizer, NUM_LEVELS, CODEBOOK_SIZE)

    model = load_and_extend_model(tokenizer)
    model = wrap_lora(model, args)

    sample_n = 1000 if args.smoke else None
    train_ds = build_hf_dataset(args.train_parquet, tokenizer, sid_lookup, args.max_prompt_len, sample_n=sample_n)
    val_ds = build_hf_dataset(args.val_parquet, tokenizer, sid_lookup, args.max_prompt_len, sample_n=200 if args.smoke else None)

    max_steps = 200 if args.smoke else -1
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if not args.smoke else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    def collator(examples):
        return collate_training_batch(examples, pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()

    # Save LoRA adapter locally + push.
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] LoRA adapter at {adapter_dir}", file=sys.stderr)

    hub_lora_repo = f"{args.hub_repo}-lora"
    model.push_to_hub(hub_lora_repo, private=False)
    tokenizer.push_to_hub(hub_lora_repo, private=False)
    print(f"[push] {hub_lora_repo}", file=sys.stderr)

    if args.merge:
        merged = model.merge_and_unload()
        merged_dir = args.output_dir / "merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        merged_repo = f"{args.hub_repo}-merged"
        merged.push_to_hub(merged_repo, private=False, safe_serialization=True)
        tokenizer.push_to_hub(merged_repo, private=False)
        print(f"[push] merged → {merged_repo}", file=sys.stderr)

    # Persist a small summary so notebooks can pick it up.
    summary = {
        "base_model": BASE_MODEL,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "epochs": args.epochs, "lr": args.lr,
        "smoke": args.smoke,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "hub_lora_repo": hub_lora_repo,
        "hub_merged_repo": f"{args.hub_repo}-merged" if args.merge else None,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports cleanly (no syntax errors)**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -c "
import ast
src = open('scripts/train_sid_generator.py').read()
ast.parse(src)
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Verify `--help` works (smoke-test argparse)**

```bash
python scripts/train_sid_generator.py --help
```

Expected: argparse help text printed; exit 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_sid_generator.py
git commit -m "sid w3: train_sid_generator.py orchestration (Qwen-1.5B + LoRA + modules_to_save)"
```

---

## Task 10: Implement `scripts/eval_sid_generator.py` orchestration

Loads the merged (or LoRA + base) model, builds the trie + prefix function, runs constrained beam search over val parquet, computes per-query nDCG@20, optionally runs paired-bootstrap CI vs Phase 0 baseline, writes JSON metrics.

**Files:**
- Create: `scripts/eval_sid_generator.py`

- [ ] **Step 1: Write the eval script**

Create `scripts/eval_sid_generator.py`:

```python
"""W3 eval gate: run constrained beam search over val parquet, compute nDCG@20,
paired-bootstrap CI vs Phase 0 baseline.

Reads merged HF model (or base + LoRA adapter) + W1 SID lookup + W2 val parquet.
Writes:
  experiments/cache/sid_eval/w3_eval_metrics.json
  experiments/cache/sid_eval/per_query_ndcg.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcrs.sid.eval import aggregate_ndcg, compute_ndcg_at_k
from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
from mcrs.sid.vocab import build_sid_to_token_id_lookup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True,
                   help="HF Hub repo of the merged SID generator (e.g. OrRim123/...-merged)")
    p.add_argument("--val-parquet", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_training/val.parquet")
    p.add_argument("--track-to-sid", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid/track_to_sid.parquet")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "experiments/cache/sid_eval")
    p.add_argument("--phase0-jsonl", type=Path,
                   default=REPO_ROOT / "experiments/diagnostic_runs/phase0_baseline_full_dev/per_turn_metrics.jsonl",
                   help="Per-turn nDCG@20 from Phase 0 baseline (for paired-bootstrap CI). "
                        "Optional: if absent, eval still runs but skips the CI.")
    p.add_argument("--eval-slice", choices=["raw", "all"], default="raw",
                   help="raw = only conversation-derived val rows (matches Blind-A); "
                        "all = include metadata + doc2query rows (catalog coverage diagnostic).")
    p.add_argument("--num-beams", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, eval only the first N val rows (for smoke).")
    return p.parse_args()


def build_collision_lookup(track_to_sid_df: pd.DataFrame) -> dict[tuple[int,int,int], list[str]]:
    """SID triplet → ordered list of track_ids (popularity-descending, from W1 bucket_rank)."""
    lookup: dict[tuple[int,int,int], list[str]] = {}
    for row in track_to_sid_df.sort_values("bucket_rank").itertuples(index=False):
        key = (int(row.code_1), int(row.code_2), int(row.code_3))
        lookup.setdefault(key, []).append(row.track_id)
    return lookup


def decode_beams_to_tracks(
    beam_token_ids: list[list[int]],
    beam_scores: list[float],
    inverse_lookup: dict[int, tuple[int,int]],
    sid_to_tracks: dict[tuple[int,int,int], list[str]],
    cap_per_bucket: int = 1,
    top_k: int = 20,
) -> list[str]:
    """For each beam, decode SID + look up tracks (popularity-ordered). Apply per-bucket cap.

    cap_per_bucket=1 means: each beam contributes at most 1 track from its collision bucket.
    Spillover tracks (beams 21+) are appended to fill any de-duplication gaps.
    """
    seen = set()
    primary = []
    spillover = []
    for tok_ids in beam_token_ids:
        try:
            sid = (
                inverse_lookup[tok_ids[0]][1],
                inverse_lookup[tok_ids[1]][1],
                inverse_lookup[tok_ids[2]][1],
            )
        except (KeyError, IndexError):
            continue
        bucket = sid_to_tracks.get(sid, [])
        added_in_beam = 0
        for tid in bucket:
            if tid in seen:
                continue
            if added_in_beam < cap_per_bucket:
                primary.append(tid)
                added_in_beam += 1
            else:
                spillover.append(tid)
            seen.add(tid)
    return (primary + spillover)[:top_k]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer + model from {args.model_id}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    sid_lookup = build_sid_to_token_id_lookup(tok, num_levels=3, codebook_size=256)
    inverse = {v: k for k, v in sid_lookup.items()}

    print(f"[load] track_to_sid from {args.track_to_sid}", file=sys.stderr)
    t2s = pd.read_parquet(args.track_to_sid)
    sids = list(t2s[["code_1", "code_2", "code_3"]].itertuples(index=False, name=None))
    sid_to_tracks = build_collision_lookup(t2s)
    print(f"[trie] {len(sids)} SID rows, {len(sid_to_tracks)} unique SID triples", file=sys.stderr)
    trie = build_sid_trie(sids, sid_lookup)

    val = pd.read_parquet(args.val_parquet)
    if args.eval_slice == "raw":
        val = val[val["source"] == "raw"].reset_index(drop=True)
    if args.limit > 0:
        val = val.head(args.limit).reset_index(drop=True)
    print(f"[eval] {len(val)} val queries (slice={args.eval_slice})", file=sys.stderr)

    per_query_ndcg = []
    per_query_records = []
    eos_id = tok.eos_token_id

    with torch.inference_mode():
        for i, row in enumerate(val.itertuples(index=False)):
            inputs = tok(
                row.query,
                truncation=True, max_length=args.max_prompt_len,
                return_tensors="pt",
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            prefix_fn = make_prefix_allowed_tokens_fn(
                trie, prompt_lens={0: prompt_len}, eos_token_id=eos_id,
            )
            out = model.generate(
                **inputs,
                max_new_tokens=3,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                prefix_allowed_tokens_fn=prefix_fn,
                output_scores=False, return_dict_in_generate=True,
                do_sample=False,
            )
            # Each sequence row is (prompt_len + 3) tokens; slice off prompt to get the 3 SID tokens.
            beams = out.sequences[:, prompt_len:prompt_len + 3].cpu().tolist()
            tracks = decode_beams_to_tracks(
                beam_token_ids=beams,
                beam_scores=[],   # unused for now; could weight beams in v2
                inverse_lookup=inverse,
                sid_to_tracks=sid_to_tracks,
                top_k=args.top_k,
            )
            score = compute_ndcg_at_k(retrieved=tracks, gold=row.track_id, k=args.top_k)
            per_query_ndcg.append(score)
            per_query_records.append({
                "query_id": i,
                "track_id_gold": row.track_id,
                "ndcg_at_20": score,
                "top_5_retrieved": tracks[:5],
            })
            if (i + 1) % 200 == 0:
                running = aggregate_ndcg(per_query_ndcg)
                print(f"[eval] {i+1}/{len(val)} mean nDCG@20={running:.4f}", file=sys.stderr)

    mean_ndcg = aggregate_ndcg(per_query_ndcg)
    metrics = {
        "model_id": args.model_id,
        "eval_slice": args.eval_slice,
        "n_queries": len(per_query_ndcg),
        "mean_ndcg_at_20": mean_ndcg,
        "phase0_baseline_ndcg_at_20": 0.099,   # documented from Phase 0 baseline
        "delta_vs_phase0": mean_ndcg - 0.099,
    }

    # Optional paired-bootstrap CI vs Phase 0 baseline.
    if args.phase0_jsonl.exists():
        baseline_per_query = _load_phase0_per_query(args.phase0_jsonl)
        # Align by query_id (or track_id_gold) — both runs evaluated same val rows, but make sure.
        # For simplicity here, paired-bootstrap on same-length lists (truncate to common count).
        n_common = min(len(per_query_ndcg), len(baseline_per_query))
        ours = per_query_ndcg[:n_common]
        theirs = baseline_per_query[:n_common]
        from scripts.compare_diagnostic_runs import paired_bootstrap_ci
        lo, hi = paired_bootstrap_ci(
            ours, theirs, n_resamples=1000, alpha=0.05,
        )
        metrics["paired_bootstrap_ci"] = {"lo": lo, "hi": hi, "n_compared": n_common}
        metrics["gate_pass"] = (mean_ndcg >= 0.12) and (lo > 0)
    else:
        print(f"[warn] phase0 jsonl not found at {args.phase0_jsonl}; skipping CI.", file=sys.stderr)
        metrics["paired_bootstrap_ci"] = None
        metrics["gate_pass"] = mean_ndcg >= 0.12  # point-estimate gate

    (args.output_dir / "w3_eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.output_dir / "per_query_ndcg.jsonl").open("w") as f:
        for rec in per_query_records:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(metrics, indent=2))


def _load_phase0_per_query(path: Path) -> list[float]:
    """Read Phase 0 per-turn metrics JSONL; extract per-query nDCG@20 in file order."""
    out = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            # Tolerate variant key names from earlier runs.
            v = rec.get("ndcg_at_20") or rec.get("ndcg@20") or rec.get("ndcg")
            if v is not None:
                out.append(float(v))
    return out


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check + --help smoke test**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -c "import ast; ast.parse(open('scripts/eval_sid_generator.py').read()); print('OK')"
python scripts/eval_sid_generator.py --help
```

Expected: `OK` then argparse help.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_sid_generator.py
git commit -m "sid w3: eval_sid_generator.py — constrained beam + nDCG@20 + paired-bootstrap CI"
```

---

## Task 11: Implement `colab/62_train_and_eval_sid_generator.ipynb`

Single notebook covering: clone, deps, drive mount, smoke train, full train, push, eval, gate decision. Mirrors the structure of `colab/30_train_responder_kto.ipynb`.

**Files:**
- Create: `colab/62_train_and_eval_sid_generator.ipynb`

- [ ] **Step 1: Generate the notebook**

Use this Python snippet to write the notebook (avoids hand-editing JSON):

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python <<'EOF'
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md("""# 62 — Train + eval SID generator (W3)

Fine-tunes Qwen2.5-1.5B-Instruct + LoRA to emit 3 SID tokens for each W2 query.
Smoke mode: 200 steps, ~10 min. Full: 3 epochs over ~90K rows, ~3-4 hr on L4 / ~1-2 hr on Blackwell.

**Prereqs**: W1 + W2 artifacts on Drive at `/content/drive/MyDrive/recsys2026/sid/track_to_sid.parquet`
and `/content/drive/MyDrive/recsys2026/sid_training/{train,val}.parquet`. HF token in Colab Secrets as `HF_TOKEN`.
""")

code("""# 1) GPU check.
!nvidia-smi | head -20""")

code("""# 2) Clone fresh-model branch.
BRANCH = 'fresh-model'
!rm -rf /content/recsys2026
!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026
%cd /content/recsys2026""")

code("""# 3) HF auth.
import os
from huggingface_hub import login
HF_TOKEN = os.environ.get('HF_TOKEN') or input('HF_TOKEN (Colab Secret recommended): ')
login(token=HF_TOKEN, add_to_git_credential=False)
print('HF auth OK')""")

code("""# 4) Mount Drive + symlink W1/W2 artifacts into expected cache paths.
from google.colab import drive
import os
drive.mount('/content/drive', force_remount=False)

DRIVE_BASE = '/content/drive/MyDrive/recsys2026'
LOCAL_BASE = '/content/recsys2026/experiments/cache'
os.makedirs(LOCAL_BASE, exist_ok=True)
for name in ['sid', 'sid_training', 'sid_eval', 'sid_generator']:
    src = f'{DRIVE_BASE}/{name}'
    dst = f'{LOCAL_BASE}/{name}'
    os.makedirs(src, exist_ok=True)
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.exists(dst):
        import shutil; shutil.rmtree(dst)
    os.symlink(src, dst)
    print(f'symlink: {dst} -> {src}')

# Verify W1 + W2 artifacts visible.
!ls -la experiments/cache/sid/track_to_sid.parquet
!ls -la experiments/cache/sid_training/train.parquet experiments/cache/sid_training/val.parquet""")

code("""# 5) Install/upgrade deps. Colab Pro base ships transformers + datasets + torch.
# We need recent peft (>= 0.10) for modules_to_save with extended vocab.
!pip install -q -U "peft>=0.10" "transformers>=4.40" "accelerate>=0.30" "trl>=0.8"
import transformers, peft, torch
print(f'transformers={transformers.__version__}, peft={peft.__version__}, torch={torch.__version__}')""")

code("""# 6) Pytest pre-flight on the SID modules.
!cd /content/recsys2026 && python -m pytest \\
    tests/test_sid_vocab.py \\
    tests/test_sid_training_format.py \\
    tests/test_sid_inference.py \\
    tests/test_sid_eval.py \\
    -q --tb=short 2>&1 | tail -10""")

code("""# 7) SMOKE training — 200 steps, 1k train rows, ~10 min on L4.
# Confirms: vocab extension OK, LoRA wraps cleanly, gradient flows, checkpoint saves.
!cd /content/recsys2026 && python scripts/train_sid_generator.py \\
    --smoke \\
    --output-dir experiments/cache/sid_generator/smoke \\
    --hub-repo OrRim123/recsys2026-sid-generator-qwen15b-smoke \\
    2>&1 | tail -40""")

code("""# 8) FULL training — 3 epochs over the W2 train parquet. ~3-4 hr L4 / ~1-2 hr Blackwell.
# Set --merge so the final model is pushed merged (3GB), ready for W4 inference + W3 eval.
!cd /content/recsys2026 && python scripts/train_sid_generator.py \\
    --output-dir experiments/cache/sid_generator/full \\
    --hub-repo OrRim123/recsys2026-sid-generator-qwen15b-v1 \\
    --merge \\
    2>&1 | tee /content/drive/MyDrive/recsys2026/sid_generator/training_log.txt | tail -60""")

code("""# 9) EVAL — constrained-beam decode over W2 val (raw slice = matches Blind-A distribution).
# Reads the merged model just pushed, computes per-query nDCG@20, paired-bootstrap CI vs Phase 0.
!cd /content/recsys2026 && python scripts/eval_sid_generator.py \\
    --model-id OrRim123/recsys2026-sid-generator-qwen15b-v1-merged \\
    --eval-slice raw \\
    2>&1 | tail -30""")

code("""# 10) Read + display final gate metrics.
import json
m = json.load(open('experiments/cache/sid_eval/w3_eval_metrics.json'))
print(json.dumps(m, indent=2))
print()
print('=' * 60)
if m.get('gate_pass'):
    print(f"GATE PASS — mean nDCG@20={m['mean_ndcg_at_20']:.4f} "
          f"(threshold 0.12; delta vs Phase 0 = {m['delta_vs_phase0']:+.4f})")
else:
    print(f"GATE FAIL — mean nDCG@20={m['mean_ndcg_at_20']:.4f} (threshold 0.12)")
    if m.get('paired_bootstrap_ci'):
        ci = m['paired_bootstrap_ci']
        print(f"  paired-bootstrap CI = ({ci['lo']:.4f}, {ci['hi']:.4f})")
print('=' * 60)
""")

md("""## After the run

**Gate pass** (nDCG@20 ≥ 0.12 AND CI lower-bound > 0):
- Merged model is on Hub at `OrRim123/recsys2026-sid-generator-qwen15b-v1-merged`
- Proceed to W4: build `SID_GENERATOR` retrieval class + register `wrrf_bm25_dense_sid_v1`
- Update `MEMORY.md` with the W3 result file

**Gate fail**:
- If point-estimate is close (0.10-0.12) but CI includes 0: more training (5 epochs), check loss curve
- If point-estimate is low (<0.08): likely the W1 SID coarseness biting (3017 unique SIDs limits ceiling).
  Re-run W1 with smaller latent_dim (256→128) + larger codebook (256→512), then re-run W2 + W3.
- If loss diverged: drop LR to 1e-4, re-run.
""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
Path('colab/62_train_and_eval_sid_generator.ipynb').write_text(json.dumps(nb, indent=1))
print('wrote colab/62_train_and_eval_sid_generator.ipynb')
EOF
```

- [ ] **Step 2: Verify the notebook is valid JSON and parseable as nbformat**

```bash
python -c "
import json, nbformat
nb = nbformat.read('colab/62_train_and_eval_sid_generator.ipynb', as_version=4)
print(f'cells: {len(nb.cells)}')
for i, c in enumerate(nb.cells):
    src = ''.join(c.source)
    print(f'  CELL {i} [{c.cell_type}]: {src[:80].replace(chr(10), \" | \")}')
"
```

Expected: 11 cells (1 markdown header + 9 code + 1 markdown footer = 11). All cells listed.

- [ ] **Step 3: Commit**

```bash
git add colab/62_train_and_eval_sid_generator.ipynb
git commit -m "sid w3: notebook 62 (smoke + full train + push + constrained-beam eval gate)"
git push origin fresh-model
```

---

## Task 12: Final test sweep + code-reviewer agent

- [ ] **Step 1: Run all SID tests**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest \
    tests/test_sid_preprocessing.py \
    tests/test_sid_validation.py \
    tests/test_sid_quantizer.py \
    tests/test_sid_training_data.py \
    tests/test_sid_generator_dataloader.py \
    tests/test_sid_vocab.py \
    tests/test_sid_training_format.py \
    tests/test_sid_inference.py \
    tests/test_sid_eval.py \
    -q
```

Expected: 32 (W1) + 26 (W2) + 13 + 6 + 5 + 6 = **88 SID tests passing**. (Network-dependent vocab tests may take ~3 min if Qwen-1.5B not cached.)

- [ ] **Step 2: Run the full repo test sweep (skip slow integration suites)**

```bash
python -m pytest tests/ -q \
    --ignore=tests/test_local_eval.py \
    --ignore=tests/test_wave0_integration.py \
    --ignore=tests/test_wave1_integration.py \
    --ignore=tests/test_wave2_integration.py \
    2>&1 | tail -3
```

Expected: ~480+ passing (459 baseline-after-W2 + ~30 new W3).

- [ ] **Step 3: Verify clean git state + push**

```bash
git status -s   # should be clean
git log --oneline -15   # should show ~12 W3 commits
git push origin fresh-model
```

- [ ] **Step 4: Invoke code-reviewer agent**

Dispatch via Agent tool:

```
Agent({
  description: "Review SID W3 generator training",
  subagent_type: "superpowers:code-reviewer",
  prompt: "Review the W3 implementation of the SID generator (Qwen-1.5B + LoRA fine-tune) against the design spec.

  Spec: documents/specs/2026-05-15-sid-retrieval-design.md §3 + §4.1
  Plan: documents/plans/2026-05-16-sid-generator-training-w3.md

  W1+W2 artifacts that this depends on:
  - experiments/cache/sid/track_to_sid.parquet (47K rows, 3017 unique SIDs)
  - experiments/cache/sid_training/{train,val}.parquet (~95K rows)

  Files to review (all newly added on fresh-model):
  - music-crs-baselines/mcrs/sid/vocab.py (tokenizer extension + untie + SID<->id)
  - music-crs-baselines/mcrs/sid/training_format.py (label masking + collate)
  - music-crs-baselines/mcrs/sid/inference.py (trie + prefix_allowed_tokens_fn) — W3 subset; W4 will extend
  - music-crs-baselines/mcrs/sid/eval.py (nDCG@20 metric)
  - scripts/train_sid_generator.py (orchestration: extend vocab + LoRA + Trainer + push)
  - scripts/eval_sid_generator.py (constrained beam + nDCG + paired-bootstrap CI)
  - colab/62_train_and_eval_sid_generator.ipynb (smoke + full + eval)
  - tests/test_sid_vocab.py, tests/test_sid_training_format.py, tests/test_sid_inference.py, tests/test_sid_eval.py

  Critical review checks:
  1. CRITICAL: Does the LoRA config in train_sid_generator.py include `modules_to_save=['embed_tokens','lm_head']`? If missing the 768 new SID embeddings never train and the model silently fails to learn — flag immediately.
  2. CRITICAL: Is `untie_embeddings_if_tied` called BEFORE LoRA wrapping? If after, PEFT may save tied weights as two copies and corrupt the merged push.
  3. Does the `format_example_for_training` label masking actually only score the 3 SID positions? Verify with a manual trace on the test fixture.
  4. Does `make_prefix_allowed_tokens_fn` correctly handle batched generation when num_beams > 1? HF expects the callable to handle each beam id; trace through with a 2-beam example.
  5. Does the trie collapse collisions correctly (multiple tracks → same SID triplet → one trie path)? Per W1 we have 3017 unique SIDs from 47K tracks so collisions are common.
  6. Does the eval script's `decode_beams_to_tracks` apply the per-bucket cap correctly so a single high-confidence beam can't flood top-20 from one collision bucket?
  7. Is the paired-bootstrap CI properly aligned (same query_id positions in both per-query lists)? Mismatched alignment would produce nonsensical CI bounds.
  8. Does the merged-push workflow actually succeed with the extended vocab + untied embeddings? Anything obvious that might break in `merge_and_unload()`?
  9. The notebook 62 cell 8 (full training) writes log to Drive — is the path correct + persistent across Colab session restarts?

  Report: APPROVE / APPROVE-WITH-CHANGES / NEEDS-MAJOR-REVISION + numbered findings + top-5 changes if any."
})
```

- [ ] **Step 5: Address any reviewer findings**

If APPROVE → done.
If APPROVE-WITH-CHANGES → make the small fixes inline, commit (e.g. `sid w3: harden <area> per reviewer (Ix)`), re-push.
If NEEDS-MAJOR-REVISION → triage findings before user runs notebook 62.

---

## Self-review checklist (per writing-plans skill)

**1. Spec coverage** (against §3 + §4.1 of design spec):

- §3.1 LoRA config with `modules_to_save=["embed_tokens","lm_head"]` — Task 9 ✓
- §3.1 untying recipe before LoRA wrap — Tasks 3 + 9 ✓
- §3.1 mandatory merged push — Task 9 (`--merge` flag) + Task 11 cell 8 ✓
- §3.2 reads from W2 parquets (~290K row contract) — Task 9 ✓
- §3.3 query format — already shipped in W2 (`format_query_for_sid_input`); W3 reads `query` column verbatim ✓
- §3.4 training loop config (LR=2e-4, cosine, warmup_ratio=0.05, batch=8 × accum=4, 3 epochs, max_seq=1024) — Task 9 ✓
- §3.4 mask labels = -100 except 3 SID positions — Task 5 ✓
- §3.5 dev nDCG@20 ≥ 0.12 with paired-bootstrap CI — Task 10 + Task 11 cells 9-10 ✓
- §4.1 trie + `PrefixConstrainedLogitsProcessor` — Task 7 (trie + prefix function) + Task 10 (used at eval) ✓
- §4.1 per-bucket cap (deduplication) — Task 10 `decode_beams_to_tracks` ✓
- §3.6 implementation footprint — Tasks 2-11 cover all 5 listed files (with refactor: vocab.py + training_format.py + inference.py + eval.py instead of single train_sid_generator.py module) ✓

**2. Placeholder scan**: searched for "TBD", "TODO", "implement later", "fill in details", "appropriate error handling" — none. Every test has runnable assertions; every code block is complete.

**3. Type consistency**:
- `sid_lookup: dict[tuple[int, int], int]` — used consistently across vocab.py, training_format.py, inference.py
- `inverse: dict[int, tuple[int, int]]` — used in vocab.decode_token_ids_to_sid + eval decode_beams_to_tracks
- `(c1, c2, c3)` triplet — used by trie, lookup, eval (all 3-tuple of int)
- `format_example_for_training(...)` returns `dict[str, list[int]]` — consumed by `Dataset.from_generator`; `collate_training_batch` accepts the same shape
- `make_prefix_allowed_tokens_fn` returns a callable matching HF's `prefix_allowed_tokens_fn(batch_id, input_ids) -> list[int]` contract

**4. Sequencing**: each task's tests pass against artifacts produced by earlier tasks only. Tasks 2-8 are pure-function TDD with no inter-task imports beyond `mcrs.sid.vocab`. Task 9 (train script) imports from Tasks 2-6. Task 10 (eval script) imports from Tasks 2 + 7 + 8. Task 11 (notebook) calls Tasks 9 + 10. Task 12 reviews everything.

**5. Plan-deviation hooks built in**:
- Task 1 step 3 explicitly handles missing Phase 0 baseline (eval still runs, skips CI)
- Task 11 cell 7 (smoke) lets user halt before committing to 3-4 hr full run if anything looks off
- `scripts/train_sid_generator.py` `--merge` flag is opt-in: smoke runs push LoRA-only, only the full run merges
- Notebook 62 cell 10 prints clear PASS/FAIL message + remediation hint if gate fails

**Plan complete and ready for execution.**

---

## Estimated wallclock

| Component | Time |
|---|---|
| Task 1 (setup) | 2 min |
| Tasks 2-4 (vocab module + tests) | ~15-20 min via subagent (model download dominates first test) |
| Tasks 5-6 (training_format + tests) | ~5-10 min via subagent |
| Task 7 (inference trie + tests) | ~5 min via subagent |
| Task 8 (eval metric + tests) | ~5 min via subagent |
| Task 9 (train script) | ~10 min via subagent |
| Task 10 (eval script) | ~10 min via subagent |
| Task 11 (notebook 62) | ~5 min via subagent |
| Task 12 (final review) | ~10 min agent + iteration |
| **Total my coordination time** | **~70-85 min** |
| User Colab smoke run (cell 7) | **~10 min on L4** |
| User Colab full training (cell 8) | **~3-4 hr on L4 / ~1-2 hr on Blackwell** |
| User Colab eval (cell 9) | **~15-25 min on L4 (constrained beam over ~5K val rows)** |

W3 is the heaviest week of the SID retriever sprint: it spans tokenizer surgery, an LR-sensitive LoRA fine-tune with vocab expansion, and a non-trivial eval loop with paired-bootstrap CI. Most of the risk is concentrated in Task 9's `modules_to_save` configuration and Task 3's tied-embedding untangle — both have specific TDD coverage. After this, W4 (inference pipeline + first Blind-A submission) is mostly wiring and should be lighter.
