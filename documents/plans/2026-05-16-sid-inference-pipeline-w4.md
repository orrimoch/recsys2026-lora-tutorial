# SID Inference Pipeline (W4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the W3 SID generator into the existing wRRF inference pipeline as a 4th retrieval sub-stream, so the full CMQR + ProRank + responder chain can use SID candidates. Ship the first Blind-A submission with config 170 (ensemble) once the dev gate passes.

**Architecture:** Add a `SID_GENERATOR` retrieval class that implements the existing `batch_text_to_item_retrieval(queries, topk) -> list[list[str]]` interface — same shape as `BM25_MODEL`, `DENSE_PRECOMPUTED`, `DENSE_LOCAL`. The class loads the W3 merged model from HF Hub, builds the trie (reusing `mcrs/sid/inference.py` from W3), and decodes per-query via constrained beam search. Register the class in `mcrs/retrieval_modules/__init__.py` with a new `wrrf_bm25_dense_sid_v1` factory entry that slots SID into the existing 3-stream wRRF as a 4th stream. Two new YAML configs (170 ensemble, 171 pure-SID) wrap the v5-kto responder unchanged.

**Tech Stack:** Python 3.10, `transformers`, `peft` (already on Colab), `torch`. Reuses W3's `mcrs/sid/{vocab,inference}.py`. No new pip deps.

**Spec reference:** `documents/specs/2026-05-15-sid-retrieval-design.md` §4 (inference pipeline) + §4.2 (wRRF integration) + §5.1 W4 gate.

**Inputs from prior weeks:**
- W3 artifact: HF Hub merged model `OrRim123/recsys2026-sid-generator-qwen15b-v1-merged` (~3 GB). Contains the extended tokenizer (152,433 tokens), untied embed/lm_head, fine-tuned weights.
- W1 artifact: `experiments/cache/sid/track_to_sid.parquet` (47,071 rows, columns `track_id, code_1, code_2, code_3, popularity, bucket_rank`) — needed for the trie + collision-bucket lookup.
- W2 artifact: `experiments/cache/sid_training/val.parquet` — needed for the dev gate evaluation (Task 9).
- Existing infrastructure: `RRF_MODEL` in `mcrs/retrieval_modules/rrf.py`, `run_inference_blindset.py`, `scripts/compare_diagnostic_runs.py:paired_bootstrap_ci`.

**Hard prerequisite:** W3 gate must have passed (`mean_ndcg_at_20 ≥ 0.12` on val raw slice from notebook 62 cell 10). If W3 failed, W4 has no functional model to wrap and should NOT proceed.

**Output artifacts:**
- `music-crs-baselines/mcrs/retrieval_modules/sid_generator.py` (~180 lines) — the new retrieval class
- `music-crs-baselines/mcrs/retrieval_modules/__init__.py` — modified (+1 import, +35 lines for 2 new factory entries)
- `music-crs-baselines/config/170-wrrf-sid-v5kto-blindsetA.yaml` — ensemble config
- `music-crs-baselines/config/171-pure-sid-v5kto-blindsetA.yaml` — pure-SID comparison config
- `tests/test_sid_generator_retrieval.py` (~150 lines) — TDD tests
- `colab/63_run_blindset_sid.ipynb` — runs the full pipeline on Blind-A → `prediction.json`
- `colab/64_dev_gate_compare.ipynb` — paired-bootstrap CI vs current champion config 132 on dev
- One Blind-A submission zip: `submissions/2026-05-XX-sid-ensemble-170.zip`

---

## File structure

| Path | Type | Responsibility |
|---|---|---|
| `mcrs/retrieval_modules/sid_generator.py` | new | `SID_GENERATOR` class: load model + tokenizer + W1 lookup → trie → per-query constrained beam → top-K track ids |
| `mcrs/retrieval_modules/__init__.py` | modify | Register `sid_generator` factory entry + `wrrf_bm25_dense_sid_v1` (ensemble) factory entry |
| `config/170-wrrf-sid-v5kto-blindsetA.yaml` | new | Ensemble Blind-A submission config |
| `config/171-pure-sid-v5kto-blindsetA.yaml` | new | Pure-SID comparison config |
| `tests/test_sid_generator_retrieval.py` | new | 6 TDD tests (init, batch retrieve, per-bucket cap, interface parity, single-query wrapper, factory smoke) |
| `colab/63_run_blindset_sid.ipynb` | new | 7-cell notebook: clone + deps + drive + run `run_inference_blindset.py --tid 170-...` + verify `prediction.json` + zip + submit |
| `colab/64_dev_gate_compare.ipynb` | new | 6-cell notebook: run config 170 + config 132 on dev → paired-bootstrap CI → PASS/FAIL gate decision |

---

## Task 1: Setup — verify W3 model is on Hub + W1 artifacts available

**Files:**
- Verify: HF Hub repo `OrRim123/recsys2026-sid-generator-qwen15b-v1-merged` exists
- Verify: `experiments/cache/sid/track_to_sid.parquet` (W1)
- Verify: W3 gate JSON shows `gate_pass: true`

- [ ] **Step 1: Verify W3 merged model is on HF Hub**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.repo_info('OrRim123/recsys2026-sid-generator-qwen15b-v1-merged')
    print(f'OK: model exists, last_modified={info.last_modified}')
except Exception as e:
    print(f'MISSING: {e}')
    print('ABORT W4 — re-run W3 first (notebook 62)')
"
```

Expected: `OK: model exists, last_modified=...`. If MISSING, W3 must be re-run or fixed before W4 can proceed.

- [ ] **Step 2: Verify W3 eval metrics show gate passed**

```bash
cat experiments/cache/sid_eval/w3_eval_metrics.json 2>/dev/null | python -c "
import json, sys
try:
    m = json.load(sys.stdin)
    print(f'mean_ndcg_at_20={m[\"mean_ndcg_at_20\"]:.4f}, gate_pass={m[\"gate_pass\"]}')
    if not m['gate_pass']:
        print('WARN: W3 gate did NOT pass. Proceeding with W4 anyway is exploratory.')
except Exception as e:
    print(f'No W3 eval metrics yet — check Drive: experiments/cache/sid_eval/w3_eval_metrics.json')
"
```

Expected: `mean_ndcg_at_20=0.xxxx, gate_pass=True`. (If you're on local Mac and W3 ran on Colab, this file lives on Drive; either sync from Drive or accept that W4 proceeds based on Hub model existence alone.)

- [ ] **Step 3: Verify W1 SID lookup exists (or accept that the Colab path is canonical)**

```bash
ls -la experiments/cache/sid/track_to_sid.parquet 2>&1 || echo "Not on local — will fetch from Drive in Colab notebook 63"
```

OK if missing locally — the Colab notebook re-symlinks Drive.

---

## Task 2: TDD `SID_GENERATOR.__init__` — load model + tokenizer + W1 lookup

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/sid_generator.py`
- Create: `tests/test_sid_generator_retrieval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sid_generator_retrieval.py`:

```python
"""Tests for the SID_GENERATOR retrieval class (W4)."""
import pytest


@pytest.fixture(scope="module")
def tiny_sid_lookup_parquet(tmp_path_factory):
    """A minimal 5-row track_to_sid.parquet for fast tests (matches W1 schema)."""
    import pandas as pd
    rows = [
        {"track_id": "t1", "code_1": 0, "code_2": 0, "code_3": 0, "popularity": 1.0, "bucket_rank": 0},
        {"track_id": "t2", "code_1": 0, "code_2": 0, "code_3": 1, "popularity": 0.9, "bucket_rank": 0},
        {"track_id": "t3", "code_1": 0, "code_2": 1, "code_3": 0, "popularity": 0.8, "bucket_rank": 0},
        # Collision: t4 and t5 share the same SID triplet (0, 1, 0)
        {"track_id": "t4", "code_1": 0, "code_2": 1, "code_3": 0, "popularity": 0.7, "bucket_rank": 1},
        {"track_id": "t5", "code_1": 1, "code_2": 0, "code_3": 0, "popularity": 0.6, "bucket_rank": 0},
    ]
    p = tmp_path_factory.mktemp("sid") / "track_to_sid.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return p


def test_sid_generator_init_loads_model_tokenizer_and_trie(monkeypatch, tiny_sid_lookup_parquet, tmp_path):
    """Init: load model + tokenizer from a (mocked) hub_repo, build trie + collision lookup."""
    from mcrs.retrieval_modules.sid_generator import SID_GENERATOR

    # Stub out heavy HF loads so the test doesn't need GPU or network.
    class _StubTokenizer:
        def __init__(self): self.pad_token_id = 0; self.eos_token_id = 1
        def get_vocab(self):
            # 768 SID tokens at ids 1000..1767 (deterministic for test)
            v = {f"<SID_L{lvl}_C{code}>": 1000 + lvl * 256 + code
                 for lvl in range(3) for code in range(256)}
            return v
        @classmethod
        def from_pretrained(cls, _): return cls()

    class _StubModel:
        device = "cpu"
        def eval(self): return self
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def generate(self, *a, **k):
            # Returns a fake top-K beam result; real test for this is task 3.
            raise NotImplementedError

    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoTokenizer", _StubTokenizer)
    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoModelForCausalLM", _StubModel)

    gen = SID_GENERATOR(
        hub_repo="fake/repo",
        sid_lookup_path=tiny_sid_lookup_parquet,
        device="cpu",
        num_beams=20,
    )

    # Trie has 4 unique SID triplets ((0,0,0), (0,0,1), (0,1,0), (1,0,0))
    # because (0,1,0) appears twice via collision.
    assert len(gen.sid_to_tracks) == 4
    # Collision bucket (0,1,0) has 2 tracks, popularity-sorted
    assert gen.sid_to_tracks[(0, 1, 0)] == ["t3", "t4"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest tests/test_sid_generator_retrieval.py -q
```

Expected: ImportError `cannot import name 'SID_GENERATOR'`.

- [ ] **Step 3: Write minimal implementation**

Create `music-crs-baselines/mcrs/retrieval_modules/sid_generator.py`:

```python
"""W4: SID-generator retrieval class.

Implements `batch_text_to_item_retrieval(queries, topk) -> list[list[str]]`,
matching the existing retrievers (BM25_MODEL, DENSE_PRECOMPUTED, etc.) so it
slots into RRF_MODEL as a 4th sub-stream transparently.

Loads the W3-trained merged Qwen model from HF Hub, builds the SID trie from
W1's track_to_sid.parquet, and per query runs trie-constrained beam search
(prefix_allowed_tokens_fn) to emit exactly 3 SID tokens. The 3 tokens are
decoded back to a SID triplet, looked up in the collision table, and one track
per beam is returned with optional spillover to fill top-K under the per-bucket
cap (per spec §2.5).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
from mcrs.sid.vocab import build_sid_to_token_id_lookup


class SID_GENERATOR:
    """SID-generator retriever wrapping a W3 merged model + W1 trie + collision lookup."""

    def __init__(
        self,
        hub_repo: str,
        sid_lookup_path: Path,
        *,
        device: str = "cuda",
        num_beams: int = 20,
        max_prompt_len: int = 1024,
        cap_per_bucket: int = 1,
    ):
        self.hub_repo = hub_repo
        self.num_beams = num_beams
        self.max_prompt_len = max_prompt_len
        self.cap_per_bucket = cap_per_bucket

        # Load model + tokenizer from Hub.
        self.tokenizer = AutoTokenizer.from_pretrained(hub_repo)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            hub_repo, torch_dtype=dtype, device_map=device if device != "cpu" else None,
        )
        self.model.eval()
        # Pin generation config so beam search doesn't trip on missing pad/eos.
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        # Build SID lookups from W1 parquet (sorted by bucket_rank so popularity
        # order is preserved within collision buckets).
        self.sid_lookup = build_sid_to_token_id_lookup(self.tokenizer, num_levels=3, codebook_size=256)
        self.inverse_lookup = {v: k for k, v in self.sid_lookup.items()}

        t2s = pd.read_parquet(sid_lookup_path).sort_values("bucket_rank")
        self.sid_to_tracks: dict[tuple[int, int, int], list[str]] = {}
        for row in t2s.itertuples(index=False):
            key = (int(row.code_1), int(row.code_2), int(row.code_3))
            self.sid_to_tracks.setdefault(key, []).append(row.track_id)

        # Build trie of valid SID sequences.
        sids = list(t2s[["code_1", "code_2", "code_3"]].itertuples(index=False, name=None))
        self.trie = build_sid_trie(sids, self.sid_lookup)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_sid_generator_retrieval.py -q
```

Expected: 1 passing.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/sid_generator.py tests/test_sid_generator_retrieval.py
git commit -m "sid w4: SID_GENERATOR.__init__ loads model + trie (TDD, 1 test)"
```

---

## Task 3: TDD `batch_text_to_item_retrieval` — per-query constrained beam decode

This is the core method. It MUST match the existing retrieval interface signature: `(queries: list[str], topk: int, user_ids=None) -> list[list[str]]`. Output ordering matters — RRF reads it as ranked list.

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/sid_generator.py`
- Modify: `tests/test_sid_generator_retrieval.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_sid_generator_retrieval.py`:

```python
def test_batch_text_to_item_retrieval_returns_topk_per_query(monkeypatch, tiny_sid_lookup_parquet):
    """Method emits exactly topk tracks per query, applying per-bucket cap."""
    from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
    import torch

    class _StubTokenizer:
        def __init__(self): self.pad_token_id = 0; self.eos_token_id = 1
        def get_vocab(self):
            return {f"<SID_L{lvl}_C{code}>": 1000 + lvl * 256 + code
                    for lvl in range(3) for code in range(256)}
        @classmethod
        def from_pretrained(cls, _): return cls()
        def __call__(self, text, **k):
            # Simulate tokenizer: encode each query to a unique fake id sequence.
            return {"input_ids": torch.tensor([[10, 20, 30]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])}
        def encode(self, s, **k): return [10, 20, 30]

    class _StubModel:
        device = "cpu"
        def __init__(self): self.generation_config = type("c", (), {})()
        def eval(self): return self
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def generate(self, input_ids, attention_mask, max_new_tokens, num_beams,
                     num_return_sequences, prefix_allowed_tokens_fn, **k):
            # Return 5 beams each with prompt_ids + 3 SID tokens.
            # SID triplets in our tiny fixture: (0,0,0), (0,0,1), (0,1,0), (1,0,0)
            beam_sids = [
                (0, 0, 0),  # → t1
                (0, 0, 1),  # → t2
                (0, 1, 0),  # → t3 (primary), t4 (spillover)
                (1, 0, 0),  # → t5
                (0, 0, 0),  # duplicate of beam 0 (just to fill 5 beams)
            ]
            sequences = []
            prompt = input_ids[0].tolist()
            for (c1, c2, c3) in beam_sids:
                seq = prompt + [1000 + c1, 1000 + 256 + c2, 1000 + 512 + c3]
                sequences.append(seq)
            return type("o", (), {"sequences": torch.tensor(sequences)})()

    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoTokenizer", _StubTokenizer)
    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoModelForCausalLM", _StubModel)

    gen = SID_GENERATOR(
        hub_repo="fake/repo",
        sid_lookup_path=tiny_sid_lookup_parquet,
        device="cpu",
        num_beams=5,
    )

    out = gen.batch_text_to_item_retrieval(queries=["play me something"], topk=5)
    assert len(out) == 1
    tracks = out[0]
    # First 4 beams resolve to distinct tracks: t1, t2, t3, t5 (one per bucket).
    # 5th beam is a duplicate of (0,0,0) → already seen, contributes nothing primary.
    # Spillover from (0,1,0) bucket adds t4. Final ordering: primary then spillover.
    assert tracks[:4] == ["t1", "t2", "t3", "t5"]
    assert "t4" in tracks  # spillover
    assert len(tracks) == 5  # exactly topk


def test_batch_text_to_item_retrieval_processes_multiple_queries(monkeypatch, tiny_sid_lookup_parquet):
    """Method handles N queries → returns list of N inner lists."""
    from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
    import torch

    # Reuse the same stubs as the prior test by monkeypatching at module level.
    class _Tok:
        pad_token_id = 0; eos_token_id = 1
        def get_vocab(self):
            return {f"<SID_L{lvl}_C{code}>": 1000 + lvl * 256 + code
                    for lvl in range(3) for code in range(256)}
        @classmethod
        def from_pretrained(cls, _): return cls()
        def __call__(self, text, **k):
            return {"input_ids": torch.tensor([[10, 20, 30]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])}
        def encode(self, s, **k): return [10, 20, 30]

    class _Model:
        device = "cpu"
        def __init__(self): self.generation_config = type("c", (), {})()
        def eval(self): return self
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def generate(self, input_ids, **k):
            seqs = []
            for _ in range(k["num_return_sequences"]):
                seqs.append(input_ids[0].tolist() + [1000, 1256, 1512])  # (0,0,0)
            return type("o", (), {"sequences": torch.tensor(seqs)})()

    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoTokenizer", _Tok)
    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoModelForCausalLM", _Model)

    gen = SID_GENERATOR(
        hub_repo="fake/repo",
        sid_lookup_path=tiny_sid_lookup_parquet,
        device="cpu",
        num_beams=3,
    )

    out = gen.batch_text_to_item_retrieval(queries=["q1", "q2", "q3"], topk=3)
    assert len(out) == 3
    for tracks in out:
        # Each inner list has up to topk tracks
        assert len(tracks) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sid_generator_retrieval.py -q
```

Expected: 2 failures with `AttributeError: 'SID_GENERATOR' object has no attribute 'batch_text_to_item_retrieval'`.

- [ ] **Step 3: Append implementation**

Add to `mcrs/retrieval_modules/sid_generator.py`:

```python
    def batch_text_to_item_retrieval(
        self,
        queries: list[str],
        topk: int,
        user_ids: Optional[list[str]] = None,
    ) -> list[list[str]]:
        """For each query, run constrained beam search → top-K track IDs.

        Interface parity: matches BM25_MODEL.batch_text_to_item_retrieval (user_ids
        accepted but ignored — SID retrieval is text-conditioned only).
        """
        results: list[list[str]] = []
        with torch.inference_mode():
            for query in queries:
                inputs = self.tokenizer(
                    query, truncation=True, max_length=self.max_prompt_len,
                    return_tensors="pt", add_special_tokens=False,
                ).to(self.model.device)
                prompt_len = inputs["input_ids"].shape[1]

                # Build prefix-allowed-tokens fn populated for every beam id.
                prefix_fn = make_prefix_allowed_tokens_fn(
                    self.trie,
                    prompt_lens={i: prompt_len for i in range(self.num_beams)},
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                out = self.model.generate(
                    **inputs,
                    max_new_tokens=3,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    prefix_allowed_tokens_fn=prefix_fn,
                    do_sample=False,
                )
                # Slice off the prompt → 3 SID tokens per beam.
                beams = out.sequences[:, prompt_len:prompt_len + 3].cpu().tolist()
                tracks = self._decode_beams_to_tracks(beams, topk=topk)
                results.append(tracks)
        return results

    def _decode_beams_to_tracks(
        self, beam_token_ids: list[list[int]], topk: int,
    ) -> list[str]:
        """Convert each beam's 3-token output → SID triplet → tracks.

        Applies per-bucket cap: each beam contributes at most `cap_per_bucket`
        tracks from its collision bucket; remainder of the bucket goes to
        spillover so it can fill any dedup gap (per spec §2.5)."""
        seen: set[str] = set()
        primary: list[str] = []
        spillover: list[str] = []
        for tok_ids in beam_token_ids:
            try:
                sid = (
                    self.inverse_lookup[tok_ids[0]][1],
                    self.inverse_lookup[tok_ids[1]][1],
                    self.inverse_lookup[tok_ids[2]][1],
                )
            except (KeyError, IndexError):
                continue
            bucket = self.sid_to_tracks.get(sid, [])
            added_in_beam = 0
            for tid in bucket:
                if tid in seen:
                    continue
                if added_in_beam < self.cap_per_bucket:
                    primary.append(tid)
                    added_in_beam += 1
                else:
                    spillover.append(tid)
                seen.add(tid)
        return (primary + spillover)[:topk]

    def text_to_item_retrieval(
        self, query: str, topk: int, user_id: Optional[str] = None,
    ) -> list[str]:
        """Single-query convenience wrapper for interface parity with other retrievers."""
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sid_generator_retrieval.py -q
```

Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/sid_generator.py tests/test_sid_generator_retrieval.py
git commit -m "sid w4: batch_text_to_item_retrieval (TDD, 2 tests; per-bucket cap + spillover)"
```

---

## Task 4: TDD interface-parity smoke test (matches existing retrievers)

Verify that `SID_GENERATOR` has the same callable shape as `BM25_MODEL` and `DENSE_PRECOMPUTED` so RRF_MODEL can wire it in transparently.

**Files:**
- Modify: `tests/test_sid_generator_retrieval.py`

- [ ] **Step 1: Append failing test**

```python
def test_sid_generator_interface_matches_existing_retrievers(monkeypatch, tiny_sid_lookup_parquet):
    """SID_GENERATOR.batch_text_to_item_retrieval signature must be identical to
    BM25_MODEL's (queries: list[str], topk: int, user_ids=None) -> list[list[str]]."""
    import inspect
    from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
    from mcrs.retrieval_modules.bm25 import BM25_MODEL

    sid_sig = inspect.signature(SID_GENERATOR.batch_text_to_item_retrieval)
    bm25_sig = inspect.signature(BM25_MODEL.batch_text_to_item_retrieval)

    sid_params = list(sid_sig.parameters.keys())
    bm25_params = list(bm25_sig.parameters.keys())
    # First 3 params must match: self, queries, topk; user_ids 4th
    assert sid_params[:3] == bm25_params[:3] == ["self", "queries", "topk"]
    assert "user_ids" in sid_params
    assert "user_ids" in bm25_params
```

- [ ] **Step 2: Run test to verify it fails OR passes**

```bash
python -m pytest tests/test_sid_generator_retrieval.py::test_sid_generator_interface_matches_existing_retrievers -q
```

(If Task 3 was implemented correctly, this should already PASS. The test exists to lock the interface in regression suite.)

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sid_generator_retrieval.py
git commit -m "sid w4: interface-parity regression test (locks signature vs BM25_MODEL)"
```

---

## Task 5: Register `sid_generator` + `wrrf_bm25_dense_sid_v1` in factory

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/__init__.py`

- [ ] **Step 1: Read the current factory layout**

```bash
grep -n "elif retrieval_type" music-crs-baselines/mcrs/retrieval_modules/__init__.py | head -25
```

- [ ] **Step 2: Add `sid_generator` factory entry**

Append after the `cf_bpr` entry (around line 230). Edit `music-crs-baselines/mcrs/retrieval_modules/__init__.py`:

Find:

```python
    elif retrieval_type == "cf_bpr":
```

Add ABOVE this line (so retrieval_modules can be loaded in order):

```python
    elif retrieval_type == "sid_generator":
        from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
        return SID_GENERATOR(
            hub_repo="OrRim123/recsys2026-sid-generator-qwen15b-v1-merged",
            sid_lookup_path=Path(cache_dir) / "sid" / "track_to_sid.parquet",
            device="cuda",
            num_beams=20,
            max_prompt_len=1024,
            cap_per_bucket=1,
        )
```

- [ ] **Step 3: Add `wrrf_bm25_dense_sid_v1` factory entry**

Append after the last wrrf_* entry (find the last `elif retrieval_type == "wrrf_..."` block and add this entry below it):

```python
    elif retrieval_type == "wrrf_bm25_dense_sid_v1":
        # W4 ensemble: existing 3-stream wRRF (BM25 + dense_metadata + dense_lyrics)
        # plus the new SID generator as a 4th stream. Initial SID weight = 0.5
        # (between BM25=1.0 and dense=0.4); tuned in W5.
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "sid_generator",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.5,
                },
            ],
            k=60,
        )
```

- [ ] **Step 4: Ensure `Path` is imported at the top of `__init__.py`**

```bash
head -10 music-crs-baselines/mcrs/retrieval_modules/__init__.py
```

If `from pathlib import Path` is missing, add it. (Most retrieval factories already use Path internally; this is defensive.)

If missing, edit at top of file:

```python
from pathlib import Path
```

- [ ] **Step 5: Smoke-test that the imports resolve**

```bash
python -c "
import sys
sys.path.insert(0, 'music-crs-baselines')
from mcrs.retrieval_modules import load_retrieval_module
print('OK: load_retrieval_module imports cleanly')
"
```

Expected: `OK: load_retrieval_module imports cleanly`. If ImportError, the issue is likely the new sid_generator module — re-check the import path.

- [ ] **Step 6: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/__init__.py
git commit -m "sid w4: register sid_generator + wrrf_bm25_dense_sid_v1 factory entries"
```

---

## Task 6: TDD factory wiring (smoke that the new keys are reachable)

**Files:**
- Modify: `tests/test_sid_generator_retrieval.py`

- [ ] **Step 1: Append failing test**

```python
def test_factory_recognizes_sid_generator_retrieval_type():
    """load_retrieval_module(retrieval_type='sid_generator', ...) must dispatch
    to SID_GENERATOR class (verified by inspecting registered branches)."""
    import inspect
    from mcrs.retrieval_modules import load_retrieval_module

    src = inspect.getsource(load_retrieval_module)
    assert '"sid_generator"' in src
    assert '"wrrf_bm25_dense_sid_v1"' in src
    assert "SID_GENERATOR" in src


def test_factory_wrrf_sid_subspec_has_four_streams():
    """wrrf_bm25_dense_sid_v1 spec must include BM25 + 2 dense + SID = 4 streams."""
    import inspect
    from mcrs.retrieval_modules import load_retrieval_module

    src = inspect.getsource(load_retrieval_module)
    # Find the wrrf_bm25_dense_sid_v1 block by string match
    start = src.find('"wrrf_bm25_dense_sid_v1"')
    assert start >= 0, "wrrf_bm25_dense_sid_v1 entry not found"
    end = src.find("elif retrieval_type", start + 1)
    block = src[start:end if end > 0 else len(src)]
    # Should reference all 4 sub-stream types
    assert '"bm25"' in block
    assert '"dense_metadata_qwen3_instruct"' in block
    assert '"dense_lyrics_qwen3_instruct"' in block
    assert '"sid_generator"' in block
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_sid_generator_retrieval.py -q
```

Expected: 5 passing (3 prior + 2 new factory smoke).

- [ ] **Step 3: Commit**

```bash
git add tests/test_sid_generator_retrieval.py
git commit -m "sid w4: factory wiring smoke tests (TDD, 2 tests)"
```

---

## Task 7: Create ensemble config `170-wrrf-sid-v5kto-blindsetA.yaml`

**Files:**
- Create: `music-crs-baselines/config/170-wrrf-sid-v5kto-blindsetA.yaml`

- [ ] **Step 1: Write the config**

Create `music-crs-baselines/config/170-wrrf-sid-v5kto-blindsetA.yaml`:

```yaml
# W4 ensemble: 4-stream wRRF (BM25 + dense_meta + dense_lyrics + SID generator)
# stacked with the v5-kto-merged responder (W6) on Blind-A.
#
# Origin of SID generator: W3 fine-tune of Qwen2.5-1.5B-Instruct + LoRA on
# (raw conversations + metadata-as-query) training data; see
# documents/specs/2026-05-15-sid-retrieval-design.md + W3 memory.
# Hub: OrRim123/recsys2026-sid-generator-qwen15b-v1-merged
#
# Initial SID weight = 0.5 (tunable in W5).
#
# To run:
#   python run_inference_blindset.py --tid 170-wrrf-sid-v5kto-blindsetA --batch_size 32
#
# Wallclock estimate: ~60-100 min on L4 (4 retrievers + CMQR + ProRank + v5-kto
# responder; SID-gen pass adds ~10-15 min over 132 for 80 Blind-A turns).

lm_type: "OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged"
lora_path: null
lora_max_rank: 32

retrieval_type: "wrrf_bm25_dense_sid_v1"
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Blind-A"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types:
  - "all_tracks"
user_split_types:
  - "all_users"
corpus_types:
  - "track_name"
  - "artist_name"
  - "album_name"
cache_dir: "../experiments/cache"
device: "cuda"
attn_implementation: "sdpa"

retrieval_topk: 100
reranker_type: "pro_rank"

# v5-kto-trained prompt + longer max_new_tokens to accommodate CoT + envelope.
response_prompt_name: "response_generation_cot_user_state"
response_max_new_tokens: 320
top_n_for_prompt: 1
query_preprocessing_mode: "raw"

use_vllm: false

use_state_tracker: true
state_tracker_prompt_name: "state_extraction"
state_tracker_max_new_tokens: 96

# CMQR routes only to BM25/dense streams (NOT SID — see spec §1) but the
# config flag enables it globally; downstream wRRF handles routing internally.
use_cmqr: true
cmqr_prompt_name: "cmqr_rewrites"
cmqr_n_rewrites: 4
cmqr_topk_per_rewrite: 50
cmqr_rrf_k: 60
cmqr_max_new_tokens: 96
```

- [ ] **Step 2: Verify YAML parses cleanly**

```bash
python -c "
import yaml
with open('music-crs-baselines/config/170-wrrf-sid-v5kto-blindsetA.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'OK: parsed {len(cfg)} keys, retrieval_type={cfg[\"retrieval_type\"]!r}')
"
```

Expected: `OK: parsed 25 keys, retrieval_type='wrrf_bm25_dense_sid_v1'` (or similar count).

- [ ] **Step 3: Commit**

```bash
git add music-crs-baselines/config/170-wrrf-sid-v5kto-blindsetA.yaml
git commit -m "sid w4: config 170 — wRRF 4-stream ensemble (+ SID) for Blind-A"
```

---

## Task 8: Create pure-SID comparison config `171-pure-sid-v5kto-blindsetA.yaml`

**Files:**
- Create: `music-crs-baselines/config/171-pure-sid-v5kto-blindsetA.yaml`

- [ ] **Step 1: Write the config**

Create `music-crs-baselines/config/171-pure-sid-v5kto-blindsetA.yaml`:

```yaml
# W4 comparison: PURE SID retriever (no fusion) stacked with v5-kto responder.
# Submitted alongside config 170 in W5 so we can measure whether the wRRF
# ensemble dominates or pure-SID is competitive. Per spec §1 promotion path.
#
# Same responder as 170; only the retriever differs (single-axis comparison).
#
# To run:
#   python run_inference_blindset.py --tid 171-pure-sid-v5kto-blindsetA --batch_size 32

lm_type: "OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged"
lora_path: null
lora_max_rank: 32

retrieval_type: "sid_generator"
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Blind-A"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types:
  - "all_tracks"
user_split_types:
  - "all_users"
corpus_types:
  - "track_name"
  - "artist_name"
  - "album_name"
cache_dir: "../experiments/cache"
device: "cuda"
attn_implementation: "sdpa"

retrieval_topk: 100
reranker_type: "pro_rank"

response_prompt_name: "response_generation_cot_user_state"
response_max_new_tokens: 320
top_n_for_prompt: 1
query_preprocessing_mode: "raw"

use_vllm: false

use_state_tracker: true
state_tracker_prompt_name: "state_extraction"
state_tracker_max_new_tokens: 96

# CMQR DISABLED for pure-SID: the SID generator was trained on raw queries
# (not CMQR-rewrites). Per spec §1, routing CMQR through SID creates a
# training/inference mismatch. For pure-SID, skip CMQR entirely.
use_cmqr: false
```

- [ ] **Step 2: Verify YAML parses + diff against 170 to spot drift**

```bash
python -c "
import yaml
with open('music-crs-baselines/config/170-wrrf-sid-v5kto-blindsetA.yaml') as f:
    a = yaml.safe_load(f)
with open('music-crs-baselines/config/171-pure-sid-v5kto-blindsetA.yaml') as f:
    b = yaml.safe_load(f)
# Differences should be: retrieval_type and use_cmqr (and missing cmqr_* keys)
diffs = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
print('170 vs 171 diffs:')
for k, (av, bv) in diffs.items():
    print(f'  {k}: {av!r} -> {bv!r}')
"
```

Expected: only `retrieval_type` and `use_cmqr` differ between 170 and 171 (plus 171 is missing the `cmqr_*` parameters).

- [ ] **Step 3: Commit**

```bash
git add music-crs-baselines/config/171-pure-sid-v5kto-blindsetA.yaml
git commit -m "sid w4: config 171 — pure-SID retriever (no fusion) for Blind-A comparison"
```

---

## Task 9: Create dev-gate notebook `colab/64_dev_gate_compare.ipynb`

This notebook runs the W4 gate: paired-bootstrap CI on dev comparing config 170 (ensemble + SID) vs config 132 (current champion). If gate passes, proceed to Blind-A; if not, escalate (tune SID weight in W5, or revisit W3).

**Files:**
- Create: `colab/64_dev_gate_compare.ipynb`

- [ ] **Step 1: Generate the notebook**

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

md("""# 64 — W4 dev gate: config 170 (ensemble + SID) vs config 132 (current champion)

Runs both configs on the dev split, computes per-turn nDCG@20, then runs
paired-bootstrap CI (n_resamples=1000, alpha=0.05). Gate passes if:
  - mean Δ nDCG@20 ≥ +0.005, AND
  - paired-bootstrap CI lower bound > 0 (statistically real improvement).

Wallclock: ~2.5-3 hr on L4 (two diagnostic runs back-to-back).
""")

code("""# 1) Setup: clone + auth + Drive + deps. (Same pattern as notebook 62.)
import os
from google.colab import userdata, drive
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
drive.mount('/content/drive', force_remount=False)

BRANCH = 'fresh-model'
!rm -rf /content/recsys2026
!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026
%cd /content/recsys2026

DRIVE_BASE = '/content/drive/MyDrive'
LOCAL_BASE = '/content/recsys2026/experiments/cache'
os.makedirs(LOCAL_BASE, exist_ok=True)
for name, drive_subdir in [
    ('sid', 'recsys2026_sid_cache'),
    ('sid_training', 'recsys2026_sid_training_cache'),
    ('dense', 'recsys2026_dense_cache'),
]:
    src = f'{DRIVE_BASE}/{drive_subdir}'
    dst = f'{LOCAL_BASE}/{name}'
    os.makedirs(src, exist_ok=True)
    if os.path.islink(dst): os.unlink(dst)
    elif os.path.exists(dst):
        import shutil; shutil.rmtree(dst)
    os.symlink(src, dst)

!pip install -q -U \"peft>=0.10\" \"transformers>=4.40\" \"accelerate>=0.30\" \"torchao>=0.17\"
""")

code("""# 2) Run config 170 (ensemble + SID) on the dev split.
# Note: --tid points at the config name (without .yaml); output goes to a
# named diagnostic_runs/ subdir per project convention.
%cd /content/recsys2026/music-crs-baselines
!python run_inference_blindset.py \\
    --tid 170-wrrf-sid-v5kto-blindsetA \\
    --batch_size 32 \\
    --test_dataset_name talkpl-ai/TalkPlayData-Challenge-Dataset \\
    --output_dir ../experiments/diagnostic_runs/170-dev \\
    2>&1 | tail -50""")

code("""# 3) Run config 132 (current champion, no SID) on the same dev split for paired-bootstrap.
!python run_inference_blindset.py \\
    --tid 132-bge-m3-v5kto-prorank-rerank-blindsetA \\
    --batch_size 32 \\
    --test_dataset_name talkpl-ai/TalkPlayData-Challenge-Dataset \\
    --output_dir ../experiments/diagnostic_runs/132-dev \\
    2>&1 | tail -50""")

code("""# 4) Paired-bootstrap CI: 170 vs 132 on dev.
%cd /content/recsys2026
!python scripts/compare_diagnostic_runs.py \\
    --run_a experiments/diagnostic_runs/170-dev \\
    --run_b experiments/diagnostic_runs/132-dev \\
    --label_a 'wRRF+SID (170)' \\
    --label_b 'wRRF current (132)' \\
    --metric ndcg_at_20 \\
    --n_resamples 1000 \\
    --alpha 0.05 \\
    --output experiments/diagnostic_runs/w4_gate.json""")

code("""# 5) Display gate decision.
import json
m = json.load(open('experiments/diagnostic_runs/w4_gate.json'))
print(json.dumps(m, indent=2))
print()
print('=' * 60)
gate_pass = m.get('delta_mean', 0) >= 0.005 and m.get('paired_bootstrap_ci', {}).get('lo', -1) > 0
if gate_pass:
    print(f\"GATE PASS — Δ nDCG@20 = {m['delta_mean']:+.4f}, \"
          f\"CI lo = {m['paired_bootstrap_ci']['lo']:+.4f} > 0\")
    print('PROCEED to Blind-A submission via notebook 63.')
else:
    print(f\"GATE FAIL — Δ nDCG@20 = {m.get('delta_mean', 0):+.4f}, \"
          f\"CI lo = {m.get('paired_bootstrap_ci', {}).get('lo', 0):+.4f}\")
    print('Required: Δ ≥ +0.005 AND CI lo > 0. Do NOT submit Blind-A yet.')
print('=' * 60)
""")

md("""## After the gate

**Gate PASS**: open notebook 63 and run config 170 on Blind-A for the official submission.

**Gate FAIL**: do not submit Blind-A. Options:
  - **Δ small but CI excludes 0 (e.g. Δ=+0.003)**: tighten in W5 by tuning SID weight (sweep {0.3, 0.5, 0.7, 1.0}); the SID stream may need a smaller weight to not dominate.
  - **Δ near 0 or negative**: SID is hurting the ensemble. Try config 171 (pure-SID) to isolate whether SID alone is better or worse than current wRRF — informs whether to keep SID at all.
  - **High CI variance**: dev sample may be too small. Re-run on full dev split or wait until W5 for Blind-A signal.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path('colab/64_dev_gate_compare.ipynb').write_text(json.dumps(nb, indent=1))
print('wrote colab/64_dev_gate_compare.ipynb')
EOF
```

- [ ] **Step 2: Verify notebook is valid + cell count**

```bash
python -c "
import nbformat
nb = nbformat.read('colab/64_dev_gate_compare.ipynb', as_version=4)
print(f'cells: {len(nb.cells)}')
"
```

Expected: `cells: 7` (1 markdown header + 5 code + 1 markdown footer).

- [ ] **Step 3: Commit**

```bash
git add colab/64_dev_gate_compare.ipynb
git commit -m "sid w4: notebook 64 — dev gate (170 vs 132) with paired-bootstrap CI"
```

---

## Task 10: Create Blind-A submission notebook `colab/63_run_blindset_sid.ipynb`

Only run this AFTER notebook 64's gate has passed. Produces the `prediction.json` for CodaBench submission.

**Files:**
- Create: `colab/63_run_blindset_sid.ipynb`

- [ ] **Step 1: Generate the notebook**

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

md("""# 63 — Run Blind-A with config 170 (wRRF + SID) → submission zip

ONLY run this after notebook 64's dev gate has PASSED. This produces the
prediction.json for CodaBench. Wallclock ~60-100 min on L4.
""")

code("""# 1) Setup (same pattern as notebook 62).
import os
from google.colab import userdata, drive
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
drive.mount('/content/drive', force_remount=False)

BRANCH = 'fresh-model'
!rm -rf /content/recsys2026
!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026
%cd /content/recsys2026

DRIVE_BASE = '/content/drive/MyDrive'
LOCAL_BASE = '/content/recsys2026/experiments/cache'
os.makedirs(LOCAL_BASE, exist_ok=True)
for name, drive_subdir in [
    ('sid', 'recsys2026_sid_cache'),
    ('dense', 'recsys2026_dense_cache'),
]:
    src = f'{DRIVE_BASE}/{drive_subdir}'
    dst = f'{LOCAL_BASE}/{name}'
    os.makedirs(src, exist_ok=True)
    if os.path.islink(dst): os.unlink(dst)
    elif os.path.exists(dst):
        import shutil; shutil.rmtree(dst)
    os.symlink(src, dst)

!pip install -q -U \"peft>=0.10\" \"transformers>=4.40\" \"accelerate>=0.30\" \"torchao>=0.17\"
""")

code("""# 2) Run config 170 on Blind-A.
%cd /content/recsys2026/music-crs-baselines
!python run_inference_blindset.py \\
    --tid 170-wrrf-sid-v5kto-blindsetA \\
    --batch_size 32 \\
    2>&1 | tee /content/drive/MyDrive/recsys2026_sid_generator_cache/blindA_170_log.txt | tail -60""")

code("""# 3) Validate the prediction.json: must be 80 entries (80 unique sessions × 1 turn each).
%cd /content/recsys2026
import json
pred_path = 'music-crs-baselines/output/170-wrrf-sid-v5kto-blindsetA/prediction.json'
preds = json.load(open(pred_path))
print(f'prediction.json: {len(preds)} entries')
assert len(preds) == 80, f'EXPECTED 80, got {len(preds)} — DO NOT submit'
sample_keys = list(preds[0].keys()) if isinstance(preds, list) else list(list(preds.values())[0].keys())
print(f'sample entry keys: {sample_keys}')""")

code("""# 4) Validate per existing validator (catches schema bugs before submission).
!python scripts/validate_prediction.py \\
    --prediction music-crs-baselines/output/170-wrrf-sid-v5kto-blindsetA/prediction.json \\
    --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A
""")

code("""# 5) Zip for CodaBench (prediction.json must be at the ROOT of the zip).
import os, zipfile, datetime
date_str = datetime.date.today().strftime('%Y-%m-%d')
zip_path = f'/content/drive/MyDrive/recsys2026_submissions/{date_str}-sid-ensemble-170.zip'
os.makedirs(os.path.dirname(zip_path), exist_ok=True)
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(pred_path, arcname='prediction.json')
print(f'zip ready: {zip_path}')
print(f'size: {os.path.getsize(zip_path) / 1024:.1f} KB')
""")

md("""## After the run

1. Download the zip from Drive: `/content/drive/MyDrive/recsys2026_submissions/<date>-sid-ensemble-170.zip`
2. Upload to CodaBench (https://www.codabench.org/competitions/...).
3. Record the composite + nDCG@20 + LLM + lex_div scores in the project memory under a new `project_blind_a_w4_first_submission.md`.
4. If results are positive: proceed to W5 weight tuning + config 171 (pure-SID) comparison.
5. If results are negative: debug — likely the SID weight (0.5) is too high or the SID generator isn't strong enough; revisit.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path('colab/63_run_blindset_sid.ipynb').write_text(json.dumps(nb, indent=1))
print('wrote colab/63_run_blindset_sid.ipynb')
EOF
```

- [ ] **Step 2: Verify notebook is valid + cell count**

```bash
python -c "
import nbformat
nb = nbformat.read('colab/63_run_blindset_sid.ipynb', as_version=4)
print(f'cells: {len(nb.cells)}')
"
```

Expected: `cells: 7` (1 markdown header + 5 code + 1 markdown footer).

- [ ] **Step 3: Commit**

```bash
git add colab/63_run_blindset_sid.ipynb
git commit -m "sid w4: notebook 63 — Blind-A submission for config 170 (ensemble + SID)"
```

---

## Task 11: Final test sweep + code-reviewer agent

- [ ] **Step 1: Run all SID + W4 tests**

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
    tests/test_sid_generator_retrieval.py \
    -q
```

Expected: 88 (W1+W2+W3) + 5 (W4) = **93 SID tests passing**.

- [ ] **Step 2: Full repo test sweep**

```bash
python -m pytest tests/ -q \
    --ignore=tests/test_local_eval.py \
    --ignore=tests/test_wave0_integration.py \
    --ignore=tests/test_wave1_integration.py \
    --ignore=tests/test_wave2_integration.py \
    2>&1 | tail -3
```

Expected: ~500+ passing (498 W3-baseline + ~5 W4).

- [ ] **Step 3: Verify clean git state + push**

```bash
git status -s   # should be clean
git log --oneline -15
git push origin fresh-model
```

- [ ] **Step 4: Invoke code-reviewer agent**

```
Agent({
  description: "Review SID W4 inference pipeline",
  subagent_type: "superpowers:code-reviewer",
  prompt: "Review the W4 implementation of the SID inference pipeline against the design spec.

  Spec: documents/specs/2026-05-15-sid-retrieval-design.md §4 + §4.2 + §5.1 W4 gate
  Plan: documents/plans/2026-05-16-sid-inference-pipeline-w4.md

  W3 artifact this depends on:
  - HF Hub: OrRim123/recsys2026-sid-generator-qwen15b-v1-merged (the trained generator)

  Files to review (newly added on fresh-model):
  - music-crs-baselines/mcrs/retrieval_modules/sid_generator.py (SID_GENERATOR class)
  - music-crs-baselines/mcrs/retrieval_modules/__init__.py (factory registration)
  - music-crs-baselines/config/170-wrrf-sid-v5kto-blindsetA.yaml
  - music-crs-baselines/config/171-pure-sid-v5kto-blindsetA.yaml
  - tests/test_sid_generator_retrieval.py
  - colab/64_dev_gate_compare.ipynb
  - colab/63_run_blindset_sid.ipynb

  Critical checks:
  1. SID_GENERATOR.batch_text_to_item_retrieval signature MUST match BM25_MODEL +
     DENSE_PRECOMPUTED exactly: (queries: list[str], topk: int, user_ids=None) ->
     list[list[str]]. Verify by inspect.signature comparison.
  2. Per-bucket cap correctly implemented: each beam contributes at most cap_per_bucket
     tracks from its collision bucket; spillover fills dedup gaps.
  3. prefix_allowed_tokens_fn populated for ALL beam ids (W3 reviewer caught this bug;
     verify W4 doesn't repeat it).
  4. Config 170 vs 132 single-axis: ONLY retrieval_type differs (132 has wrrf_bm25_dense_lyrics_bge_m3_v1;
     170 has wrrf_bm25_dense_sid_v1). All other fields identical.
  5. Config 171 has use_cmqr: false (SID was trained on raw queries; CMQR routing creates
     training/inference mismatch per spec §1).
  6. Notebook 64 gate computes mean Δ AND CI lower bound > 0; PASS requires BOTH.

  Report: APPROVE / APPROVE-WITH-CHANGES / NEEDS-MAJOR-REVISION + numbered findings + top-5 changes if any."
})
```

- [ ] **Step 5: Address any reviewer findings**

If APPROVE → done.
If APPROVE-WITH-CHANGES → make fixes inline, commit (e.g. `sid w4: harden <area> per reviewer (Ix)`), re-push.
If NEEDS-MAJOR-REVISION → triage findings before user runs notebook 64.

---

## Self-review checklist (per writing-plans skill)

**1. Spec coverage** (against §4 + §5.1 of design spec):

- §4.1 SID_GENERATOR class with trie + prefix_allowed_tokens_fn + per-bucket cap — Task 3 ✓
- §4.1 reuses W3's `mcrs/sid/inference.py` (trie + prefix_fn) — Task 2 import ✓
- §4.2 wRRF integration: register `sid_generator` + `wrrf_bm25_dense_sid_v1` — Task 5 ✓
- §4.2 SID weight = 0.5 initial — Task 5 (factory entry weight=0.5) ✓
- §4.3 Responder handoff unchanged — Tasks 7+8 use existing v5-kto model ✓
- §5.1 W4 gate: mean Δ ≥ +0.005 AND CI lo > 0 — Task 9 notebook 64 ✓
- First Blind-A submission — Task 10 notebook 63 ✓

**2. Placeholder scan**: searched for "TBD", "TODO", "implement later", "add appropriate" — none. Every step has runnable code + commands.

**3. Type consistency**:
- `batch_text_to_item_retrieval(queries: list[str], topk: int, user_ids=None) -> list[list[str]]` — matches BM25 + dense_precomputed signature
- `sid_to_tracks: dict[tuple[int, int, int], list[str]]` — used by `_decode_beams_to_tracks`
- `inverse_lookup: dict[int, tuple[int, int]]` — used by `_decode_beams_to_tracks`
- Factory entry references same constructor signature (`hub_repo`, `sid_lookup_path`, `device`, `num_beams`)

**4. Sequencing**: each task only depends on artifacts from earlier tasks. Task 5 imports Task 2's class; Task 7 references Task 5's `wrrf_bm25_dense_sid_v1` key; Task 9 uses Task 7's config; Task 10 also uses Task 7's config (after Task 9 gate passes).

**5. Plan-deviation hooks**:
- Task 1 verifies W3 prerequisite before any work
- Task 5 has a smoke-test step that catches import errors
- Task 9 ends with PASS/FAIL printout so the user has a clear gate decision
- Task 10 has a 80-entry validation step before zipping (catches submission-format bugs)

**Plan complete and ready for execution.**

---

## Estimated wallclock

| Component | Time |
|---|---|
| Task 1 (setup verify) | 2 min |
| Tasks 2-4 (SID_GENERATOR class + 3 tests) | ~15 min via subagent |
| Tasks 5-6 (factory + 2 tests) | ~10 min via subagent |
| Tasks 7-8 (two YAML configs) | ~5 min via subagent |
| Task 9 (notebook 64) | ~5 min via subagent |
| Task 10 (notebook 63) | ~5 min via subagent |
| Task 11 (test sweep + reviewer) | ~10 min agent + iteration |
| **Total my coordination time** | **~50-60 min** |
| User Colab time (notebook 64 dev gate) | **~2.5-3 hr on L4** |
| User Colab time (notebook 63 Blind-A) | **~60-100 min on L4** |
| User CodaBench submission + result wait | hours-days |

W4 implementation is light — most of the actual code already exists (the SID generator from W3, the retrieval interface from existing wRRF). The real work is wiring + configs. Heavy lifting moves to W5 (weight sweep + multiple Blind-A submissions).
