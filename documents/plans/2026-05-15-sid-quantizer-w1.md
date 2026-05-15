# SID Quantizer (W1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an RQ-VAE quantizer that turns each of 47K track embeddings (concatenated text + CF + audio) into a unique 3-token Semantic ID, validated by 4 gates, with a 3-seed reproducibility protocol.

**Architecture:** RQ-VAE (Residual Quantization VAE) using `vector-quantize-pytorch`'s `ResidualVQ` class, augmented with a Sinkhorn-uniform loss term (LC-Rec recipe motivation, λ tuned in v1) to prevent codebook collapse. Train 3 seeds, pick best by cluster-purity gate, pin chosen artifact by SHA256 hash. Output: `track_to_sid.parquet` keyed by `track_id`.

**Tech Stack:** Python 3.10, PyTorch (existing), `vector-quantize-pytorch` (new dep), `datasets` (existing), `pandas` + `pyarrow` (existing), `numpy` (existing).

**Spec reference:** `documents/specs/2026-05-15-sid-retrieval-design.md` §2 (SID Quantizer), §0.2 (TDD), §0.3 (cache discipline), §0.4 (commit cadence).

---

## File structure

| Path | Type | Responsibility |
|---|---|---|
| `mcrs/sid/__init__.py` | new | Expose `SIDQuantizer` and `SIDLookupTable` |
| `mcrs/sid/quantizer.py` | new | `SIDQuantizer` class: encode embedding → 3-token SID; `SIDLookupTable` class: decode SID → track_ids (handles collision buckets) |
| `mcrs/sid/preprocessing.py` | new | Pure functions: `concat_modalities`, `compute_collision_buckets`, `dedup_with_per_bucket_cap` |
| `mcrs/sid/validation.py` | new | Pure functions for the 4 gates: `compute_relative_mse_gate`, `validate_codebook_utilization`, `validate_cluster_purity` (gate 4 deferred to W3 per spec §2.4) |
| `scripts/build_sid_quantizer.py` | new | Orchestration: load 47K embeddings from HF → train RQ-VAE for one seed → save → run gates 1-3 |
| `scripts/pick_best_sid_quantizer.py` | new | Read 3 trained quantizers, pick best by cluster-purity, write SHA256 hash pin |
| `tests/test_sid_preprocessing.py` | new | TDD tests for the pure functions in preprocessing.py |
| `tests/test_sid_validation.py` | new | TDD tests for validation gate functions |
| `tests/test_sid_quantizer.py` | new | TDD tests for SIDQuantizer + SIDLookupTable round-trip |
| `colab/60_build_sid_quantizer.ipynb` | new | Colab wrapper: clone, deps, run train script ×3 seeds, run pick-best, push artifacts to Drive |

**Output artifacts** (all to `experiments/cache/sid/` symlinked to Drive):
- `quantizer_seed{42,123,7}.pt` — three trained RQ-VAE checkpoints
- `quantizer_seed{42,123,7}_assignments.parquet` — per-seed (track_id, code_1, code_2, code_3) assignments
- `quantizer_seed{42,123,7}_gates.json` — per-seed gate scores
- `quantizer_chosen.pt` (symlink to chosen) + `track_to_sid.parquet` (chosen seed's assignments + collision buckets) + `quantizer_chosen.sha256` (hash pin)

---

## Task 1: Setup directory structure + new dependency

**Files:**
- Create: `mcrs/sid/__init__.py` (empty)
- Modify: `pyproject.toml` (or wherever deps live; check first)

- [ ] **Step 1: Verify directory layout, create empty package**

```bash
ls /Users/orrimoch/PythonProjs/recsys2026/mcrs/ 2>/dev/null || ls /Users/orrimoch/PythonProjs/recsys2026/music-crs-baselines/mcrs/
```

Expected: a `mcrs` package directory exists (likely under `music-crs-baselines/`). Note the actual path; all `mcrs/sid/...` paths in this plan should be interpreted relative to that base.

- [ ] **Step 2: Create the `sid` subpackage**

```bash
mkdir -p music-crs-baselines/mcrs/sid
touch music-crs-baselines/mcrs/sid/__init__.py
```

- [ ] **Step 3: Add `vector-quantize-pytorch` dependency to install command**

This is a Colab notebook deploy — we install at runtime, not via pyproject. Note for `colab/60_build_sid_quantizer.ipynb`: the install cell will need:
```
!pip install -q --upgrade vector-quantize-pytorch
```
(No file edit yet; just record this for Task 11.)

- [ ] **Step 4: Verify `vector-quantize-pytorch` is installable + importable**

```bash
pip install vector-quantize-pytorch
python -c "from vector_quantize_pytorch import ResidualVQ; print('OK')"
```

Expected: `OK`. If it fails, check Python version compatibility (≥3.8 required).

- [ ] **Step 5: Commit setup**

```bash
git add music-crs-baselines/mcrs/sid/__init__.py
git commit -m "sid w1: scaffold mcrs/sid package"
```

---

## Task 2: TDD `concat_modalities` (L2-normalize + concat text + CF + audio)

**Files:**
- Create: `music-crs-baselines/mcrs/sid/preprocessing.py`
- Test: `tests/test_sid_preprocessing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sid_preprocessing.py`:

```python
"""Tests for SID quantizer preprocessing pure functions."""
import numpy as np
import pytest


def test_concat_modalities_l2_normalizes_each_modality_independently():
    """Each modality should be L2-normalized to unit norm before concatenation."""
    from mcrs.sid.preprocessing import concat_modalities

    text = np.array([3.0, 4.0])         # norm = 5
    cf = np.array([1.0, 0.0, 0.0])      # norm = 1 (already unit)
    audio = np.array([0.0, 0.0, 0.0, 1.0])  # norm = 1

    out = concat_modalities(text=text, cf=cf, audio=audio)

    # Expected: [text/5, cf, audio] = [0.6, 0.8, 1, 0, 0, 0, 0, 0, 1]
    expected = np.array([0.6, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(out, expected, atol=1e-7)


def test_concat_modalities_preserves_dimensionality():
    """Output dim equals sum of input dims."""
    from mcrs.sid.preprocessing import concat_modalities

    text = np.random.randn(1024).astype(np.float32)
    cf = np.random.randn(128).astype(np.float32)
    audio = np.random.randn(512).astype(np.float32)

    out = concat_modalities(text=text, cf=cf, audio=audio)
    assert out.shape == (1024 + 128 + 512,)


def test_concat_modalities_handles_zero_cf_imputation():
    """Cold-start CF rows are zero vectors; concat must not divide by zero."""
    from mcrs.sid.preprocessing import concat_modalities

    text = np.array([1.0, 0.0])
    cf_zero = np.zeros(3, dtype=np.float32)
    audio = np.array([1.0, 0.0])

    out = concat_modalities(text=text, cf=cf_zero, audio=audio)

    # CF segment should remain all zeros (no NaN from divide-by-zero)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[2:5], np.zeros(3))


def test_concat_modalities_returns_float32():
    """Output dtype should be float32 (memory + downstream-compat)."""
    from mcrs.sid.preprocessing import concat_modalities

    text = np.random.randn(4).astype(np.float64)
    cf = np.random.randn(2).astype(np.float64)
    audio = np.random.randn(3).astype(np.float64)

    out = concat_modalities(text=text, cf=cf, audio=audio)
    assert out.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest tests/test_sid_preprocessing.py -q
```

Expected: 4 failures with `ModuleNotFoundError: No module named 'mcrs.sid.preprocessing'`.

- [ ] **Step 3: Write minimal implementation**

Create `music-crs-baselines/mcrs/sid/preprocessing.py`:

```python
"""Pure functions for SID quantizer preprocessing + collision handling."""
from __future__ import annotations

import numpy as np


def concat_modalities(text: np.ndarray, cf: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """L2-normalize each modality independently, concatenate.

    Cold-start CF rows (zero vectors) are preserved as zeros — no divide-by-zero.
    Output dtype is float32 for memory efficiency downstream.
    """
    def _l2(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v.astype(np.float32)
        return (v / norm).astype(np.float32)

    return np.concatenate([_l2(text), _l2(cf), _l2(audio)])
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_preprocessing.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/preprocessing.py tests/test_sid_preprocessing.py
git commit -m "sid w1: concat_modalities pure function (TDD, 4 tests)"
```

---

## Task 3: TDD `compute_collision_buckets`

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/preprocessing.py`
- Modify: `tests/test_sid_preprocessing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sid_preprocessing.py`:

```python
def test_compute_collision_buckets_groups_tracks_by_sid():
    """Tracks sharing a 3-tuple SID end up in the same bucket."""
    from mcrs.sid.preprocessing import compute_collision_buckets

    track_ids = ["t1", "t2", "t3", "t4"]
    sid_assignments = [
        (10, 20, 30),  # t1 unique
        (5, 5, 5),     # t2 collides with t3
        (5, 5, 5),     # t3
        (1, 2, 3),     # t4 unique
    ]
    popularity = {"t1": 50.0, "t2": 30.0, "t3": 80.0, "t4": 10.0}

    buckets = compute_collision_buckets(track_ids, sid_assignments, popularity)

    assert buckets[(10, 20, 30)] == ["t1"]
    # t3 (popularity 80) ahead of t2 (popularity 30) within the bucket
    assert buckets[(5, 5, 5)] == ["t3", "t2"]
    assert buckets[(1, 2, 3)] == ["t4"]


def test_compute_collision_buckets_sorts_by_popularity_descending():
    """Within a collision bucket, ordering is descending popularity."""
    from mcrs.sid.preprocessing import compute_collision_buckets

    track_ids = ["a", "b", "c"]
    sid_assignments = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    popularity = {"a": 1.0, "b": 100.0, "c": 50.0}

    buckets = compute_collision_buckets(track_ids, sid_assignments, popularity)
    assert buckets[(0, 0, 0)] == ["b", "c", "a"]


def test_compute_collision_buckets_handles_missing_popularity_as_zero():
    """A track without popularity data sorts to the bottom of its bucket."""
    from mcrs.sid.preprocessing import compute_collision_buckets

    track_ids = ["a", "b"]
    sid_assignments = [(0, 0, 0), (0, 0, 0)]
    popularity = {"a": 5.0}  # b is missing

    buckets = compute_collision_buckets(track_ids, sid_assignments, popularity)
    assert buckets[(0, 0, 0)] == ["a", "b"]


def test_compute_collision_buckets_empty_input_returns_empty_dict():
    from mcrs.sid.preprocessing import compute_collision_buckets
    assert compute_collision_buckets([], [], {}) == {}
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_preprocessing.py::test_compute_collision_buckets_groups_tracks_by_sid -q
```

Expected: `ImportError: cannot import name 'compute_collision_buckets'`.

- [ ] **Step 3: Write implementation**

Append to `music-crs-baselines/mcrs/sid/preprocessing.py`:

```python
def compute_collision_buckets(
    track_ids: list[str],
    sid_assignments: list[tuple[int, int, int]],
    popularity: dict[str, float],
) -> dict[tuple[int, int, int], list[str]]:
    """Group tracks by their 3-tuple SID; sort each bucket by descending popularity.

    Tracks with missing popularity get rank 0.0 (sort to bucket bottom).
    """
    if len(track_ids) != len(sid_assignments):
        raise ValueError(
            f"track_ids ({len(track_ids)}) and sid_assignments "
            f"({len(sid_assignments)}) must be the same length"
        )
    buckets: dict[tuple[int, int, int], list[str]] = {}
    for tid, sid in zip(track_ids, sid_assignments):
        buckets.setdefault(sid, []).append(tid)
    for sid in buckets:
        buckets[sid].sort(key=lambda t: -popularity.get(t, 0.0))
    return buckets
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_preprocessing.py -q
```

Expected: `8 passed` (4 from Task 2 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/preprocessing.py tests/test_sid_preprocessing.py
git commit -m "sid w1: compute_collision_buckets (TDD, 4 tests, popularity-ordered)"
```

---

## Task 4: TDD `dedup_with_per_bucket_cap`

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/preprocessing.py`
- Modify: `tests/test_sid_preprocessing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sid_preprocessing.py`:

```python
def test_dedup_per_bucket_cap_one_keeps_first_per_bucket():
    """With cap=1, each beam contributes at most 1 track to the dedup output."""
    from mcrs.sid.preprocessing import dedup_with_per_bucket_cap

    # Each beam outputs a list of candidate track_ids from its SID's collision bucket
    beam_outputs = [
        ["t1", "t2", "t3"],   # beam 0: bucket A
        ["t4"],                # beam 1: bucket B
        ["t5", "t6"],          # beam 2: bucket C
    ]
    out = dedup_with_per_bucket_cap(beam_outputs, cap=1)
    # Each beam contributes its top-1; rest spill over to fill remaining slots
    assert out[:3] == ["t1", "t4", "t5"]


def test_dedup_per_bucket_cap_spills_extras_to_fill_remaining_slots():
    """After top-1 from each beam, extras spill in beam-then-bucket-rank order."""
    from mcrs.sid.preprocessing import dedup_with_per_bucket_cap

    beam_outputs = [
        ["t1", "t2", "t3"],   # beam 0
        ["t4", "t5"],          # beam 1
    ]
    out = dedup_with_per_bucket_cap(beam_outputs, cap=1)
    # First pass: t1, t4 (cap=1 each)
    # Spillover: t2, t3 (beam 0 extras) then t5 (beam 1 extras)
    assert out == ["t1", "t4", "t2", "t3", "t5"]


def test_dedup_per_bucket_cap_drops_duplicate_track_ids_across_beams():
    """If two beams' buckets share a track_id, it appears only once."""
    from mcrs.sid.preprocessing import dedup_with_per_bucket_cap

    beam_outputs = [
        ["t1", "t2"],
        ["t2", "t3"],   # t2 also appears here
    ]
    out = dedup_with_per_bucket_cap(beam_outputs, cap=1)
    # First pass: t1, t2 (cap=1 each)
    # Spillover: skip t2 (already added), then t3
    assert out == ["t1", "t2", "t3"]


def test_dedup_per_bucket_cap_empty_beams_input_returns_empty():
    from mcrs.sid.preprocessing import dedup_with_per_bucket_cap
    assert dedup_with_per_bucket_cap([], cap=1) == []
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_preprocessing.py::test_dedup_per_bucket_cap_one_keeps_first_per_bucket -q
```

Expected: `ImportError: cannot import name 'dedup_with_per_bucket_cap'`.

- [ ] **Step 3: Write implementation**

Append to `music-crs-baselines/mcrs/sid/preprocessing.py`:

```python
def dedup_with_per_bucket_cap(
    beam_outputs: list[list[str]],
    cap: int = 1,
) -> list[str]:
    """Deduplicate beam outputs across collision buckets with per-bucket cap.

    Algorithm:
      Pass 1: take cap items from each beam's bucket, in beam order
      Pass 2: spillover — take remaining items from each bucket, in beam-then-rank order
    Globally unique track_ids only (a track appearing in multiple beam buckets
    appears once in the output, at its earliest position).
    """
    seen: set[str] = set()
    out: list[str] = []
    # Pass 1: take cap items per beam
    for bucket in beam_outputs:
        taken = 0
        for tid in bucket:
            if taken >= cap:
                break
            if tid in seen:
                continue
            out.append(tid)
            seen.add(tid)
            taken += 1
    # Pass 2: spillover — take remaining items per beam in order
    for bucket in beam_outputs:
        for tid in bucket:
            if tid in seen:
                continue
            out.append(tid)
            seen.add(tid)
    return out
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_preprocessing.py -q
```

Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/preprocessing.py tests/test_sid_preprocessing.py
git commit -m "sid w1: dedup_with_per_bucket_cap (TDD, 4 tests; fixes v1 collision pathology)"
```

---

## Task 5: TDD `validate_codebook_utilization`

**Files:**
- Create: `music-crs-baselines/mcrs/sid/validation.py`
- Create: `tests/test_sid_validation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sid_validation.py`:

```python
"""Tests for SID quantizer validation gate functions."""
import numpy as np


def test_validate_codebook_utilization_passes_when_above_threshold():
    """All 256 codes used by ≥1 track → 100% utilization → passes 80% threshold."""
    from mcrs.sid.validation import validate_codebook_utilization

    # All 256 codes get used at least once across 1000 assignments
    assignments = list(range(256)) + list(range(256)) + [42] * (1000 - 512)
    passed, util = validate_codebook_utilization(
        assignments, codebook_size=256, threshold=0.80,
    )
    assert passed is True
    assert util == 1.0


def test_validate_codebook_utilization_fails_when_below_threshold():
    """Only 100/256 codes used → 39% utilization → fails 80% threshold."""
    from mcrs.sid.validation import validate_codebook_utilization

    assignments = list(range(100)) * 10  # only codes 0-99 used
    passed, util = validate_codebook_utilization(
        assignments, codebook_size=256, threshold=0.80,
    )
    assert passed is False
    assert abs(util - 100/256) < 1e-9


def test_validate_codebook_utilization_at_exact_threshold_passes():
    """80% utilization (e.g., 205/256) at threshold = 0.80 → passes (≥, not >)."""
    from mcrs.sid.validation import validate_codebook_utilization

    assignments = list(range(205))  # 205/256 = 0.8008
    passed, _ = validate_codebook_utilization(
        assignments, codebook_size=256, threshold=0.80,
    )
    assert passed is True
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_validation.py -q
```

Expected: `ModuleNotFoundError: No module named 'mcrs.sid.validation'`.

- [ ] **Step 3: Write implementation**

Create `music-crs-baselines/mcrs/sid/validation.py`:

```python
"""Pure functions for SID quantizer validation gates (per spec §2.4)."""
from __future__ import annotations


def validate_codebook_utilization(
    assignments: list[int],
    codebook_size: int,
    threshold: float = 0.80,
) -> tuple[bool, float]:
    """Gate 2: fraction of codebook entries used by ≥1 track must be ≥ threshold.

    Returns (passed, utilization_fraction). Direct test of Sinkhorn regularization
    (collapsed codebooks have low utilization).
    """
    used = set(assignments)
    util = len(used) / codebook_size
    return (util >= threshold, util)
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_validation.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/validation.py tests/test_sid_validation.py
git commit -m "sid w1: validate_codebook_utilization (gate 2, TDD, 3 tests)"
```

---

## Task 6: TDD `validate_cluster_purity`

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/validation.py`
- Modify: `tests/test_sid_validation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sid_validation.py`:

```python
def test_validate_cluster_purity_pure_buckets_pass():
    """When every bucket's tracks share a tag, purity = 100% → passes."""
    from mcrs.sid.validation import validate_cluster_purity

    # 5 buckets, each with tracks all sharing 'rock' tag
    buckets = {
        f"bucket_{i}": [f"t{i}_{j}" for j in range(3)]
        for i in range(5)
    }
    tag_lookup = {
        f"t{i}_{j}": ["rock", f"genre{i}"]
        for i in range(5) for j in range(3)
    }
    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=5, threshold=0.60, seed=42,
    )
    assert passed is True
    assert purity == 1.0


def test_validate_cluster_purity_random_tags_fail():
    """When tracks within buckets don't share tags, purity = 0% → fails."""
    from mcrs.sid.validation import validate_cluster_purity

    buckets = {
        f"bucket_{i}": [f"t{i}_{j}" for j in range(3)]
        for i in range(5)
    }
    # Each track has a unique tag with no overlap
    tag_lookup = {
        f"t{i}_{j}": [f"tag_{i}_{j}"]
        for i in range(5) for j in range(3)
    }
    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=5, threshold=0.60, seed=42,
    )
    assert passed is False
    assert purity < 0.60


def test_validate_cluster_purity_handles_singleton_buckets():
    """Singleton buckets count as pure (only one track, trivially shares tags with itself)."""
    from mcrs.sid.validation import validate_cluster_purity

    buckets = {f"bucket_{i}": [f"t{i}"] for i in range(5)}
    tag_lookup = {f"t{i}": [f"tag_{i}"] for i in range(5)}

    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=5, threshold=0.60, seed=42,
    )
    assert passed is True
    assert purity == 1.0


def test_validate_cluster_purity_lowercases_and_strips_tags():
    """Tag matching is case-insensitive and whitespace-stripped."""
    from mcrs.sid.validation import validate_cluster_purity

    buckets = {"b": ["t1", "t2"]}
    tag_lookup = {"t1": ["Rock"], "t2": [" rock "]}

    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=1, threshold=0.60, seed=42,
    )
    assert passed is True


def test_validate_cluster_purity_samples_subset_when_n_samples_lt_total():
    """When buckets > n_samples, only sample n_samples for evaluation."""
    from mcrs.sid.validation import validate_cluster_purity

    # 100 buckets, all pure
    buckets = {f"b_{i}": [f"t{i}_a", f"t{i}_b"] for i in range(100)}
    tag_lookup = {tid: ["rock"] for i in range(100) for tid in [f"t{i}_a", f"t{i}_b"]}

    passed, purity = validate_cluster_purity(
        buckets, tag_lookup, n_samples=10, threshold=0.60, seed=42,
    )
    assert passed is True
    assert purity == 1.0  # all sampled are pure
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_validation.py -q
```

Expected: 5 new failures with `ImportError: cannot import name 'validate_cluster_purity'`.

- [ ] **Step 3: Write implementation**

Append to `music-crs-baselines/mcrs/sid/validation.py`:

```python
import random


def validate_cluster_purity(
    buckets: dict[str, list[str]],
    tag_lookup: dict[str, list[str]],
    n_samples: int = 100,
    threshold: float = 0.60,
    seed: int = 42,
) -> tuple[bool, float]:
    """Gate 3: sample n_samples buckets; check fraction where tracks share ≥1 tag.

    Singleton buckets count as pure. Tags compared case-insensitive,
    whitespace-stripped. Returns (passed, purity_fraction).
    """
    if not buckets:
        return (True, 1.0)
    rng = random.Random(seed)
    keys = list(buckets.keys())
    sampled_keys = rng.sample(keys, min(n_samples, len(keys)))

    pure_count = 0
    for k in sampled_keys:
        members = buckets[k]
        if len(members) <= 1:
            pure_count += 1  # singleton trivially pure
            continue
        # Normalize: lowercase + strip
        tag_sets = [
            {t.strip().lower() for t in tag_lookup.get(tid, [])}
            for tid in members
        ]
        # Check intersection across ALL members has ≥1 tag
        common = set.intersection(*tag_sets) if tag_sets else set()
        if common:
            pure_count += 1
    purity = pure_count / len(sampled_keys)
    return (purity >= threshold, purity)
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_validation.py -q
```

Expected: `8 passed` (3 from Task 5 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/validation.py tests/test_sid_validation.py
git commit -m "sid w1: validate_cluster_purity (gate 3, TDD, 5 tests)"
```

---

## Task 7: TDD `compute_relative_mse_gate`

**Files:**
- Modify: `music-crs-baselines/mcrs/sid/validation.py`
- Modify: `tests/test_sid_validation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sid_validation.py`:

```python
def test_compute_relative_mse_gate_passes_when_rqvae_within_1_5x_pca():
    """Gate 1: RQ-VAE MSE ≤ 1.5 × PCA-256 baseline MSE."""
    from mcrs.sid.validation import compute_relative_mse_gate

    passed, ratio = compute_relative_mse_gate(rqvae_mse=0.020, pca_mse=0.018, multiplier=1.5)
    assert passed is True
    assert abs(ratio - 0.020 / 0.018) < 1e-9


def test_compute_relative_mse_gate_fails_when_rqvae_too_lossy():
    """RQ-VAE MSE 5× PCA → fails."""
    from mcrs.sid.validation import compute_relative_mse_gate

    passed, ratio = compute_relative_mse_gate(rqvae_mse=0.10, pca_mse=0.02, multiplier=1.5)
    assert passed is False
    assert ratio == 5.0


def test_compute_relative_mse_gate_passes_when_rqvae_better_than_pca():
    """RQ-VAE MSE < PCA → passes (ratio < 1)."""
    from mcrs.sid.validation import compute_relative_mse_gate

    passed, ratio = compute_relative_mse_gate(rqvae_mse=0.005, pca_mse=0.020, multiplier=1.5)
    assert passed is True
    assert ratio == 0.25


def test_compute_relative_mse_gate_handles_zero_pca_baseline():
    """If PCA somehow reconstructs perfectly, fall back to absolute threshold check."""
    from mcrs.sid.validation import compute_relative_mse_gate

    passed, ratio = compute_relative_mse_gate(rqvae_mse=0.001, pca_mse=0.0, multiplier=1.5)
    # With pca_mse=0, ratio is undefined; the function should pass if rqvae_mse is small enough
    # Define: pass if rqvae_mse < 0.01 (absolute fallback), else fail
    assert passed is True
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_validation.py -q
```

Expected: 4 new failures with `ImportError: cannot import name 'compute_relative_mse_gate'`.

- [ ] **Step 3: Write implementation**

Append to `music-crs-baselines/mcrs/sid/validation.py`:

```python
def compute_relative_mse_gate(
    rqvae_mse: float,
    pca_mse: float,
    multiplier: float = 1.5,
    absolute_fallback: float = 0.01,
) -> tuple[bool, float]:
    """Gate 1: RQ-VAE reconstruction MSE ≤ multiplier × PCA-256 baseline MSE.

    Falls back to absolute threshold when PCA baseline is 0 (degenerate case).
    Returns (passed, ratio_or_absolute_value).
    """
    if pca_mse == 0.0:
        # Degenerate baseline; use absolute fallback threshold.
        return (rqvae_mse < absolute_fallback, rqvae_mse)
    ratio = rqvae_mse / pca_mse
    return (ratio <= multiplier, ratio)
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_validation.py -q
```

Expected: `12 passed` (8 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/validation.py tests/test_sid_validation.py
git commit -m "sid w1: compute_relative_mse_gate (gate 1, TDD, 4 tests; relative-to-PCA per reviewer 3.1)"
```

---

## Task 8: Build `SIDQuantizer` wrapper class around `ResidualVQ`

**Files:**
- Create: `music-crs-baselines/mcrs/sid/quantizer.py`
- Test: `tests/test_sid_quantizer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sid_quantizer.py`:

```python
"""Tests for SIDQuantizer wrapper class (RQ-VAE encoder/decoder + Sinkhorn loss)."""
import numpy as np
import pytest
import torch


@pytest.fixture
def small_quantizer():
    """A tiny quantizer for fast unit tests: 16-dim input → 3 levels × 8 codes."""
    from mcrs.sid.quantizer import SIDQuantizer
    return SIDQuantizer(input_dim=16, latent_dim=8, num_levels=3, codebook_size=8, seed=42)


def test_sid_quantizer_encode_returns_3_token_tuple_per_input(small_quantizer):
    """encode() of a single embedding returns a tuple of 3 ints in [0, codebook_size)."""
    emb = np.random.randn(16).astype(np.float32)
    sid = small_quantizer.encode(emb)
    assert isinstance(sid, tuple)
    assert len(sid) == 3
    for code in sid:
        assert isinstance(code, int)
        assert 0 <= code < 8


def test_sid_quantizer_encode_batch_returns_array_shape_N_3(small_quantizer):
    """encode_batch() of N embeddings returns ndarray of shape (N, 3)."""
    embs = np.random.randn(50, 16).astype(np.float32)
    sids = small_quantizer.encode_batch(embs)
    assert sids.shape == (50, 3)
    assert sids.dtype in (np.int32, np.int64)


def test_sid_quantizer_save_load_round_trip(small_quantizer, tmp_path):
    """Saving and loading the quantizer preserves SID assignments deterministically."""
    from mcrs.sid.quantizer import SIDQuantizer

    embs = np.random.randn(20, 16).astype(np.float32)
    sids_before = small_quantizer.encode_batch(embs)

    path = tmp_path / "quantizer.pt"
    small_quantizer.save(path)
    loaded = SIDQuantizer.load(path)
    sids_after = loaded.encode_batch(embs)

    np.testing.assert_array_equal(sids_before, sids_after)


def test_sid_quantizer_train_step_returns_loss_dict(small_quantizer):
    """train_step on a batch returns a dict with the 3 loss components."""
    embs = torch.randn(8, 16, dtype=torch.float32)
    losses = small_quantizer.train_step(embs)
    assert "mse_recon" in losses
    assert "commitment" in losses
    assert "sinkhorn" in losses
    for v in losses.values():
        assert torch.is_tensor(v)


def test_sid_quantizer_encode_is_deterministic_given_seed(small_quantizer):
    """Same seed + same input = same SID."""
    from mcrs.sid.quantizer import SIDQuantizer

    quantizer_2 = SIDQuantizer(input_dim=16, latent_dim=8, num_levels=3, codebook_size=8, seed=42)
    emb = np.random.RandomState(0).randn(16).astype(np.float32)

    sid_1 = small_quantizer.encode(emb)
    sid_2 = quantizer_2.encode(emb)
    assert sid_1 == sid_2
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_sid_quantizer.py -q
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'mcrs.sid.quantizer'`.

- [ ] **Step 3: Write implementation**

Create `music-crs-baselines/mcrs/sid/quantizer.py`:

```python
"""SIDQuantizer — RQ-VAE wrapper using vector-quantize-pytorch + Sinkhorn loss term.

Spec §2.2 architecture:
  L2-norm(text) ⊕ L2-norm(CF) ⊕ L2-norm(audio)
    → MLP encoder (1664 → 512 → 256)
    → ResidualVQ × 3 levels, codebook=256 each
    → MLP decoder (256 → 512 → 1664)
  Loss = MSE_recon + 0.25 · commitment_loss + λ · sinkhorn_uniform_loss
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


def _set_seed(seed: int) -> None:
    """Pin all RNG state for reproducible RQ-VAE training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sinkhorn_uniform_loss(
    distances: torch.Tensor,
    n_iters: int = 5,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Approximate Sinkhorn-uniform loss over codebook assignments (LC-Rec recipe motivation).

    Encourages each code to receive roughly 1/K of the total assignment mass,
    preventing collapse. `distances` is a (B, K) matrix of squared distances
    from each input to each code; we Sinkhorn-normalize it and penalize
    deviation from uniform code marginals.
    """
    B, K = distances.shape
    # Convert distances to a similarity matrix (negative distance, scaled)
    log_q = -distances / max(epsilon, 1e-6)
    log_q = log_q - log_q.logsumexp(dim=1, keepdim=True)  # row-normalize
    # Sinkhorn iterations: alternately normalize rows and columns
    for _ in range(n_iters):
        # Column normalization toward uniform 1/K per column
        col_marg = log_q.logsumexp(dim=0)
        log_q = log_q - col_marg + np.log(B / K)
        # Row normalization
        row_marg = log_q.logsumexp(dim=1, keepdim=True)
        log_q = log_q - row_marg
    # The loss: KL from current column marginal to uniform 1/K
    final_col = log_q.logsumexp(dim=0)
    target = torch.full_like(final_col, fill_value=np.log(B / K))
    return (final_col.exp() * (final_col - target)).sum()


class SIDQuantizer:
    """RQ-VAE wrapper that encodes embeddings into 3-token SIDs."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        num_levels: int = 3,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        sinkhorn_lambda: float = 0.10,
        seed: int = 42,
    ) -> None:
        _set_seed(seed)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.sinkhorn_lambda = sinkhorn_lambda
        self.seed = seed

        # MLP encoder: input_dim → latent_dim (matches spec §2.2 architecture)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(latent_dim * 2, 64)),
            nn.GELU(),
            nn.Linear(max(latent_dim * 2, 64), latent_dim),
        )
        # Decoder: mirror of encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim * 2, 64)),
            nn.GELU(),
            nn.Linear(max(latent_dim * 2, 64), input_dim),
        )
        self.rvq = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_levels,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_init=True,
            kmeans_iters=10,
        )

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.rvq.parameters())

    def to(self, device):
        self.encoder.to(device); self.decoder.to(device); self.rvq.to(device)
        return self

    def train_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """One training step: encode → quantize → decode → compute losses."""
        z = self.encoder(batch)
        z_q, _, commitment_loss = self.rvq(z)
        recon = self.decoder(z_q)
        mse = ((recon - batch) ** 2).mean()
        # Sinkhorn term over the FIRST codebook's distances
        first_codebook = self.rvq.layers[0]._codebook.embed[0]  # (K, D)
        dists = ((z.unsqueeze(1) - first_codebook.unsqueeze(0)) ** 2).sum(dim=-1)  # (B, K)
        sinkhorn = _sinkhorn_uniform_loss(dists)
        return {
            "mse_recon": mse,
            "commitment": commitment_loss.mean() if commitment_loss.dim() > 0 else commitment_loss,
            "sinkhorn": self.sinkhorn_lambda * sinkhorn,
        }

    @torch.no_grad()
    def encode_batch(self, embs: np.ndarray) -> np.ndarray:
        """Encode N embeddings → (N, num_levels) array of code indices."""
        device = next(self.encoder.parameters()).device
        x = torch.from_numpy(np.asarray(embs, dtype=np.float32)).to(device)
        z = self.encoder(x)
        _, indices, _ = self.rvq(z)
        # indices is shape (B, num_levels) — one code per level
        return indices.detach().cpu().numpy().astype(np.int64)

    def encode(self, emb: np.ndarray) -> tuple[int, ...]:
        """Encode a single embedding → tuple of num_levels code indices."""
        sids = self.encode_batch(np.asarray(emb)[None, :])
        return tuple(int(c) for c in sids[0])

    def save(self, path) -> None:
        """Save the quantizer state including all learned weights + config."""
        torch.save({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "commitment_weight": self.commitment_weight,
            "sinkhorn_lambda": self.sinkhorn_lambda,
            "seed": self.seed,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "rvq": self.rvq.state_dict(),
        }, path)

    @classmethod
    def load(cls, path) -> "SIDQuantizer":
        """Load a saved quantizer."""
        ckpt = torch.load(path, weights_only=False)
        q = cls(
            input_dim=ckpt["input_dim"],
            latent_dim=ckpt["latent_dim"],
            num_levels=ckpt["num_levels"],
            codebook_size=ckpt["codebook_size"],
            commitment_weight=ckpt["commitment_weight"],
            sinkhorn_lambda=ckpt["sinkhorn_lambda"],
            seed=ckpt["seed"],
        )
        q.encoder.load_state_dict(ckpt["encoder"])
        q.decoder.load_state_dict(ckpt["decoder"])
        q.rvq.load_state_dict(ckpt["rvq"])
        return q
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python -m pytest tests/test_sid_quantizer.py -q
```

Expected: `5 passed`. (May see warnings from `vector-quantize-pytorch` about kmeans init on small batches; OK to ignore for tests.)

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/sid/quantizer.py tests/test_sid_quantizer.py
git commit -m "sid w1: SIDQuantizer (RQ-VAE + Sinkhorn loss; TDD, 5 tests, save/load round-trip)"
```

---

## Task 9: Build orchestration script `scripts/build_sid_quantizer.py`

**Files:**
- Create: `scripts/build_sid_quantizer.py`

(This is orchestration code — heavy I/O + training loop, not unit-tested directly. Validated by smoke-run in Task 12.)

- [ ] **Step 1: Write the script skeleton with CLI args + imports**

Create `scripts/build_sid_quantizer.py`:

```python
"""Train ONE SID quantizer (one seed) over the 47K-track catalog.

Loads text + CF + audio embeddings from talkpl-ai/TalkPlayData-Challenge-Track-Embeddings,
concatenates per spec §2.1, trains RQ-VAE + Sinkhorn for ~50 epochs, saves checkpoint
+ assignment parquet + per-seed gate scores.

Usage:
    python scripts/build_sid_quantizer.py --seed 42
    python scripts/build_sid_quantizer.py --seed 123
    python scripts/build_sid_quantizer.py --seed 7

After all 3 seeds done, run scripts/pick_best_sid_quantizer.py to choose + pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from sklearn.decomposition import PCA  # for gate 1 PCA baseline


def main():
    parser = argparse.ArgumentParser(description="Train one SID quantizer (one seed).")
    parser.add_argument("--seed", type=int, required=True, help="RNG seed (use 42, 123, or 7).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--sinkhorn-lambda", type=float, default=0.10)
    parser.add_argument("--cache-root", default=str(REPO_ROOT / "experiments" / "cache" / "sid"))
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Smoke test: train on first N tracks only.")
    args = parser.parse_args()

    out_dir = Path(args.cache_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build_sid_quantizer] seed={args.seed} → out_dir={out_dir}", file=sys.stderr)

    # Continued in next steps...


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the embedding-loading + concat block**

Replace the `# Continued in next steps...` placeholder in `scripts/build_sid_quantizer.py` with:

```python
    from mcrs.sid.preprocessing import concat_modalities

    print("[build_sid_quantizer] loading TalkPlayData-Challenge-Track-Embeddings...", file=sys.stderr)
    emb_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks"
    )
    if args.max_tracks is not None:
        emb_ds = emb_ds.select(range(min(args.max_tracks, len(emb_ds))))

    track_ids: list[str] = []
    fused_embs: list[np.ndarray] = []
    for row in tqdm(emb_ds, desc="concat modalities"):
        tid = row["track_id"]
        text = np.array(row["metadata-qwen3_embedding_0.6b"], dtype=np.float32)
        cf_raw = row["cf-bpr"]
        cf = np.zeros(128, dtype=np.float32) if (cf_raw is None or len(cf_raw) == 0) \
             else np.array(cf_raw, dtype=np.float32)
        audio = np.array(row["audio-laion_clap"], dtype=np.float32)
        fused = concat_modalities(text=text, cf=cf, audio=audio)
        track_ids.append(tid)
        fused_embs.append(fused)

    X = np.stack(fused_embs)  # (N, ~1664)
    print(f"[build_sid_quantizer] X shape={X.shape}", file=sys.stderr)

    # Continued in next steps...
```

- [ ] **Step 3: Add the training loop**

Append (replacing the new placeholder):

```python
    from mcrs.sid.quantizer import SIDQuantizer

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[build_sid_quantizer] device={device}", file=sys.stderr)

    quantizer = SIDQuantizer(
        input_dim=X.shape[1],
        latent_dim=args.latent_dim,
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        sinkhorn_lambda=args.sinkhorn_lambda,
        seed=args.seed,
    )
    quantizer.to(device)
    optimizer = torch.optim.AdamW(quantizer.parameters(), lr=args.lr)

    X_tensor = torch.from_numpy(X)
    n = X_tensor.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        epoch_losses = {"mse_recon": 0.0, "commitment": 0.0, "sinkhorn": 0.0}
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = X_tensor[idx].to(device)
            losses = quantizer.train_step(batch)
            total = losses["mse_recon"] + losses["commitment"] + losses["sinkhorn"]
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            for k, v in losses.items():
                epoch_losses[k] += float(v.item())
            n_batches += 1
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        print(
            f"[epoch {epoch+1:2d}/{args.epochs}] "
            f"mse={avg['mse_recon']:.5f} commit={avg['commitment']:.5f} "
            f"sinkhorn={avg['sinkhorn']:.5f}",
            file=sys.stderr,
        )

    # Continued in next steps...
```

- [ ] **Step 4: Add encoding + assignments save**

Append:

```python
    # Encode all tracks → assignments parquet
    print("[build_sid_quantizer] encoding all tracks...", file=sys.stderr)
    sids_arr = quantizer.encode_batch(X)  # (N, num_levels)
    df = pd.DataFrame({
        "track_id": track_ids,
        **{f"code_{level + 1}": sids_arr[:, level].astype(int) for level in range(args.num_levels)},
    })
    assignments_path = out_dir / f"quantizer_seed{args.seed}_assignments.parquet"
    df.to_parquet(assignments_path, index=False)
    print(f"[build_sid_quantizer] wrote {assignments_path} ({len(df)} rows)", file=sys.stderr)

    # Save quantizer
    qpath = out_dir / f"quantizer_seed{args.seed}.pt"
    quantizer.save(qpath)
    print(f"[build_sid_quantizer] wrote {qpath}", file=sys.stderr)

    # Continued in next steps...
```

- [ ] **Step 5: Add the 3 validation gate evaluations**

Append (replacing the placeholder):

```python
    from mcrs.sid.validation import (
        compute_relative_mse_gate,
        validate_codebook_utilization,
        validate_cluster_purity,
    )

    # Gate 1: relative MSE vs PCA-256 baseline.
    pca = PCA(n_components=args.latent_dim).fit(X)
    X_pca_recon = pca.inverse_transform(pca.transform(X))
    pca_mse = float(((X_pca_recon - X) ** 2).mean())
    with torch.no_grad():
        z = quantizer.encoder(torch.from_numpy(X).to(device))
        z_q, _, _ = quantizer.rvq(z)
        recon = quantizer.decoder(z_q).cpu().numpy()
    rqvae_mse = float(((recon - X) ** 2).mean())
    g1_pass, g1_ratio = compute_relative_mse_gate(rqvae_mse, pca_mse, multiplier=1.5)

    # Gate 2: codebook utilization (per level)
    g2_results = []
    for level in range(args.num_levels):
        passed, util = validate_codebook_utilization(
            sids_arr[:, level].tolist(), codebook_size=args.codebook_size, threshold=0.80,
        )
        g2_results.append({"level": level + 1, "passed": passed, "utilization": util})
    g2_pass = all(r["passed"] for r in g2_results)

    # Gate 3: cluster purity at level 1, sample 100 buckets
    print("[build_sid_quantizer] loading metadata for tag_lookup...", file=sys.stderr)
    meta_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    tag_lookup = {row["track_id"]: row.get("tag_list") or [] for row in meta_ds}

    # Build level-1 buckets
    level1_buckets: dict[int, list[str]] = {}
    for tid, sid_row in zip(track_ids, sids_arr):
        level1_buckets.setdefault(int(sid_row[0]), []).append(tid)
    g3_pass, g3_purity = validate_cluster_purity(
        {str(k): v for k, v in level1_buckets.items()},
        tag_lookup, n_samples=100, threshold=0.60, seed=args.seed,
    )

    gates = {
        "seed": args.seed,
        "rqvae_mse": rqvae_mse,
        "pca_mse": pca_mse,
        "gate_1_relative_mse": {"passed": g1_pass, "ratio": g1_ratio},
        "gate_2_codebook_utilization": {"passed": g2_pass, "per_level": g2_results},
        "gate_3_cluster_purity": {"passed": g3_pass, "purity": g3_purity},
        "all_gates_passed": g1_pass and g2_pass and g3_pass,
    }
    gates_path = out_dir / f"quantizer_seed{args.seed}_gates.json"
    gates_path.write_text(json.dumps(gates, indent=2))
    print(f"[build_sid_quantizer] wrote {gates_path}", file=sys.stderr)
    print(f"[build_sid_quantizer] all gates passed: {gates['all_gates_passed']}", file=sys.stderr)
```

- [ ] **Step 6: Verify the script syntax + CLI help work**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -c "import ast; ast.parse(open('scripts/build_sid_quantizer.py').read()); print('AST OK')"
python scripts/build_sid_quantizer.py --help 2>&1 | head -20
```

Expected: `AST OK` then a CLI help dump.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_sid_quantizer.py
git commit -m "sid w1: build_sid_quantizer.py orchestration (load embs, train, encode, 3 gates)"
```

---

## Task 10: Build `pick_best_sid_quantizer.py` for the 3-seed selection + SHA256 pinning

**Files:**
- Create: `scripts/pick_best_sid_quantizer.py`

- [ ] **Step 1: Write the script**

Create `scripts/pick_best_sid_quantizer.py`:

```python
"""Pick the best of 3 trained SID quantizers (per seed), pin by SHA256.

Reads quantizer_seed{42,123,7}_gates.json, picks the seed whose quantizer:
  1. Passes all 3 gates, AND
  2. Has the highest gate_3_cluster_purity score (per spec §2.3).

Then writes:
  - quantizer_chosen.pt (copy of chosen seed's checkpoint)
  - track_to_sid.parquet (chosen seed's assignments + collision buckets)
  - quantizer_chosen.sha256 (pinning hash)

Usage:
    python scripts/pick_best_sid_quantizer.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
sys.path.insert(0, str(BASELINES_DIR))

import pandas as pd
from datasets import load_dataset


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cache_root = REPO_ROOT / "experiments" / "cache" / "sid"
    seeds = [42, 123, 7]

    candidates: list[dict] = []
    for s in seeds:
        gates_path = cache_root / f"quantizer_seed{s}_gates.json"
        if not gates_path.exists():
            print(f"missing {gates_path} — skipping seed {s}", file=sys.stderr)
            continue
        gates = json.loads(gates_path.read_text())
        if not gates.get("all_gates_passed", False):
            print(f"seed {s} did not pass all gates", file=sys.stderr)
            continue
        candidates.append({
            "seed": s,
            "purity": gates["gate_3_cluster_purity"]["purity"],
            "gates": gates,
        })

    if not candidates:
        raise SystemExit("No seed passed all 3 validation gates. Debug quantizer config.")

    # Pick highest purity among gate-passing candidates
    best = max(candidates, key=lambda c: c["purity"])
    print(
        f"chosen: seed={best['seed']} purity={best['purity']:.3f}",
        file=sys.stderr,
    )

    # Copy chosen artifact + SHA256-pin
    src_q = cache_root / f"quantizer_seed{best['seed']}.pt"
    chosen_q = cache_root / "quantizer_chosen.pt"
    shutil.copy(src_q, chosen_q)
    sha = sha256_of_file(chosen_q)
    (cache_root / "quantizer_chosen.sha256").write_text(sha + "\n")

    # Build track_to_sid.parquet with collision buckets
    src_assign = cache_root / f"quantizer_seed{best['seed']}_assignments.parquet"
    df = pd.read_parquet(src_assign)
    # Read popularity to order collision buckets per spec §2.5
    print("[pick_best] loading metadata for popularity ordering...", file=sys.stderr)
    meta_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    popularity = {row["track_id"]: row.get("popularity") or 0.0 for row in meta_ds}
    df["popularity"] = df["track_id"].map(popularity).fillna(0.0)

    # Group collision buckets and assign uniqueness suffix
    df["sid_tuple"] = list(zip(df["code_1"], df["code_2"], df["code_3"]))
    df = df.sort_values(["sid_tuple", "popularity"], ascending=[True, False]).reset_index(drop=True)
    df["bucket_rank"] = df.groupby("sid_tuple").cumcount()  # 0=most popular, 1=next, ...

    out = df[["track_id", "code_1", "code_2", "code_3", "popularity", "bucket_rank"]]
    out_path = cache_root / "track_to_sid.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[pick_best] wrote {out_path} ({len(out)} rows)", file=sys.stderr)

    # Final summary
    n_buckets = df["sid_tuple"].nunique()
    n_collisions = (df.groupby("sid_tuple").size() > 1).sum()
    print(
        f"[pick_best] {n_buckets} unique SIDs, {n_collisions} collision buckets, "
        f"chosen sha256={sha[:12]}...",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax + CLI**

```bash
python -c "import ast; ast.parse(open('scripts/pick_best_sid_quantizer.py').read()); print('AST OK')"
python scripts/pick_best_sid_quantizer.py --help 2>&1 | head -10
```

Expected: `AST OK` (no required CLI args).

- [ ] **Step 3: Commit**

```bash
git add scripts/pick_best_sid_quantizer.py
git commit -m "sid w1: pick_best_sid_quantizer (3-seed pick by purity, SHA256 pin)"
```

---

## Task 11: Build the Colab notebook `60_build_sid_quantizer.ipynb`

**Files:**
- Create: `colab/60_build_sid_quantizer.ipynb`

- [ ] **Step 1: Write the notebook (full content)**

Create the notebook with this exact JSON content:

```bash
cat > /Users/orrimoch/PythonProjs/recsys2026/colab/60_build_sid_quantizer.ipynb <<'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 60 — SID Quantizer (W1) on Colab\n",
    "\n",
    "Trains 3 RQ-VAE quantizers (seeds 42, 123, 7) over text + CF + audio embeddings, picks the best by cluster-purity, pins by SHA256. Output: `track_to_sid.parquet` on Drive.\n",
    "\n",
    "**Wallclock**: ~10–15 min per seed on L4 (47K embeddings, batch=512, 50 epochs); ~30–45 min total for 3 seeds + pick-best.\n",
    "\n",
    "**Spec**: `documents/specs/2026-05-15-sid-retrieval-design.md` §2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1) GPU check.\n",
    "!nvidia-smi | head -10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2) Clone fresh-model branch.\n",
    "BRANCH = 'fresh-model'\n",
    "!rm -rf /content/recsys2026\n",
    "!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026\n",
    "%cd /content/recsys2026"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3) HF auth + Drive mount + symlink the SID cache to Drive.\n",
    "import os\n",
    "from google.colab import userdata, drive\n",
    "os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n",
    "drive.mount('/content/drive')\n",
    "os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'\n",
    "\n",
    "DRIVE_SID_DIR = '/content/drive/MyDrive/recsys2026_sid_cache'\n",
    "os.makedirs(DRIVE_SID_DIR, exist_ok=True)\n",
    "REPO_SID_DIR = '/content/recsys2026/experiments/cache/sid'\n",
    "os.makedirs(os.path.dirname(REPO_SID_DIR), exist_ok=True)\n",
    "if os.path.lexists(REPO_SID_DIR):\n",
    "    !rm -rf {REPO_SID_DIR}\n",
    "!ln -s {DRIVE_SID_DIR} {REPO_SID_DIR}\n",
    "print('symlinked', REPO_SID_DIR, '->', DRIVE_SID_DIR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4) Install deps.\n",
    "!pip install -q --upgrade transformers datasets 'pandas<3.0' tqdm omegaconf pyyaml numpy scipy scikit-learn pyarrow vector-quantize-pytorch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5) Smoke test: train one seed on first 500 tracks (~30 sec) — verifies the pipeline before committing 30+ min.\n",
    "!python scripts/build_sid_quantizer.py --seed 42 --epochs 5 --max-tracks 500"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5b) Inspect smoke output\n",
    "import json, os\n",
    "smoke_gates = json.load(open('experiments/cache/sid/quantizer_seed42_gates.json'))\n",
    "print(json.dumps(smoke_gates, indent=2))\n",
    "print('all gates passed (on 500-track smoke):', smoke_gates['all_gates_passed'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6) Full run, all 3 seeds. ~10-15 min each on L4.\n",
    "# Wipes the smoke artifacts first.\n",
    "!rm -f experiments/cache/sid/quantizer_seed*\n",
    "for seed in [42, 123, 7]:\n",
    "    print(f'\\n=== seed {seed} ===')\n",
    "    !python scripts/build_sid_quantizer.py --seed {seed}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7) Pick best of 3 + pin SHA256.\n",
    "!python scripts/pick_best_sid_quantizer.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8) Final inspection — what landed on Drive\n",
    "!ls -la experiments/cache/sid/\n",
    "import pandas as pd\n",
    "df = pd.read_parquet('experiments/cache/sid/track_to_sid.parquet')\n",
    "print(f'\\ntrack_to_sid.parquet: {len(df)} rows')\n",
    "print(df.head(10))\n",
    "n_unique = df.groupby(['code_1', 'code_2', 'code_3']).ngroups\n",
    "n_collisions = (df.groupby(['code_1', 'code_2', 'code_3']).size() > 1).sum()\n",
    "print(f'\\nunique SIDs: {n_unique}, collision buckets: {n_collisions}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF
```

- [ ] **Step 2: Validate notebook JSON**

```bash
python -c "import json; nb=json.load(open('colab/60_build_sid_quantizer.ipynb')); print(f'OK, {len(nb[\"cells\"])} cells')"
```

Expected: `OK, 10 cells`.

- [ ] **Step 3: Commit + push so Colab can pull**

```bash
git add colab/60_build_sid_quantizer.ipynb
git commit -m "sid w1: notebook 60 (smoke + 3-seed train + pick-best + inspect)"
git push origin fresh-model
```

---

## Task 12: Local smoke test on small subset (verify wiring before user runs full on Colab)

**Files:** none modified — verification only

- [ ] **Step 1: Run the test suite first**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest tests/test_sid_preprocessing.py tests/test_sid_validation.py tests/test_sid_quantizer.py -v
```

Expected: all 24 tests green (12 preprocessing + 7 validation + 5 quantizer).

- [ ] **Step 2: Smoke-run the full orchestration on a tiny subset locally (Mac MPS)**

```bash
python scripts/build_sid_quantizer.py --seed 42 --epochs 3 --max-tracks 200
```

Expected output highlights:
- `[build_sid_quantizer] X shape=(200, 1664)` (or whatever the actual concat dim is)
- 3 epoch training loops printing decreasing MSE
- `wrote experiments/cache/sid/quantizer_seed42_assignments.parquet (200 rows)`
- `wrote experiments/cache/sid/quantizer_seed42_gates.json`
- `all gates passed: ...` (may be False on 200 tracks; that's OK for smoke — full 47K is the real test)

- [ ] **Step 3: Inspect the smoke artifacts**

```bash
ls -la experiments/cache/sid/ && python -c "
import json, pandas as pd
g = json.load(open('experiments/cache/sid/quantizer_seed42_gates.json'))
print('gates:', json.dumps(g, indent=2))
df = pd.read_parquet('experiments/cache/sid/quantizer_seed42_assignments.parquet')
print('df shape:', df.shape, 'cols:', df.columns.tolist())
print(df.head())
"
```

Expected: gates JSON has all 3 gate sub-results; parquet has columns `track_id`, `code_1`, `code_2`, `code_3` with 200 rows.

- [ ] **Step 4: Cleanup smoke artifacts (don't commit them)**

```bash
rm -rf experiments/cache/sid/quantizer_seed42* experiments/cache/sid/quantizer_chosen* experiments/cache/sid/track_to_sid.parquet
```

- [ ] **Step 5: Commit any final adjustments + push**

If any bugs found in steps 1-3, fix them, re-run tests, then commit:
```bash
git add -p   # carefully review any changes
git commit -m "sid w1: smoke-test fixes"
git push origin fresh-model
```

If no bugs, the prior commits already cover everything.

---

## Task 13: Final commit + invoke code-reviewer agent

**Files:** none — workflow gate per spec §0.4

- [ ] **Step 1: Final test sweep**

```bash
python -m pytest tests/ -q --ignore=tests/test_local_eval.py --ignore=tests/test_wave0_integration.py --ignore=tests/test_wave1_integration.py --ignore=tests/test_wave2_integration.py 2>&1 | tail -3
```

Expected: `XXX passed` where XXX includes the 24 new tests added in this plan (392 baseline + 24 = 416 total expected).

- [ ] **Step 2: Status check + confirm everything is committed and pushed**

```bash
git status -s   # should be clean
git log --oneline -10   # should show this plan's ~13 commits
git push origin fresh-model
```

- [ ] **Step 3: Invoke code-reviewer agent (per spec §0.4)**

Dispatch via Agent tool:

```
Agent({
  description: "Review SID quantizer W1 implementation",
  subagent_type: "superpowers:code-reviewer",
  prompt: "Review the W1 implementation of the SID quantizer for the RecSys 2026 generative-retrieval project against the design spec and the project's working methodology.

  Spec: documents/specs/2026-05-15-sid-retrieval-design.md (especially §2 SID Quantizer)
  Plan: documents/plans/2026-05-15-sid-quantizer-w1.md
  
  Files to review (newly added on fresh-model branch):
  - music-crs-baselines/mcrs/sid/__init__.py
  - music-crs-baselines/mcrs/sid/preprocessing.py
  - music-crs-baselines/mcrs/sid/quantizer.py
  - music-crs-baselines/mcrs/sid/validation.py
  - scripts/build_sid_quantizer.py
  - scripts/pick_best_sid_quantizer.py
  - tests/test_sid_preprocessing.py
  - tests/test_sid_validation.py
  - tests/test_sid_quantizer.py
  - colab/60_build_sid_quantizer.ipynb

  Working methodology to verify against: feedback_recsys_working_methodology.md.
  
  Check:
  1. Does the implementation match the spec §2 architecture exactly?
  2. Are the 4 validation gates (1: relative MSE, 2: codebook utilization, 3: cluster purity, 4: deferred to W3) wired correctly?
  3. Does the 3-seed protocol + SHA256 pinning work as designed?
  4. Are all pure functions TDD'd with adequate edge-case coverage?
  5. Does the Colab notebook correctly symlink to Drive (per the path-bug lesson from notebook 52)?
  6. Are there any silent failures or design drift from the spec to flag?

  Report: APPROVE / APPROVE-WITH-CHANGES / NEEDS-MAJOR-REVISION + numbered findings + top-5 changes if any."
})
```

- [ ] **Step 4: Address any reviewer findings**

If APPROVE → done.
If APPROVE-WITH-CHANGES → make the small fixes, commit, re-push.
If NEEDS-MAJOR-REVISION → triage findings; major issues block the user's Colab run.

- [ ] **Step 5: Hand off to user for Colab execution**

Send the user:
- Notebook 60 link: `https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-model/colab/60_build_sid_quantizer.ipynb`
- W1 expected wallclock: ~30–45 min on L4 (3 seeds × ~10 min each + pick-best ~2 min)
- W1 success criteria: at least 1 of 3 seeds passes all 3 gates; `track_to_sid.parquet` (~47K rows) on Drive; SHA256 pinned

---

## Self-review checklist (per writing-plans skill)

**1. Spec coverage** (against §2 of `documents/specs/2026-05-15-sid-retrieval-design.md`):

- §2.1 input modalities (text + CF + audio L2-norm concat) — Task 2 (`concat_modalities`) + Task 9 (script loads correct columns) ✓
- §2.2 RQ-VAE + Sinkhorn + commitment loss — Task 8 (`SIDQuantizer`) ✓
- §2.3 3-seed protocol + SHA256 pinning — Task 9 (`--seed` arg) + Task 10 (`pick_best_sid_quantizer.py`) + Task 11 notebook cell 6 ✓
- §2.4 gate 1 (relative MSE) — Task 7 (`compute_relative_mse_gate`) + Task 9 step 5 ✓
- §2.4 gate 2 (codebook utilization ≥ 80%) — Task 5 + Task 9 step 5 ✓
- §2.4 gate 3 (cluster purity ≥ 60%) — Task 6 + Task 9 step 5 ✓
- §2.4 gate 4 (search/rec balance) — DEFERRED to W3 per spec ("moved to §3.5") ✓
- §2.5 collision handling (per-bucket cap, popularity ordering) — Task 3 (`compute_collision_buckets`) + Task 4 (`dedup_with_per_bucket_cap`) + Task 10 (parquet write with bucket_rank) ✓
- §2.6 implementation footprint — files match (mcrs/sid/* + scripts + tests + notebook 60) ✓

**2. Placeholder scan**: searched for "TBD", "TODO", "implement later", vague "add appropriate error handling" — none present. Every step has actual code or actual command.

**3. Type consistency**: 
- `concat_modalities(text, cf, audio) → np.ndarray (float32)` — used consistently in Tasks 2, 9
- `SIDQuantizer.encode(emb) → tuple[int, ...]` — used in Tasks 8, 10
- `SIDQuantizer.encode_batch(embs) → np.ndarray (N, num_levels)` — used in Tasks 8, 9, 10
- `compute_collision_buckets(track_ids, sid_assignments, popularity) → dict[tuple, list[str]]` — Task 3, 10 (note: Task 10 uses pandas groupby instead of calling this function directly because it's working with full DataFrames, not lists; this is fine — both produce equivalent results, the pure function is for in-memory use cases like the inference dedup)
- `dedup_with_per_bucket_cap(beam_outputs, cap) → list[str]` — Task 4; will be used in W4 inference (not consumed in W1 directly)
- All validation functions return `(bool, float)` consistently — Tasks 5, 6, 7

**4. Sequencing**: Each task only depends on earlier tasks' artifacts. Task 9 imports from `mcrs.sid.preprocessing` (Tasks 2-4) and `mcrs.sid.validation` (Tasks 5-7) and `mcrs.sid.quantizer` (Task 8). Task 10 reads files written by Task 9. Task 11 calls Task 9 + Task 10. Task 12 verifies Task 9 end-to-end. Task 13 reviews everything.

**Plan complete and ready for execution.**
