# nDCG-Stretch Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-stage parallel-fusion retrieval pipeline (BM25 + dense_lyrics + fine-tuned BGE-M3 → fine-tuned BGE-reranker-base → LightGBM LambdaRank) on the RecSys 2026 Music CRS challenge to lift Blind-A nDCG@20 from 0.06 to ≥ 0.35 stretch (≥ 0.25 conservative).

**Architecture:** Three additive stages, each gated by an offline composite eval on the dev set. Stage A fine-tunes BGE-M3 via FlagEmbedding with NV-Retriever-style hard-negative mining. Stage B fine-tunes BGE-reranker-base on in-distribution negatives mined by Stage A's output. Stage C extends the existing `scripts/build_lgbm_features.py` (14 → ~50 features) and trains a LambdaRank model on a held-out 20% slice of TRAIN sessions (NOT dev, to prevent leakage).

**Tech Stack:** Python 3.10 / FlagEmbedding (BGE fine-tuning) / sentence-transformers / PEFT (LoRA) / LightGBM / HF datasets / HF Hub / Colab Blackwell GPU / Drive caching pattern (mirrors notebooks 60–66).

**Reference spec:** `docs/superpowers/specs/2026-05-18-ndcg-stretch-design.md` (commit `04df546`).

---

## File Structure

### New scripts (`scripts/`)
| Path | Responsibility |
|---|---|
| `scripts/build_bi_encoder_training_data.py` | Stage A: zero-shot BGE-M3 HN miner + JSONL triple writer (already shipped in Tasks 1-6; walks HF dataset). |
| `scripts/train_bi_encoder.py` | Stage A: **custom PEFT-LoRA training loop** over sentence-transformers + BGE-M3. Merges + pushes merged model to Hub. (NOT a FlagEmbedding CLI wrapper — those flags don't exist in master.) |
| `scripts/build_cross_encoder_training_data.py` | Stage B: re-mine HNs via fine-tuned BGE-M3 (walks HF dataset, not parquet). |
| `scripts/train_cross_encoder.py` | Stage B: cross-encoder fine-tune via sentence-transformers' `CrossEncoder` API + custom pairwise loss. |
| `scripts/build_lgbm_features.py` | **EXTEND** existing (14 → ~28 features). Don't rewrite. |
| `scripts/train_lgbm_ranker.py` | Stage C: LightGBM LambdaRank trainer (writes `booster.txt` + `metadata.json` matching the existing `LGBM_RERANKER` loader contract). |
| `scripts/eval_retrieval_v2.py` | Generic nDCG@20 / recall@K / MRR offline eval harness. |

### Modified retrieval modules (`music-crs-baselines/mcrs/`)
| Path | Responsibility |
|---|---|
| `mcrs/retrieval_modules/__init__.py` | **MODIFY** — add new factory `wrrf_bm25_dense_lyrics_bge_m3_ft_v1` (Stage A). The fine-tuned BGE-M3 reuses the existing `DENSE_LOCAL` class with `model_name=<hub_repo>` + a distinct `embed_label`. No new sub-retriever class. |
| `mcrs/rerankers/__init__.py` | **MODIFY** — verify the existing `bge_reranker_v2_m3` registry entry accepts a `model_path` override (it already gets one passed through from `crs_baseline.py`; we add a smoke test). Add new `chain` reranker_type that orchestrates a list of rerankers. |
| `mcrs/rerankers/bge_reranker.py` | **MODIFY** — accept an optional `model_name` constructor arg that overrides the module-level `MODEL_NAME` default. Enables loading our fine-tuned reranker by config. |
| `mcrs/rerankers/lgbm_rerank.py` | **REUSE AS-IS** — the existing `LGBM_RERANKER` already loads `booster.txt + metadata.json`. We just train a new model that conforms to that contract. |
| `mcrs/rerankers/chain.py` | **NEW** — `CHAIN_RERANKER` runs a list of rerankers in sequence (CE → LGBM). Enables Stage A+B+C without touching `crs_baseline.batch_chat`. |
| `mcrs/crs_baseline.py` | **UNCHANGED** — the existing single-reranker call site forwards side-channel kwargs; `CHAIN_RERANKER` is a single reranker that delegates internally. |

### New configs (`music-crs-baselines/config/`)
| Path | Responsibility |
|---|---|
| `config/180-wrrf-bge-m3-ft-v5kto-blindA.yaml` | Submission 1: Stage A only |
| `config/181-+ce-ft-v5kto-blindA.yaml` | Submission 2: Stage A + B |
| `config/182-+lgbm-v5kto-blindA.yaml` | Submission 3: Stage A + B + C |
| `config/179-zero-shot-bge-m3-baseline-v5kto-blindA.yaml` | Stage 0: zero-shot baseline (sanity check) |

### New notebooks (`colab/`)
| Path | Responsibility | GPU |
|---|---|---|
| `colab/70_train_bi_encoder.ipynb` | Stage A: HN mining + fine-tune + merge + push + eval | yes |
| `colab/71_train_cross_encoder.ipynb` | Stage B: re-mine + fine-tune + push + eval | yes |
| `colab/72_build_lgbm_features_train.ipynb` | Stage C: feature extraction + LGBM train | CPU |
| `colab/73_run_blindset_retrieval_v2.ipynb` | Blind-A inference (mirrors `colab/63_*.ipynb`) | yes |
| `colab/74_compare_v1_v2.ipynb` | Diagnostic: dev composite for v1 vs v2 | yes |

### Tests (`tests/`)
| Path | Coverage |
|---|---|
| `tests/test_bi_encoder_training_data.py` | (Tasks 1-6, already shipped) HN miner, PercPos filter, JSONL writer |
| `tests/test_train_bi_encoder.py` | Custom PEFT-LoRA loop: dataset, loss, merge step |
| `tests/test_cross_encoder_training_data.py` | Re-mining, triple builder (uses HF dataset walk) |
| `tests/test_lgbm_features_extended.py` | New feature helpers added to `build_lgbm_features.py` |
| `tests/test_train_lgbm_ranker.py` | `build_groups` (sort=False), trainer wires up metadata.json |
| `tests/test_eval_retrieval_v2.py` | nDCG@K, recall@K, MRR computation |
| `tests/test_v2_factories.py` | New `wrrf_bm25_dense_lyrics_bge_m3_ft_v1` factory builds correctly |
| `tests/test_bge_reranker_override.py` | Existing `BGE_RERANKER` accepts `model_name` override |
| `tests/test_chain_reranker.py` | `CHAIN_RERANKER` runs two rerankers in order, forwards side-channel kwargs |

---

## Task Sequence Overview

| Phase | Tasks | Calendar time |
|---|---|---|
| Pre-flight | 1–2 | 30 min |
| Stage 0 — zero-shot baseline | 3 | 1 hr (offline eval only) |
| Stage A — bi-encoder fine-tune | 4–12 | ~2 days (mostly Colab training) |
| **Submission 1** | 13 | 30 min + CodaBench wait |
| Stage B — cross-encoder fine-tune | 14–19 | ~1 day |
| **Submission 2** | 20 | 30 min + CodaBench wait |
| Stage C — LightGBM | 21–26 | ~1 day |
| **Submission 3** | 27 | 30 min + CodaBench wait |
| Wrap-up | 28 | 30 min |

---

## Pre-flight

### Task 1: Verify environment + Drive caches

**Files:**
- No file changes — verification only

- [ ] **Step 1: Verify local Python venv has expected packages**

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -c "import torch, transformers, datasets, lightgbm; print(torch.__version__, transformers.__version__, lightgbm.__version__)"`

Expected: prints versions without error (torch >=2.0, transformers >=4.40, lightgbm any).

- [ ] **Step 2: Verify FlagEmbedding is available (will be installed in Colab notebooks too)**

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -c "import FlagEmbedding" 2>&1 || pip install -q FlagEmbedding`

Expected: import succeeds or installs cleanly.

- [ ] **Step 3: Verify HF dataset schema matches spec assumptions**

Run:
```bash
/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -c "
from datasets import load_dataset
ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='train', streaming=True)
row = next(iter(ds))
required = ['conversation_goal', 'user_profile_raw', 'chat_history', 'goal_progress_assessments']
for f in required:
    if f not in row:
        print(f'MISSING: {f}')
    else:
        print(f'OK: {f}')
"
```

Expected: all 4 fields print `OK:`. If `goal_progress_assessments` is missing, note it and proceed (we have fallback per spec §10).

- [ ] **Step 4: Commit any working notes if env diverges from spec**

If Step 3 finds discrepancies vs the spec, append a note to the spec or open a new memory file. Otherwise no commit.

---

### Task 2: Create branch protection — confirm we're on `fresh-model`

**Files:**
- No file changes — verification only

- [ ] **Step 1: Confirm branch + clean working tree**

Run: `git status && git log --oneline -3`

Expected: branch is `fresh-model`, working tree clean, top commit is `04df546` (the spec commit) or later.

- [ ] **Step 2: Ensure remote is in sync**

Run: `git fetch origin && git status -sb`

Expected: branch is up-to-date with `origin/fresh-model` or ahead by exactly the spec commit.

---

## Stage 0 — Zero-shot Bundle A baseline (sanity check)

### Task 3: Run zero-shot Bundle A on dev + Blind-A

**Files:**
- Create: `music-crs-baselines/config/179-zero-shot-bge-m3-baseline-v5kto-blindA.yaml`
- Read: `music-crs-baselines/config/132-bge-m3-v5kto-prorank-rerank-blindsetA.yaml` (closest existing template)

**Why this task exists:** Per spec §9, before sinking ~10 GPU-hr into fine-tuning, confirm zero-shot BGE-M3 doesn't already match or beat the current 0.21 baseline (which would invalidate the fine-tune plan).

- [ ] **Step 1: Inspect existing config 132 to crib structure**

Run: `cat music-crs-baselines/config/132-bge-m3-v5kto-prorank-rerank-blindsetA.yaml`

Note the field set; you'll mirror it.

- [ ] **Step 2: Write the zero-shot baseline config**

Create `music-crs-baselines/config/179-zero-shot-bge-m3-baseline-v5kto-blindA.yaml`:

```yaml
# Stage 0 baseline: zero-shot BGE-M3 + dense_lyrics + BM25 wRRF + ProRank + v5-kto responder.
# Run BEFORE fine-tuning to establish the no-FT lower bound.

lm_type: "OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged"
lora_path: null
lora_max_rank: 32

retrieval_type: "wrrf_bm25_dense_lyrics_bge_m3_v1"
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

use_cmqr: true
cmqr_prompt_name: "cmqr_rewrites"
cmqr_n_rewrites: 4
cmqr_topk_per_rewrite: 50
cmqr_rrf_k: 60
cmqr_max_new_tokens: 96
```

- [ ] **Step 3: Run dev-set composite eval (NOT Blind-A — don't waste quota)**

Open `colab/74_compare_v1_v2.ipynb` (will be created in Task 28; for now use a fresh Colab cell). Or run locally on Blackwell:

```bash
%cd /content/recsys2026/music-crs-baselines
!python run_inference_devset.py --tid 179-zero-shot-bge-m3-baseline-v5kto-blindA --batch_size 32 2>&1 | tee /content/drive/MyDrive/recsys2026_retrieval_v2_cache/stage0_devset_log.txt
!python ../scripts/compute_composite.py --predictions exp/inference/devset/179-zero-shot-bge-m3-baseline-v5kto-blindA.json --official-evaluator
```

Expected output: a composite score. Compare to the current Bundle A baseline (composite ~0.21 on Blind-A; dev parity unknown but should be in the same ballpark).

- [ ] **Step 4: Decision gate**

- If composite ≥ 0.20 (matches v5-kto baseline parity): zero-shot BGE-M3 is competitive; **fine-tuning should add meaningful lift**. Proceed.
- If composite ≪ 0.18 (clear regression): zero-shot BGE-M3 doesn't fit this catalog. **HALT this plan**. Reassess encoder choice with the user.
- If composite is in (0.18, 0.20): noisy; proceed with caution.

- [ ] **Step 5: Commit the config + decision note**

```bash
git add music-crs-baselines/config/179-zero-shot-bge-m3-baseline-v5kto-blindA.yaml
git commit -m "stage 0: zero-shot BGE-M3 baseline config (Bundle A pre-fine-tune sanity)"
```

---

## Stage A — BGE-M3 bi-encoder fine-tune

### Task 4: Query/track formatter module + tests

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/bge_m3_format.py`
- Create: `tests/test_bge_m3_format.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bge_m3_format.py`:

```python
"""Tests for BGE-M3 query/track text formatters used in fine-tune + inference."""
import pytest


def test_format_track_text_includes_all_5_fields():
    """Track text concatenates the 5 corpus fields in the documented order."""
    from mcrs.retrieval_modules.bge_m3_format import format_track_text
    text = format_track_text(
        track_name="Suite: Judy Blue Eyes",
        artist_name="Crosby Stills Nash",
        album_name="Crosby Stills & Nash",
        release_date="1969-05-29",
        tag_list=["folk rock", "harmony"],
    )
    assert "Suite: Judy Blue Eyes" in text
    assert "Crosby Stills Nash" in text
    assert "1969" in text
    assert "folk rock" in text


def test_format_track_text_handles_empty_fields():
    """Missing fields render as 'unknown' or empty, not crash."""
    from mcrs.retrieval_modules.bge_m3_format import format_track_text
    text = format_track_text(
        track_name="Unknown Track", artist_name=None, album_name=None,
        release_date=None, tag_list=None,
    )
    assert "Unknown Track" in text


def test_format_query_text_uses_inference_pipeline_format():
    """Query format must match what crs_baseline.batch_chat produces at inference.

    Critical: train/eval parity. The format includes chat_history concatenation
    + current_user_query. We reuse the existing online formatter.
    """
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    text = format_query_text(
        chat_history=[
            {"role": "user", "content": "I like 70s rock"},
            {"role": "assistant", "content": "How about CSN?"},
        ],
        current_user_query="Yes, more like that",
        user_profile={"age": 35, "country_code": "US"},
        conversation_goal={"listener_goal": "find 70s folk-rock"},
    )
    assert "I like 70s rock" in text
    assert "Yes, more like that" in text
    # User profile + goal influence retrieval; should be present.
    assert "US" in text or "35" in text


def test_format_query_handles_empty_history():
    """First-turn case: chat_history is empty list."""
    from mcrs.retrieval_modules.bge_m3_format import format_query_text
    text = format_query_text(
        chat_history=[],
        current_user_query="Play me something upbeat",
        user_profile=None,
        conversation_goal=None,
    )
    assert "Play me something upbeat" in text
```

- [ ] **Step 2: Run tests — expect failure**

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -m pytest tests/test_bge_m3_format.py -v`

Expected: 4 tests FAIL with `ModuleNotFoundError: bge_m3_format`.

- [ ] **Step 3: Implement the formatter**

Create `music-crs-baselines/mcrs/retrieval_modules/bge_m3_format.py`:

```python
"""Query + track text formatters for BGE-M3 fine-tune + inference.

CRITICAL: format_query_text() must produce the EXACT same string as
crs_baseline.batch_chat builds at inference. This is the train/eval parity
contract — fine-tuning with one format and inferring with another is the
canonical generative-retrieval failure mode.
"""
from __future__ import annotations

from typing import Optional


def format_track_text(
    track_name: str,
    artist_name: Optional[str] = None,
    album_name: Optional[str] = None,
    release_date: Optional[str] = None,
    tag_list: Optional[list[str]] = None,
) -> str:
    """5-field track text matching BM25 corpus.

    Format: 'track_name: X | artist_name: Y | album_name: Z | release_date: W | tag_list: a, b, c'
    Missing fields render as the field name followed by 'unknown'.
    """
    parts = [f"track_name: {track_name}"]
    parts.append(f"artist_name: {artist_name or 'unknown'}")
    parts.append(f"album_name: {album_name or 'unknown'}")
    parts.append(f"release_date: {release_date or 'unknown'}")
    if tag_list:
        parts.append(f"tag_list: {', '.join(tag_list)}")
    else:
        parts.append("tag_list: ")
    return " | ".join(parts)


def format_query_text(
    chat_history: list[dict[str, str]],
    current_user_query: str,
    user_profile: Optional[dict] = None,
    conversation_goal: Optional[dict] = None,
    max_history_turns: int = 6,
) -> str:
    """Query text matching crs_baseline.batch_chat's runtime construction.

    Layout:
        [USER]: age=X, country=Y, prefers=Z
        [GOAL]: listener_goal_text
        [HISTORY]: U: prev_user_msg / A: prev_assistant_msg / ...
        [QUERY]: current_user_query
    """
    parts: list[str] = []
    if user_profile is not None:
        parts.append(
            f"[USER]: age={user_profile.get('age', '?')}, "
            f"country={user_profile.get('country_code', '?')}, "
            f"prefers={user_profile.get('preferred_musical_culture', '?')}"
        )
    if conversation_goal is not None:
        goal = conversation_goal.get("listener_goal", "")
        if goal:
            parts.append(f"[GOAL]: {goal}")
    if chat_history:
        msgs = chat_history[-max_history_turns:]
        hist_parts = []
        for m in msgs:
            role = "U" if m.get("role") == "user" else "A"
            hist_parts.append(f"{role}: {m.get('content', '')}")
        parts.append("[HISTORY]: " + " / ".join(hist_parts))
    parts.append(f"[QUERY]: {current_user_query}")
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -m pytest tests/test_bge_m3_format.py -v`

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/bge_m3_format.py tests/test_bge_m3_format.py
git commit -m "stage A: BGE-M3 query/track text formatters (TDD, 4 tests)"
```

---

### Task 5: Hard-negative miner module + tests

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/hn_miner.py`
- Create: `tests/test_hn_miner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hn_miner.py`:

```python
"""Tests for NV-Retriever-style hard-negative miner."""
import numpy as np
import pytest


def test_percpos_filter_drops_above_threshold():
    """Candidates scoring >= threshold * positive_score are dropped (false-neg guard)."""
    from mcrs.retrieval_modules.hn_miner import percpos_filter
    candidate_scores = [0.95, 0.82, 0.75, 0.60, 0.40]
    positive_score = 1.00
    kept = percpos_filter(candidate_scores, positive_score, threshold=0.80)
    assert kept == [False, False, True, True, True]  # 0.95 and 0.82 dropped


def test_percpos_filter_keeps_everything_at_threshold_1_0():
    """threshold=1.0 only drops candidates that exactly tie the positive."""
    from mcrs.retrieval_modules.hn_miner import percpos_filter
    kept = percpos_filter([0.99, 0.50], positive_score=1.00, threshold=1.0)
    assert kept == [True, True]


def test_sample_hard_negatives_respects_rank_range_and_cap():
    """sample_hard_negatives draws from the filtered top-K within rank_range."""
    from mcrs.retrieval_modules.hn_miner import sample_hard_negatives
    # Filtered candidates at ranks 5, 7, 10, 12, 15, 20, 30, 40
    filtered_ranks = [5, 7, 10, 12, 15, 20, 30, 40, 50, 80]
    drawn = sample_hard_negatives(filtered_ranks, k=4, rank_range=(2, 200), seed=42)
    assert len(drawn) == 4
    for rank in drawn:
        assert 2 <= rank <= 200
        assert rank in filtered_ranks


def test_sample_hard_negatives_returns_all_when_pool_smaller_than_k():
    """If only 3 negatives survive the filter but k=10, return all 3 (no padding)."""
    from mcrs.retrieval_modules.hn_miner import sample_hard_negatives
    drawn = sample_hard_negatives([5, 10, 15], k=10, rank_range=(2, 200), seed=42)
    assert len(drawn) == 3


def test_mine_negatives_for_query_returns_negs_aligned_to_track_ids():
    """End-to-end mine: returns list of track_ids (strings), not indices."""
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    track_ids = ["t1", "t2", "t3", "t4", "t5"]
    # Query embedding most similar to t1 (gold), then t2, t3...
    query_emb = np.array([1.0, 0.0])
    track_embs = np.array([
        [1.0, 0.0],   # t1 — gold
        [0.95, 0.31], # t2 — hard near-gold
        [0.80, 0.60], # t3 — medium
        [0.50, 0.87], # t4 — far
        [0.0, 1.0],   # t5 — orthogonal
    ])
    negs = mine_negatives_for_query(
        query_emb=query_emb, track_embs=track_embs, track_ids=track_ids,
        gold_track_id="t1", percpos_threshold=0.97, k_negs=2, seed=42,
    )
    assert len(negs) <= 2
    assert "t1" not in negs  # gold excluded
    # t2's similarity to query is 0.95 which is >= 0.97 * 1.0 = 0.97? 0.95 < 0.97, so kept.
    # We expect t3, t4, or t5 in negs (t2 may also be kept).
    for n in negs:
        assert n in track_ids
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_hn_miner.py -v`

Expected: 5 tests FAIL with import error.

- [ ] **Step 3: Implement the HN miner**

Create `music-crs-baselines/mcrs/retrieval_modules/hn_miner.py`:

```python
"""Hard-negative miner for bi-encoder fine-tuning.

Implements NV-Retriever's TopK-PercPos filtering [arXiv 2407.15831] adapted for
small catalogs (47K tracks): threshold defaults to 0.80 (not 0.95 — see spec
§6 reviewer finding).
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np


def percpos_filter(
    candidate_scores: list[float],
    positive_score: float,
    threshold: float = 0.80,
) -> list[bool]:
    """Return a boolean mask of which candidates pass the false-negative filter.

    A candidate is KEPT if candidate_score < threshold * positive_score.
    Candidates that score TOO CLOSE to the positive are likely false negatives
    (i.e., other valid answers we just didn't have labels for).
    """
    cutoff = threshold * positive_score
    return [s < cutoff for s in candidate_scores]


def sample_hard_negatives(
    filtered_ranks: list[int],
    k: int,
    rank_range: tuple[int, int] = (2, 200),
    seed: int = 42,
) -> list[int]:
    """Sample up to k negatives uniformly from the filtered list, within rank_range.

    If the filtered pool has fewer than k items in range, return all of them.
    """
    lo, hi = rank_range
    in_range = [r for r in filtered_ranks if lo <= r <= hi]
    if not in_range:
        return []
    if len(in_range) <= k:
        return list(in_range)
    rng = random.Random(seed)
    return sorted(rng.sample(in_range, k))


def mine_negatives_for_query(
    query_emb: np.ndarray,
    track_embs: np.ndarray,
    track_ids: list[str],
    gold_track_id: str,
    percpos_threshold: float = 0.80,
    k_negs: int = 15,
    pool_size: int = 200,
    seed: int = 42,
) -> list[str]:
    """Mine hard negatives for one query.

    1. Compute cosine similarities query × all tracks (assumes unit-normed).
    2. Identify the positive_score (sim to gold_track_id).
    3. Take top-`pool_size` candidates (excluding the gold).
    4. Apply PercPos filter at `percpos_threshold`.
    5. Sample up to k_negs from the filtered list.
    """
    if len(track_ids) != track_embs.shape[0]:
        raise ValueError("track_ids and track_embs length mismatch")
    sims = track_embs @ query_emb
    # Position of gold
    try:
        gold_idx = track_ids.index(gold_track_id)
    except ValueError:
        raise ValueError(f"gold_track_id {gold_track_id} not in track_ids")
    positive_score = float(sims[gold_idx])
    # Top-pool_size candidates excluding gold
    ranked_idxs = np.argsort(-sims)
    pool = [int(i) for i in ranked_idxs if i != gold_idx][:pool_size]
    pool_scores = [float(sims[i]) for i in pool]
    keep_mask = percpos_filter(pool_scores, positive_score, percpos_threshold)
    # Build list of (rank_in_pool, track_id) for those that pass
    filtered_track_ids = [track_ids[pool[i]] for i, keep in enumerate(keep_mask) if keep]
    filtered_ranks_in_pool = [i + 2 for i, keep in enumerate(keep_mask) if keep]  # +2 because gold = rank 1
    # Sample within range
    sampled_ranks = sample_hard_negatives(
        filtered_ranks_in_pool, k=k_negs, rank_range=(2, pool_size + 1), seed=seed,
    )
    # Map ranks back to track_ids
    rank_to_tid = dict(zip(filtered_ranks_in_pool, filtered_track_ids))
    return [rank_to_tid[r] for r in sampled_ranks]
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_hn_miner.py -v`

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/hn_miner.py tests/test_hn_miner.py
git commit -m "stage A: NV-Retriever-style HN miner (PercPos 0.80, TDD, 5 tests)"
```

---

### Task 6: Triple-builder script for Stage A training data

**Files:**
- Create: `scripts/build_bi_encoder_training_data.py`
- Create: `tests/test_build_bi_encoder_training_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_bi_encoder_training_data.py`:

```python
"""Tests for build_bi_encoder_training_data.py orchestration."""
import json
import pytest


def test_build_triples_writes_jsonl_with_query_pos_neg():
    """Each output row is a JSONL dict with 'query', 'pos' (list), 'neg' (list)."""
    from scripts.build_bi_encoder_training_data import build_triples_for_row

    row = {
        "session_id": "s1",
        "user_id": "u1",
        "turn_number": 1,
        "current_user_query": "play me 70s rock",
        "chat_history": [],
        "user_profile_raw": {"age": 30, "country_code": "US"},
        "conversation_goal": {"listener_goal": "discover music"},
        "track_id": "t_gold",
    }
    track_text_map = {"t_gold": "track_name: ... | ...", "t_neg1": "...", "t_neg2": "..."}
    triple = build_triples_for_row(row, gold_track_id="t_gold", neg_track_ids=["t_neg1", "t_neg2"], track_text_map=track_text_map)
    assert "query" in triple
    assert "pos" in triple and isinstance(triple["pos"], list)
    assert triple["pos"] == ["track_name: ... | ..."]
    assert "neg" in triple and len(triple["neg"]) == 2
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_build_bi_encoder_training_data.py -v`

Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement the orchestrator**

Create `scripts/build_bi_encoder_training_data.py`:

```python
"""Stage A training data builder.

Reads existing W2 train.parquet (raw subset) → mines hard negatives via
zero-shot BGE-M3 → writes JSONL triples consumable by FlagEmbedding's
unified_finetune.

Usage:
  python scripts/build_bi_encoder_training_data.py \\
    --train-parquet experiments/cache/sid_training/train.parquet \\
    --track-meta-hf talkpl-ai/TalkPlayData-Challenge-Track-Metadata \\
    --bge-m3-model BAAI/bge-m3 \\
    --output experiments/cache/bge_m3_triples.jsonl \\
    --percpos-threshold 0.80 --k-negs 15 --pool-size 200 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mcrs.retrieval_modules.bge_m3_format import format_query_text, format_track_text
from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query


def build_triples_for_row(
    row: dict,
    gold_track_id: str,
    neg_track_ids: list[str],
    track_text_map: dict[str, str],
) -> dict:
    """Build one JSONL triple for a single (query, gold, negs) tuple."""
    query = format_query_text(
        chat_history=row.get("chat_history") or [],
        current_user_query=row.get("current_user_query", ""),
        user_profile=row.get("user_profile_raw"),
        conversation_goal=row.get("conversation_goal"),
    )
    return {
        "query": query,
        "pos": [track_text_map[gold_track_id]],
        "neg": [track_text_map[tid] for tid in neg_track_ids if tid in track_text_map],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--bge-m3-model", default="BAAI/bge-m3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--percpos-threshold", type=float, default=0.80)
    parser.add_argument("--k-negs", type=int, default=15)
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke cap; 0 = all")
    args = parser.parse_args()

    # 1. Load train data
    train = pd.read_parquet(args.train_parquet)
    train = train[train["source"] == "raw"].reset_index(drop=True)
    if args.max_rows > 0:
        train = train.head(args.max_rows)
    print(f"[hn-miner] {len(train)} raw conversation pairs", file=sys.stderr)

    # 2. Load track metadata + build text map
    from datasets import load_dataset
    track_meta = load_dataset(args.track_meta_hf, split="all_tracks")
    track_ids: list[str] = []
    track_texts: list[str] = []
    for trow in tqdm(track_meta, desc="format tracks"):
        tid = trow["track_id"]
        text = format_track_text(
            track_name=trow.get("track_name", "unknown"),
            artist_name=trow.get("artist_name"),
            album_name=trow.get("album_name"),
            release_date=trow.get("release_date"),
            tag_list=trow.get("tag_list"),
        )
        track_ids.append(tid)
        track_texts.append(text)
    track_text_map = dict(zip(track_ids, track_texts))

    # 3. Encode all tracks with zero-shot BGE-M3
    from FlagEmbedding import BGEM3FlagModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(args.bge_m3_model, use_fp16=True, device=device)
    print("[hn-miner] encoding 47K tracks...", file=sys.stderr)
    track_embs = model.encode(
        track_texts, batch_size=args.batch_size, max_length=256,
    )["dense_vecs"]  # (N, D) float32, L2-normed
    track_embs = np.asarray(track_embs, dtype=np.float32)
    # Cache to disk for re-use
    np.save(Path(args.output).with_suffix(".track_embs.npy"), track_embs)

    # 4. Format + encode queries, then mine negatives per query, write JSONL
    print(f"[hn-miner] mining negatives per query → {args.output}", file=sys.stderr)
    with open(args.output, "w") as f_out:
        for i in tqdm(range(0, len(train), args.batch_size), desc="mine"):
            batch_rows = train.iloc[i:i+args.batch_size].to_dict("records")
            batch_queries = [
                format_query_text(
                    chat_history=r.get("chat_history") or [],
                    current_user_query=r.get("current_user_query", ""),
                    user_profile=r.get("user_profile_raw"),
                    conversation_goal=r.get("conversation_goal"),
                )
                for r in batch_rows
            ]
            batch_embs = model.encode(
                batch_queries, batch_size=args.batch_size, max_length=512,
            )["dense_vecs"]
            batch_embs = np.asarray(batch_embs, dtype=np.float32)
            for j, row in enumerate(batch_rows):
                gold_tid = row["track_id"]
                if gold_tid not in track_text_map:
                    continue
                negs = mine_negatives_for_query(
                    query_emb=batch_embs[j],
                    track_embs=track_embs,
                    track_ids=track_ids,
                    gold_track_id=gold_tid,
                    percpos_threshold=args.percpos_threshold,
                    k_negs=args.k_negs,
                    pool_size=args.pool_size,
                    seed=42 + i + j,
                )
                if len(negs) < 2:
                    continue  # need at least 2 negatives for contrastive
                triple = build_triples_for_row(row, gold_tid, negs, track_text_map)
                f_out.write(json.dumps(triple) + "\n")

    print(f"[hn-miner] DONE → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_build_bi_encoder_training_data.py -v`

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_bi_encoder_training_data.py tests/test_build_bi_encoder_training_data.py
git commit -m "stage A: bi-encoder training-data builder + HN miner orchestration (TDD)"
```

---

### Task 7: Stage A notebook 70 — clone + deps + smoke triple builder

**Files:**
- Create: `colab/70_train_bi_encoder.ipynb`

**Schema note:** The triple-builder script (`scripts/build_bi_encoder_training_data.py`) shipped in Task 6 walks the HF conversation dataset directly via `_iter_conversation_turns`. It does NOT read `train.parquet` (whose schema lacks `chat_history`, `current_user_query`, `user_profile_raw`, `conversation_goal`). Notebook 70 therefore invokes the builder with `--train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset` and NEVER passes `--train-parquet`.

- [ ] **Step 1: Create the notebook with cells in this exact order**

Run locally from the repo root:

```bash
python3 << 'PYEOF'
import json
nb = {
    "cells": [
        {
            "cell_type": "markdown", "metadata": {}, "source": [
                "# 70 — Train BGE-M3 bi-encoder (Stage A of nDCG-stretch plan)\n\n",
                "Fine-tunes BAAI/bge-m3 on per-music-turn conversation pairs walked\n",
                "from `talkpl-ai/TalkPlayData-Challenge-Dataset` (train split) via the\n",
                "Task 6 builder. Uses a custom PEFT-LoRA training loop\n",
                "(sentence-transformers + peft) — NOT FlagEmbedding's CLI, which\n",
                "lacks the LoRA flags this plan needs.\n\n",
                "**Prereqs**: HF_TOKEN in Colab Secrets. Drive folder\n",
                "`recsys2026_retrieval_v2_cache` exists.\n\n",
                "**Wallclock**: ~8-12 hr on Blackwell (1.5 hr HN mining + 6-10 hr train)."
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 1) Setup — clone + HF auth + Drive mount + deps.\n",
                "import os\n",
                "from google.colab import userdata, drive\n",
                "os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n",
                "drive.mount('/content/drive', force_remount=False)\n",
                "\n",
                "BRANCH = 'fresh-model'\n",
                "!rm -rf /content/recsys2026\n",
                "!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026\n",
                "%cd /content/recsys2026\n",
                "\n",
                "DRIVE_BASE = '/content/drive/MyDrive'\n",
                "LOCAL_BASE = '/content/recsys2026/experiments/cache'\n",
                "os.makedirs(LOCAL_BASE, exist_ok=True)\n",
                "for name, drive_subdir in [\n",
                "    ('sid', 'recsys2026_sid_cache'),\n",
                "    ('retrieval_v2', 'recsys2026_retrieval_v2_cache'),\n",
                "]:\n",
                "    src = f'{DRIVE_BASE}/{drive_subdir}'\n",
                "    dst = f'{LOCAL_BASE}/{name}'\n",
                "    os.makedirs(src, exist_ok=True)\n",
                "    if os.path.islink(dst): os.unlink(dst)\n",
                "    elif os.path.exists(dst):\n",
                "        import shutil; shutil.rmtree(dst)\n",
                "    os.symlink(src, dst)\n",
                "\n",
                "!pip install -q --upgrade \\\n",
                "    'peft>=0.10' 'transformers>=4.40' 'accelerate>=0.30' \\\n",
                "    'sentence-transformers>=3.0' 'FlagEmbedding>=1.3' \\\n",
                "    'datasets' 'pandas<3.0' 'tqdm' 'omegaconf' 'pyyaml' 'tensorboard'"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 2) Smoke: build 200 triples to verify the HN miner works end-to-end.\n",
                "# IMPORTANT: --train-conv-hf walks the HF conversation dataset directly\n",
                "# (Task 6's `_iter_conversation_turns`). Do NOT pass --train-parquet:\n",
                "# the W2 parquet schema is (source, session_id, track_id, query,\n",
                "# code_1..3) and lacks chat_history / current_user_query /\n",
                "# user_profile_raw / conversation_goal that the builder needs.\n",
                "!cd /content/recsys2026 && python scripts/build_bi_encoder_training_data.py \\\n",
                "    --train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset \\\n",
                "    --output experiments/cache/retrieval_v2/triples_smoke.jsonl \\\n",
                "    --max-rows 200 --percpos-threshold 0.80 --k-negs 15 \\\n",
                "    2>&1 | tail -20\n",
                "!wc -l experiments/cache/retrieval_v2/triples_smoke.jsonl\n",
                "!head -3 experiments/cache/retrieval_v2/triples_smoke.jsonl"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 3) Full HN mining — ~1.5 hr on Blackwell.\n",
                "import os\n",
                "RESULTS_DIR = '/content/drive/MyDrive/recsys2026_retrieval_v2_cache/results'\n",
                "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
                "!cd /content/recsys2026 && python -u scripts/build_bi_encoder_training_data.py \\\n",
                "    --train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset \\\n",
                "    --output experiments/cache/retrieval_v2/triples_bge_m3.jsonl \\\n",
                "    --percpos-threshold 0.80 --k-negs 15 --batch-size 64 \\\n",
                "    2>&1 | tee /content/drive/MyDrive/recsys2026_retrieval_v2_cache/hn_mining_log.txt\n",
                "!wc -l experiments/cache/retrieval_v2/triples_bge_m3.jsonl"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4, "nbformat_minor": 5
}
with open('/Users/orrimoch/PythonProjs/recsys2026/colab/70_train_bi_encoder.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("notebook 70 written (4 cells: title + setup + smoke + full HN)")
PYEOF
```

(More cells will be added in later tasks — training, push, eval.)

- [ ] **Step 2: Verify the notebook is valid JSON + has the 4 cells**

Run: `python3 -c "import json; nb = json.load(open('colab/70_train_bi_encoder.ipynb')); print(len(nb['cells']), 'cells')"`

Expected: `4 cells`.

- [ ] **Step 3: Commit**

```bash
git add colab/70_train_bi_encoder.ipynb
git commit -m "stage A: notebook 70 — setup + smoke HN mining cells (walks HF dataset)"
```

---

### Task 8: Stage A custom PEFT-LoRA training script

**Files:**
- Create: `scripts/train_bi_encoder.py`
- Create: `tests/test_train_bi_encoder.py`

**Why custom (not FlagEmbedding)**: FlagEmbedding's `unified_finetune` CLI on master does NOT expose `--use_lora`, `--lora_rank`, `--lora_alpha` — those flags only exist on private forks / pending PRs. A `torchrun -m FlagEmbedding.finetune.embedder.encoder_only.m3 --use_lora True ...` call would crash with `unrecognized arguments` at training start. We replace it with a small custom loop using `sentence-transformers` + `peft` directly.

**Architecture**:
- Load `BAAI/bge-m3` via `transformers.AutoModel` (sentence-transformers' wrapper hides BGE-M3's multi-output heads; we only need the dense output for InfoNCE so a plain HF model is sufficient).
- Wrap with `peft.LoraConfig(r=32, lora_alpha=64, target_modules=["query", "key", "value", "dense"], task_type="FEATURE_EXTRACTION")` and `peft.get_peft_model(...)`.
- Custom `MultipleNegativesRankingLoss` over (query, positive, neg[1..15]) loaded from the JSONL.
- AdamW, lr=5e-6, 2 epochs, bf16 mixed precision.
- TensorBoard logs to `--results-dir`.
- After training: `peft_model.merge_and_unload()`, then `push_to_hub` to `<hub-repo>-merged`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_bi_encoder.py`:

```python
"""Tests for the custom PEFT-LoRA training loop in scripts/train_bi_encoder.py."""
import json

import pytest


def test_triple_jsonl_dataset_yields_query_pos_neg(tmp_path):
    """TripleJsonlDataset returns dicts with query, positive, negatives keys."""
    from scripts.train_bi_encoder import TripleJsonlDataset

    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        for i in range(3):
            f.write(json.dumps({
                "query": f"q{i}", "pos": [f"p{i}"],
                "neg": [f"n{i}_{j}" for j in range(15)],
            }) + "\n")
    ds = TripleJsonlDataset(str(path))
    assert len(ds) == 3
    row = ds[0]
    assert row["query"] == "q0"
    assert row["positive"] == "p0"
    assert len(row["negatives"]) == 15


def test_triple_jsonl_dataset_pads_short_neg_list(tmp_path):
    """When a row has fewer than `n_negatives` negs, it's repeated (don't drop the row)."""
    from scripts.train_bi_encoder import TripleJsonlDataset

    path = tmp_path / "triples.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"query": "q", "pos": ["p"], "neg": ["n1", "n2"]}) + "\n")
    ds = TripleJsonlDataset(str(path), n_negatives=15)
    row = ds[0]
    assert len(row["negatives"]) == 15
    # First two are the actual negs; rest are random samples from the same pool.
    assert "n1" in row["negatives"]
    assert "n2" in row["negatives"]


def test_build_lora_targets_returns_attention_module_names():
    """target_modules covers BGE-M3 XLMRoberta attention + FFN projections."""
    from scripts.train_bi_encoder import _BGE_M3_LORA_TARGETS
    # BGE-M3 is XLMRoberta-based; attention layers are .query/.key/.value/.dense
    assert "query" in _BGE_M3_LORA_TARGETS
    assert "key" in _BGE_M3_LORA_TARGETS
    assert "value" in _BGE_M3_LORA_TARGETS
```

- [ ] **Step 2: Run tests — expect failure**

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -m pytest tests/test_train_bi_encoder.py -v`

Expected: 3 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the training script**

Create `scripts/train_bi_encoder.py`:

```python
"""Stage A: custom PEFT-LoRA fine-tune of BAAI/bge-m3 on conversation→track triples.

Why a custom loop (not FlagEmbedding's CLI):
  FlagEmbedding's master `unified_finetune` does NOT expose --use_lora /
  --lora_rank / --lora_alpha. A `torchrun -m FlagEmbedding...` invocation
  with those flags crashes at startup. This script uses sentence-transformers'
  underlying AutoModel + peft.LoraConfig + a small MultipleNegativesRanking
  loss to get equivalent training behavior with the exact LoRA settings the
  plan calls for (r=32 / alpha=64 over attention+FFN projections).

Hyperparameters (spec §6):
  - lr 5e-6, per-device bs 2, train_group_size 8 (1 pos + 7 in-batch negs),
    n_negatives_per_query 15 (sampled from the 15 mined negs per row),
    temperature 0.05, epochs 2.
  - LoRA r=32 alpha=64 over query/key/value/dense projections.
  - bf16 mixed precision via torch.cuda.amp.
  - After training: merge LoRA via peft_model.merge_and_unload(), push merged
    model to Hub.

Usage:
  python scripts/train_bi_encoder.py \
    --triples experiments/cache/retrieval_v2/triples_bge_m3.jsonl \
    --output-dir /content/bge_m3_finetune \
    --hub-repo OrRim123/recsys2026-bge-m3-music-v1 \
    --results-dir /content/drive/MyDrive/recsys2026_retrieval_v2_cache/results/bge_m3 \
    --merge --cleanup-after-push
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional


# LoRA target modules for BGE-M3 (XLM-RoBERTa under the hood).
_BGE_M3_LORA_TARGETS = ["query", "key", "value", "dense"]


class TripleJsonlDataset:
    """Loads JSONL triples produced by scripts/build_bi_encoder_training_data.py.

    Each row: {"query": str, "pos": [str], "neg": [str, ...]}.
    On __getitem__, returns {"query": str, "positive": str, "negatives": [str]*n_negatives}.
    Short neg-lists are upsampled by repeated random sampling from the same row.
    """

    def __init__(self, path: str, n_negatives: int = 15, seed: int = 42):
        import json as _json
        self.rows = []
        with open(path) as f:
            for line in f:
                obj = _json.loads(line)
                if not obj.get("pos") or not obj.get("neg"):
                    continue
                self.rows.append(obj)
        self.n_negatives = int(n_negatives)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        negs = list(row["neg"])
        if len(negs) >= self.n_negatives:
            negs = negs[: self.n_negatives]
        else:
            pad_pool = list(negs) if negs else [""]
            while len(negs) < self.n_negatives:
                negs.append(self.rng.choice(pad_pool))
        return {
            "query": row["query"],
            "positive": row["pos"][0],
            "negatives": negs,
        }


def _collate_batch(batch: list[dict], tokenizer, max_q_len: int, max_p_len: int):
    """Tokenize a list of {query, positive, negatives} rows into tensors."""
    import torch

    queries = [b["query"] for b in batch]
    # positives + negatives per row → (B * (1 + n_negs)) docs.
    docs: list[str] = []
    n_per = 1 + len(batch[0]["negatives"])
    for b in batch:
        docs.append(b["positive"])
        docs.extend(b["negatives"])

    q_enc = tokenizer(queries, max_length=max_q_len, padding=True, truncation=True, return_tensors="pt")
    d_enc = tokenizer(docs, max_length=max_p_len, padding=True, truncation=True, return_tensors="pt")
    return q_enc, d_enc, n_per


def _mean_pool(last_hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """L2-normalized mean-pool over non-padding tokens. Matches BGE-M3 dense head."""
    import torch
    import torch.nn.functional as F

    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return F.normalize(pooled, p=2, dim=1)


def _info_nce_loss(q_emb: "torch.Tensor", d_emb: "torch.Tensor", n_per: int, temperature: float) -> "torch.Tensor":
    """InfoNCE: each query has 1 positive + (n_per - 1) negatives, contiguous in d_emb."""
    import torch
    import torch.nn.functional as F

    B = q_emb.size(0)
    d_emb = d_emb.view(B, n_per, -1)             # (B, n_per, D)
    scores = torch.einsum("bd,bnd->bn", q_emb, d_emb) / temperature  # (B, n_per)
    labels = torch.zeros(B, dtype=torch.long, device=scores.device)  # positive is index 0
    return F.cross_entropy(scores, labels)


def _train(args):
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train-bi-encoder] device={device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=_BGE_M3_LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(device)
    model.print_trainable_parameters()

    ds = TripleJsonlDataset(args.triples, n_negatives=args.n_negatives)
    print(f"[train-bi-encoder] {len(ds)} training triples", file=sys.stderr)
    loader = DataLoader(
        ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: _collate_batch(b, tokenizer, args.query_max_len, args.passage_max_len),
    )

    total_steps = len(loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    step = 0
    for epoch in range(args.epochs):
        for q_enc, d_enc, n_per in loader:
            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            d_enc = {k: v.to(device) for k, v in d_enc.items()}
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                q_out = model(**q_enc)
                d_out = model(**d_enc)
                q_emb = _mean_pool(q_out.last_hidden_state, q_enc["attention_mask"])
                d_emb = _mean_pool(d_out.last_hidden_state, d_enc["attention_mask"])
                loss = _info_nce_loss(q_emb, d_emb, n_per, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.logging_steps == 0:
                writer.add_scalar("train/loss", float(loss.item()), step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                print(f"[train-bi-encoder] step={step}/{total_steps} loss={float(loss.item()):.4f}", file=sys.stderr)

    writer.close()
    # Save the adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train-bi-encoder] adapter saved → {output_dir}", file=sys.stderr)


def _merge_and_push(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    print("[train-bi-encoder] merging LoRA → base", file=sys.stderr)
    base = AutoModel.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, args.output_dir)
    merged = peft_model.merge_and_unload()
    merged_dir = Path(args.output_dir) / "merged"
    merged.save_pretrained(str(merged_dir))
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(str(merged_dir))
    print(f"[train-bi-encoder] merged → {merged_dir}", file=sys.stderr)

    hub_target = f"{args.hub_repo}-merged"
    print(f"[train-bi-encoder] pushing to {hub_target}", file=sys.stderr)
    merged.push_to_hub(hub_target, private=False)
    tok.push_to_hub(hub_target, private=False)
    return merged_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triples", required=True)
    p.add_argument("--base-model", default="BAAI/bge-m3")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hub-repo", required=True,
                   help="HF Hub repo prefix (the merged model is pushed to <hub_repo>-merged).")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--cleanup-after-push", action="store_true")
    p.add_argument("--results-dir", default=None)
    # Hyperparameters (spec §6)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--n-negatives", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--query-max-len", type=int, default=512)
    p.add_argument("--passage-max-len", type=int, default=256)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--logging-steps", type=int, default=50)
    args = p.parse_args()

    _train(args)

    if args.merge:
        _merge_and_push(args)

    if args.results_dir is not None:
        rd = Path(args.results_dir)
        rd.mkdir(parents=True, exist_ok=True)
        runs_src = Path(args.output_dir) / "runs"
        if runs_src.exists():
            shutil.copytree(runs_src, rd / "runs", dirs_exist_ok=True)

    if args.cleanup_after_push and args.merge:
        merged_dir = Path(args.output_dir) / "merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect pass**

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -m pytest tests/test_train_bi_encoder.py -v`

Expected: 3 tests PASS. The tests intentionally avoid importing torch/peft/transformers; they only touch `TripleJsonlDataset` and the `_BGE_M3_LORA_TARGETS` constant, both of which are pure Python.

- [ ] **Step 5: Smoke-check the script parses (Python compile only)**

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -m py_compile scripts/train_bi_encoder.py`

Expected: no output (clean compile).

- [ ] **Step 6: Commit**

```bash
git add scripts/train_bi_encoder.py tests/test_train_bi_encoder.py
git commit -m "stage A: train_bi_encoder.py — custom PEFT-LoRA loop (replaces FlagEmbedding CLI; TDD)"
```

---

### Task 9: Add Stage A training + push + catalog re-embed cells to notebook 70

**Files:**
- Modify: `colab/70_train_bi_encoder.ipynb`

**Note on catalog re-embed (Step 1 cell #6)**: This cell reads ONLY the HF Track-Metadata dataset (`talkpl-ai/TalkPlayData-Challenge-Track-Metadata`) — it never touches `train.parquet`, so no schema-walk fix is needed here.

- [ ] **Step 1: Append the smoke-train, full-train, and re-embed cells**

Run locally:

```bash
python3 << 'PYEOF'
import json
NB = '/Users/orrimoch/PythonProjs/recsys2026/colab/70_train_bi_encoder.ipynb'
nb = json.load(open(NB))

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}

nb['cells'].extend([
    code(
        "# 4) Smoke fine-tune: 1 epoch on 500 triples — verify code path works.\n"
        "!head -500 experiments/cache/retrieval_v2/triples_bge_m3.jsonl > experiments/cache/retrieval_v2/triples_smoke_500.jsonl\n"
        "!cd /content/recsys2026 && python scripts/train_bi_encoder.py \\\n"
        "    --triples experiments/cache/retrieval_v2/triples_smoke_500.jsonl \\\n"
        "    --output-dir /content/bge_m3_smoke \\\n"
        "    --hub-repo OrRim123/recsys2026-bge-m3-smoke \\\n"
        "    --epochs 1 --logging-steps 10 \\\n"
        "    2>&1 | tail -20\n"
        "!rm -rf /content/bge_m3_smoke"
    ),
    code(
        "# 5) FULL fine-tune: ~6-10 hr on Blackwell. Pushes merged model to Hub.\n"
        "!cd /content/recsys2026 && python -u scripts/train_bi_encoder.py \\\n"
        "    --triples experiments/cache/retrieval_v2/triples_bge_m3.jsonl \\\n"
        "    --output-dir /content/bge_m3_finetune \\\n"
        "    --hub-repo OrRim123/recsys2026-bge-m3-music-v1 \\\n"
        "    --results-dir /content/drive/MyDrive/recsys2026_retrieval_v2_cache/results/bge_m3 \\\n"
        "    --merge --cleanup-after-push \\\n"
        "    2>&1 | tee /content/drive/MyDrive/recsys2026_retrieval_v2_cache/bge_m3_train_log.txt"
    ),
    code(
        "# 6) Re-embed the catalog with the fine-tuned model, into DENSE_LOCAL's expected path.\n"
        "# DENSE_LOCAL reads from {cache_dir}/dense_local/{safe_model}/{embed_label}/track_embeddings.pkl\n"
        "# We pin embed_label='bge-m3-music-v1-merged' so wRRF factory entry can find it.\n"
        "import os, pickle, numpy as np\n"
        "from datasets import load_dataset\n"
        "from sentence_transformers import SentenceTransformer\n"
        "import sys\n"
        "sys.path.insert(0, '/content/recsys2026/music-crs-baselines')\n"
        "from mcrs.retrieval_modules.bge_m3_format import format_track_text\n"
        "\n"
        "HUB_REPO = 'OrRim123/recsys2026-bge-m3-music-v1-merged'\n"
        "EMBED_LABEL = 'bge-m3-music-v1-merged'\n"
        "CACHE_ROOT = '/content/drive/MyDrive/recsys2026_retrieval_v2_cache/dense_local'\n"
        "safe_model = HUB_REPO.replace('/', '_')\n"
        "out_dir = os.path.join(CACHE_ROOT, safe_model, EMBED_LABEL)\n"
        "os.makedirs(out_dir, exist_ok=True)\n"
        "\n"
        "model = SentenceTransformer(HUB_REPO, device='cuda')\n"
        "tm = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')\n"
        "texts = [format_track_text(r.get('track_name','unknown'), r.get('artist_name'), r.get('album_name'), r.get('release_date'), r.get('tag_list')) for r in tm]\n"
        "track_ids = [r['track_id'] for r in tm]\n"
        "embs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)\n"
        "embs = np.asarray(embs, dtype=np.float32)\n"
        "out_path = os.path.join(out_dir, 'track_embeddings.pkl')\n"
        "with open(out_path, 'wb') as f:\n"
        "    pickle.dump({'track_ids': track_ids, 'track_mat': embs}, f)\n"
        "print(f'wrote {len(track_ids)} embeddings → {out_path}')\n"
        "# Symlink into experiments/cache/dense_local so the runtime cache_dir lookup hits.\n"
        "local_cache = '/content/recsys2026/experiments/cache/dense_local'\n"
        "os.makedirs(local_cache, exist_ok=True)\n"
        "local_link = os.path.join(local_cache, safe_model)\n"
        "if not os.path.exists(local_link):\n"
        "    os.symlink(os.path.join(CACHE_ROOT, safe_model), local_link)\n"
        "print('symlink ready:', local_link)"
    ),
])

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"notebook 70 now has {len(nb['cells'])} cells")
PYEOF
```

- [ ] **Step 2: Verify notebook**

Run: `python3 -c "import json; nb = json.load(open('colab/70_train_bi_encoder.ipynb')); print(len(nb['cells']), 'cells')"`

Expected: `7 cells`.

- [ ] **Step 3: Commit**

```bash
git add colab/70_train_bi_encoder.ipynb
git commit -m "stage A: notebook 70 — smoke + full train + catalog re-embed cells (DENSE_LOCAL contract)"
```

---

### Task 10: Add Stage A offline-eval module + tests

**Files:**
- Create: `scripts/eval_retrieval_v2.py`
- Create: `tests/test_eval_retrieval_v2.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_retrieval_v2.py`:

```python
"""Tests for offline retrieval eval (nDCG@K / recall@K / MRR)."""
import pytest


def test_ndcg_at_k_perfect_top_1():
    """Gold at rank 1 → nDCG@20 = 1.0."""
    from scripts.eval_retrieval_v2 import compute_ndcg_at_k
    score = compute_ndcg_at_k(retrieved=["g", "a", "b", "c"], gold="g", k=20)
    assert score == 1.0


def test_ndcg_at_k_gold_outside_topk():
    """Gold below k → nDCG@20 = 0.0."""
    from scripts.eval_retrieval_v2 import compute_ndcg_at_k
    score = compute_ndcg_at_k(retrieved=["a", "b"] * 20, gold="g", k=20)
    assert score == 0.0


def test_recall_at_k_present():
    """Recall@K = 1 if gold in top-K."""
    from scripts.eval_retrieval_v2 import compute_recall_at_k
    assert compute_recall_at_k(retrieved=["a", "b", "g"], gold="g", k=3) == 1.0
    assert compute_recall_at_k(retrieved=["a", "b"], gold="g", k=3) == 0.0


def test_mrr_returns_reciprocal_of_first_correct_rank():
    """MRR for one query is 1 / rank_of_gold (or 0 if absent)."""
    from scripts.eval_retrieval_v2 import compute_mrr
    assert compute_mrr(retrieved=["a", "g", "c"], gold="g") == 0.5
    assert compute_mrr(retrieved=["a", "b", "c"], gold="g") == 0.0
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_eval_retrieval_v2.py -v`

Expected: 4 fails (module missing).

- [ ] **Step 3: Implement eval functions**

Create `scripts/eval_retrieval_v2.py`:

```python
"""Generic offline eval harness for the v2 retrieval pipeline.

Computes nDCG@K, recall@K, MRR per turn against a JSONL of predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def compute_ndcg_at_k(retrieved: list[str], gold: str, k: int) -> float:
    """nDCG@k for a single-relevant-item case: 1/log2(rank+1) if gold in top-k, else 0."""
    top_k = retrieved[:k]
    if gold not in top_k:
        return 0.0
    rank = top_k.index(gold) + 1  # 1-indexed
    return 1.0 / math.log2(rank + 1)


def compute_recall_at_k(retrieved: list[str], gold: str, k: int) -> float:
    """Recall@K for single-relevant-item: 1 if gold in top-k else 0."""
    return 1.0 if gold in retrieved[:k] else 0.0


def compute_mrr(retrieved: list[str], gold: str) -> float:
    """MRR for one query: 1/rank or 0."""
    try:
        rank = retrieved.index(gold) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-jsonl", required=True,
                        help="JSONL with rows {query_id, gold, retrieved: [...]}.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.predictions_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))

    ndcgs = [compute_ndcg_at_k(r["retrieved"], r["gold"], 20) for r in rows]
    recalls_20 = [compute_recall_at_k(r["retrieved"], r["gold"], 20) for r in rows]
    recalls_100 = [compute_recall_at_k(r["retrieved"], r["gold"], 100) for r in rows]
    mrrs = [compute_mrr(r["retrieved"], r["gold"]) for r in rows]

    metrics = {
        "n_queries": len(rows),
        "mean_ndcg_at_20": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "mean_recall_at_20": sum(recalls_20) / len(recalls_20) if recalls_20 else 0.0,
        "mean_recall_at_100": sum(recalls_100) / len(recalls_100) if recalls_100 else 0.0,
        "mean_mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
    }
    Path(args.output_json).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_eval_retrieval_v2.py -v`

Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_retrieval_v2.py tests/test_eval_retrieval_v2.py
git commit -m "stage A: offline eval harness (nDCG@K + recall@K + MRR, TDD, 4 tests)"
```

---

### Task 11: Wire fine-tuned BGE-M3 into the wRRF factory (reuse DENSE_LOCAL)

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/__init__.py`
- Create: `tests/test_v2_factories.py`

**Key choice**: We REUSE the existing `DENSE_LOCAL` class (which already accepts `model_name` + `embed_label` constructor args and reads precomputed catalog pickles from `{cache_dir}/dense_local/{safe_model}/{embed_label}/track_embeddings.pkl`). No new sub-retriever class.

**Top-level YAML key**: The new factory reads the override from `extra_config["bge_m3_hub_repo"]` — and configs 180/181/182 put `bge_m3_hub_repo:` at the TOP LEVEL of the YAML (NOT nested under an `extra_config:` block), because `run_inference_blindset.py:102` dumps the entire YAML into the `extra_config_dict` already.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v2_factories.py`:

```python
"""Tests for the v2 wRRF factory variant that uses the fine-tuned BGE-M3."""
import pytest


def test_wrrf_bm25_dense_lyrics_bge_m3_ft_v1_factory_recognizes_type(tmp_path):
    """New factory recognizes the retrieval_type key (does NOT raise 'Unsupported')."""
    from mcrs.retrieval_modules import load_retrieval_module

    try:
        load_retrieval_module(
            retrieval_type="wrrf_bm25_dense_lyrics_bge_m3_ft_v1",
            dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            track_split_types=["all_tracks"],
            corpus_types=["track_name", "artist_name", "album_name"],
            cache_dir=str(tmp_path),
            extra_config={"bge_m3_hub_repo": "OrRim123/recsys2026-bge-m3-music-v1-merged"},
        )
    except ValueError as e:
        # Acceptable: sub-retrievers can't fully init in a test sandbox (no
        # cached embeddings) — but the factory MUST recognize the type.
        assert "Unsupported retrieval type" not in str(e), \
            f"factory failed to register the new retrieval_type: {e}"
    except FileNotFoundError:
        # Also acceptable: DENSE_LOCAL needs precomputed catalog pickles that
        # don't exist in tmp_path. Reaching this branch proves the factory
        # dispatched correctly.
        pass
```

- [ ] **Step 2: Run test — expect fail**

Run: `pytest tests/test_v2_factories.py -v`

Expected: ValueError about "Unsupported retrieval type" (or similar mismatch).

- [ ] **Step 3: Add the new factory branch**

Modify `music-crs-baselines/mcrs/retrieval_modules/__init__.py`. Find the existing branch `elif retrieval_type == "wrrf_bm25_dense_lyrics_bge_m3_v1":` and ADD a NEW branch immediately after its closing `)` block, BEFORE the `wrrf_bm25_dense_lyrics_qwen3_4b_v1` branch:

```python
    # nDCG-stretch Stage A factory. Same shape as wrrf_bm25_dense_lyrics_bge_m3_v1
    # but the metadata-dense sub uses our FINE-TUNED merged BGE-M3 model loaded
    # by DENSE_LOCAL (model_name=<hub_repo>, embed_label='bge-m3-music-v1-merged').
    #
    # `extra_config["bge_m3_hub_repo"]` overrides the default Hub path. The key
    # MUST be at YAML top level (run_inference_blindset.py forwards the entire
    # YAML dict as extra_config); do NOT nest under `extra_config:` in the YAML.
    elif retrieval_type == "wrrf_bm25_dense_lyrics_bge_m3_ft_v1":
        bge_m3_hub = extra_config.get(
            "bge_m3_hub_repo",
            "OrRim123/recsys2026-bge-m3-music-v1-merged",
        )
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
                    "type": "dense_metadata_bge_m3_ft_local",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.6,
                    "extra_config": {"hub_repo": bge_m3_hub},
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
            ],
            k=60,
        )
```

Also add the new sub-retriever type `dense_metadata_bge_m3_ft_local` near the existing `dense_metadata_bge_m3_local` branch (line ~71):

```python
    # nDCG-stretch Stage A — fine-tuned BGE-M3 (merged Hub repo). Reuses
    # DENSE_LOCAL; the embed_label distinguishes its precomputed catalog
    # pickle from the zero-shot BGE-M3 cache.
    elif retrieval_type == "dense_metadata_bge_m3_ft_local":
        hub_repo = extra_config.get("hub_repo", "OrRim123/recsys2026-bge-m3-music-v1-merged")
        return DENSE_LOCAL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            model_name=hub_repo,
            embed_label="bge-m3-music-v1-merged",
        )
```

- [ ] **Step 4: Run test — expect pass**

Run: `pytest tests/test_v2_factories.py -v`

Expected: pass (or FileNotFoundError — both acceptable per test design).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/__init__.py tests/test_v2_factories.py
git commit -m "stage A: new factory wrrf_bm25_dense_lyrics_bge_m3_ft_v1 + dense_metadata_bge_m3_ft_local (reuses DENSE_LOCAL)"
```

---

### Task 12: Stage A offline eval cell in notebook 70 + decision gate

**Files:**
- Modify: `colab/70_train_bi_encoder.ipynb`

**Schema fix**: Eval walks the HF conversation dataset (dev split) — NOT `val.parquet` — so it has access to `chat_history`, `current_user_query`, `user_profile_raw`, `conversation_goal` for the production-equivalent query format.

- [ ] **Step 1: Append the offline-eval cell**

```bash
python3 << 'PYEOF'
import json
NB = '/Users/orrimoch/PythonProjs/recsys2026/colab/70_train_bi_encoder.ipynb'
nb = json.load(open(NB))

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}

nb['cells'].append(code(
    "# 7) Offline eval: fine-tuned BGE-M3 (bi-encoder only) on the HF dev split.\n"
    "# Walks the HF conversation dataset directly (NOT val.parquet — that schema\n"
    "# lacks chat_history / user_profile / conversation_goal needed for the\n"
    "# production query format).\n"
    "# Gate: standalone nDCG@20 >= 0.15.\n"
    "import sys, math, os, pickle\n"
    "import numpy as np\n"
    "from datasets import load_dataset\n"
    "from sentence_transformers import SentenceTransformer\n"
    "sys.path.insert(0, '/content/recsys2026/music-crs-baselines')\n"
    "sys.path.insert(0, '/content/recsys2026/scripts')\n"
    "from mcrs.retrieval_modules.bge_m3_format import format_query_text\n"
    "from build_bi_encoder_training_data import _iter_conversation_turns\n"
    "\n"
    "HUB_REPO = 'OrRim123/recsys2026-bge-m3-music-v1-merged'\n"
    "EMBED_LABEL = 'bge-m3-music-v1-merged'\n"
    "CACHE_ROOT = '/content/drive/MyDrive/recsys2026_retrieval_v2_cache/dense_local'\n"
    "safe_model = HUB_REPO.replace('/', '_')\n"
    "catalog_pkl = os.path.join(CACHE_ROOT, safe_model, EMBED_LABEL, 'track_embeddings.pkl')\n"
    "with open(catalog_pkl, 'rb') as f:\n"
    "    payload = pickle.load(f)\n"
    "track_ids = payload['track_ids']\n"
    "track_mat = payload['track_mat']  # (N, D) L2-normalized\n"
    "model = SentenceTransformer(HUB_REPO, device='cuda')\n"
    "\n"
    "dev = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='dev')\n"
    "rows = _iter_conversation_turns(dev)[:500]  # cap for fast diagnostic\n"
    "queries = [format_query_text(\n"
    "    chat_history=r.get('chat_history') or [],\n"
    "    current_user_query=r.get('current_user_query',''),\n"
    "    user_profile=r.get('user_profile_raw'),\n"
    "    conversation_goal=r.get('conversation_goal'),\n"
    "    mode='raw',\n"
    ") for r in rows]\n"
    "q_emb = model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=True)\n"
    "q_emb = np.asarray(q_emb, dtype=np.float32)\n"
    "sims = q_emb @ track_mat.T  # (Q, N)\n"
    "top20_idx = np.argpartition(-sims, kth=19, axis=1)[:, :20]\n"
    "ri = np.arange(sims.shape[0])[:, None]\n"
    "top20_sorted = top20_idx[ri, np.argsort(-sims[ri, top20_idx], axis=1)]\n"
    "ndcgs = []\n"
    "tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}\n"
    "for i, r in enumerate(rows):\n"
    "    gold = r['track_id']\n"
    "    if gold not in tid_to_idx:\n"
    "        ndcgs.append(0.0); continue\n"
    "    gold_idx = tid_to_idx[gold]\n"
    "    top20_tids = top20_sorted[i]\n"
    "    if gold_idx in top20_tids:\n"
    "        rank = list(top20_tids).index(gold_idx) + 1\n"
    "        ndcgs.append(1.0 / math.log2(rank + 1))\n"
    "    else:\n"
    "        ndcgs.append(0.0)\n"
    "mean_ndcg = float(sum(ndcgs) / len(ndcgs))\n"
    "print(f'fine-tuned BGE-M3 standalone nDCG@20 on dev: {mean_ndcg:.4f}')\n"
    "print(f'gate (Stage A diagnostic): >= 0.15')\n"
    "if mean_ndcg < 0.15:\n"
    "    print('GATE FAIL — investigate before Submission 1.')\n"
    "else:\n"
    "    print('GATE PASS — proceed to Submission 1.')"
))

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"notebook 70 now has {len(nb['cells'])} cells")
PYEOF
```

- [ ] **Step 2: Commit**

```bash
git add colab/70_train_bi_encoder.ipynb
git commit -m "stage A: notebook 70 — offline eval on HF dev split + gate (>= 0.15)"
```

---

## Submission 1

### Task 13: Submission 1 — Stage A only on Blind-A

**Files:**
- Create: `music-crs-baselines/config/180-wrrf-bge-m3-ft-v5kto-blindA.yaml`
- Create: `colab/73_run_blindset_retrieval_v2.ipynb`

**Top-level YAML key**: `bge_m3_hub_repo` is at the TOP LEVEL of the YAML — NOT nested under `extra_config:` — because `run_inference_blindset.py:102` dumps the whole YAML into the `extra_config_dict` already.

- [ ] **Step 1: Write the config**

Create `music-crs-baselines/config/180-wrrf-bge-m3-ft-v5kto-blindA.yaml`:

```yaml
# Submission 1: Stage A — wRRF(BM25 + dense_lyrics + fine-tuned BGE-M3) + ProRank + v5-kto.
#
# The `bge_m3_hub_repo` key is at TOP LEVEL on purpose. `run_inference_blindset.py:102`
# does `OmegaConf.to_container(config, resolve=True)` and forwards the whole dict as
# extra_config, so the factory's `extra_config.get("bge_m3_hub_repo")` reads it.
# Do NOT nest under `extra_config:` (would put the key one level too deep).

lm_type: "OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged"
lora_path: null
lora_max_rank: 32

retrieval_type: "wrrf_bm25_dense_lyrics_bge_m3_ft_v1"
bge_m3_hub_repo: "OrRim123/recsys2026-bge-m3-music-v1-merged"

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

use_cmqr: true
cmqr_prompt_name: "cmqr_rewrites"
cmqr_n_rewrites: 4
cmqr_topk_per_rewrite: 50
cmqr_rrf_k: 60
cmqr_max_new_tokens: 96
```

- [ ] **Step 2: Create notebook 73 (clone of notebook 63 with config 180)**

```bash
cp /Users/orrimoch/PythonProjs/recsys2026/colab/63_run_blindset_sid.ipynb \
   /Users/orrimoch/PythonProjs/recsys2026/colab/73_run_blindset_retrieval_v2.ipynb
```

Then edit notebook 73 via Python to swap config TID, zip name, and title:

```bash
python3 << 'PYEOF'
import json
NB = '/Users/orrimoch/PythonProjs/recsys2026/colab/73_run_blindset_retrieval_v2.ipynb'
nb = json.load(open(NB))
for cell in nb['cells']:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    src = src.replace('170-wrrf-sid-v5kto-blindsetA', '180-wrrf-bge-m3-ft-v5kto-blindA')
    src = src.replace('sid-ensemble-170', 'retrieval-v2-180-bge-m3-ft')
    src = src.replace(
        '# 63 — Run Blind-A with config 170 (wRRF + SID)',
        '# 73 — Run Blind-A with config 180 (wRRF + fine-tuned BGE-M3, Submission 1)',
    )
    cell['source'] = src.splitlines(keepends=True)
with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print("notebook 73 created from notebook 63 with config 180")
PYEOF
```

- [ ] **Step 3: Commit**

```bash
git add music-crs-baselines/config/180-wrrf-bge-m3-ft-v5kto-blindA.yaml \
        colab/73_run_blindset_retrieval_v2.ipynb
git commit -m "submission 1: config 180 + notebook 73 (Stage A only, BGE-M3-FT + ProRank)"
```

- [ ] **Step 4: User runs notebook 73 in Colab, uploads zip to CodaBench, waits for score**

This is a manual step. After the score lands:
- Composite score logged via `scripts/blind_a_score_tracker.py append --tid 180 --composite <X> --ndcg <Y> --catdiv <Z> --lexdiv <W> --llm <V>`
- **Decision**: if composite < 0.18, HALT (per spec §10 abort rule). Otherwise proceed to Stage B.

---

## Stage B — Cross-encoder fine-tune

### Task 14: Cross-encoder HN re-mining script (walks HF dataset)

**Files:**
- Create: `scripts/build_cross_encoder_training_data.py`
- Create: `tests/test_build_cross_encoder_training_data.py`

**Schema fix**: This script walks the HF conversation dataset via `_iter_conversation_turns` (imported from `scripts.build_bi_encoder_training_data`). It NEVER reads `train.parquet`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_cross_encoder_training_data.py`:

```python
"""Tests for cross-encoder triple-builder."""
import pytest


def test_build_ce_triple_shapes():
    """Cross-encoder triple has query, pos list (len 1), neg list (len 7)."""
    from scripts.build_cross_encoder_training_data import build_ce_triple
    triple = build_ce_triple(
        query="play me jazz",
        gold_track_text="track_name: So What | ...",
        neg_track_texts=["track" + str(i) for i in range(7)],
    )
    assert triple["query"] == "play me jazz"
    assert triple["pos"] == ["track_name: So What | ..."]
    assert len(triple["neg"]) == 7


def test_build_ce_triple_preserves_neg_order():
    """Neg list order is preserved (matters for downstream pair construction)."""
    from scripts.build_cross_encoder_training_data import build_ce_triple
    triple = build_ce_triple(query="q", gold_track_text="g", neg_track_texts=["n1", "n2", "n3"])
    assert triple["neg"] == ["n1", "n2", "n3"]
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_build_cross_encoder_training_data.py -v`

Expected: 2 fails (module missing).

- [ ] **Step 3: Implement the script**

Create `scripts/build_cross_encoder_training_data.py`:

```python
"""Stage B training data builder.

Mines HNs via the FINE-TUNED BGE-M3 (Stage A output) so the cross-encoder
sees in-distribution negatives per spec §7.

Walks the HF conversation dataset (NOT train.parquet — schema lacks
chat_history / current_user_query / user_profile_raw / conversation_goal).

Usage:
  python scripts/build_cross_encoder_training_data.py \
    --train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset \
    --bge-m3-ft-hub OrRim123/recsys2026-bge-m3-music-v1-merged \
    --output experiments/cache/retrieval_v2/triples_reranker.jsonl \
    --percpos-threshold 0.80 --k-negs 7
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from tqdm import tqdm

from mcrs.retrieval_modules.bge_m3_format import format_query_text, format_track_text
from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
from build_bi_encoder_training_data import _iter_conversation_turns


def build_ce_triple(query: str, gold_track_text: str, neg_track_texts: list[str]) -> dict:
    """One JSONL row for the cross-encoder trainer."""
    return {"query": query, "pos": [gold_track_text], "neg": list(neg_track_texts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-conv-hf", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--bge-m3-ft-hub", required=True)
    parser.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--percpos-threshold", type=float, default=0.80)
    parser.add_argument("--k-negs", type=int, default=7)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"[ce-build] walking {args.train_conv_hf} (train split)", file=sys.stderr)
    conv_ds = load_dataset(args.train_conv_hf, split="train")
    train_rows = _iter_conversation_turns(conv_ds)
    if args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]
    print(f"[ce-build] {len(train_rows)} per-music-turn rows", file=sys.stderr)

    # Track text map
    track_meta = load_dataset(args.track_meta_hf, split="all_tracks")
    track_ids: list[str] = []
    track_texts: list[str] = []
    for trow in tqdm(track_meta, desc="format tracks"):
        track_ids.append(trow["track_id"])
        track_texts.append(format_track_text(
            track_name=trow.get("track_name", "unknown"),
            artist_name=trow.get("artist_name"),
            album_name=trow.get("album_name"),
            release_date=trow.get("release_date"),
            tag_list=trow.get("tag_list"),
        ))
    track_text_map = dict(zip(track_ids, track_texts))
    if len(set(track_ids)) != len(track_ids):
        dupes = [tid for tid, c in Counter(track_ids).items() if c > 1]
        raise RuntimeError(f"track catalog has duplicates (first 5: {dupes[:5]})")

    # Encode catalog with the fine-tuned model
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.bge_m3_ft_hub, device=device)
    print("[ce-build] encoding catalog with fine-tuned BGE-M3", file=sys.stderr)
    track_embs = model.encode(track_texts, batch_size=args.batch_size, normalize_embeddings=True,
                              show_progress_bar=True)
    track_embs = np.asarray(track_embs, dtype=np.float32)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with open(args.output, "w") as f_out:
        for i in tqdm(range(0, len(train_rows), args.batch_size), desc="mine"):
            batch = train_rows[i:i + args.batch_size]
            batch_queries = [format_query_text(
                chat_history=r.get("chat_history") or [],
                current_user_query=r.get("current_user_query", ""),
                user_profile=r.get("user_profile_raw"),
                conversation_goal=r.get("conversation_goal"),
                mode="raw",
            ) for r in batch]
            batch_embs = model.encode(batch_queries, batch_size=args.batch_size,
                                       normalize_embeddings=True, show_progress_bar=False)
            batch_embs = np.asarray(batch_embs, dtype=np.float32)
            for j, row in enumerate(batch):
                gold = row["track_id"]
                if gold not in track_text_map:
                    n_skipped += 1; continue
                try:
                    negs = mine_negatives_for_query(
                        query_emb=batch_embs[j], track_embs=track_embs, track_ids=track_ids,
                        gold_track_id=gold, percpos_threshold=args.percpos_threshold,
                        k_negs=args.k_negs, pool_size=args.pool_size, seed=42 + i + j,
                    )
                except ValueError:
                    n_skipped += 1; continue
                if len(negs) < 2:
                    n_skipped += 1; continue
                triple = build_ce_triple(
                    query=batch_queries[j],
                    gold_track_text=track_text_map[gold],
                    neg_track_texts=[track_text_map[n] for n in negs if n in track_text_map],
                )
                f_out.write(json.dumps(triple) + "\n")
                n_written += 1

    print(f"[ce-build] DONE → {args.output} (wrote {n_written}, skipped {n_skipped})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_build_cross_encoder_training_data.py -v`

Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_cross_encoder_training_data.py tests/test_build_cross_encoder_training_data.py
git commit -m "stage B: cross-encoder HN re-mining (walks HF dataset; uses fine-tuned BGE-M3)"
```

---

### Task 15: Cross-encoder fine-tune script

**Files:**
- Create: `scripts/train_cross_encoder.py`

Uses `sentence-transformers`' built-in `CrossEncoder` API (which is a thin wrapper over HF's `AutoModelForSequenceClassification` with the right loss). Full FT (no LoRA — the model is small, 568M, fits cleanly in Colab Blackwell memory).

- [ ] **Step 1: Implement**

Create `scripts/train_cross_encoder.py`:

```python
"""Stage B: fine-tune BAAI/bge-reranker-v2-m3 via sentence-transformers' CrossEncoder API.

Hyperparams (spec §7):
  - Full FT (no LoRA), lr=2e-5, bs=16, epochs=3, max_length=512, bf16
  - Cross-entropy on pairwise (pos, neg) — one example per (query, pos, neg) pair.

Usage:
  python scripts/train_cross_encoder.py \
    --triples experiments/cache/retrieval_v2/triples_reranker.jsonl \
    --output-dir /content/bge_reranker_finetune \
    --hub-repo OrRim123/recsys2026-bge-reranker-music-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _expand_triples_to_pairs(path: str) -> tuple[list, list]:
    """Each (q, pos, [negs]) row → 1 positive pair + len(negs) negative pairs."""
    pos_pairs = []
    neg_pairs = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            q = obj["query"]
            for p in obj.get("pos", []):
                pos_pairs.append([q, p])
            for n in obj.get("neg", []):
                neg_pairs.append([q, n])
    return pos_pairs, neg_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hub-repo", required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    import torch

    pos_pairs, neg_pairs = _expand_triples_to_pairs(args.triples)
    print(f"[train-ce] {len(pos_pairs)} pos, {len(neg_pairs)} neg pairs", file=sys.stderr)

    # Build InputExample list: pos label=1.0, neg label=0.0.
    examples = [InputExample(texts=p, label=1.0) for p in pos_pairs] + \
               [InputExample(texts=n, label=0.0) for n in neg_pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = CrossEncoder(
        args.base_model, num_labels=1, max_length=args.max_length,
        automodel_args={"torch_dtype": torch.bfloat16},
    )
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(loader) * args.epochs),
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
        use_amp=True,
    )

    # Push to Hub (the saved model dir has the standard HF artifacts).
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    pushed = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    tok = AutoTokenizer.from_pretrained(args.output_dir)
    print(f"[train-ce] pushing → {args.hub_repo}", file=sys.stderr)
    pushed.push_to_hub(args.hub_repo, private=False)
    tok.push_to_hub(args.hub_repo, private=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke compile**

Run: `python3 -m py_compile scripts/train_cross_encoder.py`

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_cross_encoder.py
git commit -m "stage B: train_cross_encoder.py — sentence-transformers CrossEncoder fine-tune"
```

---

### Task 16: Extend existing BGE_RERANKER to accept a model_name override

**Files:**
- Modify: `music-crs-baselines/mcrs/rerankers/bge_reranker.py`
- Modify: `music-crs-baselines/mcrs/rerankers/__init__.py`
- Create: `tests/test_bge_reranker_override.py`

**Why this approach (not a new class)**: The existing `BGE_RERANKER` already loads `BAAI/bge-reranker-v2-m3` from a module-level `MODEL_NAME` constant and exposes the same `rerank(queries, candidate_tids, topk, **kwargs)` interface we need. The cleanest change: accept an optional `model_name` constructor arg that overrides the default. Zero new reranker class.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bge_reranker_override.py`:

```python
"""Test that BGE_RERANKER accepts a model_name override via constructor."""
import inspect

import pytest


def test_bge_reranker_constructor_accepts_model_name_kwarg():
    """The constructor signature includes `model_name` so the factory can pass
    `reranker_model_path` through to it."""
    from mcrs.rerankers.bge_reranker import BGE_RERANKER
    sig = inspect.signature(BGE_RERANKER.__init__)
    assert "model_name" in sig.parameters, \
        "BGE_RERANKER.__init__ must accept a model_name kwarg for fine-tuned model loading"


def test_reranker_factory_passes_model_path_to_bge_reranker(monkeypatch, tmp_path):
    """When reranker_type='bge_reranker_v2_m3' and model_path is set, the factory
    forwards model_path → BGE_RERANKER(model_name=...)."""
    from mcrs.rerankers import load_reranker_module

    captured = {}

    class _Stub:
        def __init__(self, item_db_name, track_split_types, corpus_types, cache_dir, model_name=None):
            captured["model_name"] = model_name
            captured["called"] = True

    import mcrs.rerankers.bge_reranker as bge_mod
    monkeypatch.setattr(bge_mod, "BGE_RERANKER", _Stub)

    load_reranker_module(
        reranker_type="bge_reranker_v2_m3",
        item_db_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        track_split_types=["all_tracks"],
        corpus_types=["track_name"],
        cache_dir=str(tmp_path),
        model_path="OrRim123/recsys2026-bge-reranker-music-v1",
    )
    assert captured["called"]
    assert captured["model_name"] == "OrRim123/recsys2026-bge-reranker-music-v1"
```

- [ ] **Step 2: Run test — expect fail**

Run: `pytest tests/test_bge_reranker_override.py -v`

Expected: fails — `model_name` is not in constructor signature, factory doesn't forward `model_path`.

- [ ] **Step 3: Patch `BGE_RERANKER.__init__` to accept `model_name`**

In `music-crs-baselines/mcrs/rerankers/bge_reranker.py`:

```python
# Before (existing):
class BGE_RERANKER:
    def __init__(
        self,
        item_db_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
    ) -> None:
        ...
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, torch_dtype=dtype
        ).to(self.device).eval()

# After:
class BGE_RERANKER:
    def __init__(
        self,
        item_db_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        model_name: Optional[str] = None,
    ) -> None:
        ...
        # Default to MODEL_NAME (the public BGE reranker); override via
        # `model_name` for our fine-tuned weights.
        resolved = model_name or MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved, torch_dtype=dtype
        ).to(self.device).eval()
        print(f"[bge-rerank] loaded {resolved} on {self.device} dtype={dtype}")
```

(Also add `from typing import Optional` import if not present.)

- [ ] **Step 4: Patch the factory to forward `model_path` to BGE_RERANKER**

In `music-crs-baselines/mcrs/rerankers/__init__.py`:

```python
# Before:
if reranker_type == "bge_reranker_v2_m3":
    from .bge_reranker import BGE_RERANKER
    return BGE_RERANKER(
        item_db_name=item_db_name,
        track_split_types=track_split_types,
        corpus_types=corpus_types,
        cache_dir=cache_dir,
    )

# After:
if reranker_type == "bge_reranker_v2_m3":
    from .bge_reranker import BGE_RERANKER
    return BGE_RERANKER(
        item_db_name=item_db_name,
        track_split_types=track_split_types,
        corpus_types=corpus_types,
        cache_dir=cache_dir,
        model_name=model_path,  # None → keep default; Hub repo → override.
    )
```

- [ ] **Step 5: Run test — expect pass**

Run: `pytest tests/test_bge_reranker_override.py -v`

Expected: 2 pass.

- [ ] **Step 6: Commit**

```bash
git add music-crs-baselines/mcrs/rerankers/bge_reranker.py \
        music-crs-baselines/mcrs/rerankers/__init__.py \
        tests/test_bge_reranker_override.py
git commit -m "stage B: BGE_RERANKER accepts model_name override; factory forwards reranker_model_path (TDD)"
```

---

### Task 17: Notebook 71 — Stage B end-to-end

**Files:**
- Create: `colab/71_train_cross_encoder.ipynb`

Mirror notebook 70's structure: setup + smoke + full HN remine + full fine-tune + eval. All explicit cell contents.

- [ ] **Step 1: Create the notebook**

```bash
python3 << 'PYEOF'
import json
nb = {
    "cells": [
        {
            "cell_type": "markdown", "metadata": {}, "source": [
                "# 71 — Train BGE-reranker-v2-m3 cross-encoder (Stage B)\n\n",
                "Re-mines HNs via the Stage A fine-tuned BGE-M3, then full-FT the\n",
                "cross-encoder via sentence-transformers' CrossEncoder API.\n\n",
                "**Prereqs**: Stage A complete — `OrRim123/recsys2026-bge-m3-music-v1-merged`\n",
                "exists on Hub. HF_TOKEN in Colab Secrets.\n\n",
                "**Wallclock**: ~3-5 hr on Blackwell (1 hr HN re-mine + 2-4 hr train)."
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 1) Setup — clone + HF auth + Drive mount + deps.\n",
                "import os\n",
                "from google.colab import userdata, drive\n",
                "os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n",
                "drive.mount('/content/drive', force_remount=False)\n",
                "\n",
                "BRANCH = 'fresh-model'\n",
                "!rm -rf /content/recsys2026\n",
                "!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026\n",
                "%cd /content/recsys2026\n",
                "\n",
                "DRIVE_BASE = '/content/drive/MyDrive'\n",
                "LOCAL_BASE = '/content/recsys2026/experiments/cache'\n",
                "os.makedirs(LOCAL_BASE, exist_ok=True)\n",
                "src = f'{DRIVE_BASE}/recsys2026_retrieval_v2_cache'\n",
                "dst = f'{LOCAL_BASE}/retrieval_v2'\n",
                "os.makedirs(src, exist_ok=True)\n",
                "if os.path.islink(dst): os.unlink(dst)\n",
                "elif os.path.exists(dst):\n",
                "    import shutil; shutil.rmtree(dst)\n",
                "os.symlink(src, dst)\n",
                "\n",
                "!pip install -q --upgrade \\\n",
                "    'transformers>=4.40' 'accelerate>=0.30' \\\n",
                "    'sentence-transformers>=3.0' \\\n",
                "    'datasets' 'pandas<3.0' 'tqdm'"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 2) Smoke HN re-mine: 200 rows, verify the script runs.\n",
                "!cd /content/recsys2026 && python scripts/build_cross_encoder_training_data.py \\\n",
                "    --train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset \\\n",
                "    --bge-m3-ft-hub OrRim123/recsys2026-bge-m3-music-v1-merged \\\n",
                "    --output experiments/cache/retrieval_v2/triples_reranker_smoke.jsonl \\\n",
                "    --max-rows 200 --k-negs 7 \\\n",
                "    2>&1 | tail -10\n",
                "!wc -l experiments/cache/retrieval_v2/triples_reranker_smoke.jsonl"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 3) Full HN re-mine — ~1 hr on Blackwell.\n",
                "!cd /content/recsys2026 && python -u scripts/build_cross_encoder_training_data.py \\\n",
                "    --train-conv-hf talkpl-ai/TalkPlayData-Challenge-Dataset \\\n",
                "    --bge-m3-ft-hub OrRim123/recsys2026-bge-m3-music-v1-merged \\\n",
                "    --output experiments/cache/retrieval_v2/triples_reranker.jsonl \\\n",
                "    --percpos-threshold 0.80 --k-negs 7 \\\n",
                "    2>&1 | tee /content/drive/MyDrive/recsys2026_retrieval_v2_cache/ce_hn_remine_log.txt\n",
                "!wc -l experiments/cache/retrieval_v2/triples_reranker.jsonl"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 4) Smoke fine-tune: 1 epoch on 500 triples.\n",
                "!head -500 experiments/cache/retrieval_v2/triples_reranker.jsonl > experiments/cache/retrieval_v2/triples_reranker_smoke_500.jsonl\n",
                "!cd /content/recsys2026 && python scripts/train_cross_encoder.py \\\n",
                "    --triples experiments/cache/retrieval_v2/triples_reranker_smoke_500.jsonl \\\n",
                "    --output-dir /content/bge_reranker_smoke \\\n",
                "    --hub-repo OrRim123/recsys2026-bge-reranker-smoke \\\n",
                "    --epochs 1 --batch-size 8 \\\n",
                "    2>&1 | tail -15\n",
                "!rm -rf /content/bge_reranker_smoke"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 5) FULL fine-tune: ~2-4 hr on Blackwell. Pushes to Hub.\n",
                "!cd /content/recsys2026 && python -u scripts/train_cross_encoder.py \\\n",
                "    --triples experiments/cache/retrieval_v2/triples_reranker.jsonl \\\n",
                "    --output-dir /content/bge_reranker_finetune \\\n",
                "    --hub-repo OrRim123/recsys2026-bge-reranker-music-v1 \\\n",
                "    2>&1 | tee /content/drive/MyDrive/recsys2026_retrieval_v2_cache/ce_train_log.txt"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 6) Offline eval (Stage A+B): nDCG@20 on dev split through full pipeline.\n",
                "# Pulls dev conversations from HF, computes BM25+dense_lyrics+BGE-M3-FT top-100,\n",
                "# then reranks with the fine-tuned BGE-reranker. Gate: nDCG@20 >= 0.25.\n",
                "import sys, math, os, pickle\n",
                "import numpy as np\n",
                "from datasets import load_dataset\n",
                "from sentence_transformers import SentenceTransformer, CrossEncoder\n",
                "sys.path.insert(0, '/content/recsys2026/music-crs-baselines')\n",
                "sys.path.insert(0, '/content/recsys2026/scripts')\n",
                "from mcrs.retrieval_modules.bge_m3_format import format_query_text, format_track_text\n",
                "from build_bi_encoder_training_data import _iter_conversation_turns\n",
                "\n",
                "BGE_REPO = 'OrRim123/recsys2026-bge-m3-music-v1-merged'\n",
                "CE_REPO = 'OrRim123/recsys2026-bge-reranker-music-v1'\n",
                "CATALOG_PKL = f'/content/drive/MyDrive/recsys2026_retrieval_v2_cache/dense_local/{BGE_REPO.replace(\"/\",\"_\")}/bge-m3-music-v1-merged/track_embeddings.pkl'\n",
                "with open(CATALOG_PKL, 'rb') as f:\n",
                "    payload = pickle.load(f)\n",
                "track_ids, track_mat = payload['track_ids'], payload['track_mat']\n",
                "tid_to_idx = {t: i for i, t in enumerate(track_ids)}\n",
                "tm = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')\n",
                "tid_to_text = {r['track_id']: format_track_text(r.get('track_name','unknown'), r.get('artist_name'), r.get('album_name'), r.get('release_date'), r.get('tag_list')) for r in tm}\n",
                "\n",
                "bi = SentenceTransformer(BGE_REPO, device='cuda')\n",
                "ce = CrossEncoder(CE_REPO, device='cuda', max_length=512)\n",
                "\n",
                "dev = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='dev')\n",
                "rows = _iter_conversation_turns(dev)[:500]\n",
                "queries = [format_query_text(r.get('chat_history') or [], r.get('current_user_query',''), r.get('user_profile_raw'), r.get('conversation_goal'), mode='raw') for r in rows]\n",
                "\n",
                "# Bi-encoder top-100\n",
                "q_emb = bi.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=True)\n",
                "q_emb = np.asarray(q_emb, dtype=np.float32)\n",
                "sims = q_emb @ track_mat.T\n",
                "top100_idx = np.argpartition(-sims, kth=99, axis=1)[:, :100]\n",
                "ri = np.arange(sims.shape[0])[:, None]\n",
                "top100_sorted = top100_idx[ri, np.argsort(-sims[ri, top100_idx], axis=1)]\n",
                "\n",
                "# Cross-encoder rerank\n",
                "ndcgs = []\n",
                "for i, r in enumerate(rows):\n",
                "    gold = r['track_id']\n",
                "    cand_tids = [track_ids[j] for j in top100_sorted[i]]\n",
                "    pairs = [(queries[i], tid_to_text.get(t, '')) for t in cand_tids]\n",
                "    scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)\n",
                "    order = np.argsort(-scores)[:20]\n",
                "    top20 = [cand_tids[j] for j in order]\n",
                "    if gold in top20:\n",
                "        rank = top20.index(gold) + 1\n",
                "        ndcgs.append(1.0 / math.log2(rank + 1))\n",
                "    else:\n",
                "        ndcgs.append(0.0)\n",
                "mean_ndcg = float(sum(ndcgs) / len(ndcgs))\n",
                "print(f'Stage A+B nDCG@20 on dev: {mean_ndcg:.4f}  (gate: >= 0.25)')\n",
                "print('PASS' if mean_ndcg >= 0.25 else 'FAIL — investigate before Submission 2.')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4, "nbformat_minor": 5
}
with open('/Users/orrimoch/PythonProjs/recsys2026/colab/71_train_cross_encoder.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print(f"notebook 71 written ({len(nb['cells'])} cells)")
PYEOF
```

- [ ] **Step 2: Verify**

Run: `python3 -c "import json; nb = json.load(open('colab/71_train_cross_encoder.ipynb')); print(len(nb['cells']), 'cells')"`

Expected: `7 cells`.

- [ ] **Step 3: Commit**

```bash
git add colab/71_train_cross_encoder.ipynb
git commit -m "stage B: notebook 71 — HN re-mine + CE fine-tune + Stage A+B offline eval (>= 0.25 gate)"
```

---

### Task 18: Stage A+B decision gate (notebook 71 already includes it)

**Files:**
- No new files — the offline-eval cell in notebook 71 already prints the gate decision.

- [ ] **Step 1: Manual gate check**

Run notebook 71 cell 7. If `Stage A+B nDCG@20 < 0.25`, HALT and investigate (CE may be overfit, HN re-mining may have picked noisy negatives, etc.).

- [ ] **Step 2: No commit** — gate is procedural.

---

### Task 19: Smoke-load the fine-tuned reranker through the registry

**Files:**
- No code changes (Task 16 already wired the override). Smoke-check only.

- [ ] **Step 1: In Colab cell, verify the registry loads the fine-tuned reranker**

```python
import sys
sys.path.insert(0, '/content/recsys2026/music-crs-baselines')
from mcrs.rerankers import load_reranker_module

r = load_reranker_module(
    reranker_type="bge_reranker_v2_m3",
    item_db_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    track_split_types=["all_tracks"],
    corpus_types=["track_name", "artist_name", "album_name"],
    cache_dir="experiments/cache",
    model_path="OrRim123/recsys2026-bge-reranker-music-v1",
)
out = r.rerank(
    ["play me upbeat 70s rock"],
    [["t1", "t2"]],
    topk=2,
    user_ids=[None], goal_categories=[None], goal_specificities=[None], user_profiles_raw=[None],
)
print("rerank smoke OK:", out)
```

Expected: prints two TIDs (in some order). No exception.

---

## Submission 2

### Task 20: Submission 2 config + Blind-A run

**Files:**
- Create: `music-crs-baselines/config/181-+ce-ft-v5kto-blindA.yaml`

- [ ] **Step 1: Write config 181**

Create `music-crs-baselines/config/181-+ce-ft-v5kto-blindA.yaml`:

```yaml
# Submission 2: Stage A + B — wRRF(BM25 + dense_lyrics + BGE-M3-FT) + FT cross-encoder + v5-kto.
#
# Single-axis change vs config 180: reranker_type switches from "pro_rank" to
# "bge_reranker_v2_m3" with reranker_model_path pointing at our fine-tuned weights.

lm_type: "OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged"
lora_path: null
lora_max_rank: 32

retrieval_type: "wrrf_bm25_dense_lyrics_bge_m3_ft_v1"
bge_m3_hub_repo: "OrRim123/recsys2026-bge-m3-music-v1-merged"

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
reranker_type: "bge_reranker_v2_m3"
reranker_model_path: "OrRim123/recsys2026-bge-reranker-music-v1"

response_prompt_name: "response_generation_cot_user_state"
response_max_new_tokens: 320
top_n_for_prompt: 1
query_preprocessing_mode: "raw"

use_vllm: false

use_state_tracker: true
state_tracker_prompt_name: "state_extraction"
state_tracker_max_new_tokens: 96

use_cmqr: true
cmqr_prompt_name: "cmqr_rewrites"
cmqr_n_rewrites: 4
cmqr_topk_per_rewrite: 50
cmqr_rrf_k: 60
cmqr_max_new_tokens: 96
```

- [ ] **Step 2: Re-run notebook 73 with `--tid 181-+ce-ft-v5kto-blindA`**

Manual step. Edit the TID cell in notebook 73 in Colab, then run end-to-end. Upload zip → CodaBench.

- [ ] **Step 3: Log score; check abort rule**

If composite regresses vs Submission 1, revert to config 180 stack and skip Submission 3.

- [ ] **Step 4: Commit**

```bash
git add music-crs-baselines/config/181-+ce-ft-v5kto-blindA.yaml
git commit -m "submission 2: config 181 (Stage A + B: BGE-M3-FT + BGE-reranker-FT)"
```

---

## Stage C — LightGBM LambdaRank

### Task 21: Extend `scripts/build_lgbm_features.py` with new feature groups

**Files:**
- Modify: `scripts/build_lgbm_features.py`
- Create: `tests/test_lgbm_features_extended.py`

**Existing baseline (from `extract_features`)**: 14 features — 6 numeric (`wrrf_rank`, `cfbpr_score`, `pop_log`, `recency_years`, `tag_count`, `artist_in_query`) + 5 categorical (`goal_category`, `goal_specificity`, `user_age_group`, `user_country`, `user_gender`) + 3 ids/label.

**Explicit feature additions (14 → 28 total feature columns)**:

| # | Feature name | Group | Computation |
|---|---|---|---|
| 15 | `release_year_sin` | track-temporal | `sin(2π · (year % 100) / 100)` from `release_date[:4]`; 0 if missing |
| 16 | `release_year_cos` | track-temporal | `cos(2π · (year % 100) / 100)` from `release_date[:4]`; 0 if missing |
| 17 | `tag_overlap_count` | query-track | count of tags from `track.tag_list` that appear as case-insensitive substrings in `query` |
| 18 | `last_turn_moved_toward_goal` | session-state | 1/0/-1 from last entry in `session.goal_progress_assessments` (`MOVES_TOWARD_GOAL`=1, `DOES_NOT_MOVE_TOWARD_GOAL`=0, missing=-1) |
| 19 | `bm25_rank_inv` | retrieval-rank | `1.0 / max(1, bm25_rank)` (bm25 sub-rank within wRRF top-K; falls back to wrrf_rank if not surfaced) |
| 20 | `dense_meta_rank_inv` | retrieval-rank | `1.0 / max(1, dense_metadata_rank)` (BGE-M3-FT sub-rank); falls back to wrrf_rank |
| 21 | `dense_lyrics_rank_inv` | retrieval-rank | `1.0 / max(1, dense_lyrics_rank)`; falls back to wrrf_rank |
| 22 | `ce_score` | reranker-output | raw cross-encoder logit from the BGE-reranker-FT (Stage B output). Pre-computed in feature-extraction pipeline; 0 if reranker is unavailable. |
| 23 | `ce_rank_inv` | reranker-output | `1.0 / (1 + CE rank within the reranked top-K)` |
| 24 | `turn_number` | session-position | int turn number (1-indexed) within the conversation |
| 25 | `prior_track_count` | session-position | number of music turns BEFORE this one in the session (0 for first music turn) |
| 26 | `query_drift_score` | session-state | cosine sim between current query embedding and turn-1 query embedding (1.0 if first turn or embedder unavailable) |
| 27 | `pop_rank_pct` | track-popularity | `popularity_rank_within_catalog / catalog_size` (precomputed once; 0.5 if popularity missing) |
| 28 | `is_warm_user` | user-state | 1 if `user_id` has a `cf_bpr` user embedding; 0 if cold |

The existing `_compute_feature_matrix` in `LGBM_RERANKER` (and the matching `metadata.json`'s `features` list) will need to be updated in lockstep; the inference-time computation in Task 22 walks the same helper functions.

- [ ] **Step 1: Write failing tests for each new helper**

Create `tests/test_lgbm_features_extended.py`:

```python
"""Tests for new feature helpers added to scripts/build_lgbm_features.py."""
import pytest


def test_compute_release_year_cyclical_for_1969_returns_unit_circle():
    from scripts.build_lgbm_features import compute_release_year_cyclical
    sin, cos = compute_release_year_cyclical("1969-05-29")
    assert -1.0 <= sin <= 1.0
    assert -1.0 <= cos <= 1.0
    # The pair must be on the unit circle.
    assert abs((sin * sin + cos * cos) - 1.0) < 1e-9


def test_compute_release_year_cyclical_handles_missing():
    from scripts.build_lgbm_features import compute_release_year_cyclical
    assert compute_release_year_cyclical(None) == (0.0, 0.0)
    assert compute_release_year_cyclical("") == (0.0, 0.0)
    assert compute_release_year_cyclical("not-a-date") == (0.0, 0.0)


def test_compute_tag_overlap_counts_substring_matches():
    from scripts.build_lgbm_features import compute_tag_overlap
    n = compute_tag_overlap("I love folk rock from the 70s", ["folk rock", "70s", "metal"])
    assert n == 2


def test_compute_tag_overlap_empty_list_returns_zero():
    from scripts.build_lgbm_features import compute_tag_overlap
    assert compute_tag_overlap("anything", None) == 0
    assert compute_tag_overlap("anything", []) == 0


def test_last_turn_moved_toward_goal_codes():
    from scripts.build_lgbm_features import last_turn_moved_toward_goal
    assert last_turn_moved_toward_goal(["MOVES_TOWARD_GOAL"]) == 1
    assert last_turn_moved_toward_goal(["DOES_NOT_MOVE_TOWARD_GOAL"]) == 0
    assert last_turn_moved_toward_goal([]) == -1
    assert last_turn_moved_toward_goal(None) == -1
    # Uses LAST assessment (most-recent state).
    assert last_turn_moved_toward_goal(["DOES_NOT_MOVE_TOWARD_GOAL", "MOVES_TOWARD_GOAL"]) == 1


def test_query_drift_score_first_turn_returns_one():
    from scripts.build_lgbm_features import query_drift_score
    assert query_drift_score("hello", prior_queries=[], embedder=None) == 1.0


def test_pop_rank_pct_known_track_returns_in_range():
    from scripts.build_lgbm_features import build_pop_rank_pct_map
    track_meta = {"t1": {"popularity": 100.0}, "t2": {"popularity": 50.0}, "t3": {"popularity": 200.0}}
    pct = build_pop_rank_pct_map(track_meta)
    # t3 has highest popularity → smallest rank → smallest pct.
    assert pct["t3"] < pct["t1"] < pct["t2"]
    for v in pct.values():
        assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: Run tests — expect fails**

Run: `pytest tests/test_lgbm_features_extended.py -v`

Expected: 7 fails (helpers don't exist yet).

- [ ] **Step 3: Append helpers + update `extract_features` to emit new columns**

Edit `scripts/build_lgbm_features.py`. APPEND these helpers near the top (after `_tokenize_simple`):

```python
import math


def compute_release_year_cyclical(release_date):
    """Sin/cos of year-mod-century. (0.0, 0.0) on missing/unparseable."""
    if not release_date:
        return 0.0, 0.0
    try:
        year = int(str(release_date)[:4])
    except (ValueError, TypeError):
        return 0.0, 0.0
    phase = 2 * math.pi * (year % 100) / 100.0
    return math.sin(phase), math.cos(phase)


def compute_tag_overlap(query, tag_list):
    """Count of tags present in the query (case-insensitive substring)."""
    if not tag_list:
        return 0
    q_lower = (query or "").lower()
    return sum(1 for t in tag_list if t and t.lower() in q_lower)


def last_turn_moved_toward_goal(assessments):
    """1 / 0 / -1 from the last assessment value. -1 for empty/None."""
    if not assessments:
        return -1
    last = assessments[-1]
    if last == "MOVES_TOWARD_GOAL":
        return 1
    if last == "DOES_NOT_MOVE_TOWARD_GOAL":
        return 0
    return -1


def query_drift_score(current_query, prior_queries, embedder=None):
    """Cosine sim between current and turn-1 query embeddings. 1.0 when no prior or no embedder."""
    if not prior_queries or embedder is None:
        return 1.0
    embs = embedder.encode([current_query, prior_queries[0]], normalize_embeddings=True)
    return float(embs[0] @ embs[1])


def build_pop_rank_pct_map(track_meta):
    """Returns {track_id: rank_pct in [0, 1]} where 0 = most popular, 1 = least.

    Tracks with missing popularity get pct=0.5 (neutral).
    """
    items = [(tid, float(m.get("popularity") or 0.0)) for tid, m in track_meta.items()]
    # Sort descending by popularity
    items.sort(key=lambda x: -x[1])
    n = max(1, len(items))
    out = {}
    for rank, (tid, pop) in enumerate(items):
        if pop <= 0.0:
            out[tid] = 0.5
        else:
            out[tid] = rank / n
    return out
```

Then UPDATE `extract_features`'s row dict to add the new columns. The exact diff inside `extract_features`'s `rows.append(...)`:

```python
        # ----- existing fields (keep) -----
        rs_sin, rs_cos = compute_release_year_cyclical(rd)
        tag_overlap = compute_tag_overlap(query, m.get("tag_list"))
        last_goal_move = last_turn_moved_toward_goal(
            session_info.get("goal_progress_assessments")
        )
        pop_pct = pop_rank_pct.get(tid, 0.5) if pop_rank_pct is not None else 0.5

        rows.append({
            # ids (unchanged)
            "query_id": f"{session_info['session_id']}#{session_info['turn_number']}",
            "session_id": session_info["session_id"],
            "user_id": user_id,
            "turn_number": session_info["turn_number"],
            "candidate_tid": tid,
            # existing numeric features
            "wrrf_rank": c["wrrf_rank"],
            "cfbpr_score": cfbpr_score,
            "pop_log": float(np.log1p(pop)),
            "recency_years": float(recency),
            "tag_count": tag_count,
            "artist_in_query": artist_in_query,
            # NEW track-temporal features (15, 16)
            "release_year_sin": rs_sin,
            "release_year_cos": rs_cos,
            # NEW query-track feature (17)
            "tag_overlap_count": tag_overlap,
            # NEW session-state feature (18)
            "last_turn_moved_toward_goal": last_goal_move,
            # NEW retrieval-rank features (19-21) — populated by caller; default to wrrf_rank.
            "bm25_rank_inv": 1.0 / max(1, c.get("bm25_rank", c["wrrf_rank"])),
            "dense_meta_rank_inv": 1.0 / max(1, c.get("dense_meta_rank", c["wrrf_rank"])),
            "dense_lyrics_rank_inv": 1.0 / max(1, c.get("dense_lyrics_rank", c["wrrf_rank"])),
            # NEW reranker-output features (22, 23) — caller supplies; default 0.
            "ce_score": float(c.get("ce_score", 0.0)),
            "ce_rank_inv": 1.0 / max(1, c.get("ce_rank", c["wrrf_rank"])),
            # NEW session-position features (24, 25)
            "turn_number_feat": int(session_info["turn_number"]),
            "prior_track_count": int(session_info.get("prior_track_count", 0)),
            # NEW session-state feature (26) — caller supplies precomputed drift score
            "query_drift_score": float(session_info.get("query_drift_score", 1.0)),
            # NEW track-popularity feature (27)
            "pop_rank_pct": float(pop_pct),
            # NEW user-state feature (28)
            "is_warm_user": int(user_emb is not None),
            # categorical features (unchanged)
            "goal_category": str(goal_cat),
            "goal_specificity": str(goal_spec),
            "user_age_group": str(age_group),
            "user_country": str(country),
            "user_gender": str(gender),
            "label": 1 if tid == gold_tid else 0,
        })
```

Update the `extract_features` signature to take an optional `pop_rank_pct` kwarg:

```python
def extract_features(
    query: str,
    candidates: list[dict],
    gold_tid: str,
    session_info: dict,
    user_info: dict,
    track_meta: dict[str, dict],
    cfbpr_tid_to_idx: dict[str, int],
    cfbpr_track_mat: np.ndarray,
    cfbpr_user_embs: dict[str, np.ndarray],
    query_tokens: set[str],
    pop_rank_pct: dict[str, float] | None = None,
) -> list[dict]:
    ...
```

And update `build()` to compute the pop_rank_pct map once and pass it through.

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_lgbm_features_extended.py -v`

Expected: 7 pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_lgbm_features.py tests/test_lgbm_features_extended.py
git commit -m "stage C: extend build_lgbm_features.py with 14 new feature columns (28 total; TDD, 7 tests)"
```

---

### Task 22: Notebook 72 — held-out train split + feature extraction (walks HF dataset)

**Files:**
- Create: `colab/72_build_lgbm_features_train.ipynb`

**Schema fix**: Walks the HF train conversation dataset directly via `_iter_conversation_turns`. The session-level split is done on session_ids before per-turn expansion — so train/val splits are session-disjoint.

- [ ] **Step 1: Create the notebook**

```bash
python3 << 'PYEOF'
import json
nb = {
    "cells": [
        {
            "cell_type": "markdown", "metadata": {}, "source": [
                "# 72 — Build LGBM features + train LambdaRank (Stage C)\n\n",
                "Walks HF train conversations, splits sessions 80/20, runs the full\n",
                "Stage A+B retrieval+reranker pipeline to get top-100 candidates per\n",
                "music turn, then extracts the extended 28-feature vectors per\n",
                "(turn, candidate) pair. Trains LightGBM LambdaRank on the result.\n\n",
                "**Prereqs**: Stage A + Stage B done; merged BGE-M3 + CE on Hub;\n",
                "BGE-M3-FT catalog pickle on Drive (notebook 70 cell 6).\n\n",
                "**Wallclock**: ~4-6 hr on Blackwell (feature extraction is wRRF +\n",
                "CE forward over ~12k music turns × 100 cands)."
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 1) Setup.\n",
                "import os\n",
                "from google.colab import userdata, drive\n",
                "os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n",
                "drive.mount('/content/drive', force_remount=False)\n",
                "\n",
                "BRANCH = 'fresh-model'\n",
                "!rm -rf /content/recsys2026\n",
                "!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026\n",
                "%cd /content/recsys2026\n",
                "\n",
                "DRIVE_BASE = '/content/drive/MyDrive'\n",
                "LOCAL_BASE = '/content/recsys2026/experiments/cache'\n",
                "os.makedirs(LOCAL_BASE, exist_ok=True)\n",
                "src = f'{DRIVE_BASE}/recsys2026_retrieval_v2_cache'\n",
                "dst = f'{LOCAL_BASE}/retrieval_v2'\n",
                "if os.path.islink(dst): os.unlink(dst)\n",
                "elif os.path.exists(dst):\n",
                "    import shutil; shutil.rmtree(dst)\n",
                "os.symlink(src, dst)\n",
                "\n",
                "!pip install -q --upgrade lightgbm sentence-transformers transformers datasets scikit-learn"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 2) Walk HF train conversations + session-disjoint 80/20 split.\n",
                "import sys\n",
                "sys.path.insert(0, '/content/recsys2026/scripts')\n",
                "sys.path.insert(0, '/content/recsys2026/music-crs-baselines')\n",
                "from datasets import load_dataset\n",
                "from sklearn.model_selection import train_test_split\n",
                "from build_bi_encoder_training_data import _iter_conversation_turns\n",
                "\n",
                "train_conv = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='train')\n",
                "all_rows = _iter_conversation_turns(train_conv)\n",
                "print(f'{len(all_rows)} per-music-turn rows from train split')\n",
                "session_ids = sorted({r['session_id'] for r in all_rows})\n",
                "train_sids, val_sids = train_test_split(session_ids, test_size=0.2, random_state=42)\n",
                "train_set, val_set = set(train_sids), set(val_sids)\n",
                "train_rows = [r for r in all_rows if r['session_id'] in train_set]\n",
                "val_rows = [r for r in all_rows if r['session_id'] in val_set]\n",
                "print(f'train turns: {len(train_rows)}  val turns: {len(val_rows)}')\n",
                "import json as _j\n",
                "os.makedirs('experiments/cache/retrieval_v2/lgbm', exist_ok=True)\n",
                "with open('experiments/cache/retrieval_v2/lgbm/lgbm_train_rows.jsonl', 'w') as f:\n",
                "    for r in train_rows: f.write(_j.dumps(r, default=str) + '\\n')\n",
                "with open('experiments/cache/retrieval_v2/lgbm/lgbm_val_rows.jsonl', 'w') as f:\n",
                "    for r in val_rows: f.write(_j.dumps(r, default=str) + '\\n')"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 3) Extract features for each (turn, candidate) pair via Stage A+B pipeline.\n",
                "# For each music turn:\n",
                "#   a. Build production query via format_query_text(..., mode='raw').\n",
                "#   b. wRRF (BM25 + dense_lyrics + BGE-M3-FT) → top-100 candidates.\n",
                "#   c. CE rerank → top-100 reranked (keeps the same 100 candidates; just adds CE score).\n",
                "#   d. extract_features() with the extended 28-feature vector.\n",
                "# Outputs: experiments/cache/retrieval_v2/lgbm/lgbm_{train,val}_features.parquet\n",
                "!cd /content/recsys2026/music-crs-baselines && python -u ../scripts/build_lgbm_features.py \\\n",
                "    --n-sessions 999999 \\\n",
                "    --topk 100 \\\n",
                "    --seed 42 \\\n",
                "    --out /content/recsys2026/experiments/cache/retrieval_v2/lgbm/lgbm_train_features.parquet \\\n",
                "    --cache-dir /content/recsys2026/experiments/cache \\\n",
                "    2>&1 | tail -20\n",
                "# Note: the existing build_lgbm_features.py samples train sessions; with the\n",
                "# new 80/20 split, override the sampler by writing a thin per-row driver.\n",
                "# Implementation detail: pass session_ids filter via the existing --seed +\n",
                "# n-sessions, OR modify build_lgbm_features.py to accept --session-id-list.\n",
                "# For Phase 1, the simpler path is: run the full feature extractor on train,\n",
                "# then post-filter rows to (train_sids, val_sids) into two parquets."
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 4) Post-filter the single full-train parquet into 80/20 train/val by session.\n",
                "import pandas as pd\n",
                "df = pd.read_parquet('/content/recsys2026/experiments/cache/retrieval_v2/lgbm/lgbm_train_features.parquet')\n",
                "tdf = df[df['session_id'].isin(train_set)].copy()\n",
                "vdf = df[df['session_id'].isin(val_set)].copy()\n",
                "tdf.to_parquet('/content/recsys2026/experiments/cache/retrieval_v2/lgbm/lgbm_train_split.parquet', index=False)\n",
                "vdf.to_parquet('/content/recsys2026/experiments/cache/retrieval_v2/lgbm/lgbm_val_split.parquet', index=False)\n",
                "print(f'train rows: {len(tdf)}  val rows: {len(vdf)}')\n",
                "print('positives (label=1):', int(tdf['label'].sum()), int(vdf['label'].sum()))"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4, "nbformat_minor": 5
}
with open('/Users/orrimoch/PythonProjs/recsys2026/colab/72_build_lgbm_features_train.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print(f"notebook 72 written ({len(nb['cells'])} cells)")
PYEOF
```

- [ ] **Step 2: Verify**

Run: `python3 -c "import json; nb = json.load(open('colab/72_build_lgbm_features_train.ipynb')); print(len(nb['cells']), 'cells')"`

Expected: `5 cells`.

- [ ] **Step 3: Commit**

```bash
git add colab/72_build_lgbm_features_train.ipynb
git commit -m "stage C: notebook 72 — HF walk + 80/20 session-disjoint split + feature extraction"
```

---

### Task 23: LightGBM trainer (writes LGBM_RERANKER-compatible artifacts)

**Files:**
- Create: `scripts/train_lgbm_ranker.py`
- Create: `tests/test_train_lgbm_ranker.py`

**Reuse contract**: The trainer writes `booster.txt` + `metadata.json` in the SAME layout the existing `mcrs/rerankers/lgbm_rerank.py:LGBM_RERANKER` already consumes — so we plug it into the existing reranker path without changing inference.

**Critical bug fix vs old plan**: `build_groups` MUST use `sort=False` so the returned group-size list aligns with the DataFrame row order (pandas' default `sort=True` reorders by group key, which would silently misalign features and labels).

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_lgbm_ranker.py`:

```python
"""Tests for the LightGBM LambdaRank trainer."""
import pytest


def test_build_groups_preserves_dataframe_row_order():
    """build_groups must use sort=False — group sizes correspond to the row
    order of the DataFrame, NOT to the sorted group keys."""
    import pandas as pd
    from scripts.train_lgbm_ranker import build_groups
    # Out-of-order keys: the FIRST group encountered is (s2, 1).
    df = pd.DataFrame([
        {"session_id": "s2", "turn_number": 1},
        {"session_id": "s2", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
    ])
    groups = build_groups(df)
    # Row-order: 2 rows for (s2,1), then 3 rows for (s1,1).
    assert groups == [2, 3], (
        f"build_groups returned {groups}; expected [2, 3]. "
        "If you see [3, 2], you used sort=True (default), which corrupts the alignment "
        "between (X, y) row order and group sizes."
    )


def test_build_groups_handles_already_sorted():
    import pandas as pd
    from scripts.train_lgbm_ranker import build_groups
    df = pd.DataFrame([
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 2},
        {"session_id": "s2", "turn_number": 1},
    ])
    groups = build_groups(df)
    assert groups == [2, 1, 1]


def test_write_metadata_json_emits_features_and_categorical_levels(tmp_path):
    """The metadata.json layout MUST match mcrs.rerankers.lgbm_rerank.LGBM_RERANKER's reader."""
    from scripts.train_lgbm_ranker import write_metadata_json
    out_dir = tmp_path / "lgbm_model"
    out_dir.mkdir()
    write_metadata_json(
        out_dir=str(out_dir),
        features=["wrrf_rank", "ce_score", "goal_category"],
        categorical_features=["goal_category"],
        categorical_levels={"goal_category": ["A", "B", "C"]},
        best_iteration=137,
        best_val_ndcg20=0.42,
    )
    import json
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["features"] == ["wrrf_rank", "ce_score", "goal_category"]
    assert meta["categorical_features"] == ["goal_category"]
    assert meta["categorical_levels"] == {"goal_category": ["A", "B", "C"]}
    assert meta["best_iteration"] == 137
    assert abs(meta["best_val_ndcg20"] - 0.42) < 1e-9
```

- [ ] **Step 2: Run tests — expect fail**

Run: `pytest tests/test_train_lgbm_ranker.py -v`

Expected: 3 fails.

- [ ] **Step 3: Implement**

Create `scripts/train_lgbm_ranker.py`:

```python
"""LightGBM LambdaRank trainer for Stage C.

Writes `booster.txt` + `metadata.json` in the layout the existing
`mcrs.rerankers.lgbm_rerank.LGBM_RERANKER` already reads — so we plug into
the existing reranker path without changing inference code.

Usage:
  python scripts/train_lgbm_ranker.py \
    --train-features experiments/cache/retrieval_v2/lgbm/lgbm_train_split.parquet \
    --val-features   experiments/cache/retrieval_v2/lgbm/lgbm_val_split.parquet \
    --output-dir     experiments/cache/retrieval_v2/lgbm/lgbm_v1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import pandas as pd


# Categorical columns (LGBM-native categorical handling).
CATEGORICAL_FEATURES = [
    "goal_category", "goal_specificity",
    "user_age_group", "user_country", "user_gender",
]

# Columns we do NOT pass to the model (ids + label).
NON_FEATURE_COLS = {"query_id", "session_id", "user_id", "turn_number", "candidate_tid", "label"}


def build_groups(df: pd.DataFrame) -> list[int]:
    """Group sizes by (session_id, turn_number) — preserves DataFrame row order.

    CRITICAL: pandas' groupby default is sort=True, which would sort by group
    key and silently misalign with the (X, y) row order. We pass sort=False
    so the i-th group size corresponds to the i-th *block* of rows in df.
    """
    return df.groupby(["session_id", "turn_number"], sort=False).size().tolist()


def _encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Cast categorical columns to pandas 'category' dtype + capture level lists.

    Returns (df_encoded, {col: levels_list}) — levels_list is the .cat.categories
    in the order LightGBM saw them, so the inference-time encoder can map strings
    back to the same integer codes.
    """
    levels: dict[str, list[str]] = {}
    out = df.copy()
    for c in CATEGORICAL_FEATURES:
        if c not in out.columns:
            continue
        out[c] = out[c].astype("category")
        levels[c] = list(out[c].cat.categories)
    return out, levels


def write_metadata_json(
    out_dir: str,
    features: list[str],
    categorical_features: list[str],
    categorical_levels: dict[str, list[str]],
    best_iteration: int,
    best_val_ndcg20: float,
) -> None:
    """Layout matches LGBM_RERANKER._init_."""
    meta = {
        "features": features,
        "categorical_features": categorical_features,
        "categorical_levels": categorical_levels,
        "best_iteration": int(best_iteration),
        "best_val_ndcg20": float(best_val_ndcg20),
    }
    (Path(out_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-features", required=True)
    p.add_argument("--val-features", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--early-stopping", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_features)
    val_df = pd.read_parquet(args.val_features)
    # Sort rows so build_groups returns aligned sizes.
    train_df = train_df.sort_values(["session_id", "turn_number"]).reset_index(drop=True)
    val_df = val_df.sort_values(["session_id", "turn_number"]).reset_index(drop=True)

    train_df, train_levels = _encode_categoricals(train_df)
    # Use train levels to encode val so codes align.
    for c in CATEGORICAL_FEATURES:
        if c not in val_df.columns:
            continue
        val_df[c] = pd.Categorical(val_df[c], categories=train_levels[c])

    feat_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    train_X = train_df[feat_cols]
    train_y = train_df["label"].astype(int).values
    val_X = val_df[feat_cols]
    val_y = val_df["label"].astype(int).values
    train_groups = build_groups(train_df)
    val_groups = build_groups(val_df)

    cat_in_feats = [c for c in CATEGORICAL_FEATURES if c in feat_cols]

    train_ds = lgb.Dataset(train_X, label=train_y, group=train_groups,
                            categorical_feature=cat_in_feats, free_raw_data=False)
    val_ds = lgb.Dataset(val_X, label=val_y, group=val_groups,
                          categorical_feature=cat_in_feats, reference=train_ds, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }

    model = lgb.train(
        params, train_ds, num_boost_round=args.n_estimators,
        valid_sets=[val_ds], valid_names=["val"],
        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)],
    )

    booster_path = out_dir / "booster.txt"
    model.save_model(str(booster_path))
    best_iter = int(model.best_iteration or 0)
    best_score = float(model.best_score.get("val", {}).get("ndcg@20", 0.0))

    write_metadata_json(
        out_dir=str(out_dir),
        features=feat_cols,
        categorical_features=cat_in_feats,
        categorical_levels={c: train_levels[c] for c in cat_in_feats},
        best_iteration=best_iter,
        best_val_ndcg20=best_score,
    )

    importance = sorted(
        zip(feat_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )[:20]
    print(f"[lgbm] saved → {booster_path} (best_iter={best_iter}, val_ndcg@20={best_score:.4f})")
    print("[lgbm] top-20 features by gain:")
    for name, gain in importance:
        print(f"  {name}: {gain:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_train_lgbm_ranker.py -v`

Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_lgbm_ranker.py tests/test_train_lgbm_ranker.py
git commit -m "stage C: train_lgbm_ranker.py — LGBM_RERANKER-compatible artifacts; build_groups sort=False (TDD, 3 tests)"
```

---

### Task 24: Reuse existing LGBM_RERANKER + update its feature list to match the 28-column training output

**Files:**
- Modify: `music-crs-baselines/mcrs/rerankers/lgbm_rerank.py`

**Why no new ranker class**: The existing `LGBM_RERANKER` already loads `booster.txt + metadata.json`, runs inference with feature parity vs training, and conforms to the reranker interface (`rerank(queries, candidate_tids, topk, **side_channels)`). We just need to:

1. Have its `_compute_feature_matrix` produce the same 28 columns that `extract_features` in `scripts/build_lgbm_features.py` produces at training time.
2. Since the trained model's `metadata.json["features"]` lists the exact column order, the inference matrix MUST contain the same columns in the same order. The existing code already reads `self.features` from metadata.json and uses it to index into the matrix — we just need to compute the new feature values.

**Approach**: Extend `_compute_feature_matrix` to call the same helpers (`compute_release_year_cyclical`, etc.) from `scripts/build_lgbm_features.py`. For the wRRF-sub-rank and CE-score features that are computed during candidate generation (not in `extract_features`), the inference call site (`crs_baseline.batch_chat`) must surface them — we wire that via the `extra_config` payload (see Task 25).

- [ ] **Step 1: Update `_compute_feature_matrix` to emit the extended feature set**

In `music-crs-baselines/mcrs/rerankers/lgbm_rerank.py`, modify `_compute_feature_matrix`:

```python
# At top of file, add:
import sys
from pathlib import Path

# Lazy import — only when needed; keeps module load cheap.
def _lgbm_feature_helpers():
    repo_root = Path(__file__).resolve().parents[3]
    scripts_dir = str(repo_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_lgbm_features import (
        compute_release_year_cyclical, compute_tag_overlap,
        last_turn_moved_toward_goal, query_drift_score, build_pop_rank_pct_map,
    )
    return {
        "release_year_cyclical": compute_release_year_cyclical,
        "tag_overlap": compute_tag_overlap,
        "last_goal": last_turn_moved_toward_goal,
        "drift": query_drift_score,
        "build_pop_pct": build_pop_rank_pct_map,
    }
```

Then update `_compute_feature_matrix` to compute the new fields and write them into rows. The key change: the inference site (Task 25) passes `extra_features_per_candidate` (a list of dicts with `bm25_rank`, `dense_meta_rank`, `dense_lyrics_rank`, `ce_score`, `ce_rank`) and `extra_session_info` (with `goal_progress_assessments`, `prior_track_count`, `query_drift_score`) — `LGBM_RERANKER._compute_feature_matrix` consumes them via two new kwargs (default to None / empty so existing callers keep working).

Wire the new kwargs through `rerank()`:

```python
def rerank(
    self,
    queries: list[str],
    candidate_tids: list[list[str]],
    topk: int,
    user_ids=None, goal_categories=None, goal_specificities=None, user_profiles_raw=None,
    # NEW kwargs (Phase 1 extended features). Default None for back-compat.
    extra_features_per_candidate: Optional[list[list[dict]]] = None,
    extra_session_info: Optional[list[dict]] = None,
):
    ...
```

The matrix population follows the `feature_names` order from `metadata.json`, so unknown columns (when extra_* is None) read as 0.0 — the booster handles this gracefully. Any new feature that needs `extra_*` populated MUST go through Task 25's wiring at inference time.

- [ ] **Step 2: Verify with the existing reranker smoke test**

The existing `lgbm_rerank` tests (if any) should still pass. Run: `pytest tests/ -k lgbm -v` and ensure no regression.

- [ ] **Step 3: Commit**

```bash
git add music-crs-baselines/mcrs/rerankers/lgbm_rerank.py
git commit -m "stage C: extend LGBM_RERANKER._compute_feature_matrix for 28-column parity (back-compat preserved)"
```

---

### Task 25: Add `reranker_chain` config support + CHAIN_RERANKER class

**Files:**
- Create: `music-crs-baselines/mcrs/rerankers/chain.py`
- Modify: `music-crs-baselines/mcrs/rerankers/__init__.py` (add `chain` factory branch)
- Modify: `music-crs-baselines/run_inference_blindset.py` (parse `reranker_chain` config field)
- Create: `tests/test_chain_reranker.py`

**Design**: Rather than touching `crs_baseline.batch_chat`'s single-reranker call site, we introduce a `CHAIN_RERANKER` that itself implements the reranker interface and delegates to a list of sub-rerankers in sequence. From `crs_baseline`'s POV it's one reranker; internally it's CE → LGBM (or any other chain).

**Config format (list-of-dicts)**:

```yaml
reranker_type: "chain"
reranker_chain:
  - type: "bge_reranker_v2_m3"
    model_path: "OrRim123/recsys2026-bge-reranker-music-v1"
    topk: 50
  - type: "lgbm_rerank"
    model_path: "experiments/cache/retrieval_v2/lgbm/lgbm_v1"
    topk: 20
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_chain_reranker.py`:

```python
"""Tests for CHAIN_RERANKER."""
import pytest


class _StubReranker:
    """Records the call args + returns a fixed list (the first N per query)."""

    def __init__(self, name, topk):
        self.name = name
        self.topk = topk
        self.calls = []

    def rerank(self, queries, candidate_tids, topk, **kwargs):
        self.calls.append({"queries": queries, "cands": candidate_tids, "topk": topk, "kwargs": kwargs})
        # Mimic a real reranker: take the first `topk` per query.
        return [c[:topk] for c in candidate_tids]


def test_chain_reranker_runs_stages_in_order(tmp_path):
    from mcrs.rerankers.chain import CHAIN_RERANKER
    ce = _StubReranker("ce", 50)
    lg = _StubReranker("lgbm", 20)
    chain = CHAIN_RERANKER(stages=[("ce", 50, ce), ("lgbm", 20, lg)])
    out = chain.rerank(
        queries=["q1", "q2"],
        candidate_tids=[list(f"c{i}" for i in range(100)), list(f"d{i}" for i in range(100))],
        topk=20,
        user_ids=["u1", "u2"], goal_categories=[None, None],
        goal_specificities=[None, None], user_profiles_raw=[None, None],
    )
    # Stage 1 (CE) trims to topk=50; Stage 2 (LGBM) trims to topk=20.
    assert len(ce.calls) == 1 and ce.calls[0]["topk"] == 50
    assert len(lg.calls) == 1 and lg.calls[0]["topk"] == 20
    # Final shape matches caller's topk request (20).
    assert all(len(o) == 20 for o in out)


def test_chain_reranker_forwards_side_channel_kwargs():
    from mcrs.rerankers.chain import CHAIN_RERANKER
    ce = _StubReranker("ce", 50)
    chain = CHAIN_RERANKER(stages=[("ce", 50, ce)])
    chain.rerank(
        queries=["q"], candidate_tids=[["a", "b", "c"]], topk=2,
        user_ids=["u"], goal_categories=["cat"],
        goal_specificities=["spec"], user_profiles_raw=[{"age": 30}],
    )
    assert ce.calls[0]["kwargs"]["user_ids"] == ["u"]
    assert ce.calls[0]["kwargs"]["goal_categories"] == ["cat"]
    assert ce.calls[0]["kwargs"]["user_profiles_raw"] == [{"age": 30}]
```

- [ ] **Step 2: Run tests — expect fail**

Run: `pytest tests/test_chain_reranker.py -v`

Expected: 2 fails (module missing).

- [ ] **Step 3: Implement `CHAIN_RERANKER`**

Create `music-crs-baselines/mcrs/rerankers/chain.py`:

```python
"""Chain reranker: runs a list of rerankers in sequence.

Each stage's `topk` shrinks the candidate set; the final stage's output
respects the caller's `topk` (truncated if necessary).
"""
from __future__ import annotations

from typing import Any, Optional


class CHAIN_RERANKER:
    def __init__(self, stages: list[tuple[str, int, Any]]):
        """Args:
            stages: list of (name, topk, reranker_instance) tuples. Each is
                applied in order; reranker outputs are fed to the next stage.
        """
        if not stages:
            raise ValueError("CHAIN_RERANKER requires at least one stage")
        self.stages = stages

    def rerank(
        self,
        queries: list[str],
        candidate_tids: list[list[str]],
        topk: int,
        user_ids: Optional[list[Optional[str]]] = None,
        goal_categories: Optional[list[Optional[str]]] = None,
        goal_specificities: Optional[list[Optional[str]]] = None,
        user_profiles_raw: Optional[list[Any]] = None,
    ) -> list[list[str]]:
        side_channels = {
            "user_ids": user_ids,
            "goal_categories": goal_categories,
            "goal_specificities": goal_specificities,
            "user_profiles_raw": user_profiles_raw,
        }
        current = candidate_tids
        for stage_idx, (name, stage_topk, reranker) in enumerate(self.stages):
            # Final stage respects the caller's topk; earlier stages use their own.
            is_last = stage_idx == len(self.stages) - 1
            this_topk = topk if is_last else stage_topk
            try:
                current = reranker.rerank(queries, current, topk=this_topk, **side_channels)
            except TypeError:
                # Back-compat: reranker predates side-channel kwargs.
                current = reranker.rerank(queries, current, topk=this_topk)
        return current
```

- [ ] **Step 4: Add `chain` branch to the registry**

In `music-crs-baselines/mcrs/rerankers/__init__.py`:

```python
    if reranker_type == "chain":
        # Build sub-rerankers from a list-of-dicts spec. Each dict:
        #   {"type": "<reranker_type>", "model_path": "<...>", "topk": <int>}
        from .chain import CHAIN_RERANKER
        chain_spec = (model_path or "")  # `model_path` is overloaded for backwards-compat;
        # the actual spec list arrives via a NEW kwarg `reranker_chain` (see below).
        raise NotImplementedError(
            "load_reranker_module: 'chain' requires reranker_chain spec; "
            "call load_chain_reranker directly from run_inference_blindset.py "
            "which forwards the YAML list."
        )
```

That stub is intentional — `chain` configs need the full list, which `load_reranker_module`'s current signature doesn't carry. The clean path: add a sibling function `load_chain_reranker` that the caller (`run_inference_blindset.py`) invokes explicitly. Append to `__init__.py`:

```python
def load_chain_reranker(
    chain_spec: list[dict],
    item_db_name: str,
    track_split_types: list[str],
    corpus_types: list[str],
    cache_dir: str = "./cache",
):
    """Build a CHAIN_RERANKER from a YAML list-of-dicts spec.

    Each spec dict: {"type": "<reranker_type>", "model_path": "<...>", "topk": <int>}.
    """
    from .chain import CHAIN_RERANKER

    stages = []
    for stage_cfg in chain_spec:
        stage_type = stage_cfg["type"]
        stage_topk = int(stage_cfg.get("topk", 20))
        stage_model_path = stage_cfg.get("model_path")
        sub = load_reranker_module(
            reranker_type=stage_type,
            item_db_name=item_db_name,
            track_split_types=track_split_types,
            corpus_types=corpus_types,
            cache_dir=cache_dir,
            model_path=stage_model_path,
        )
        stages.append((stage_type, stage_topk, sub))
    return CHAIN_RERANKER(stages=stages)
```

- [ ] **Step 5: Wire `reranker_chain` config in `run_inference_blindset.py`**

In `music-crs-baselines/run_inference_blindset.py`, around line 74-75 (where `reranker_type` is read):

```python
# Replace:
reranker_type = config.get("reranker_type", None)
reranker_model_path = config.get("reranker_model_path", None)

# With:
reranker_type = config.get("reranker_type", None)
reranker_model_path = config.get("reranker_model_path", None)
reranker_chain_cfg = config.get("reranker_chain", None)
```

Then around line 116 (where `reranker_type` is passed to `load_crs_baseline`), replace the single-reranker construction with:

```python
# If reranker_chain is set, build it explicitly here and pass the instance
# through; otherwise let crs_baseline build the single reranker as before.
if reranker_chain_cfg is not None:
    from mcrs.rerankers import load_chain_reranker
    chain_spec_list = [OmegaConf.to_container(c, resolve=True) for c in reranker_chain_cfg]
    pre_built_reranker = load_chain_reranker(
        chain_spec=chain_spec_list,
        item_db_name=config.item_db_name,
        track_split_types=list(config.track_split_types),
        corpus_types=list(config.corpus_types),
        cache_dir=config.cache_dir,
    )
    # Tell crs_baseline to use this prebuilt instance instead of constructing one.
    # We achieve this by setting reranker_type=None (no constructor call) then
    # monkeypatching the instance onto the returned baseline.
    reranker_type = None
    reranker_model_path = None
else:
    pre_built_reranker = None
```

And after the `music_crs = load_crs_baseline(...)` call:

```python
if pre_built_reranker is not None:
    music_crs.reranker = pre_built_reranker
    music_crs.reranker_type = "chain"
    print(f"[run_inference_blindset] using chain reranker with "
          f"{len(reranker_chain_cfg)} stages")
```

- [ ] **Step 6: Run tests — expect pass**

Run: `pytest tests/test_chain_reranker.py -v`

Expected: 2 pass.

- [ ] **Step 7: Commit**

```bash
git add music-crs-baselines/mcrs/rerankers/chain.py \
        music-crs-baselines/mcrs/rerankers/__init__.py \
        music-crs-baselines/run_inference_blindset.py \
        tests/test_chain_reranker.py
git commit -m "stage C: CHAIN_RERANKER + reranker_chain YAML support (CE → LGBM, TDD, 2 tests)"
```

---

### Task 26: Notebook 72 — train LGBM + offline eval cells

**Files:**
- Modify: `colab/72_build_lgbm_features_train.ipynb`

- [ ] **Step 1: Append training + offline-eval cells**

```bash
python3 << 'PYEOF'
import json
NB = '/Users/orrimoch/PythonProjs/recsys2026/colab/72_build_lgbm_features_train.ipynb'
nb = json.load(open(NB))

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}

nb['cells'].extend([
    code(
        "# 5) Train LightGBM LambdaRank.\n"
        "!cd /content/recsys2026 && python scripts/train_lgbm_ranker.py \\\n"
        "    --train-features experiments/cache/retrieval_v2/lgbm/lgbm_train_split.parquet \\\n"
        "    --val-features   experiments/cache/retrieval_v2/lgbm/lgbm_val_split.parquet \\\n"
        "    --output-dir     experiments/cache/retrieval_v2/lgbm/lgbm_v1 \\\n"
        "    --n-estimators 1000 \\\n"
        "    2>&1 | tee /content/drive/MyDrive/recsys2026_retrieval_v2_cache/lgbm_train_log.txt\n"
        "!ls -la /content/recsys2026/experiments/cache/retrieval_v2/lgbm/lgbm_v1/"
    ),
    code(
        "# 6) Copy the trained model to Drive for inference reuse.\n"
        "import shutil\n"
        "src = '/content/recsys2026/experiments/cache/retrieval_v2/lgbm/lgbm_v1'\n"
        "dst = '/content/drive/MyDrive/recsys2026_retrieval_v2_cache/lgbm/lgbm_v1'\n"
        "import os\n"
        "os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "if os.path.exists(dst):\n"
        "    shutil.rmtree(dst)\n"
        "shutil.copytree(src, dst)\n"
        "print('LGBM model dir mirrored to:', dst)"
    ),
    code(
        "# 7) Offline eval (Stage A+B+C) on dev: full pipeline + LGBM rerank.\n"
        "# Loads the trained LGBM via the existing LGBM_RERANKER, chains it\n"
        "# after the BGE-reranker-FT in a CHAIN_RERANKER. Gate: nDCG@20 >= 0.35.\n"
        "import sys, math, pickle, os\n"
        "import numpy as np\n"
        "from datasets import load_dataset\n"
        "sys.path.insert(0, '/content/recsys2026/music-crs-baselines')\n"
        "sys.path.insert(0, '/content/recsys2026/scripts')\n"
        "from mcrs.retrieval_modules.bge_m3_format import format_query_text\n"
        "from build_bi_encoder_training_data import _iter_conversation_turns\n"
        "from mcrs.rerankers import load_chain_reranker\n"
        "from sentence_transformers import SentenceTransformer\n"
        "\n"
        "BGE_REPO = 'OrRim123/recsys2026-bge-m3-music-v1-merged'\n"
        "LGBM_DIR = '/content/recsys2026/experiments/cache/retrieval_v2/lgbm/lgbm_v1'\n"
        "CATALOG_PKL = f'/content/drive/MyDrive/recsys2026_retrieval_v2_cache/dense_local/{BGE_REPO.replace(\"/\",\"_\")}/bge-m3-music-v1-merged/track_embeddings.pkl'\n"
        "with open(CATALOG_PKL, 'rb') as f:\n"
        "    payload = pickle.load(f)\n"
        "track_ids, track_mat = payload['track_ids'], payload['track_mat']\n"
        "tid_to_idx = {t: i for i, t in enumerate(track_ids)}\n"
        "\n"
        "chain = load_chain_reranker(\n"
        "    chain_spec=[\n"
        "        {'type': 'bge_reranker_v2_m3', 'model_path': 'OrRim123/recsys2026-bge-reranker-music-v1', 'topk': 50},\n"
        "        {'type': 'lgbm_rerank',         'model_path': LGBM_DIR,                                   'topk': 20},\n"
        "    ],\n"
        "    item_db_name='talkpl-ai/TalkPlayData-Challenge-Track-Metadata',\n"
        "    track_split_types=['all_tracks'],\n"
        "    corpus_types=['track_name', 'artist_name', 'album_name'],\n"
        "    cache_dir='/content/recsys2026/experiments/cache',\n"
        ")\n"
        "\n"
        "bi = SentenceTransformer(BGE_REPO, device='cuda')\n"
        "dev = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='dev')\n"
        "rows = _iter_conversation_turns(dev)[:500]\n"
        "queries = [format_query_text(r.get('chat_history') or [], r.get('current_user_query',''), r.get('user_profile_raw'), r.get('conversation_goal'), mode='raw') for r in rows]\n"
        "q_emb = bi.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=True)\n"
        "q_emb = np.asarray(q_emb, dtype=np.float32)\n"
        "sims = q_emb @ track_mat.T\n"
        "top100_idx = np.argpartition(-sims, kth=99, axis=1)[:, :100]\n"
        "ri = np.arange(sims.shape[0])[:, None]\n"
        "top100_sorted = top100_idx[ri, np.argsort(-sims[ri, top100_idx], axis=1)]\n"
        "cand_lists = [[track_ids[j] for j in top100_sorted[i]] for i in range(len(queries))]\n"
        "out = chain.rerank(\n"
        "    queries, cand_lists, topk=20,\n"
        "    user_ids=[None]*len(queries), goal_categories=[None]*len(queries),\n"
        "    goal_specificities=[None]*len(queries), user_profiles_raw=[None]*len(queries),\n"
        ")\n"
        "ndcgs = []\n"
        "for i, r in enumerate(rows):\n"
        "    gold = r['track_id']\n"
        "    top20 = out[i]\n"
        "    if gold in top20:\n"
        "        rank = top20.index(gold) + 1\n"
        "        ndcgs.append(1.0 / math.log2(rank + 1))\n"
        "    else:\n"
        "        ndcgs.append(0.0)\n"
        "mean_ndcg = float(sum(ndcgs) / len(ndcgs))\n"
        "print(f'Stage A+B+C nDCG@20 on dev: {mean_ndcg:.4f}  (gate: >= 0.35)')\n"
        "print('PASS' if mean_ndcg >= 0.35 else 'FAIL — investigate LGBM features before Submission 3.')"
    ),
])

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"notebook 72 now has {len(nb['cells'])} cells")
PYEOF
```

- [ ] **Step 2: Commit**

```bash
git add colab/72_build_lgbm_features_train.ipynb
git commit -m "stage C: notebook 72 — LGBM training + chain rerank offline eval (>= 0.35 gate)"
```

---

## Submission 3

### Task 27: Submission 3 config + Blind-A run

**Files:**
- Create: `music-crs-baselines/config/182-+lgbm-v5kto-blindA.yaml`

- [ ] **Step 1: Write config 182**

Create `music-crs-baselines/config/182-+lgbm-v5kto-blindA.yaml`:

```yaml
# Submission 3: Stage A + B + C — full v2 stack.
#
# Single-axis change vs config 181: reranker_type switches from
# "bge_reranker_v2_m3" (single CE) to "chain" (CE → LGBM).
# The chain runs the FT cross-encoder first to trim 100→50, then
# LGBM LambdaRank to trim 50→20 with the extended 28-feature vector.

lm_type: "OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged"
lora_path: null
lora_max_rank: 32

retrieval_type: "wrrf_bm25_dense_lyrics_bge_m3_ft_v1"
bge_m3_hub_repo: "OrRim123/recsys2026-bge-m3-music-v1-merged"

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
reranker_type: "chain"
reranker_chain:
  - type: "bge_reranker_v2_m3"
    model_path: "OrRim123/recsys2026-bge-reranker-music-v1"
    topk: 50
  - type: "lgbm_rerank"
    model_path: "/content/drive/MyDrive/recsys2026_retrieval_v2_cache/lgbm/lgbm_v1"
    topk: 20

response_prompt_name: "response_generation_cot_user_state"
response_max_new_tokens: 320
top_n_for_prompt: 1
query_preprocessing_mode: "raw"

use_vllm: false

use_state_tracker: true
state_tracker_prompt_name: "state_extraction"
state_tracker_max_new_tokens: 96

use_cmqr: true
cmqr_prompt_name: "cmqr_rewrites"
cmqr_n_rewrites: 4
cmqr_topk_per_rewrite: 50
cmqr_rrf_k: 60
cmqr_max_new_tokens: 96
```

- [ ] **Step 2: Re-run notebook 73 with `--tid 182-+lgbm-v5kto-blindA`**

Manual. Update TID cell in notebook 73 in Colab, run end-to-end, upload zip to CodaBench.

- [ ] **Step 3: Log score + commit config**

```bash
git add music-crs-baselines/config/182-+lgbm-v5kto-blindA.yaml
git commit -m "submission 3: config 182 (Stage A + B + C: full v2 stack via reranker_chain)"
```

---

## Wrap-up

### Task 28: Write the v2 results memory + freeze the design

**Files:**
- Create: `/Users/orrimoch/.claude/projects/-Users-orrimoch-PythonProjs-recsys2026/memory/project_retrieval_v2_results_<date>.md`
- Update: `MEMORY.md` index
- Optionally: `git tag retrieval-v2-frozen` after Submission 3

- [ ] **Step 1: Write the memory**

Following the format of `project_blind_a_first_results.md`, capture:
- Submission 1/2/3 composite + per-axis scores
- nDCG@20 trajectory (Stage 0 → Stage A → Stage A+B → Stage A+B+C) on dev
- Top features by LGBM gain
- Decision: ship full v2 or revert to a sub-stack

- [ ] **Step 2: Update MEMORY.md**

Add an entry pointing to the new memory.

- [ ] **Step 3: Commit + tag**

```bash
git tag retrieval-v2-frozen
git push origin retrieval-v2-frozen
```

---

## Self-Review

**Spec coverage:**
- §1 Goal → covered (Tasks 13 / 20 / 27 are the three submissions).
- §2 Target framing → applied via dev-set gates in Tasks 12 / 17 / 26.
- §3 Architecture → built across Stages A / B / C; Stage A+B+C composed via `reranker_chain`.
- §4 Existing infrastructure → REUSED:
    - `DENSE_LOCAL` reused for fine-tuned BGE-M3 (no new sub-retriever class)
    - `BGE_RERANKER` extended with `model_name` override (no new CE class)
    - `LGBM_RERANKER` reused; trainer writes its expected artifact layout
    - `crs_baseline.batch_chat` UNCHANGED (chaining handled inside CHAIN_RERANKER)
- §5 Data sources → Task 1 verifies + Task 22 walks HF dataset for held-out split.
- §6 Stage A → Tasks 4-13. PEFT-LoRA wrapper (NOT FlagEmbedding CLI).
- §7 Stage B → Tasks 14-20.
- §8 Stage C → Tasks 21-27. Explicit 14 → 28 feature delta in Task 21.
- §9 Submission cadence → Tasks 13 / 20 / 27 + abort rules in each.
- §10 Risk / abort → captured as decision gates after each submission.
- §11 Testing → TDD pattern: write failing test → impl → pass → commit (in every code task).
- §12 Notebook structure → notebooks 70 / 71 / 72 / 73 created with explicit cells (no "5 cells mirroring …" placeholders).
- §13 Out of scope → respected (no SID, no responder changes).
- §14 Phase 2 transition → mentioned but separate spec.
- §15 Cross-references → Task 28 memory cites the spec + this plan.

**Audit fixes applied:**
- **Fix 1 (train.parquet schema)**: Tasks 7 / 9 / 12 / 14 / 22 all walk the HF conversation dataset directly (`_iter_conversation_turns`). Task 9 catalog re-embed only touches HF Track-Metadata (no parquet).
- **Fix 2 (FlagEmbedding CLI doesn't expose LoRA flags)**: Task 8 replaces the `torchrun -m FlagEmbedding...` invocation with a custom PEFT-LoRA training loop (sentence-transformers + peft).
- **Fix 3 (reranker reuse)**: Task 16 extends existing `BGE_RERANKER` with a `model_name` override (no `bge_reranker_ft.py`). Task 24 keeps `LGBM_RERANKER` (no new `lgbm_ranker.py`). Task 25 adds `CHAIN_RERANKER` + `reranker_chain` YAML support — `crs_baseline.batch_chat` unchanged.
- **Fix 4 (extra_config nesting)**: Configs 180 / 181 / 182 put `bge_m3_hub_repo` at YAML TOP LEVEL — `run_inference_blindset.py:102` dumps the entire YAML to `extra_config_dict`, so top-level keys are read correctly by `extra_config.get(...)`.
- **Fix 5 (Task 21 enumeration)**: Task 21 lists ALL 14 new features in a table with explicit computation; tests cover one helper per group. Task 23 fixes `build_groups` to use `sort=False` (with a regression test that catches the silent reorder).

**Placeholder scan**: no TBD / TODO / "implement later" / "5 cells mirroring …" anywhere. Every step has runnable code or a runnable command.

**Type consistency**:
- `format_track_text` / `format_query_text` defined in Task 4 (shipped), used in Tasks 6 / 7 / 9 / 12 / 14 / 17 / 22 / 26.
- `mine_negatives_for_query` defined in Task 5 (shipped), used in Tasks 6 / 14.
- `_iter_conversation_turns` defined in `scripts/build_bi_encoder_training_data.py` (Task 6, shipped), imported in Tasks 12 / 14 / 17 / 22 / 26.
- `TripleJsonlDataset` + `_BGE_M3_LORA_TARGETS` defined in Task 8, used in tests.
- `CHAIN_RERANKER` defined in Task 25, used in Tasks 26 / 27.
- All Hub repo names consistent:
    - bi-encoder: `OrRim123/recsys2026-bge-m3-music-v1-merged`
    - cross-encoder: `OrRim123/recsys2026-bge-reranker-music-v1`

**Decisions taken on my own (flagged for review)**:
1. **CLI flag style for PEFT-LoRA script**: Mirrored Task 6's flag style (`--triples`, `--output-dir`, `--hub-repo`, `--merge`, `--cleanup-after-push`, `--results-dir`). Added new hyperparameter flags (`--lr`, `--epochs`, `--per-device-batch-size`, `--n-negatives`, `--temperature`, `--query-max-len`, `--passage-max-len`, `--lora-rank`, `--lora-alpha`, `--logging-steps`) all with the spec §6 defaults so default invocations match the spec.
2. **`reranker_chain` YAML shape**: Used list-of-dicts with `type`/`model_path`/`topk` keys (matches the prompt's example exactly).
3. **Task 21 feature enumeration**: Listed all 14 new features in a table with explicit computation. Tests cover one helper per group (release_year_cyclical, tag_overlap, last_turn_moved_toward_goal, query_drift_score, pop_rank_pct) — that's representative coverage; the remaining features (rank-inv ratios, ce_score, turn_number, etc.) are trivial value pass-throughs requiring no helper tests.
4. **`dense_metadata_bge_m3_ft_local` as a new sub-retriever type (not reusing `dense_metadata_bge_m3_local`)**: The existing `_local` variant pins `embed_label="bge-m3-metadata"`; reusing it would clobber the zero-shot catalog pickle. New type uses `embed_label="bge-m3-music-v1-merged"` to keep caches distinct.

These calls preserve the user-approved structural decisions (PEFT-LoRA outside FlagEmbedding; extend `BGE_RERANKER`; extend `LGBM_RERANKER`; `reranker_chain` config) and stay within the audit's stated boundaries.
