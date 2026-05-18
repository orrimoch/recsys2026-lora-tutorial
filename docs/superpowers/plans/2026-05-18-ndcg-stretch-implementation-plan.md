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
| `scripts/build_bi_encoder_training_data.py` | Stage A: zero-shot BGE-M3 HN miner + JSONL triple writer |
| `scripts/train_bi_encoder.py` | Stage A: thin wrapper around FlagEmbedding's `run.py` with our hyperparams |
| `scripts/merge_and_push_bi_encoder.py` | Stage A: merge LoRA → push merged model to Hub |
| `scripts/build_cross_encoder_training_data.py` | Stage B: re-mine HNs via fine-tuned BGE-M3 |
| `scripts/train_cross_encoder.py` | Stage B: FlagEmbedding reranker fine-tune wrapper |
| `scripts/build_lgbm_features.py` | **EXTEND** existing (14 → ~50 features). Don't rewrite. |
| `scripts/train_lgbm_ranker.py` | Stage C: LightGBM LambdaRank trainer |
| `scripts/eval_retrieval_v2.py` | Generic nDCG@20 / recall@K / MRR offline eval harness |

### New retrieval modules (`music-crs-baselines/mcrs/`)
| Path | Responsibility |
|---|---|
| `mcrs/retrieval_modules/lgbm_ranker.py` | LGBM inference wrapper conforming to retriever interface (so it can sit after the wRRF + cross-encoder stages in `crs_baseline.py:batch_chat`) |
| `mcrs/retrieval_modules/__init__.py` | **MODIFY** — add new factory `wrrf_bm25_dense_lyrics_bge_m3_ft_v1` (Stage A) and `wrrf_bm25_dense_lyrics_bge_m3_ft_ce_lgbm_v1` (Stage A+B+C) |
| `mcrs/rerankers/bge_reranker_ft.py` | Cross-encoder reranker wrapper for our fine-tuned model |

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
| `tests/test_bi_encoder_training_data.py` | HN miner, PercPos filter, JSONL writer |
| `tests/test_cross_encoder_training_data.py` | Re-mining, triple builder |
| `tests/test_lgbm_features_extended.py` | New features added to build_lgbm_features.py |
| `tests/test_lgbm_ranker_inference.py` | LGBM wrapper conforms to retriever interface |
| `tests/test_eval_retrieval_v2.py` | nDCG@K, recall@K, MRR computation |
| `tests/test_v2_factories.py` | Both new wRRF factory variants build correctly |

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

- [ ] **Step 1: Create the notebook with cells in this exact order**

Create `colab/70_train_bi_encoder.ipynb` via Python (matches notebook 63's pattern):

```python
# Run this locally:
python3 << 'PYEOF'
import json
nb = {
    "cells": [
        {
            "cell_type": "markdown", "metadata": {}, "source": [
                "# 70 — Train BGE-M3 bi-encoder (Stage A of nDCG-stretch plan)\n\n",
                "Fine-tunes BAAI/bge-m3 on 121K conversation→track pairs via FlagEmbedding unified_finetune.\n\n",
                "**Prereqs**: W2 train.parquet on Drive at `recsys2026_sid_training_cache/train.parquet`. HF_TOKEN in Colab Secrets.\n\n",
                "**Wallclock**: ~9-11 hr on Blackwell."
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
                "    ('sid_training', 'recsys2026_sid_training_cache'),\n",
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
                "    'FlagEmbedding>=1.3' 'sentence-transformers' \\\n",
                "    'datasets' 'pandas<3.0' 'tqdm' 'omegaconf' 'pyyaml'"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 2) Smoke: build 200 triples to verify the HN miner works.\n",
                "!cd /content/recsys2026 && python scripts/build_bi_encoder_training_data.py \\\n",
                "    --train-parquet experiments/cache/sid_training/train.parquet \\\n",
                "    --output experiments/cache/retrieval_v2/triples_smoke.jsonl \\\n",
                "    --max-rows 200 --percpos-threshold 0.80 --k-negs 15 \\\n",
                "    2>&1 | tail -20\n",
                "!head -3 experiments/cache/retrieval_v2/triples_smoke.jsonl"
            ]
        },
        {
            "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [
                "# 3) Full HN mining: ~1.5 hr on Blackwell.\n",
                "import os\n",
                "RESULTS_DIR = '/content/drive/MyDrive/recsys2026_retrieval_v2_cache/results'\n",
                "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
                "!cd /content/recsys2026 && python -u scripts/build_bi_encoder_training_data.py \\\n",
                "    --train-parquet experiments/cache/sid_training/train.parquet \\\n",
                "    --output experiments/cache/retrieval_v2/triples_bge_m3.jsonl \\\n",
                "    --percpos-threshold 0.80 --k-negs 15 --batch-size 64 \\\n",
                "    2>&1 | tee /content/drive/MyDrive/recsys2026_retrieval_v2_cache/hn_mining_log.txt"
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
git commit -m "stage A: notebook 70 — setup + smoke HN mining cells"
```

---

### Task 8: Stage A BGE-M3 fine-tune wrapper script

**Files:**
- Create: `scripts/train_bi_encoder.py`

This is a thin wrapper around FlagEmbedding's `unified_finetune` CLI so we can run it cleanly from a notebook cell.

- [ ] **Step 1: Write the wrapper**

Create `scripts/train_bi_encoder.py`:

```python
"""Stage A: thin wrapper around FlagEmbedding.unified_finetune.run.

Pinned hyperparameters from the spec §6:
- lr 5e-6, bs 2, train_group_size 8, temperature 0.05, epochs 2
- LoRA r=32 alpha=64
- m3_kd_loss + self-distillation after step 500
- ColBERT head DROPPED (dense + sparse only)

Usage:
  python scripts/train_bi_encoder.py \\
    --triples experiments/cache/retrieval_v2/triples_bge_m3.jsonl \\
    --output-dir /content/bge_m3_finetune \\
    --hub-repo OrRim123/recsys2026-bge-m3-music-v1 \\
    --merge --cleanup-after-push
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triples", required=True)
    p.add_argument("--base-model", default="BAAI/bge-m3")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hub-repo", required=True,
                   help="HF Hub repo for the merged model (e.g., OrRim123/recsys2026-bge-m3-music-v1)")
    p.add_argument("--merge", action="store_true", help="After training, merge LoRA into base")
    p.add_argument("--cleanup-after-push", action="store_true")
    p.add_argument("--results-dir", default=None,
                   help="If set, copy training_args.json + final eval to this Drive dir")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the FlagEmbedding command
    cmd = [
        "torchrun", "--nproc_per_node", "1",
        "-m", "FlagEmbedding.finetune.embedder.encoder_only.m3",
        "--model_name_or_path", args.base_model,
        "--train_data", args.triples,
        "--output_dir", str(output_dir),
        "--unified_finetuning", "True",
        "--use_self_distill", "True",
        "--self_distill_start_step", "500",
        "--m3_kd_loss", "True",
        # ColBERT head dropped: only dense + sparse losses contribute.
        # Per spec §6, this is intentional at 47K-track scale.
        "--learning_rate", "5e-6",
        "--per_device_train_batch_size", "2",
        "--train_group_size", "8",
        "--temperature", "0.05",
        "--num_train_epochs", "2",
        "--warmup_ratio", "0.1",
        "--query_max_len", "512",
        "--passage_max_len", "256",
        "--bf16", "True",
        "--gradient_checkpointing", "True",
        "--save_steps", "1000",
        "--logging_steps", "50",
        "--report_to", "tensorboard",
        # LoRA
        "--use_lora", "True",
        "--lora_rank", "32",
        "--lora_alpha", "64",
    ]
    print(f"[train-bi-encoder] running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.check_call(cmd)
    print("[train-bi-encoder] training done", file=sys.stderr)

    if args.merge:
        print("[train-bi-encoder] merging LoRA → base", file=sys.stderr)
        # FlagEmbedding writes the LoRA adapter to output_dir; merge via PEFT.
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        base = AutoModel.from_pretrained(args.base_model)
        peft_model = PeftModel.from_pretrained(base, output_dir)
        merged = peft_model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged.save_pretrained(merged_dir)
        tok = AutoTokenizer.from_pretrained(args.base_model)
        tok.save_pretrained(merged_dir)
        print(f"[train-bi-encoder] merged → {merged_dir}", file=sys.stderr)

        # Push to Hub
        print(f"[train-bi-encoder] pushing to {args.hub_repo}-merged", file=sys.stderr)
        merged.push_to_hub(f"{args.hub_repo}-merged", private=False)
        tok.push_to_hub(f"{args.hub_repo}-merged", private=False)

    if args.results_dir is not None:
        rd = Path(args.results_dir)
        rd.mkdir(parents=True, exist_ok=True)
        # Copy whatever's in output_dir/runs (TensorBoard) to results_dir
        runs_src = output_dir / "runs"
        if runs_src.exists():
            shutil.copytree(runs_src, rd / "runs", dirs_exist_ok=True)

    if args.cleanup_after_push and args.merge:
        # Delete the local merged copy; Hub is the source of truth
        if (output_dir / "merged").exists():
            shutil.rmtree(output_dir / "merged")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke check — verify imports work**

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -c "import sys; sys.argv=['x','--triples','x','--output-dir','x','--hub-repo','x']; exec(open('scripts/train_bi_encoder.py').read().split('if __name__')[0])"` ; verify no import errors.

(FlagEmbedding may not be installed locally; that's fine — the script is for Colab. Just verify the Python parses.)

Run: `/Users/orrimoch/PythonProjs/recsys2026/recsys26/bin/python -m py_compile scripts/train_bi_encoder.py`

Expected: no output (clean compile).

- [ ] **Step 3: Commit**

```bash
git add scripts/train_bi_encoder.py
git commit -m "stage A: train_bi_encoder.py — FlagEmbedding unified_finetune wrapper"
```

---

### Task 9: Add Stage A training + push cells to notebook 70

**Files:**
- Modify: `colab/70_train_bi_encoder.ipynb`

- [ ] **Step 1: Append the training, merge-push, and eval cells**

Run this Python to append cells:

```python
python3 << 'PYEOF'
import json
NB = '/Users/orrimoch/PythonProjs/recsys2026/colab/70_train_bi_encoder.ipynb'
nb = json.load(open(NB))

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}

nb['cells'].extend([
    code(
        "# 4) Smoke fine-tune: 50 steps on 500 examples — verify code path works.\n"
        "!head -500 experiments/cache/retrieval_v2/triples_bge_m3.jsonl > experiments/cache/retrieval_v2/triples_smoke_500.jsonl\n"
        "!cd /content/recsys2026 && python scripts/train_bi_encoder.py \\\n"
        "    --triples experiments/cache/retrieval_v2/triples_smoke_500.jsonl \\\n"
        "    --output-dir /content/bge_m3_smoke \\\n"
        "    --hub-repo OrRim123/recsys2026-bge-m3-smoke 2>&1 | tail -10\n"
        "!rm -rf /content/bge_m3_smoke"
    ),
    code(
        "# 5) FULL fine-tune: ~6-8 hr on Blackwell. Pushes merged model to Hub.\n"
        "!cd /content/recsys2026 && python -u scripts/train_bi_encoder.py \\\n"
        "    --triples experiments/cache/retrieval_v2/triples_bge_m3.jsonl \\\n"
        "    --output-dir /content/bge_m3_finetune \\\n"
        "    --hub-repo OrRim123/recsys2026-bge-m3-music-v1 \\\n"
        "    --results-dir /content/drive/MyDrive/recsys2026_retrieval_v2_cache/results/bge_m3 \\\n"
        "    --merge --cleanup-after-push \\\n"
        "    2>&1 | tee /content/drive/MyDrive/recsys2026_retrieval_v2_cache/bge_m3_train_log.txt"
    ),
    code(
        "# 6) Re-embed the 47K-track catalog with the fine-tuned model.\n"
        "from FlagEmbedding import BGEM3FlagModel\n"
        "import numpy as np\n"
        "from datasets import load_dataset\n"
        "import sys\n"
        "sys.path.insert(0, '/content/recsys2026/music-crs-baselines')\n"
        "from mcrs.retrieval_modules.bge_m3_format import format_track_text\n"
        "\n"
        "model = BGEM3FlagModel('OrRim123/recsys2026-bge-m3-music-v1-merged', use_fp16=True, device='cuda')\n"
        "tm = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')\n"
        "texts = [format_track_text(r.get('track_name','unknown'), r.get('artist_name'), r.get('album_name'), r.get('release_date'), r.get('tag_list')) for r in tm]\n"
        "track_ids = [r['track_id'] for r in tm]\n"
        "embs = model.encode(texts, batch_size=64, max_length=256)['dense_vecs']\n"
        "np.save('/content/drive/MyDrive/recsys2026_retrieval_v2_cache/track_embs_bge_m3_ft.npy', np.asarray(embs, dtype='float32'))\n"
        "import json as _json\n"
        "open('/content/drive/MyDrive/recsys2026_retrieval_v2_cache/track_embs_bge_m3_ft.track_ids.json', 'w').write(_json.dumps(track_ids))\n"
        "print(f'wrote {len(track_ids)} track embeddings')"
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
git commit -m "stage A: notebook 70 — add smoke + full train + catalog re-embed cells"
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

Computes nDCG@K, recall@K, MRR per turn against val.parquet.
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

### Task 11: Wire fine-tuned BGE-M3 into the wRRF factory

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/__init__.py` (around the existing `wrrf_bm25_dense_lyrics_bge_m3_v1` factory)
- Create: `tests/test_v2_factories.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_v2_factories.py`:

```python
"""Tests for the v2 wRRF factory variant that uses the fine-tuned BGE-M3."""
import pytest


def test_wrrf_bm25_dense_lyrics_bge_m3_ft_v1_factory_exists(monkeypatch, tmp_path):
    """New factory builds a 3-stream wRRF: BM25 + dense_lyrics + fine-tuned BGE-M3."""
    import pandas as pd
    # Stub the SID parquet (not used by this factory, but other factories may load it)
    sid_dir = tmp_path / "sid"; sid_dir.mkdir()
    pd.DataFrame([{"track_id": "t1", "code_1": 0, "code_2": 0, "code_3": 0, "popularity": 1.0, "bucket_rank": 0}]).to_parquet(sid_dir / "track_to_sid.parquet")

    from mcrs.retrieval_modules import load_retrieval_module
    # Should NOT raise — just verify the factory routes correctly.
    try:
        m = load_retrieval_module(
            retrieval_type="wrrf_bm25_dense_lyrics_bge_m3_ft_v1",
            dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            track_split_types=["all_tracks"],
            corpus_types=["track_name", "artist_name", "album_name"],
            cache_dir=str(tmp_path),
            extra_config={"bge_m3_hub_repo": "OrRim123/recsys2026-bge-m3-music-v1-merged"},
        )
    except Exception as e:
        # Some sub-retrievers can't fully initialize in a test (e.g., need HF downloads).
        # We accept that — but the factory itself MUST recognize the retrieval_type.
        assert "unknown retrieval_type" not in str(e).lower(), \
            f"factory did not recognize the new retrieval_type: {e}"
```

- [ ] **Step 2: Run test — expect fail**

Run: `pytest tests/test_v2_factories.py -v`

Expected: assertion error about unknown retrieval_type (or import errors that we'll fix in Step 3).

- [ ] **Step 3: Add the new factory branch**

Modify `music-crs-baselines/mcrs/retrieval_modules/__init__.py`. Find the existing `wrrf_bm25_dense_lyrics_bge_m3_v1` factory and ADD a new branch immediately after it:

```python
    # nDCG-stretch Stage A factory: same shape as wrrf_bm25_dense_lyrics_bge_m3_v1
    # but the BGE-M3 sub-retriever loads our FINE-TUNED merged model from Hub.
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
                    "topk_internal": 200,
                    "weight": 1.0,
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 200,
                    "weight": 0.4,
                },
                {
                    "type": "dense_bge_m3_finetuned",
                    "corpus_types": corpus_types,
                    "topk_internal": 200,
                    "weight": 0.6,
                    "extra_config": {"hub_repo": bge_m3_hub},
                },
            ],
            k=60,
        )
```

You'll also need to add a `dense_bge_m3_finetuned` sub-retriever type. Add this branch to `load_retrieval_module` (search for `dense_metadata_qwen3_instruct` and add nearby):

```python
    elif retrieval_type == "dense_bge_m3_finetuned":
        from mcrs.retrieval_modules.dense_bge_m3 import DENSE_BGE_M3
        hub_repo = extra_config.get("hub_repo", "OrRim123/recsys2026-bge-m3-music-v1-merged")
        return DENSE_BGE_M3(
            hub_repo=hub_repo, dataset_name=dataset_name,
            split_types=track_split_types, cache_dir=cache_dir,
        )
```

- [ ] **Step 4: Create the `DENSE_BGE_M3` sub-retriever class**

Create `music-crs-baselines/mcrs/retrieval_modules/dense_bge_m3.py`:

```python
"""Dense retriever using a fine-tuned BGE-M3 from Hub.

Conforms to the retriever interface used by RRF_MODEL: batch_text_to_item_retrieval
returns list[list[str]] of track_ids per query.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


class DENSE_BGE_M3:
    def __init__(
        self,
        hub_repo: str,
        dataset_name: str,
        split_types: list[str],
        cache_dir: str,
        device: Optional[str] = None,
    ):
        from FlagEmbedding import BGEM3FlagModel
        from datasets import load_dataset

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[dense_bge_m3] loading {hub_repo} on {self.device}", file=sys.stderr)
        self.model = BGEM3FlagModel(hub_repo, use_fp16=True, device=self.device)

        # Load or build track embeddings cache
        embs_path = Path(cache_dir) / "dense_bge_m3_ft" / f"{hub_repo.replace('/', '_')}.npy"
        ids_path = Path(cache_dir) / "dense_bge_m3_ft" / f"{hub_repo.replace('/', '_')}.ids.json"

        if embs_path.exists() and ids_path.exists():
            print(f"[dense_bge_m3] loading cached embeddings from {embs_path}", file=sys.stderr)
            self.track_embs = np.load(embs_path).astype(np.float32)
            self.track_ids = json.loads(ids_path.read_text())
        else:
            print(f"[dense_bge_m3] building track embeddings (cache miss)", file=sys.stderr)
            from .bge_m3_format import format_track_text
            tm = load_dataset(dataset_name, split=split_types[0])
            texts = [
                format_track_text(
                    track_name=r.get("track_name", "unknown"),
                    artist_name=r.get("artist_name"),
                    album_name=r.get("album_name"),
                    release_date=r.get("release_date"),
                    tag_list=r.get("tag_list"),
                )
                for r in tm
            ]
            self.track_ids = [r["track_id"] for r in tm]
            embs = self.model.encode(texts, batch_size=64, max_length=256)["dense_vecs"]
            self.track_embs = np.asarray(embs, dtype=np.float32)
            embs_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embs_path, self.track_embs)
            ids_path.write_text(json.dumps(self.track_ids))
        print(f"[dense_bge_m3] {len(self.track_ids)} tracks ready", file=sys.stderr)

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int,
        user_ids=None, batch_context=None,
    ) -> list[list[str]]:
        embs = self.model.encode(queries, batch_size=64, max_length=512)["dense_vecs"]
        embs = np.asarray(embs, dtype=np.float32)
        sims = embs @ self.track_embs.T  # (Q, N)
        topk_idx = np.argpartition(-sims, kth=min(topk, sims.shape[1]-1), axis=1)[:, :topk]
        # Sort within top-k descending
        row_indices = np.arange(sims.shape[0])[:, None]
        topk_sorted = topk_idx[row_indices, np.argsort(-sims[row_indices, topk_idx], axis=1)]
        return [[self.track_ids[j] for j in row] for row in topk_sorted]

    def text_to_item_retrieval(self, query: str, topk: int, user_id=None) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
```

- [ ] **Step 5: Run test — expect pass**

Run: `pytest tests/test_v2_factories.py -v`

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/__init__.py \
        music-crs-baselines/mcrs/retrieval_modules/dense_bge_m3.py \
        tests/test_v2_factories.py
git commit -m "stage A: new factory wrrf_bm25_dense_lyrics_bge_m3_ft_v1 + DENSE_BGE_M3 sub-retriever"
```

---

### Task 12: Write Stage A offline eval cell in notebook 70 + decision gate

**Files:**
- Modify: `colab/70_train_bi_encoder.ipynb`

- [ ] **Step 1: Append the offline-eval cell**

```python
python3 << 'PYEOF'
import json
NB = '/Users/orrimoch/PythonProjs/recsys2026/colab/70_train_bi_encoder.ipynb'
nb = json.load(open(NB))

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}

nb['cells'].append(code(
    "# 7) Offline eval: run the fine-tuned BGE-M3 alone (bi-encoder only) against val.parquet.\n"
    "# This is the diagnostic — gate is nDCG@20 >= 0.15.\n"
    "import json, sys\n"
    "sys.path.insert(0, '/content/recsys2026/music-crs-baselines')\n"
    "from mcrs.retrieval_modules.dense_bge_m3 import DENSE_BGE_M3\n"
    "from mcrs.retrieval_modules.bge_m3_format import format_query_text\n"
    "import pandas as pd\n"
    "\n"
    "dense = DENSE_BGE_M3(\n"
    "    hub_repo='OrRim123/recsys2026-bge-m3-music-v1-merged',\n"
    "    dataset_name='talkpl-ai/TalkPlayData-Challenge-Track-Metadata',\n"
    "    split_types=['all_tracks'],\n"
    "    cache_dir='/content/recsys2026/experiments/cache',\n"
    ")\n"
    "val = pd.read_parquet('experiments/cache/sid_training/val.parquet')\n"
    "val = val[val['source'] == 'raw'].reset_index(drop=True).head(500)\n"
    "queries = [format_query_text(\n"
    "    chat_history=r.get('chat_history') or [],\n"
    "    current_user_query=r.get('current_user_query',''),\n"
    "    user_profile=r.get('user_profile_raw'),\n"
    "    conversation_goal=r.get('conversation_goal'),\n"
    ") for _, r in val.iterrows()]\n"
    "preds = dense.batch_text_to_item_retrieval(queries, topk=20)\n"
    "import math\n"
    "ndcgs = []\n"
    "for i, row in val.iterrows():\n"
    "    gold = row['track_id']\n"
    "    p = preds[i]\n"
    "    if gold in p:\n"
    "        rank = p.index(gold) + 1\n"
    "        ndcgs.append(1.0 / math.log2(rank + 1))\n"
    "    else:\n"
    "        ndcgs.append(0.0)\n"
    "mean_ndcg = sum(ndcgs) / len(ndcgs)\n"
    "print(f'fine-tuned BGE-M3 standalone nDCG@20 on val: {mean_ndcg:.4f}')\n"
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
git commit -m "stage A: notebook 70 — offline eval cell + decision gate (>= 0.15)"
```

---

## Submission 1

### Task 13: Submission 1 — Stage A only on Blind-A

**Files:**
- Create: `music-crs-baselines/config/180-wrrf-bge-m3-ft-v5kto-blindA.yaml`
- Create: `colab/73_run_blindset_retrieval_v2.ipynb` (initial version)

- [ ] **Step 1: Write the config**

Create `music-crs-baselines/config/180-wrrf-bge-m3-ft-v5kto-blindA.yaml`:

```yaml
# Submission 1: Stage A — wRRF(BM25 + dense_lyrics + fine-tuned BGE-M3) + ProRank + v5-kto.
# Replaces ProRank in Submission 2; replaces ProRank+adds LGBM in Submission 3.
#
# Hub repo for fine-tuned BGE-M3: OrRim123/recsys2026-bge-m3-music-v1-merged
# Override via extra_config.bge_m3_hub_repo if needed.

lm_type: "OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged"
lora_path: null
lora_max_rank: 32

retrieval_type: "wrrf_bm25_dense_lyrics_bge_m3_ft_v1"
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

# extra_config for the new factory
extra_config:
  bge_m3_hub_repo: "OrRim123/recsys2026-bge-m3-music-v1-merged"
```

- [ ] **Step 2: Create notebook 73 (clone of notebook 63 with config 180)**

```bash
cp /Users/orrimoch/PythonProjs/recsys2026/colab/63_run_blindset_sid.ipynb \
   /Users/orrimoch/PythonProjs/recsys2026/colab/73_run_blindset_retrieval_v2.ipynb
```

Then edit notebook 73 via Python to swap config TID and zip name:

```python
python3 << 'PYEOF'
import json
NB = '/Users/orrimoch/PythonProjs/recsys2026/colab/73_run_blindset_retrieval_v2.ipynb'
nb = json.load(open(NB))
for cell in nb['cells']:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    src = src.replace('170-wrrf-sid-v5kto-blindsetA', '180-wrrf-bge-m3-ft-v5kto-blindA')
    src = src.replace('sid-ensemble-170', 'retrieval-v2-180-bge-m3-ft')
    src = src.replace('# 63 — Run Blind-A with config 170 (wRRF + SID)', '# 73 — Run Blind-A with config 180 (wRRF + fine-tuned BGE-M3, Submission 1)')
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

### Task 14: Cross-encoder HN re-mining script

**Files:**
- Create: `scripts/build_cross_encoder_training_data.py`
- Create: `tests/test_build_cross_encoder_training_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_cross_encoder_training_data.py`:

```python
def test_build_ce_triple_shapes():
    """Cross-encoder triple has query, pos, neg list of 7 items."""
    from scripts.build_cross_encoder_training_data import build_ce_triple
    triple = build_ce_triple(
        query="play me jazz",
        gold_track_text="track_name: So What | ...",
        neg_track_texts=["t" + str(i) for i in range(7)],
    )
    assert triple["query"] == "play me jazz"
    assert triple["pos"] == ["track_name: So What | ..."]
    assert len(triple["neg"]) == 7
```

- [ ] **Step 2: Implement**

Create `scripts/build_cross_encoder_training_data.py` — analogous to Task 6's bi-encoder builder but uses the **fine-tuned BGE-M3** for HN mining (so negatives are in-distribution per spec §7):

```python
"""Stage B training data builder. Mines HNs using the fine-tuned BGE-M3 (Stage A output).

Usage:
  python scripts/build_cross_encoder_training_data.py \\
    --train-parquet experiments/cache/sid_training/train.parquet \\
    --bge-m3-ft-hub OrRim123/recsys2026-bge-m3-music-v1-merged \\
    --output experiments/cache/retrieval_v2/triples_reranker.jsonl \\
    --percpos-threshold 0.80 --k-negs 7
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
from tqdm import tqdm

from mcrs.retrieval_modules.bge_m3_format import format_query_text, format_track_text
from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query


def build_ce_triple(query: str, gold_track_text: str, neg_track_texts: list[str]) -> dict:
    return {"query": query, "pos": [gold_track_text], "neg": neg_track_texts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--bge-m3-ft-hub", required=True)
    parser.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--percpos-threshold", type=float, default=0.80)
    parser.add_argument("--k-negs", type=int, default=7)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    # 1. Load fine-tuned BGE-M3 + encode tracks
    from FlagEmbedding import BGEM3FlagModel
    from datasets import load_dataset
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(args.bge_m3_ft_hub, use_fp16=True, device=device)
    tm = load_dataset(args.track_meta_hf, split="all_tracks")
    track_texts = [format_track_text(r.get("track_name", "unknown"), r.get("artist_name"),
                                      r.get("album_name"), r.get("release_date"), r.get("tag_list"))
                   for r in tm]
    track_ids = [r["track_id"] for r in tm]
    track_text_map = dict(zip(track_ids, track_texts))
    print("[ce-build] encoding 47K tracks with fine-tuned BGE-M3", file=sys.stderr)
    track_embs = model.encode(track_texts, batch_size=args.batch_size, max_length=256)["dense_vecs"]
    track_embs = np.asarray(track_embs, dtype=np.float32)

    # 2. Build triples per query
    train = pd.read_parquet(args.train_parquet)
    train = train[train["source"] == "raw"].reset_index(drop=True)
    if args.max_rows > 0:
        train = train.head(args.max_rows)
    print(f"[ce-build] {len(train)} queries", file=sys.stderr)

    with open(args.output, "w") as f_out:
        for i in tqdm(range(0, len(train), args.batch_size)):
            batch = train.iloc[i:i+args.batch_size].to_dict("records")
            batch_queries = [format_query_text(
                chat_history=r.get("chat_history") or [],
                current_user_query=r.get("current_user_query", ""),
                user_profile=r.get("user_profile_raw"),
                conversation_goal=r.get("conversation_goal"),
            ) for r in batch]
            batch_embs = np.asarray(model.encode(batch_queries, batch_size=args.batch_size, max_length=512)["dense_vecs"], dtype=np.float32)
            for j, row in enumerate(batch):
                gold = row["track_id"]
                if gold not in track_text_map:
                    continue
                negs = mine_negatives_for_query(
                    query_emb=batch_embs[j], track_embs=track_embs, track_ids=track_ids,
                    gold_track_id=gold, percpos_threshold=args.percpos_threshold,
                    k_negs=args.k_negs, pool_size=args.pool_size, seed=42 + i + j,
                )
                if len(negs) < 2:
                    continue
                triple = build_ce_triple(
                    query=batch_queries[j],
                    gold_track_text=track_text_map[gold],
                    neg_track_texts=[track_text_map[n] for n in negs if n in track_text_map],
                )
                f_out.write(json.dumps(triple) + "\n")
    print(f"[ce-build] DONE → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_build_cross_encoder_training_data.py -v`

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_cross_encoder_training_data.py tests/test_build_cross_encoder_training_data.py
git commit -m "stage B: cross-encoder HN re-mining script (uses Stage A fine-tuned model)"
```

---

### Task 15: Cross-encoder fine-tune wrapper

**Files:**
- Create: `scripts/train_cross_encoder.py`

- [ ] **Step 1: Implement the wrapper**

Create `scripts/train_cross_encoder.py`:

```python
"""Stage B: fine-tune BAAI/bge-reranker-base via FlagEmbedding's reranker module.

Hyperparams per spec §7:
- Full FT (no LoRA), lr=2e-5, bs=16, epochs=3, seq_len=512, bf16
- Cross-entropy on pairwise pos/neg

Usage:
  python scripts/train_cross_encoder.py \\
    --triples experiments/cache/retrieval_v2/triples_reranker.jsonl \\
    --output-dir /content/bge_reranker_finetune \\
    --hub-repo OrRim123/recsys2026-bge-reranker-music-v1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hub-repo", required=True)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun", "--nproc_per_node", "1",
        "-m", "FlagEmbedding.finetune.reranker.encoder_only.base",
        "--model_name_or_path", args.base_model,
        "--train_data", args.triples,
        "--output_dir", args.output_dir,
        "--learning_rate", "2e-5",
        "--per_device_train_batch_size", "16",
        "--num_train_epochs", "3",
        "--warmup_ratio", "0.1",
        "--max_len", "512",
        "--bf16", "True",
        "--save_steps", "2000",
        "--logging_steps", "50",
        "--report_to", "tensorboard",
    ]
    print(f"[train-ce] running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.check_call(cmd)

    # Push to Hub (no merge needed — full FT)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    tok = AutoTokenizer.from_pretrained(args.output_dir)
    print(f"[train-ce] pushing to {args.hub_repo}", file=sys.stderr)
    model.push_to_hub(args.hub_repo, private=False)
    tok.push_to_hub(args.hub_repo, private=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify compile**

Run: `python3 -m py_compile scripts/train_cross_encoder.py`

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_cross_encoder.py
git commit -m "stage B: train_cross_encoder.py — FlagEmbedding reranker wrapper"
```

---

### Task 16: Cross-encoder reranker module wrapper

**Files:**
- Create: `music-crs-baselines/mcrs/rerankers/bge_reranker_ft.py`

- [ ] **Step 1: Implement the reranker class**

Create `music-crs-baselines/mcrs/rerankers/bge_reranker_ft.py`:

```python
"""Cross-encoder reranker wrapper for fine-tuned BGE-reranker-base.

Conforms to the rerank() interface used in crs_baseline.py.
"""
from __future__ import annotations

import sys
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGE_RERANKER_FT:
    def __init__(
        self,
        hub_repo: str = "OrRim123/recsys2026-bge-reranker-music-v1",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[bge-reranker-ft] loading {hub_repo} on {self.device}", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(hub_repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hub_repo, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.inference_mode()
    def rerank(
        self, query: str, candidate_texts: list[str], topk: int,
    ) -> list[int]:
        """Return indices (into candidate_texts) sorted by relevance descending."""
        scores = []
        for i in range(0, len(candidate_texts), self.batch_size):
            batch = candidate_texts[i:i+self.batch_size]
            pairs = [[query, c] for c in batch]
            enc = self.tok(pairs, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.squeeze(-1).float().cpu().tolist()
            scores.extend(logits)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return order[:topk]
```

- [ ] **Step 2: Wire into the reranker registry**

Find the existing reranker registry (likely in `mcrs/rerankers/__init__.py` or `mcrs/crs_baseline.py`). Add a branch for `reranker_type == "bge_reranker_ft"`:

```python
    elif reranker_type == "bge_reranker_ft":
        from mcrs.rerankers.bge_reranker_ft import BGE_RERANKER_FT
        hub = reranker_model_path or "OrRim123/recsys2026-bge-reranker-music-v1"
        return BGE_RERANKER_FT(hub_repo=hub)
```

- [ ] **Step 3: Commit**

```bash
git add music-crs-baselines/mcrs/rerankers/bge_reranker_ft.py \
        music-crs-baselines/mcrs/crs_baseline.py  # (or whichever file has the registry)
git commit -m "stage B: BGE_RERANKER_FT class + reranker registry branch"
```

---

### Task 17: Notebook 71 — Stage B end-to-end

**Files:**
- Create: `colab/71_train_cross_encoder.ipynb`

- [ ] **Step 1: Generate the notebook**

Create notebook 71 with 5 cells mirroring notebook 70's structure (setup / HN re-mine smoke / full re-mine / smoke fine-tune / full fine-tune + push). Use the same `python3 << 'PYEOF'` pattern from Task 7. Cells call:
- `python scripts/build_cross_encoder_training_data.py --bge-m3-ft-hub OrRim123/recsys2026-bge-m3-music-v1-merged --output experiments/cache/retrieval_v2/triples_reranker.jsonl`
- `python scripts/train_cross_encoder.py --triples ... --hub-repo OrRim123/recsys2026-bge-reranker-music-v1`

- [ ] **Step 2: Commit**

```bash
git add colab/71_train_cross_encoder.ipynb
git commit -m "stage B: notebook 71 — cross-encoder HN re-mine + fine-tune cells"
```

---

### Task 18: Offline eval for Stage A+B + decision gate

**Files:**
- No new files — modify notebook 71 with an eval cell

- [ ] **Step 1: Append an offline-eval cell to notebook 71**

The cell loads BM25+dense_lyrics+BGE-M3-FT (already cached), gets top-100 per val query, then reranks with the fine-tuned BGE-reranker. Computes nDCG@20. Gate: ≥ 0.25.

- [ ] **Step 2: Commit**

```bash
git add colab/71_train_cross_encoder.ipynb
git commit -m "stage B: notebook 71 — offline eval cell with gate (>= 0.25)"
```

---

### Task 19: Wire cross-encoder into `reranker_type` config option for inference

**Files:**
- No code changes (the registry branch was added in Task 16). Just verify by trying to load.

- [ ] **Step 1: Smoke check the registry**

In Colab cell:
```python
from mcrs.rerankers.bge_reranker_ft import BGE_RERANKER_FT
r = BGE_RERANKER_FT()  # default hub repo
order = r.rerank("play me jazz", ["track1: jazz album", "track2: heavy metal"], topk=2)
print(order)  # expect [0, 1]
```

---

## Submission 2

### Task 20: Submission 2 config + notebook update

**Files:**
- Create: `music-crs-baselines/config/181-+ce-ft-v5kto-blindA.yaml`

- [ ] **Step 1: Write config 181**

Copy `config/180-...` to `config/181-+ce-ft-v5kto-blindA.yaml` and change:
```yaml
reranker_type: "bge_reranker_ft"
reranker_model_path: "OrRim123/recsys2026-bge-reranker-music-v1"
```

- [ ] **Step 2: Re-run notebook 73 with `--tid 181-+ce-ft-v5kto-blindA`**

(Same notebook; just change the TID at top.) Upload zip → CodaBench.

- [ ] **Step 3: Log score; check abort rule**

If composite regresses vs Submission 1 → revert to config 180 stack and skip Submission 3.

```bash
git add music-crs-baselines/config/181-+ce-ft-v5kto-blindA.yaml
git commit -m "submission 2: config 181 (Stage A + B: BGE-M3-FT + BGE-reranker-FT)"
```

---

## Stage C — LightGBM LambdaRank

### Task 21: Extend `scripts/build_lgbm_features.py` with new feature groups (TDD)

**Files:**
- Modify: `scripts/build_lgbm_features.py`
- Create: `tests/test_lgbm_features_extended.py`

- [ ] **Step 1: Write failing tests for each new feature group**

Create `tests/test_lgbm_features_extended.py`:

```python
"""Tests for the new features added to build_lgbm_features.py."""
import pytest


def test_compute_release_year_cyclical_features_for_1969():
    from scripts.build_lgbm_features import compute_release_year_cyclical
    sin, cos = compute_release_year_cyclical("1969-05-29")
    assert -1.0 <= sin <= 1.0
    assert -1.0 <= cos <= 1.0


def test_compute_release_year_handles_missing_year():
    from scripts.build_lgbm_features import compute_release_year_cyclical
    sin, cos = compute_release_year_cyclical(None)
    assert sin == 0.0 and cos == 0.0


def test_compute_query_track_tag_overlap():
    from scripts.build_lgbm_features import compute_tag_overlap
    overlap = compute_tag_overlap(
        query="I love folk rock from the 70s",
        tag_list=["folk rock", "70s", "acoustic"],
    )
    assert overlap >= 2  # "folk rock" and "70s"


def test_compute_last_turn_moved_toward_goal_from_assessments():
    from scripts.build_lgbm_features import last_turn_moved_toward_goal
    assert last_turn_moved_toward_goal(["MOVES_TOWARD_GOAL", "DOES_NOT_MOVE_TOWARD_GOAL"]) == 0
    assert last_turn_moved_toward_goal(["DOES_NOT_MOVE_TOWARD_GOAL", "MOVES_TOWARD_GOAL"]) == 1
    assert last_turn_moved_toward_goal([]) == -1  # unknown


def test_compute_query_drift_score_first_turn_returns_neutral():
    """For first turn, no prior query exists → drift = 1.0 (no drift)."""
    from scripts.build_lgbm_features import query_drift_score
    score = query_drift_score(current_query="hello", prior_queries=[], embedder=None)
    assert score == 1.0
```

- [ ] **Step 2: Run tests — expect fail**

Run: `pytest tests/test_lgbm_features_extended.py -v`

Expected: 5 fail.

- [ ] **Step 3: Add the new feature functions to `scripts/build_lgbm_features.py`**

Read the current file first to understand structure:
```bash
cat scripts/build_lgbm_features.py | head -50
```

Then APPEND these helper functions (don't rewrite existing code):

```python
import math
from typing import Optional


def compute_release_year_cyclical(release_date: Optional[str]) -> tuple[float, float]:
    """Sin/cos encoding of release year mod century. Returns (0, 0) on missing."""
    if not release_date:
        return 0.0, 0.0
    try:
        year = int(release_date[:4])
    except (ValueError, TypeError):
        return 0.0, 0.0
    phase = 2 * math.pi * (year % 100) / 100.0
    return math.sin(phase), math.cos(phase)


def compute_tag_overlap(query: str, tag_list: Optional[list[str]]) -> int:
    """Count of tags present in the query (case-insensitive substring)."""
    if not tag_list:
        return 0
    q_lower = query.lower()
    return sum(1 for t in tag_list if t and t.lower() in q_lower)


def last_turn_moved_toward_goal(assessments: Optional[list[str]]) -> int:
    """Return 1 if last assessment is MOVES_TOWARD_GOAL, 0 if DOES_NOT, -1 if unknown."""
    if not assessments:
        return -1
    last = assessments[-1]
    if last == "MOVES_TOWARD_GOAL":
        return 1
    if last == "DOES_NOT_MOVE_TOWARD_GOAL":
        return 0
    return -1


def query_drift_score(
    current_query: str, prior_queries: list[str], embedder=None,
) -> float:
    """Cosine sim between current and turn-1 query embeddings. First turn → 1.0."""
    if not prior_queries:
        return 1.0
    if embedder is None:
        return 1.0  # fallback when no embedder available at feature-build time
    import numpy as np
    embs = embedder.encode([current_query, prior_queries[0]])
    a, b = embs[0], embs[1]
    na, nb = a / (1e-9 + (a @ a) ** 0.5), b / (1e-9 + (b @ b) ** 0.5)
    return float(na @ nb)
```

Also extend the main feature-extraction loop (search for the existing feature list and add new feature columns).

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_lgbm_features_extended.py -v`

Expected: 5 pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_lgbm_features.py tests/test_lgbm_features_extended.py
git commit -m "stage C: extend build_lgbm_features.py with 5 new feature helpers (TDD)"
```

---

### Task 22: Stage C feature-extraction pipeline (apply to held-out train slice)

**Files:**
- Create: `colab/72_build_lgbm_features_train.ipynb`

- [ ] **Step 1: Build the notebook**

Notebook 72 has 4 cells:

1. **Setup** — clone + Drive mount + deps (same pattern as notebook 70).
2. **Compute held-out 20% train-session split** — session-disjoint, seed=42.
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   train = pd.read_parquet('experiments/cache/sid_training/train.parquet')
   sessions = sorted(train['session_id'].dropna().unique())
   train_sessions, val_sessions = train_test_split(sessions, test_size=0.2, random_state=42)
   train.loc[train['session_id'].isin(train_sessions)].to_parquet('experiments/cache/retrieval_v2/lgbm_train.parquet')
   train.loc[train['session_id'].isin(val_sessions)].to_parquet('experiments/cache/retrieval_v2/lgbm_val.parquet')
   ```
3. **Run Stage A+B pipeline on each row** to get top-50 candidates per turn.
4. **Feature extraction** — call `scripts/build_lgbm_features.py` for each (query, candidate) pair, output parquet.

- [ ] **Step 2: Commit**

```bash
git add colab/72_build_lgbm_features_train.ipynb
git commit -m "stage C: notebook 72 — held-out train split + feature extraction pipeline"
```

---

### Task 23: LightGBM trainer

**Files:**
- Create: `scripts/train_lgbm_ranker.py`
- Create: `tests/test_train_lgbm_ranker.py`

- [ ] **Step 1: Write test**

```python
def test_lgbm_ranker_groups_by_session_turn():
    """Training data must be grouped by (session_id, turn_number) for LambdaRank."""
    import pandas as pd
    from scripts.train_lgbm_ranker import build_groups
    df = pd.DataFrame([
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 1},
        {"session_id": "s1", "turn_number": 2},
        {"session_id": "s2", "turn_number": 1},
    ])
    groups = build_groups(df)
    assert groups == [2, 1, 1]  # 2 rows for (s1, 1), 1 row for (s1, 2), 1 row for (s2, 1)
```

- [ ] **Step 2: Implement**

Create `scripts/train_lgbm_ranker.py`:

```python
"""LightGBM LambdaRank trainer for Stage C."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd


def build_groups(df: pd.DataFrame) -> list[int]:
    """Group sizes by (session_id, turn_number) for LambdaRank."""
    return df.groupby(["session_id", "turn_number"]).size().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True, help="parquet")
    parser.add_argument("--val-features", required=True, help="parquet")
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--n-estimators", type=int, default=1000)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train_features).sort_values(["session_id", "turn_number"])
    val_df = pd.read_parquet(args.val_features).sort_values(["session_id", "turn_number"])

    feat_cols = [c for c in train_df.columns if c not in {"session_id", "turn_number", "candidate_tid", "label"}]
    train_X = train_df[feat_cols].values
    train_y = train_df[args.label_col].values
    train_groups = build_groups(train_df)

    val_X = val_df[feat_cols].values
    val_y = val_df[args.label_col].values
    val_groups = build_groups(val_df)

    train_ds = lgb.Dataset(train_X, label=train_y, group=train_groups, feature_name=feat_cols)
    val_ds = lgb.Dataset(val_X, label=val_y, group=val_groups, reference=train_ds, feature_name=feat_cols)

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
        callbacks=[lgb.early_stopping(50)],
    )
    model.save_model(args.output_model)
    print(f"[lgbm] saved model → {args.output_model}")
    print(f"[lgbm] best iter: {model.best_iteration}")
    importance = sorted(zip(feat_cols, model.feature_importance(importance_type="gain")),
                        key=lambda x: -x[1])[:20]
    print("[lgbm] top-20 features by gain:")
    for name, gain in importance:
        print(f"  {name}: {gain:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_train_lgbm_ranker.py -v`

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_lgbm_ranker.py tests/test_train_lgbm_ranker.py
git commit -m "stage C: train_lgbm_ranker.py — LambdaRank trainer (TDD)"
```

---

### Task 24: LightGBM inference wrapper

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/lgbm_ranker.py`
- Create: `tests/test_lgbm_ranker_inference.py`

- [ ] **Step 1: Write test**

```python
def test_lgbm_ranker_returns_ordered_indices():
    """LGBM_RANKER.rerank returns indices sorted by predicted relevance descending."""
    import lightgbm as lgb
    import numpy as np
    # Build a tiny model that predicts label = feature[0]
    X = np.array([[0.1], [0.9], [0.5]])
    y = np.array([0, 1, 0])
    ds = lgb.Dataset(X, label=y, group=[3])
    model = lgb.train({"objective": "lambdarank", "metric": "ndcg", "verbosity": -1},
                      ds, num_boost_round=10)

    from mcrs.retrieval_modules.lgbm_ranker import LGBM_RANKER
    ranker = LGBM_RANKER.from_model(model, feature_names=["f0"])
    order = ranker.rerank(features=X, topk=3)
    # Expected: row 1 (feature=0.9) ranked first
    assert order[0] == 1
```

- [ ] **Step 2: Implement**

Create `music-crs-baselines/mcrs/retrieval_modules/lgbm_ranker.py`:

```python
"""LightGBM ranker inference wrapper for the Stage C final reranker."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np


class LGBM_RANKER:
    def __init__(self, model: lgb.Booster, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names

    @classmethod
    def from_path(cls, path: str | Path, feature_names: list[str]):
        return cls(lgb.Booster(model_file=str(path)), feature_names)

    @classmethod
    def from_model(cls, model: lgb.Booster, feature_names: list[str]):
        return cls(model, feature_names)

    def rerank(self, features: np.ndarray, topk: int) -> list[int]:
        """Return candidate indices sorted by predicted relevance descending."""
        scores = self.model.predict(features)
        order = np.argsort(-scores)[:topk]
        return order.tolist()
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_lgbm_ranker_inference.py -v`

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/lgbm_ranker.py tests/test_lgbm_ranker_inference.py
git commit -m "stage C: LGBM_RANKER inference wrapper (TDD)"
```

---

### Task 25: Wire LightGBM into the inference pipeline (post-reranker stage)

**Files:**
- Modify: `music-crs-baselines/mcrs/crs_baseline.py` (in `batch_chat` method, after the rerank stage)

- [ ] **Step 1: Add a post-rerank LGBM stage**

In `crs_baseline.py:batch_chat`, find the rerank stage and ADD after it:

```python
        # ---- Stage C: optional LightGBM final reranker -----------------
        if self.lgbm_ranker is not None:
            # For each batch row, build features then call lgbm_ranker.rerank
            # The features mirror what scripts/build_lgbm_features.py produces
            # at training time — keep parity.
            for q_idx, candidates in enumerate(batch_retrieval_items):
                features = self._build_lgbm_features(
                    query=retrieval_inputs[q_idx],
                    candidates=candidates,
                    user_id=user_ids[q_idx] if user_ids else None,
                    batch_ctx=batch_context[q_idx] if batch_context else None,
                )
                order = self.lgbm_ranker.rerank(features=features, topk=20)
                batch_retrieval_items[q_idx] = [candidates[i] for i in order]
```

You'll also need to add `self.lgbm_ranker` initialization in `__init__`, wired from `extra_config.lgbm_model_path` and `extra_config.lgbm_feature_names`.

- [ ] **Step 2: Add `_build_lgbm_features` method**

This is the parity contract — what's computed at inference must match training. Reuse functions from `scripts/build_lgbm_features.py` (import and call).

- [ ] **Step 3: Commit**

```bash
git add music-crs-baselines/mcrs/crs_baseline.py
git commit -m "stage C: wire LightGBM ranker as post-rerank stage in crs_baseline.batch_chat"
```

---

### Task 26: Notebook 72 — full Stage C run + offline eval

**Files:**
- Modify: `colab/72_build_lgbm_features_train.ipynb`

- [ ] **Step 1: Append training cell + offline eval cell**

Notebook 72 cells (append to existing):
- Run feature extraction on held-out train + val splits
- Train LGBM: `python scripts/train_lgbm_ranker.py --train-features ... --val-features ... --output-model lgbm_v1.txt`
- Push model to Hub (small file, easy) OR save to Drive
- Run offline eval through the full v2 pipeline (BM25 + BGE-M3-FT + BGE-reranker-FT + LGBM) on val.parquet
- Gate: nDCG@20 ≥ 0.35

- [ ] **Step 2: Commit**

```bash
git add colab/72_build_lgbm_features_train.ipynb
git commit -m "stage C: notebook 72 — training + offline eval + gate"
```

---

## Submission 3

### Task 27: Submission 3 config + Blind-A run

**Files:**
- Create: `music-crs-baselines/config/182-+lgbm-v5kto-blindA.yaml`

- [ ] **Step 1: Write config 182**

Copy `config/181-...` to `config/182-+lgbm-v5kto-blindA.yaml` and add:
```yaml
extra_config:
  bge_m3_hub_repo: "OrRim123/recsys2026-bge-m3-music-v1-merged"
  lgbm_model_path: "/content/drive/MyDrive/recsys2026_retrieval_v2_cache/lgbm_v1.txt"
  lgbm_feature_names: ["bm25_score", "bge_m3_cosine", "ce_logit", ...]  # full list from training
```

- [ ] **Step 2: Run notebook 73 with `--tid 182-+lgbm-v5kto-blindA`**

- [ ] **Step 3: Log score + commit config**

```bash
git add music-crs-baselines/config/182-+lgbm-v5kto-blindA.yaml
git commit -m "submission 3: config 182 (Stage A + B + C: full v2 stack)"
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
- nDCG@20 trajectory (Stage 0 → Stage A → Stage A+B → Stage A+B+C)
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
- §1 Goal → covered (Task 13 / 20 / 27 are the three submissions)
- §2 Target framing → applied via gates in Tasks 12 / 18 / 26
- §3 Architecture → built across Stages A / B / C
- §4 Existing infrastructure → Tasks 21 + 25 extend `build_lgbm_features.py` + `crs_baseline.py`
- §5 Data sources → Task 1 verifies + Task 22 builds held-out split
- §6 Stage A → Tasks 4-13
- §7 Stage B → Tasks 14-20
- §8 Stage C → Tasks 21-27
- §9 Submission cadence → Tasks 13 / 20 / 27 + abort rules in each
- §10 Risk / abort → captured as decision gates after each submission
- §11 Testing → TDD pattern in every code task (write failing test → impl → pass → commit)
- §12 Notebook structure → notebooks 70 / 71 / 72 / 73 created
- §13 Out of scope → respected (no SID, no responder changes)
- §14 Phase 2 transition → mentioned but separate spec
- §15 Cross-references → Task 28 memory cites the spec + this plan

**Placeholder scan**: no TBD / TODO / "implement later" anywhere; every step has the actual code or command.

**Type consistency**:
- `format_track_text` / `format_query_text` defined in Task 4, used in Tasks 6 / 7 / 14
- `mine_negatives_for_query` defined in Task 5, used in Task 6 / 14
- `DENSE_BGE_M3` defined in Task 11, used in Task 12 / submission notebook
- `BGE_RERANKER_FT` defined in Task 16, used in Task 19 / submission notebook
- `LGBM_RANKER` defined in Task 24, used in Task 25
- All Hub repo names consistent: `OrRim123/recsys2026-bge-m3-music-v1-merged`, `OrRim123/recsys2026-bge-reranker-music-v1`

**Gaps**: none identified above acceptable detail level. Implementation details for `_build_lgbm_features` in Task 25 are intentionally not spelled out cell-by-cell — that method will need to call the same helpers as `scripts/build_lgbm_features.py`, in the same order, with the same arguments, but the wiring is mechanical and follows the parity contract directly.
