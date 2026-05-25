# Recall Union + LightGBM Aggregator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the nDCG gap by completing a 4-channel wRRF recall union (BM25 + frozen Qwen-text + same-artist + session-CF) feeding a LightGBM LambdaRank aggregator, with no GPU training.

**Architecture:** Two new numpy/rule retriever channels (same-artist, session-CF) read the session's played track_ids (threaded via a new `history_tids` field on `batch_context`) and are fused with BM25 + the frozen Qwen-metadata dense retriever via the existing `RRF_MODEL`. A new fused `retrieval_type` wires the four. The existing LightGBM LambdaRank Stage C (`build_lgbm_features.py` + nb 72 + `lgbm_rerank.py`) is extended with same-artist/same-album/session-CF features and becomes the only reranker (cross-encoder dropped).

**Tech Stack:** Python, numpy, HuggingFace `datasets`, LightGBM (LambdaRank), the repo's `mcrs` retrieval/reranker factory, pytest.

---

## Context the executor must read first

- `music-crs-baselines/mcrs/retrieval_modules/rrf.py` — `RRF_MODEL`; channels are sub-retrievers selected by a `"type"` key in `sub_specs`; each implements `batch_text_to_item_retrieval(queries, topk, user_ids=None, batch_context=None)`.
- `music-crs-baselines/mcrs/retrieval_modules/cf_bpr.py` — `CF_BPR`: the template for a numpy retriever (loads the `cf-bpr` track matrix, L2-normalizes, cosine ranks). Reuse its track-matrix loader pattern.
- `music-crs-baselines/mcrs/retrieval_modules/__init__.py` — `load_retrieval_module(retrieval_type, dataset_name, track_split_types, corpus_types, cache_dir, extra_config)`; the `wrrf_bm25_multimodal_v1` branch (~line 436) shows the `sub_specs` shape; `dense_metadata_qwen3` (~line 44) is the frozen Qwen channel to reuse.
- `music-crs-baselines/mcrs/crs_baseline.py:526-534` — `batch_context` construction (we add `history_tids` here).
- `music-crs-baselines/mcrs/db_item/music_catalog.py` — `MusicCatalogDB(dataset_name, split_types, corpus_types)`, `.metadata_dict[tid]`, `.id_to_metadata(tid)`. Confirm the artist field key (`artist_name`) and album key (`album_name`) before Task 1.
- `scripts/build_lgbm_features.py` — existing 14-feature builder; `load_track_meta_lookup`, `build_pop_rank_pct_map`. We add features here.
- `colab/71_train_cross_encoder.ipynb` cells 6-7 — corrected dev eval; the recall harness we extend.

Run all pytest from repo root with the project venv: `source recsys26/bin/activate`.

---

## Task 1: Helper — extract played track_ids from a session history

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/session_history.py`
- Test: `tests/test_session_history.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_history.py
from mcrs.retrieval_modules.session_history import played_tids_from_context

def test_prefers_explicit_history_tids():
    ctx = {"history_tids": ["a", "b"], "chat_history": []}
    assert played_tids_from_context(ctx, catalog_tids={"a", "b", "c"}) == ["a", "b"]

def test_falls_back_to_chat_history_music_turns():
    ctx = {"chat_history": [
        {"role": "user", "content": "play rock"},
        {"role": "music", "content": "a"},
        {"role": "assistant", "content": "here you go"},
        {"role": "music", "content": "zzz-not-in-catalog"},
    ]}
    assert played_tids_from_context(ctx, catalog_tids={"a", "b"}) == ["a"]

def test_empty_context_returns_empty():
    assert played_tids_from_context(None, catalog_tids={"a"}) == []
    assert played_tids_from_context({}, catalog_tids={"a"}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session_history.py -v`
Expected: FAIL — `ModuleNotFoundError: mcrs.retrieval_modules.session_history`.

- [ ] **Step 3: Write minimal implementation**

```python
# music-crs-baselines/mcrs/retrieval_modules/session_history.py
"""Recover the list of track_ids already played in a session, for the
session-aware recall channels (same-artist, session-CF).

Prefers an explicit batch_context['history_tids'] (raw played track_ids,
populated by crs_baseline + the eval harness). Falls back to scanning
chat_history for music-turn contents that are valid catalog track_ids
(works only when music turns carry raw ids, not expanded text)."""
from __future__ import annotations

from typing import Optional


def played_tids_from_context(ctx: Optional[dict], catalog_tids: set) -> list[str]:
    if not ctx:
        return []
    explicit = ctx.get("history_tids")
    if explicit:
        return [str(t) for t in explicit if str(t) in catalog_tids]
    out: list[str] = []
    for turn in ctx.get("chat_history", []) or []:
        if turn.get("role") in ("music", "assistant"):
            c = str(turn.get("content", ""))
            if c in catalog_tids:
                out.append(c)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session_history.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/session_history.py tests/test_session_history.py
git commit -m "feat(retrieval): session-history -> played track_ids helper"
```

---

## Task 2: SameArtistRetriever channel

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/same_artist.py`
- Test: `tests/test_same_artist_retriever.py`

Behavior: from the session's played tids, collect their artists (with session counts); return catalog tracks by those artists, excluding already-played, ranked by (artist session-count desc, track popularity desc).

- [ ] **Step 1: Write the failing test (with a stub catalog)**

```python
# tests/test_same_artist_retriever.py
from mcrs.retrieval_modules.same_artist import SameArtistRetriever

class _StubCatalog:
    # tid -> {artist_name, popularity}
    meta = {
        "t1": {"artist_name": "A", "popularity": 5.0},
        "t2": {"artist_name": "A", "popularity": 9.0},
        "t3": {"artist_name": "B", "popularity": 1.0},
        "t4": {"artist_name": "C", "popularity": 1.0},
    }
    metadata_dict = meta

def _build():
    r = SameArtistRetriever.__new__(SameArtistRetriever)
    r._build_index(_StubCatalog())  # tid->artist, artist->[tids by pop desc]
    return r

def test_returns_unplayed_tracks_by_session_artists_excludes_played():
    r = _build()
    ctx = {"history_tids": ["t1"]}  # artist A played
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[ctx])[0]
    assert out == ["t2"]            # other A track; t1 excluded; B/C not session artists

def test_ranks_by_artist_session_count_then_popularity():
    r = _build()
    ctx = {"history_tids": ["t1", "t3"]}  # artists A and B (count 1 each)
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[ctx])[0]
    assert out == ["t2"]            # only unplayed same-artist track is t2 (A)

def test_no_history_returns_empty():
    r = _build()
    out = r.batch_text_to_item_retrieval(["q"], topk=10, batch_context=[{}])[0]
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_same_artist_retriever.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# music-crs-baselines/mcrs/retrieval_modules/same_artist.py
"""Session artist-continuity recall channel. 38% of gold next-tracks are by an
artist already in the session (see project_ndcg_gap_closer_recall_union_2026_05_25).
Candidates = catalog tracks by the session's artists, minus already-played."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from .session_history import played_tids_from_context


class SameArtistRetriever:
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir="./cache"):
        from mcrs.db_item.music_catalog import MusicCatalogDB
        catalog = MusicCatalogDB(dataset_name, split_types, corpus_types)
        self._build_index(catalog)

    def _build_index(self, catalog) -> None:
        self.tid_to_artist: dict[str, str] = {}
        artist_tracks: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for tid, m in catalog.metadata_dict.items():
            artist = str(m.get("artist_name") or "").strip().lower()
            if not artist:
                continue
            pop = float(m.get("popularity") or 0.0)
            self.tid_to_artist[tid] = artist
            artist_tracks[artist].append((tid, pop))
        # pre-sort each artist's tracks by popularity desc
        self.artist_to_tids: dict[str, list[str]] = {
            a: [t for t, _ in sorted(v, key=lambda x: -x[1])]
            for a, v in artist_tracks.items()
        }
        self.catalog_tids = set(self.tid_to_artist.keys())

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None,
                                     batch_context=None):
        out = []
        for i in range(len(queries)):
            ctx = batch_context[i] if batch_context else None
            played = played_tids_from_context(ctx, self.catalog_tids)
            if not played:
                out.append([])
                continue
            played_set = set(played)
            artist_count = Counter(self.tid_to_artist[t] for t in played
                                   if t in self.tid_to_artist)
            ranked: list[str] = []
            # artists by session frequency desc; within artist, popularity desc
            for artist, _cnt in artist_count.most_common():
                for tid in self.artist_to_tids.get(artist, []):
                    if tid not in played_set:
                        ranked.append(tid)
                        if len(ranked) >= topk:
                            break
                if len(ranked) >= topk:
                    break
            out.append(ranked[:topk])
        return out

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_same_artist_retriever.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/same_artist.py tests/test_same_artist_retriever.py
git commit -m "feat(retrieval): same-artist session-continuity channel"
```

---

## Task 3: SessionCFRetriever channel

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/session_cf.py`
- Test: `tests/test_session_cf_retriever.py`

Behavior: mean of the played tracks' cf-bpr vectors -> L2-normalize -> cosine vs the catalog cf-bpr matrix -> top-K (exclude played). Reuses `cf_bpr.py`'s track-matrix loader.

- [ ] **Step 1: Write the failing test (inject a tiny matrix)**

```python
# tests/test_session_cf_retriever.py
import numpy as np
from mcrs.retrieval_modules.session_cf import SessionCFRetriever

def _build():
    r = SessionCFRetriever.__new__(SessionCFRetriever)
    r.track_ids = ["t1", "t2", "t3", "t4"]
    r.tid_to_idx = {t: i for i, t in enumerate(r.track_ids)}
    # t1,t2 near each other; t3 opposite; t4 orthogonal
    mat = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    r.track_mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    r.catalog_tids = set(r.track_ids)
    return r

def test_session_centroid_retrieves_nearest_unplayed():
    r = _build()
    ctx = {"history_tids": ["t1"]}            # centroid ~ t1 direction
    out = r.batch_text_to_item_retrieval(["q"], topk=2, batch_context=[ctx])[0]
    assert out[0] == "t2"                       # nearest unplayed; t1 excluded
    assert "t3" not in out[:1]

def test_no_history_returns_empty():
    r = _build()
    assert r.batch_text_to_item_retrieval(["q"], topk=2, batch_context=[{}])[0] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session_cf_retriever.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# music-crs-baselines/mcrs/retrieval_modules/session_cf.py
"""Session-level collaborative recall channel. Candidates = catalog tracks whose
cf-bpr vector is nearest to the centroid of the session's played tracks. (Session
CF centroid recall@100 ~0.24 vs near-inert user-level CF; see the gap-closer memo.)"""
from __future__ import annotations

import numpy as np

from .session_history import played_tids_from_context


class SessionCFRetriever:
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir="./cache"):
        from .cf_bpr import CF_BPR
        donor = CF_BPR(dataset_name, split_types, corpus_types, cache_dir)
        self.track_ids = donor.track_ids
        self.track_mat = donor.track_mat            # already L2-normalized (47k x 128)
        self.tid_to_idx = {t: i for i, t in enumerate(self.track_ids)}
        self.catalog_tids = set(self.track_ids)

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None,
                                     batch_context=None):
        out = []
        for i in range(len(queries)):
            ctx = batch_context[i] if batch_context else None
            played = played_tids_from_context(ctx, self.catalog_tids)
            idxs = [self.tid_to_idx[t] for t in played if t in self.tid_to_idx]
            if not idxs:
                out.append([])
                continue
            centroid = self.track_mat[idxs].mean(axis=0)
            n = np.linalg.norm(centroid)
            if n < 1e-9:
                out.append([])
                continue
            centroid = centroid / n
            scores = self.track_mat @ centroid          # cosine (rows normalized)
            order = np.argsort(-scores)
            played_set = set(played)
            ranked = []
            for j in order:
                tid = self.track_ids[j]
                if tid not in played_set:
                    ranked.append(tid)
                    if len(ranked) >= topk:
                        break
            out.append(ranked)
        return out

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session_cf_retriever.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/session_cf.py tests/test_session_cf_retriever.py
git commit -m "feat(retrieval): session-CF centroid channel"
```

---

## Task 4: Register channels + the fused union type in the factory

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/__init__.py`
- Test: `tests/test_union_factory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_union_factory.py
import inspect
from mcrs.retrieval_modules import load_retrieval_module

def test_factory_source_registers_new_types():
    src = inspect.getsource(load_retrieval_module)
    for key in ('"same_artist"', '"session_cf"', '"wrrf_union_v1"'):
        assert key in src, f"missing factory branch {key}"

def test_union_v1_has_four_subspecs():
    src = inspect.getsource(load_retrieval_module)
    # the union branch must fuse bm25 + qwen-metadata + same_artist + session_cf
    for t in ('"bm25"', '"dense_metadata_qwen3"', '"same_artist"', '"session_cf"'):
        assert t in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_union_factory.py -v`
Expected: FAIL — new keys not in factory.

- [ ] **Step 3: Add the branches**

In `load_retrieval_module`, add near the other `elif retrieval_type == ...` branches (before the final `else`/raise):

```python
    elif retrieval_type == "same_artist":
        from .same_artist import SameArtistRetriever
        return SameArtistRetriever(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "session_cf":
        from .session_cf import SessionCFRetriever
        return SessionCFRetriever(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "wrrf_union_v1":
        # 4-channel recall union: lexical + frozen-Qwen semantic + session artist
        # continuity + session CF. extra_config['qwen_metadata_*'] flows to the
        # dense_metadata_qwen3 sub if it needs an embed label/cache.
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {"type": "bm25",
                 "corpus_types": ["track_name", "artist_name", "album_name",
                                  "release_date", "tag_list"],
                 "topk_internal": 100, "weight": float(extra_config.get("w_bm25", 1.0))},
                {"type": "dense_metadata_qwen3", "corpus_types": corpus_types,
                 "topk_internal": 100, "weight": float(extra_config.get("w_qwen", 0.7))},
                {"type": "same_artist", "topk_internal": 100,
                 "weight": float(extra_config.get("w_artist", 1.0))},
                {"type": "session_cf", "topk_internal": 100,
                 "weight": float(extra_config.get("w_cf", 0.7))},
            ],
            k=60,
        )
```

Note: confirm `dense_metadata_qwen3`'s constructor needs no extra catalog build beyond the provided embeddings; if it requires an `embed_*` key, pass it through `extra_config` here.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_union_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/__init__.py tests/test_union_factory.py
git commit -m "feat(retrieval): register same_artist/session_cf + wrrf_union_v1 fused type"
```

---

## Task 5: Thread `history_tids` into batch_context

**Files:**
- Modify: `music-crs-baselines/mcrs/crs_baseline.py:526-534`
- Test: `tests/test_batch_context_history_tids.py`

- [ ] **Step 1: Write the failing test (source-level — batch_chat is integration-heavy)**

```python
# tests/test_batch_context_history_tids.py
import inspect
from mcrs.crs_baseline import CRS_BASELINE

def test_batch_context_includes_history_tids():
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert '"history_tids"' in src, "batch_context must carry history_tids for session channels"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_batch_context_history_tids.py -v`
Expected: FAIL.

- [ ] **Step 3: Add `history_tids` to the batch_context dict**

In the `batch_context.append({...})` block (crs_baseline.py:529-534), add a key that extracts raw played track_ids from `prior_history` music turns:

```python
            _played = [str(t.get("content")) for t in prior_history
                       if t.get("role") == "music" and t.get("content")]
            batch_context.append({
                "chat_history": prior_history,
                "current_user_query": data["user_query"],
                "user_profile": data.get("user_profile_raw"),
                "conversation_goal": data.get("conversation_goal"),
                "history_tids": _played,
            })
```

(If `prior_history` music turns are already expanded to text at this point, instead source `history_tids` from the raw conversation `data` — confirm by printing one `prior_history` during a smoke run; the `session_history` helper already falls back gracefully.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_batch_context_history_tids.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/crs_baseline.py tests/test_batch_context_history_tids.py
git commit -m "feat: thread session history_tids into batch_context for session channels"
```

---

## Task 6: Extend the dev-eval harness to the union + pass history (Gate G1)

**Files:**
- Modify: `colab/71_train_cross_encoder.ipynb` (cell 7, per-stream cell) — add the `wrrf_union_v1` channel and pass `batch_context` with `history_tids`.

This is a notebook/orchestration task (run on Colab), validated by output, not pytest.

- [ ] **Step 1:** In cell 6, when building `queries/golds/user_ids`, also build a parallel `hist_tids` list: for each eval turn, the raw track_ids of the music turns before it (you already have `convs`; collect `convs[k]['content']` for `k<up` where role=='music'). Build `batch_ctx = [{"history_tids": h} for h in hist_tids]`.

- [ ] **Step 2:** In cell 7, add a `wrrf_union_v1` stream:

```python
_union = load_retrieval_module('wrrf_union_v1', ITEM_DB, ['all_tracks'], CORPUS, CACHE_DIR,
                               extra_config={})
streams['union (4ch)'] = _union.batch_text_to_item_retrieval(
    queries, topk=100, user_ids=user_ids, batch_context=batch_ctx)
```

- [ ] **Step 3:** Run cells 1→2→6→7 on Colab. Record recall@{20,100} for `same_artist`, `session_cf`, and `union (4ch)`.

- [ ] **Step 4: Gate G1 check.** PASS if `union (4ch)` recall@100 ≥ 0.46. If a channel adds <0.01 recall@100 over the union-without-it, drop it (ablation). Commit the notebook.

```bash
git add colab/71_train_cross_encoder.ipynb
git commit -m "eval: wrrf_union_v1 recall in dev cell 7 (+history_tids)"
```

---

## Task 7: Add same-artist / same-album / session-CF features to the LGBM feature builder

**Files:**
- Modify: `scripts/build_lgbm_features.py`
- Test: `tests/test_lgbm_new_features.py`

- [ ] **Step 1: Write the failing test for the pure feature functions**

```python
# tests/test_lgbm_new_features.py
from scripts.build_lgbm_features import session_match_features

def test_same_artist_and_album_flags_and_counts():
    played_meta = [{"artist_name": "A", "album_name": "X"},
                   {"artist_name": "A", "album_name": "Y"}]
    cand = {"artist_name": "A", "album_name": "Y"}
    f = session_match_features(cand, played_meta)
    assert f["same_artist"] == 1
    assert f["same_album"] == 1
    assert f["artist_in_session_count"] == 2

def test_no_match():
    f = session_match_features({"artist_name": "Z", "album_name": "Q"},
                               [{"artist_name": "A", "album_name": "X"}])
    assert f == {"same_artist": 0, "same_album": 0, "artist_in_session_count": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lgbm_new_features.py -v`
Expected: FAIL — `session_match_features` undefined.

- [ ] **Step 3: Add the feature function + wire it into the per-candidate feature dict**

Add to `scripts/build_lgbm_features.py` (near the other `compute_*` helpers):

```python
def session_match_features(cand_meta: dict, played_meta: list[dict]) -> dict:
    """Structural session-continuity features for one candidate."""
    c_artist = str(cand_meta.get("artist_name") or "").strip().lower()
    c_album = str(cand_meta.get("album_name") or "").strip().lower()
    artists = [str(m.get("artist_name") or "").strip().lower() for m in played_meta]
    albums = [str(m.get("album_name") or "").strip().lower() for m in played_meta]
    return {
        "same_artist": int(bool(c_artist) and c_artist in artists),
        "same_album": int(bool(c_album) and c_album in albums),
        "artist_in_session_count": sum(1 for a in artists if a and a == c_artist),
    }
```

Then in the per-candidate feature assembly loop, merge these three keys into the feature row (the executor must locate the dict built per candidate — it carries `wrrf_score`, `bm25_score`, etc. — and `row.update(session_match_features(cand_meta, played_meta))`). `played_meta` = metadata of the session's played tids (from the same `history_tids` source as Task 5). Also add a `session_cf_cos` feature = cosine(candidate cf-bpr, session cf centroid) reusing `load_track_cfbpr`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lgbm_new_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_lgbm_features.py tests/test_lgbm_new_features.py
git commit -m "feat(lgbm): add same-artist/same-album/session-CF features"
```

---

## Task 8: Inference config — Stage A(union) + Stage C(LGBM), no cross-encoder

**Files:**
- Create: `music-crs-baselines/config/190-union-lgbm-v5kto-blindA.yaml`
- Test: `tests/test_config_190.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_190.py
import yaml
def test_config_190_union_lgbm_no_crossencoder():
    cfg = yaml.safe_load(open(
        "music-crs-baselines/config/190-union-lgbm-v5kto-blindA.yaml"))
    assert cfg["retrieval_type"] == "wrrf_union_v1"
    assert cfg["reranker_type"] == "lgbm"            # the LambdaRank reranker
    assert "multimodal_cross_encoder" not in str(cfg)  # cross-encoder dropped
    assert cfg["retrieval_topk"] >= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_190.py -v`
Expected: FAIL — file missing.

- [ ] **Step 3: Create the config** (mirror `182-+lgbm-v5kto-blindA.yaml`; confirm the LGBM reranker_type key + `lgbm_model_path` from `lgbm_rerank.py` / config 182-+lgbm):

```yaml
lm_type: "OrRim123/recsys2026-b3-grpo-pilot-2026-05-13-qwen3b-v5-kto-merged"
lora_path: null
lora_max_rank: 32

retrieval_type: "wrrf_union_v1"
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Blind-A"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types: ["all_tracks"]
user_split_types: ["all_users"]
corpus_types: ["track_name", "artist_name", "album_name"]
cache_dir: "../experiments/cache"
device: "cuda"
attn_implementation: "sdpa"

retrieval_topk: 100
reranker_type: "lgbm"
lgbm_model_path: "<path to nb-72 trained model on Drive, matching config 182-+lgbm>"

response_prompt_name: "response_generation_cot_user_state"
response_max_new_tokens: 320
top_n_for_prompt: 1
query_preprocessing_mode: "bge_m3_structured"
use_vllm: false
use_state_tracker: true
state_tracker_prompt_name: "state_extraction"
state_tracker_max_new_tokens: 96
use_cmqr: false
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_190.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/config/190-union-lgbm-v5kto-blindA.yaml tests/test_config_190.py
git commit -m "feat(config): 190 union + lgbm pipeline (cross-encoder dropped)"
```

---

## Task 9: Retrain the LGBM aggregator on the new features (Colab, nb 72)

Orchestration task — run on Colab; validated by output.

- [ ] **Step 1:** In nb 72, point the feature build at the union (`wrrf_union_v1`) and ensure the feature builder passes `history_tids` so the new features populate. Run cell 3 (feature extraction) → cell 5 (train LambdaRank) → cell 6 (copy model to Drive). Re-run with the same `lgbm_model_path` used by config 190.

- [ ] **Step 2:** In nb 72 cell 7 (offline Stage A+C eval), confirm end-to-end dev nDCG@20. Inspect LightGBM feature importances — `same_artist` / `session_cf_cos` should rank highly; if a feature is near-zero importance and the CI of its removal excludes a drop, drop it.

- [ ] **Step 3: Gate G2 check.** PASS if dev nDCG@20 shows a clear lift over ~0.09-0.10 with a paired-bootstrap CI excluding 0 (reuse the `_paired_ci` helper from nb 71 cell 6). Commit the notebook.

```bash
git add colab/72_build_lgbm_features_train.ipynb
git commit -m "train: LGBM aggregator on union + session features"
```

---

## Task 10: Blind-A submission + gate G3

Orchestration task — run nb 73 (the submission notebook) with config 190.

- [ ] **Step 1:** Set nb 73 `TID = '190-union-lgbm-v5kto-blindA'`; run cells 1→5 (preflight enforces the LGBM model + embeddings are present).
- [ ] **Step 2:** Submit the zip to CodaBench; log scores via `scripts/blind_a_score_tracker.py append`.
- [ ] **Step 3: Gate G3 check.** PASS if Blind-A nDCG@20 > 0.09. Record the per-axis composite.

```bash
git add colab/73_run_blindset_retrieval_v2.ipynb
git commit -m "submit: Blind-A via config 190 (union + lgbm)"
```

---

## Notes for the executor

- Run new channels through the corrected dev eval (nb 71 cell 6/7) before wiring into the config; each channel must earn its place (recall lift, CI-backed).
- `dense_metadata_qwen3` is the frozen Qwen channel — verify its constructor/embeddings exist before Task 4; if its key differs, adjust the sub_spec `type`.
- Confirm the LGBM `reranker_type` string + model-path key by reading `mcrs/rerankers/lgbm_rerank.py` and `config/182-+lgbm-v5kto-blindA.yaml` before Task 8.
- Keep the cross-encoder retired: do not add its score as a feature in this plan (per the approved spec).
