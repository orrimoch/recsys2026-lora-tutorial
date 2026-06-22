# K2 Chat & Session Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add behavioral/session and assistant↔user-chat-derived features to K2 (the LightGBM reranker) to lift final-turn nDCG@20, since the label space is exhausted and features are the largest untapped surface.

**Architecture:** Two feature families. (1) Pure history/chat features computed inside `FeatureBuilder.build` from causal `TurnContext` fields (`history_tids`, `utterances`, `goal`) + catalog metadata — unit-testable, leak-free by construction. (2) Embedding-similarity features (chat-vs-candidate, session-vs-candidate) injected as `score_fns` closures from the notebook, exactly like the existing `dense_cos` (frozen encoder → leak-free, auto-normalized per turn). No model-derived/trained feature is added, so no OOF is required in this plan.

**Tech Stack:** Python 3.10, LightGBM LambdaRank, NumPy, sentence-transformers (BGE encoder already loaded in the notebook), pytest.

## Global Constraints

- All features MUST be causal: use only `ctx.utterances` (turns 1..t), `ctx.history_tids` (plays up to t), `ctx.goal`, and frozen catalog/encoders. Never the gold or any future turn. (Enforced structurally — `TurnContext.__post_init__` already guards utterance causality.)
- Every new feature MUST be a per-candidate value that can differ within a group. Group-constant features (same value for all candidates of one turn) give LambdaRank no ordering signal and MUST NOT be added as standalone columns.
- New feature names are APPENDED to `FeatureBuilder.feature_names` (never reordered) — the saved model pins `feature_names_` and `rerank()` enforces train==serve parity.
- Any notebook that SERVES K2 (`nb/phase3_blindA_submission.ipynb`) MUST build the identical `score_fns` set, or the `rerank()` feature-parity guard hard-fails at serve.
- Metadata fields are LIST-valued (`artist_name`, `track_name`, `tag_list`). Always coerce with the existing `_artists()` / `_first()` helpers.
- Gate every change on the `GATE final-turn (Blind-A proxy)` nDCG@20 line in `nb/phase2_rerank.ipynb`, never all-turns.
- Tests run in the base conda env where `lightgbm` is installed: `python -m pytest tests/test_lgbm_reranker.py tests/test_features.py -q`.

---

## File Structure

- `mcrs/rerank/features.py` — MODIFY `FeatureBuilder.__init__` (append feature names) and `FeatureBuilder.build` (compute new pure features). Add one private helper `_chat_text(ctx)`.
- `tests/test_features.py` — CREATE (no dedicated FeatureBuilder test file exists today; behavioral feature assertions live here).
- `nb/phase2_rerank.ipynb` — MODIFY cell `cell-12` (train): add `chat_doc_cos` + `session_emb_cos` to `score_fns`; MODIFY `cell-14` if needed (eval reuses the same `fb`, so usually no change).
- `nb/phase3_blindA_submission.ipynb` — MODIFY the serve-side `score_fns` construction to mirror cell-12 (parity).

**Pre-req gate:** Step 2 (Run A binary baseline) from the prior stage MUST be recorded first — its number is the comparison point and its top-10 feature importances tell us which families to prioritize. Do not start Task 1 until Run A's `GATE final-turn` nDCG@20 is known.

---

### Task 1: Behavioral history features (replay + artist affinity)

Adds four per-candidate features from `ctx.history_tids` + catalog: `is_replay`, `artist_play_count`, `last_artist_match`, `artist_recency`. Rationale: music sessions are repeat- and artist-sticky; a simple in-session repeat/artist signal beats SOTA sequence models in sequential music rec (arxiv 2409.04329), and same-artist affinity is the one label signal that ever showed promise. These richen the existing binary `artist_in_history`.

**Files:**
- Modify: `mcrs/rerank/features.py` (`FeatureBuilder.__init__` feature_names list ~lines 53-65; `FeatureBuilder.build` ~lines 75-113)
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `TurnContext.history_tids: list[str]`, `catalog.metadata(tid) -> dict`, existing `_artists(meta) -> set`.
- Produces: feature keys `is_replay`, `artist_play_count`, `last_artist_match`, `artist_recency` in every `Candidate.features`, and the same four names appended to `FeatureBuilder.feature_names`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
from mcrs.contracts import Candidate, TurnContext, UserProfile
from mcrs.rerank.features import FeatureBuilder
from mcrs.data.catalog import Catalog


def _ctx(history, utterances=None, goal=None, turn=None):
    turn = turn if turn is not None else (len(utterances) if utterances else 1)
    utterances = utterances if utterances is not None else ["q"] * turn
    return TurnContext("s", "u", turn, utterances, goal,
                       UserProfile("u", 1, "f", "US", []), list(history), "warm")


def _cat(rows):
    return Catalog(rows)


def test_behavioral_history_features():
    # history: A1 (artist A), B1 (artist B), A2 (artist A, most recent)
    cat = _cat([
        {"track_id": "A1", "artist_name": ["A"]},
        {"track_id": "B1", "artist_name": ["B"]},
        {"track_id": "A2", "artist_name": ["A"]},
        {"track_id": "A3", "artist_name": ["A"]},   # candidate by artist A, never played
        {"track_id": "B1", "artist_name": ["B"]},
        {"track_id": "C9", "artist_name": ["C"]},   # candidate, unrelated artist
    ])
    fb = FeatureBuilder(cat, ["bm25"])
    ctx = _ctx(history=["A1", "B1", "A2"])
    cands = [Candidate("A2"), Candidate("A3"), Candidate("B1"), Candidate("C9")]
    fb.build(ctx, cands)
    f = {c.track_id: c.features for c in cands}
    # is_replay: only A2 and B1 are in history
    assert f["A2"]["is_replay"] == 1.0 and f["B1"]["is_replay"] == 1.0
    assert f["A3"]["is_replay"] == 0.0 and f["C9"]["is_replay"] == 0.0
    # artist_play_count: A-tracks share artist with 2 history plays (A1,A2); B with 1; C with 0
    assert f["A3"]["artist_play_count"] == 2.0
    assert f["B1"]["artist_play_count"] == 1.0
    assert f["C9"]["artist_play_count"] == 0.0
    # last_artist_match: last play is A2 (artist A) -> A-tracks match, others don't
    assert f["A3"]["last_artist_match"] == 1.0 and f["C9"]["last_artist_match"] == 0.0
    assert f["B1"]["last_artist_match"] == 0.0
    # artist_recency: A last appeared at distance 0 (A2 is last) -> 1.0; B at distance 1 -> 0.5; C -> 0
    assert f["A3"]["artist_recency"] == 1.0
    assert f["B1"]["artist_recency"] == 0.5
    assert f["C9"]["artist_recency"] == 0.0


def test_behavioral_features_in_feature_names_and_empty_history_safe():
    cat = _cat([{"track_id": "x", "artist_name": ["Z"]}])
    fb = FeatureBuilder(cat, ["bm25"])
    for name in ("is_replay", "artist_play_count", "last_artist_match", "artist_recency"):
        assert name in fb.feature_names
    ctx = _ctx(history=[])                      # cold turn, no history
    cands = [Candidate("x")]
    fb.build(ctx, cands)
    f = cands[0].features
    assert f["is_replay"] == 0.0 and f["artist_play_count"] == 0.0
    assert f["last_artist_match"] == 0.0 and f["artist_recency"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -q`
Expected: FAIL with `KeyError: 'is_replay'` (feature not computed yet).

- [ ] **Step 3: Append the four feature names**

In `mcrs/rerank/features.py`, `FeatureBuilder.__init__`, extend the `self.feature_names` list. Add this line right after the existing `"popularity_percentile", "recency"]` group (before `+ self.score_names`):

```python
            # session/behavioral: in-session replay + artist affinity (causal, leak-free)
            + ["is_replay", "artist_play_count", "last_artist_match", "artist_recency"]
```

- [ ] **Step 4: Compute the features in `build`**

In `FeatureBuilder.build`, after the existing `hist_artists` loop (~line 79) add a per-position history artist list and a history set:

```python
        hist_set = set(ctx.history_tids)
        hist_artist_sets = [
            (_artists(self.catalog.metadata(h)) if (self.catalog is not None and h in self.catalog) else set())
            for h in ctx.history_tids
        ]
        last_artists = hist_artist_sets[-1] if hist_artist_sets else set()
```

Then inside the `for c in candidates:` loop, after the existing `f = {...}` dict is assigned (after line 98), add:

```python
            cand_artists = _artists(meta)
            f["is_replay"] = 1.0 if c.track_id in hist_set else 0.0
            f["artist_play_count"] = float(sum(1 for hs in hist_artist_sets if hs & cand_artists))
            f["last_artist_match"] = 1.0 if (cand_artists & last_artists) else 0.0
            rec = 0.0
            for dist, hs in enumerate(reversed(hist_artist_sets)):
                if hs & cand_artists:
                    rec = 1.0 / (1.0 + dist)
                    break
            f["artist_recency"] = rec
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py -q`
Expected: PASS (both tests).

- [ ] **Step 6: Run the full reranker suite (no regressions)**

Run: `python -m pytest tests/test_lgbm_reranker.py tests/test_features.py -q`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add mcrs/rerank/features.py tests/test_features.py
git commit -m "feat(k2): behavioral history features (replay + artist affinity)"
```

---

### Task 2: Chat-mention features (assistant↔user dialogue signals)

Adds two per-candidate features from the conversation text: `artist_mentioned_in_chat`, `track_mentioned_in_chat`. Rationale: the user's idea — the dialogue often names the artist/track the user wants ("play something like Radiohead"), a strong per-candidate relevance cue the rank/popularity features can't express. Per-candidate (differs within a group), causal (chat is available at serve), leak-free.

**Files:**
- Modify: `mcrs/rerank/features.py` (`FeatureBuilder.__init__` feature_names; `FeatureBuilder.build`; add `_chat_text` helper)
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `TurnContext.utterances: list[str]`, `TurnContext.goal: Optional[str]`, `catalog.metadata(tid)` for `artist_name`/`track_name`, existing `_first()`.
- Produces: feature keys `artist_mentioned_in_chat`, `track_mentioned_in_chat`; same names appended to `feature_names`; a private `FeatureBuilder._chat_text(ctx) -> str` returning the lowercased concatenation of utterances + goal.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py  (append)
def test_chat_mention_features():
    cat = _cat([
        {"track_id": "rh", "artist_name": ["Radiohead"], "track_name": ["Creep"]},
        {"track_id": "ot", "artist_name": ["Other Band"], "track_name": ["Unrelated"]},
    ])
    fb = FeatureBuilder(cat, ["bm25"])
    ctx = _ctx(history=[], utterances=["play something like radiohead please"], goal="upbeat creep vibes")
    cands = [Candidate("rh"), Candidate("ot")]
    fb.build(ctx, cands)
    f = {c.track_id: c.features for c in cands}
    assert f["rh"]["artist_mentioned_in_chat"] == 1.0     # "radiohead" appears in utterance
    assert f["ot"]["artist_mentioned_in_chat"] == 0.0
    assert f["rh"]["track_mentioned_in_chat"] == 1.0      # "creep" appears in goal text
    assert f["ot"]["track_mentioned_in_chat"] == 0.0


def test_chat_mention_short_name_not_false_matched_and_no_goal_safe():
    # short artist "U2" (len<3) must NOT substring-match random text; missing goal must not crash
    cat = _cat([{"track_id": "u2", "artist_name": ["U2"], "track_name": ["One"]}])
    fb = FeatureBuilder(cat, ["bm25"])
    ctx = _ctx(history=[], utterances=["i want a quiet untune evening"], goal=None)
    cands = [Candidate("u2")]
    fb.build(ctx, cands)
    f = cands[0].features
    assert f["artist_mentioned_in_chat"] == 0.0   # "u2" guarded by min-length, no false hit on "untune"
    assert "track_mentioned_in_chat" in f          # no crash with goal=None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -q`
Expected: FAIL with `KeyError: 'artist_mentioned_in_chat'`.

- [ ] **Step 3: Add the `_chat_text` helper**

In `mcrs/rerank/features.py`, add a method to `FeatureBuilder` (place it just above `build`):

```python
    @staticmethod
    def _chat_text(ctx: TurnContext) -> str:
        """Lowercased concatenation of the causal dialogue (utterances 1..t) + goal text."""
        parts = list(ctx.utterances) + ([ctx.goal] if ctx.goal else [])
        return " ".join(parts).lower()
```

- [ ] **Step 4: Append the two feature names**

In `FeatureBuilder.__init__`, extend the same behavioral group added in Task 1:

```python
            # chat-derived: does the dialogue name the candidate's artist/track (causal, leak-free)
            + ["artist_mentioned_in_chat", "track_mentioned_in_chat"]
```

- [ ] **Step 5: Compute the features in `build`**

In `FeatureBuilder.build`, before the `for c in candidates:` loop, precompute the chat text once:

```python
        chat = self._chat_text(ctx)
```

Inside the loop, after the Task-1 block, add:

```python
            cand_track_names = meta.get("track_name") or []
            cand_track_names = cand_track_names if isinstance(cand_track_names, list) else [cand_track_names]
            f["artist_mentioned_in_chat"] = 1.0 if any(
                len(a) >= 3 and a.lower() in chat for a in cand_artists) else 0.0
            f["track_mentioned_in_chat"] = 1.0 if any(
                len(str(t)) >= 3 and str(t).lower() in chat for t in cand_track_names) else 0.0
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py -q`
Expected: PASS (all tests).

- [ ] **Step 7: Run the full reranker suite**

Run: `python -m pytest tests/test_lgbm_reranker.py tests/test_features.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add mcrs/rerank/features.py tests/test_features.py
git commit -m "feat(k2): chat-mention features (artist/track named in dialogue)"
```

---

### Task 3: Embedding-similarity features (chat-vs-candidate, session-vs-candidate)

Adds two `score_fns` injected from the notebook: `chat_doc_cos` (cosine between a mean-pooled full-conversation embedding and the candidate's doc embedding) and `session_emb_cos` (cosine between the mean embedding of in-session played tracks and the candidate's doc embedding). Rationale: `dense_cos` uses the QueryBuilder query (often recency-focused); a full-conversation vector and a "what's been played" vector capture intent and session vibe that a single query misses. These follow the exact `dense_cos` pattern (frozen BGE encoder + `doc_mat`), so they are leak-free and auto-get `_norm` columns.

**Files:**
- Modify: `nb/phase2_rerank.ipynb` cell `cell-12` (add the two closures to `score_fns`)
- Modify: `nb/phase3_blindA_submission.ipynb` (mirror the two closures in the serve-side `score_fns` — parity)
- Test: `tests/test_features.py` (a unit test that `score_fns` flow through to `feature_names` + `_norm`, using a stub fn — the closures themselves are notebook-level and validated by the smoke run)

- [ ] **Step 1: Write the failing test (score_fn plumbing for the new names)**

```python
# tests/test_features.py  (append)
def test_injected_chat_and_session_score_fns_get_columns_and_norm():
    cat = _cat([{"track_id": "x", "artist_name": ["Z"]}])
    fb = FeatureBuilder(cat, ["bm25"], score_fns={
        "chat_doc_cos": lambda ctx, tid: 0.7,
        "session_emb_cos": lambda ctx, tid: 0.2,
    })
    for name in ("chat_doc_cos", "session_emb_cos", "chat_doc_cos_norm", "session_emb_cos_norm"):
        assert name in fb.feature_names
    ctx = _ctx(history=[])
    cands = [Candidate("x")]
    fb.build(ctx, cands)
    assert cands[0].features["chat_doc_cos"] == 0.7
    assert cands[0].features["session_emb_cos"] == 0.2
```

- [ ] **Step 2: Run test to verify it passes immediately**

Run: `python -m pytest tests/test_features.py::test_injected_chat_and_session_score_fns_get_columns_and_norm -q`
Expected: PASS (the `score_fns` mechanism already exists; this test pins that the new names are first-class and locks the contract Task-3 notebook code relies on). If it fails, the `score_fns` wiring regressed — stop and fix `features.py`.

- [ ] **Step 3: Add the closures in `nb/phase2_rerank.ipynb` cell-12**

In cell-12, immediately AFTER the existing `def dense_cos(ctx, tid): ...` definition and BEFORE `score_fns={'dense_cos': dense_cos}`, add:

```python
# full-conversation intent vector vs candidate doc (complements dense_cos's focused query)
_chatcache={}
def _chat_vec(ctx):
    key=(ctx.session_id,ctx.turn_number)
    v=_chatcache.get(key)
    if v is None:
        text=' '.join(list(ctx.utterances)+([ctx.goal] if ctx.goal else [])) or ' '
        v=model.encode([DENSE_QUERY_PREFIX+text], normalize_embeddings=True)[0]
        _chatcache[key]=v
    return v
def chat_doc_cos(ctx, tid):
    j=cat.id_to_index.get(tid)
    if j is None: return 0.0
    return float(_chat_vec(ctx) @ doc_mat[j])

# session vibe: mean doc-embedding of in-session played tracks vs candidate doc
import numpy as _np
def _sess_vec(ctx):
    idx=[cat.id_to_index[h] for h in ctx.history_tids if h in cat.id_to_index]
    if not idx: return None
    m=doc_mat[idx].mean(0); n=_np.linalg.norm(m)
    return m/n if n>0 else None
def session_emb_cos(ctx, tid):
    j=cat.id_to_index.get(tid)
    if j is None: return 0.0
    sv=_sess_vec(ctx)
    return float(sv @ doc_mat[j]) if sv is not None else 0.0
```

Then change the `score_fns` line to include them:

```python
score_fns={'dense_cos': dense_cos, 'chat_doc_cos': chat_doc_cos, 'session_emb_cos': session_emb_cos}
```

Note: the `_eval_set` helper in cell-14 calls `_qcache.clear()` per split; add `_chatcache.clear()` next to it so the chat cache never serves a stale split's vectors:

In cell-14, find `_qcache.clear(); precompute_qvecs(turns)` and change to:

```python
    _qcache.clear(); _chatcache.clear(); precompute_qvecs(turns)
```

- [ ] **Step 4: Mirror the closures in `nb/phase3_blindA_submission.ipynb`**

Find the serve-side `score_fns` construction (the cell that builds `FeatureBuilder` for the blind harness, mirroring phase2's `dense_cos`). Add the identical `chat_doc_cos` and `session_emb_cos` closures (same code, using that notebook's `model`, `cat`, `doc_mat`, `DENSE_QUERY_PREFIX`) and include them in `score_fns`. This is REQUIRED — without it the loaded K2's `feature_names_` will include `chat_doc_cos`/`session_emb_cos`/their `_norm`, but the serve builder won't produce them, and `LGBMReranker.rerank()`'s `assert_feature_parity()` will raise `ValueError` naming the missing columns.

- [ ] **Step 5: Smoke-validate parity locally (no GPU)**

Run this guard script to confirm the train-side and serve-side feature specs would match (catches a missed phase3 edit before a Colab run):

```bash
python - <<'PY'
import re
tr=open('nb/phase2_rerank.ipynb').read()
sv=open('nb/phase3_blindA_submission.ipynb').read()
for fn in ('chat_doc_cos','session_emb_cos'):
    assert f"'{fn}'" in tr or f'"{fn}"' in tr, f'{fn} missing in phase2 score_fns'
    assert f"'{fn}'" in sv or f'"{fn}"' in sv, f'{fn} missing in phase3 (serve) score_fns -> parity guard WILL fail'
print('parity OK: chat_doc_cos + session_emb_cos present in both notebooks')
PY
```

Expected: `parity OK: ...`. If it asserts, Step 4 was missed.

- [ ] **Step 6: Commit**

```bash
git add nb/phase2_rerank.ipynb nb/phase3_blindA_submission.ipynb tests/test_features.py
git commit -m "feat(k2): chat/session embedding-cosine features (train + serve parity)"
```

---

### Task 4: Gate run + feature triage

Run the new spine on Colab, read the final-turn gate and feature importances, and KEEP only features that help (drop dead weight to avoid overfdit and serve cost). This is an experiment task; its deliverable is a recorded decision committed as a notebook-config note.

**Files:**
- Modify: `nb/phase2_rerank.ipynb` (config note recording which features were kept/dropped)

- [ ] **Step 1: Push and pull**

```bash
git push origin fresh-start
```
On Colab: re-run the clone/install cell (does `git pull`), restart per the OOM-hygiene cell.

- [ ] **Step 2: Run the gate (winning objective from Stage-2 Run A/B)**

Set `K2_OBJECTIVE` to whichever won Stage 2. Run cells: data/channels → ColBERT → train (cell-12) → eval (cell-14).
Record: `GATE final-turn (Blind-A proxy)` nDCG@20 and the top-15 feature importances.

- [ ] **Step 3: Compare to the binary baseline**

Decision rule: keep the new feature set only if final-turn nDCG@20 > the Stage-2 Run A baseline by more than run-to-run noise (~±0.001). Inspect importances: any new feature with importance 0 (never split on) is dead — note it for removal. If a whole family (e.g. `session_emb_cos`) is dead AND the gate didn't move, remove that family in a follow-up commit to cut serve cost.

- [ ] **Step 4: Commit the decision**

Update the cell-12 (or a markdown cell) with a dated note: which families were kept, the gate delta vs binary baseline, and which (if any) were dropped. Commit:

```bash
git add nb/phase2_rerank.ipynb
git commit -m "exp(k2): chat/session feature gate result + triage note"
git push origin fresh-start
```

---

## Self-Review

**Spec coverage:**
- Behavioral/session features (research Tier-1) → Task 1 (replay, artist affinity) + Task 3 (`session_emb_cos`). ✓
- Chat-based aggregated features (user's idea) → Task 2 (artist/track mention) + Task 3 (`chat_doc_cos`). ✓
- Per-channel rank/source features → already present (`rank_inv__{l}`, `n_channels_hit`, `n_channels_top10`); not re-added (would duplicate). Noted, no task needed.
- Leak-safety / OOF guardrail → all features are non-model-derived (pure functions + frozen encoders), so OOF is explicitly not required this plan; the constraint is documented in Global Constraints. ✓
- Final-turn-only gating → Task 4 + Global Constraints. ✓
- Train==serve parity → Task 3 Steps 4-5 (phase3 mirror + smoke guard). ✓

**Placeholder scan:** No TBD/TODO; every code step shows the exact code; every run step shows the command + expected output. ✓

**Type consistency:** Feature key names are identical across `__init__` appends, `build` assignments, and tests (`is_replay`, `artist_play_count`, `last_artist_match`, `artist_recency`, `artist_mentioned_in_chat`, `track_mentioned_in_chat`, `chat_doc_cos`, `session_emb_cos`). `_chat_text` defined in Task 2 Step 3, used in Task 2 Step 5. ✓

**Open design choices (resolved with defaults; revisit after Task 4 numbers):**
- Mention match is normalized substring with a min-length≥3 guard (cheap, some false positives). If importances show it noisy, upgrade to word-boundary/token match.
- `session_emb_cos` pools over the doc (text) embedding space (same `doc_mat` as `dense_cos`); an audio/CF-space variant is a possible follow-up if the text-space version helps.
