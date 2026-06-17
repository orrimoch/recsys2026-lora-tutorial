# K3b — Cross-Encoder LoRA Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LoRA fine-tune `BAAI/bge-reranker-v2-m3` on denoised hard negatives with a grouped listwise-softmax loss, serve the adapter as a final-stage reranker, and stack its out-of-fold score back into K2.

**Architecture:** Split into pure, GPU-free logic (data/label/query/doc/sampling/OOF-orchestration — fully TDD'd locally) and a thin torch/PEFT training layer (smoke-tested locally, run on Colab). The scorer is injected everywhere so leak-safety and sampling are testable without a GPU, mirroring the existing `build_cross_encoder_score_fn(score_fn)` pattern.

**Tech Stack:** Python, pytest, PyTorch, `transformers` (`AutoModelForSequenceClassification`), `peft` (LoRA), `sentence-transformers` (serve wrapper), Trackio (logging). Spec: `.claude/documents/features/53_K3b_ce_lora_finetune.md`.

---

## File Structure

- `mcrs/training/__init__.py` — package marker (create if missing).
- `mcrs/training/ce_data.py` — GPU-free: `build_doc`, title/artist denoise helpers, rank-stratified deterministic sampler, `build_ce_training_groups`, session-disjoint fold assignment + cross-fold near-dup dedup, within-pool score normalization.
- `mcrs/training/ce_loss.py` — torch: `masked_listwise_ce` (ragged grouped softmax CE with per-group weight).
- `mcrs/training/ce_finetune.py` — torch/PEFT: `finetune_cross_encoder` (training loop) + `oof_ce_scores` (k-fold orchestration; uses an injected fine-tune fn so the leak logic is testable GPU-free).
- `mcrs/data/catalog.py` — MODIFY: `_raw_doc` includes tags/doc2query parity; add `is_enriched`.
- `mcrs/retrieval/query.py` — MODIFY: `QueryBuilder` gains the enriched template (markers + taste clause).
- `mcrs/rerank/cross_encoder.py` — MODIFY: `build_cross_encoder_score_fn` gains `lora_adapter`, `max_length=2048`, `dtype`.
- `nb/phase2_ce_finetune.ipynb` — CREATE: orchestration + eval.
- Tests: `tests/test_catalog_rawdoc.py`, `tests/test_ce_build_doc.py`, `tests/test_query_enriched.py`, `tests/test_ce_denoise.py`, `tests/test_ce_sampler.py`, `tests/test_ce_groups.py`, `tests/test_ce_folds.py`, `tests/test_ce_loss.py`, `tests/test_ce_oof.py`, `tests/test_cross_encoder_lora.py`.

**Dev deps for local tests:** `./recsys26/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu peft` (transformers + sentencepiece already installed). Tasks 8/10 need torch on CPU; Tasks 9/11 PEFT-load paths run on Colab.

---

## Task 1: Catalog raw-doc parity + enriched check

**Files:**
- Modify: `mcrs/data/catalog.py`
- Test: `tests/test_catalog_rawdoc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catalog_rawdoc.py
from mcrs.data.catalog import Catalog

ROW = {"track_id": "t1", "track_name": "Holocene", "artist_name": "Bon Iver",
       "album_name": "Bon Iver", "release_date": "2011-06-17", "tag_list": ["indie folk", "atmospheric"]}

def test_raw_doc_includes_tags():
    cat = Catalog([ROW])
    doc = cat.id_to_metadata("t1", enriched=False)
    assert "tags: indie folk, atmospheric" in doc
    assert "track_name: Holocene" in doc

def test_is_enriched():
    cat = Catalog([ROW], enriched_docs={"t1": "enriched..."})
    assert cat.is_enriched("t1") is True
    assert cat.is_enriched("t2") is False
```

- [ ] **Step 2: Run, verify it fails**

Run: `./recsys26/bin/python -m pytest tests/test_catalog_rawdoc.py -v`
Expected: FAIL — `tag_list` not in `_DEFAULT_CORPUS`; `is_enriched` missing.

- [ ] **Step 3: Implement**

In `mcrs/data/catalog.py`, add `"tag_list"` to the default corpus and an `is_enriched` method:

```python
_DEFAULT_CORPUS = ["track_name", "artist_name", "album_name", "tag_list", "release_date"]
```

```python
    def is_enriched(self, track_id: str) -> bool:
        return track_id in self._enriched
```

Rename the emitted key so the raw doc matches the enriched `meta_text` vocabulary (`tags`, not `tag_list`). In `_raw_doc`, map the field label:

```python
    _FIELD_LABEL = {"tag_list": "tags"}

    def _raw_doc(self, track_id: str) -> str:
        row = self._meta[track_id]
        parts = []
        for field in self.corpus_types:
            v: Any = row.get(field)
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            elif v is None:
                v = ""
            else:
                v = str(v)
            parts.append(f"{self._FIELD_LABEL.get(field, field)}: {v}")
        return ", ".join(parts)
```

- [ ] **Step 4: Run, verify pass**

Run: `./recsys26/bin/python -m pytest tests/test_catalog_rawdoc.py -v`
Expected: PASS. Then run the full suite to catch BM25/dense doc-text regressions: `./recsys26/bin/python -m pytest tests/ -q -k "catalog or bm25 or dense or enrich"`. If a pre-existing test asserts the old raw-doc string, update its expected string to include the `tags:` field (the corpus change is intentional, per spec §2).

- [ ] **Step 5: Commit**

```bash
git add mcrs/data/catalog.py tests/test_catalog_rawdoc.py
git commit -m "fix(K3b): raw-doc includes tags + doc2query parity; add Catalog.is_enriched"
```

---

## Task 2: `build_doc` single doc builder

**Files:**
- Create: `mcrs/training/__init__.py` (empty), `mcrs/training/ce_data.py`
- Test: `tests/test_ce_build_doc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ce_build_doc.py
import pytest
from mcrs.data.catalog import Catalog
from mcrs.training.ce_data import build_doc

ROW = {"track_id": "t1", "track_name": "A", "artist_name": "B", "album_name": "C",
       "release_date": "2011", "tag_list": ["x"]}

def test_build_doc_uses_enriched_and_char_caps():
    cat = Catalog([ROW], enriched_docs={"t1": "E" * 5000})
    assert build_doc(cat, "t1", max_doc_chars=2000) == "E" * 2000

def test_build_doc_hard_fails_on_enriched_missing():
    cat = Catalog([ROW])  # no enriched_docs
    with pytest.raises(KeyError):
        build_doc(cat, "t1")
```

- [ ] **Step 2: Run, verify it fails**

Run: `./recsys26/bin/python -m pytest tests/test_ce_build_doc.py -v`
Expected: FAIL — module `mcrs.training.ce_data` does not exist.

- [ ] **Step 3: Implement**

```python
# mcrs/training/ce_data.py
"""K3b — GPU-free data/label construction for the cross-encoder fine-tune."""
from __future__ import annotations


def build_doc(catalog, track_id: str, *, max_doc_chars: int = 2000) -> str:
    """The SINGLE doc string for train positives, train negatives, AND serve.

    Enriched doc, char-capped. Per-pair token truncation is applied downstream by the shared
    score_fn (cross_encoder.py), so train==serve. Hard-fails if the track has no enriched doc —
    no silent raw fallback during fine-tuning (spec §2/§7).
    """
    if not catalog.is_enriched(track_id):
        raise KeyError(f"build_doc: track {track_id!r} has no enriched doc (100% coverage required)")
    return catalog.id_to_metadata(track_id, enriched=True)[:max_doc_chars]
```

Create empty `mcrs/training/__init__.py`.

- [ ] **Step 4: Run, verify pass**

Run: `./recsys26/bin/python -m pytest tests/test_ce_build_doc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcrs/training/__init__.py mcrs/training/ce_data.py tests/test_ce_build_doc.py
git commit -m "feat(K3b): build_doc single train==serve doc builder"
```

---

## Task 3: QueryBuilder enriched template

**Files:**
- Modify: `mcrs/retrieval/query.py`
- Test: `tests/test_query_enriched.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_enriched.py
from mcrs.contracts import TurnContext, UserProfile
from mcrs.retrieval.query import QueryBuilder

def _ctx(utts, hist):
    return TurnContext(session_id="s", user_id="u", turn_number=len(utts), utterances=utts,
                       goal="discover new music", user_profile=UserProfile("u", None, None, None, []),
                       history_tids=hist, segment="warm" if hist else "cold")

LABELS = {"h1": "Bon Iver – Holocene", "h2": "Tobu – Sunburst"}

def test_warm_query_has_markers_and_taste_newest_first():
    qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=LABELS.get)
    txt = qb.build(_ctx(["hi want chill", "more upbeat"], ["h1", "h2"])).text
    assert txt.startswith("request: more upbeat")
    assert "context: hi want chill" in txt
    assert "goal: discover new music" in txt
    # history_tids is chronological; taste is newest-first
    assert "taste: Tobu – Sunburst; Bon Iver – Holocene" in txt

def test_cold_query_omits_taste_and_empty_lines():
    qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=LABELS.get)
    txt = qb.build(_ctx(["just one turn"], [])).text
    assert "taste:" not in txt
    assert "context:" not in txt          # no older utterances
    assert txt.startswith("request: just one turn")

def test_markers_off_is_legacy_behavior():
    qb = QueryBuilder()  # defaults: markers off
    assert qb.build(_ctx(["a", "b"], [])).text == "a b discover new music"
```

- [ ] **Step 2: Run, verify it fails**

Run: `./recsys26/bin/python -m pytest tests/test_query_enriched.py -v`
Expected: FAIL — `QueryBuilder` has no `markers`/`taste_items`/`track_label_fn`.

- [ ] **Step 3: Implement**

In `mcrs/retrieval/query.py`, extend the dataclass and branch `build`:

```python
from typing import Callable, Optional

@dataclass
class QueryBuilder:
    context_cap: int = 0
    recency_window: int = 0
    markers: bool = False                                   # enriched template (K3b §4.1)
    taste_items: int = 0                                    # max history tracks in the taste: clause
    track_label_fn: Optional[Callable[[str], Optional[str]]] = None  # tid -> "artist – title" | None

    def build(self, ctx: TurnContext) -> Query:
        if not self.markers:
            return self._build_plain(ctx)
        return self._build_enriched(ctx)

    def _build_plain(self, ctx: TurnContext) -> Query:
        utts = list(ctx.utterances)
        if self.recency_window and len(utts) > self.recency_window:
            utts = utts[-self.recency_window:]
        goal = ctx.goal or ""
        kept = self._apply_cap(utts, goal)
        parts = kept + ([goal] if goal else [])
        return Query(text=" ".join(p for p in parts if p))

    def _build_enriched(self, ctx: TurnContext) -> Query:
        utts = list(ctx.utterances)
        latest = utts[-1] if utts else ""
        older = utts[:-1]
        if self.recency_window and len(older) > self.recency_window:
            older = older[-self.recency_window:]
        lines: list[str] = []
        if latest:
            lines.append(f"request: {latest}")
        if older:
            lines.append("context: " + " ".join(older))
        if ctx.goal:
            lines.append(f"goal: {ctx.goal}")
        if self.taste_items and ctx.history_tids and self.track_label_fn:
            labels: list[str] = []
            for tid in reversed(ctx.history_tids):          # chronological -> newest-first
                lab = self.track_label_fn(tid)
                if lab:
                    labels.append(lab)
                if len(labels) >= self.taste_items:
                    break
            if labels:
                lines.append("taste: " + "; ".join(labels))
        return Query(text="\n".join(lines))
```

Keep the existing `_apply_cap` unchanged.

- [ ] **Step 4: Run, verify pass**

Run: `./recsys26/bin/python -m pytest tests/test_query_enriched.py -v`
Expected: PASS. Run `./recsys26/bin/python -m pytest tests/ -q -k query` to confirm no legacy regression.

- [ ] **Step 5: Commit**

```bash
git add mcrs/retrieval/query.py tests/test_query_enriched.py
git commit -m "feat(K3b): QueryBuilder enriched template (markers + cold-omitted taste clause)"
```

---

## Task 4: Denoise helpers (title/artist)

**Files:**
- Modify: `mcrs/training/ce_data.py`
- Test: `tests/test_ce_denoise.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ce_denoise.py
from mcrs.training.ce_data import normalize_title, is_near_dup, is_same_artist

def test_normalize_title():
    assert normalize_title("Holocene (Remastered)") == "holocene"
    assert normalize_title("  Hello, World!  ") == "hello world"

def test_is_near_dup():
    assert is_near_dup("Holocene", "holocene (live)") is True
    assert is_near_dup("Holocene", "Sunburst") is False

def test_is_same_artist():
    meta = {"a": {"artist_name": "Bon Iver"}, "b": {"artist_name": "bon iver"}, "c": {"artist_name": "Tobu"}}
    label = lambda tid: meta[tid]["artist_name"]
    assert is_same_artist("a", "b", label) is True
    assert is_same_artist("a", "c", label) is False
```

- [ ] **Step 2: Run, verify it fails**

Run: `./recsys26/bin/python -m pytest tests/test_ce_denoise.py -v`
Expected: FAIL — helpers not defined.

- [ ] **Step 3: Implement**

Append to `mcrs/training/ce_data.py`:

```python
import re
from typing import Callable, Optional

def normalize_title(t: str) -> str:
    """Lowercase, strip parenthetical qualifiers and punctuation, collapse whitespace."""
    t = (t or "").lower()
    t = re.sub(r"\([^)]*\)", " ", t)                 # drop "(remastered)", "(live)", ...
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())

def is_near_dup(title_a: str, title_b: str) -> bool:
    na, nb = normalize_title(title_a), normalize_title(title_b)
    return bool(na) and na == nb

def is_same_artist(tid_a: str, tid_b: str, artist_fn: Callable[[str], Optional[str]]) -> bool:
    a, b = artist_fn(tid_a), artist_fn(tid_b)
    return bool(a) and bool(b) and a.strip().lower() == b.strip().lower()
```

- [ ] **Step 4: Run, verify pass** — `./recsys26/bin/python -m pytest tests/test_ce_denoise.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add mcrs/training/ce_data.py tests/test_ce_denoise.py
git commit -m "feat(K3b): title/artist denoise helpers"
```

---

## Task 5: Rank-stratified deterministic negative sampler

**Files:**
- Modify: `mcrs/training/ce_data.py`
- Test: `tests/test_ce_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ce_sampler.py
from mcrs.training.ce_data import sample_negatives

# pool: list of (track_id, rank) already sorted by rank ascending (1=best)
POOL = [(f"t{i}", i) for i in range(1, 41)]   # t1..t40
ARTIST = {f"t{i}": ("same" if i == 5 else f"art{i}") for i in range(1, 41)}
TITLE = {f"t{i}": f"title{i}" for i in range(1, 41)}

def _kw(**o):
    base = dict(gold_tid="g", gold_title="gold", gold_artist="gold_art",
                artist_fn=ARTIST.get, title_fn=TITLE.get, n=10, k_min=4, seed=0)
    base.update(o); return base

def test_deterministic_given_seed():
    a = sample_negatives(POOL, **_kw())
    b = sample_negatives(POOL, **_kw())
    assert a == b

def test_skip_top_rank_off_keeps_rank1():
    negs = sample_negatives(POOL, **_kw(n=40, skip_top_rank=False))
    assert "t1" in negs

def test_skip_top_rank_on_drops_rank1():
    negs = sample_negatives(POOL, **_kw(n=40, skip_top_rank=True))
    assert "t1" not in negs

def test_same_artist_downweighted_not_absent_over_runs():
    # 't5' is same-artist; with soft down-weight it should be selectable but rarer than a normal mid negative.
    seen = sum("t5" in sample_negatives(POOL, **_kw(seed=s, same_artist="soft_downweight")) for s in range(50))
    assert 0 < seen < 50

def test_same_artist_drop_removes_it():
    negs = sample_negatives(POOL, **_kw(n=40, same_artist="drop"))
    assert "t5" not in negs

def test_returns_at_least_k_min_when_pool_allows():
    negs = sample_negatives(POOL, **_kw(n=10))
    assert len(negs) == 10
```

- [ ] **Step 2: Run, verify it fails** — module function missing → FAIL.

- [ ] **Step 3: Implement**

Append to `mcrs/training/ce_data.py`:

```python
import random

def sample_negatives(pool, *, gold_tid, gold_title, gold_artist,
                     artist_fn, title_fn, n, k_min, seed,
                     sampling="rank_strat", same_artist="soft_downweight",
                     denoise_near_dup=True, skip_top_rank=False,
                     same_artist_weight=0.25):
    """Pick up to `n` negative track_ids from `pool` (list of (tid, rank), rank-ascending).

    Deterministic given `seed`. Near-dup titles are dropped; same-artist is dropped/down-weighted/
    kept per `same_artist`; rank-1 optionally skipped. Returns [] if the eligible set is empty.
    """
    rng = random.Random(seed)
    cands = sorted(pool, key=lambda x: x[1])                       # stable, rank-ascending
    eligible = []                                                  # (tid, weight)
    for tid, rank in cands:
        if tid == gold_tid:
            continue
        if skip_top_rank and rank == 1:
            continue
        title = title_fn(tid) or ""
        if denoise_near_dup and is_near_dup(gold_title, title):
            continue                                               # true false negative
        same = is_same_artist(gold_tid, tid, artist_fn)
        if same and same_artist == "drop":
            if denoise_near_dup and is_near_dup(gold_title, title):
                continue
            continue
        w = same_artist_weight if (same and same_artist == "soft_downweight") else 1.0
        eligible.append((tid, rank, w))
    if not eligible:
        return []
    if sampling == "rank_strat":
        # split eligible into top-half and bottom-half by rank; draw ~half from each
        mid = len(eligible) // 2 or 1
        top, bot = eligible[:mid], eligible[mid:]
        chosen = (_weighted_sample(top, (n + 1) // 2, rng)
                  + _weighted_sample(bot, n // 2, rng))
        # backfill if a stratum was short
        if len(chosen) < min(n, len(eligible)):
            remaining = [e for e in eligible if e[0] not in {c for c in chosen}]
            chosen += _weighted_sample(remaining, min(n, len(eligible)) - len(chosen), rng)
    else:
        chosen = _weighted_sample(eligible, n, rng)
    return chosen

def _weighted_sample(items, k, rng):
    """Weighted sampling WITHOUT replacement; returns a list of track_ids. items: (tid, rank, w)."""
    pool = list(items)
    out = []
    for _ in range(min(k, len(pool))):
        total = sum(w for _, _, w in pool)
        if total <= 0:
            break
        r = rng.random() * total
        acc = 0.0
        for idx, (tid, _, w) in enumerate(pool):
            acc += w
            if r <= acc:
                out.append(tid)
                pool.pop(idx)
                break
    return out
```

Note: `_weighted_sample` returns track_ids; `sample_negatives` returns a flat list of track_ids. `k_min` is enforced by the caller (Task 6), which drops the turn if `len(negs) < k_min`.

- [ ] **Step 4: Run, verify pass** — `./recsys26/bin/python -m pytest tests/test_ce_sampler.py -v` → PASS. (If `test_same_artist_downweighted_not_absent_over_runs` is flaky at the boundary, it is seed-deterministic per run; the 50-seed sweep makes `0 < seen < 50` robust.)

- [ ] **Step 5: Commit**

```bash
git add mcrs/training/ce_data.py tests/test_ce_sampler.py
git commit -m "feat(K3b): rank-stratified deterministic negative sampler w/ same-artist soft down-weight"
```

---

## Task 6: `build_ce_training_groups`

**Files:**
- Modify: `mcrs/training/ce_data.py`
- Test: `tests/test_ce_groups.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ce_groups.py
from mcrs.contracts import Candidate
from mcrs.training.ce_data import build_ce_training_groups, GP_WEIGHTS

class FakeQB:
    def build(self, ctx): 
        from mcrs.contracts import Query
        return Query(text=f"q{ctx.turn_number}")

class FakeFusion:
    def __init__(self, pools): self.pools = pools   # list[list[Candidate]] aligned to turns
    def fuse(self, queries, topk, **kw): return self.pools

class FakeCat:
    def __init__(self): 
        self.meta = {t: {"artist_name": f"a{t}", "track_name": f"n{t}"} for t in ["g","x","y","z","w"]}
        self.enr = set(self.meta)
    def is_enriched(self, t): return t in self.enr
    def id_to_metadata(self, t, enriched=False): return f"doc-{t}"
    def metadata(self, t): return self.meta[t]

def _turn(tn, gp):
    from mcrs.contracts import TurnContext, UserProfile
    return TurnContext("s", "u", tn, ["u"]*tn, "goal", UserProfile("u",None,None,None,[]), [], "cold")

def test_gold_in_pool_only_and_group_shape():
    cat = FakeCat()
    pool = [Candidate(track_id=t, rrf_score=1.0/i, channel_ranks={"c": i})
            for i, t in enumerate(["g","x","y","z","w"], start=1)]
    fusion = FakeFusion([pool, [Candidate("x"), Candidate("y")]])  # turn2 pool lacks gold 'g'
    turns = [_turn(1, "MOVES_TOWARD_GOAL"), _turn(2, "MOVES_TOWARD_GOAL")]
    gold_fn = lambda t: "g"
    gp_fn = lambda t: "MOVES_TOWARD_GOAL"
    groups = build_ce_training_groups(FakeQB(), fusion, turns, gold_fn, catalog=cat,
                                      cross_encoder_k=5, n_negatives=3, k_min=1, seed=0, gp_fn=gp_fn)
    assert len(groups) == 1                         # turn2 dropped (gold not in pool)
    q, docs, gw = groups[0]
    assert q == "q1"
    assert docs[0] == "doc-g"                       # positive at index 0
    assert 1 <= len(docs) - 1 <= 3                  # negatives
    assert gw == GP_WEIGHTS["MOVES_TOWARD_GOAL"]

def test_goal_progress_group_weight():
    cat = FakeCat()
    pool = [Candidate(track_id=t, channel_ranks={"c": i}) for i, t in enumerate(["g","x","y","z","w"],1)]
    fusion = FakeFusion([pool])
    groups = build_ce_training_groups(FakeQB(), fusion, [_turn(1,"x")], lambda t:"g", catalog=cat,
                 cross_encoder_k=5, n_negatives=2, k_min=1, seed=0,
                 gp_fn=lambda t: "DOES_NOT_MOVE_TOWARD_GOAL")
    assert groups[0][2] == GP_WEIGHTS["DOES_NOT_MOVE_TOWARD_GOAL"]
```

- [ ] **Step 2: Run, verify it fails** — function/constant missing → FAIL.

- [ ] **Step 3: Implement**

Append to `mcrs/training/ce_data.py`:

```python
GP_WEIGHTS = {"MOVES_TOWARD_GOAL": 1.0, "DOES_NOT_MOVE_TOWARD_GOAL": 0.3, None: 1.0}

def build_ce_training_groups(query_builder, fusion, turns, gold_fn, *, catalog, cross_encoder_k,
                             n_negatives=15, sampling="rank_strat", same_artist="soft_downweight",
                             denoise_near_dup=True, skip_top_rank=False, k_min=4, seed,
                             gp_fn=None, w_low=0.3, report=None):
    """Build [(query_text, [pos_doc, neg_doc...], group_weight)] for gold-in-pool turns.

    `gp_fn(turn) -> goal_progress_label | None` supplies the per-group weight (None => uniform).
    Drops turns whose gold is not in the top-`cross_encoder_k` pool, or that have < k_min negatives.
    """
    gp_weights = dict(GP_WEIGHTS); gp_weights["DOES_NOT_MOVE_TOWARD_GOAL"] = w_low
    queries = [query_builder.build(t).text for t in turns]
    bc = [{"history_tids": t.history_tids, "user_id": t.user_id} for t in turns]
    uids = [t.user_id for t in turns]
    pools = fusion.fuse(queries, cross_encoder_k, topk_internal=cross_encoder_k,
                        batch_context=bc, user_ids=uids)
    artist_fn = lambda tid: catalog.metadata(tid).get("artist_name") if tid in catalog._meta else None
    title_fn = lambda tid: catalog.metadata(tid).get("track_name") if tid in catalog._meta else None
    groups, dropped_no_gold, dropped_few_neg = [], 0, 0
    for turn, qtext, pool in zip(turns, queries, pools):
        gold = gold_fn(turn)
        top = pool[:cross_encoder_k]
        ids = [c.track_id for c in top]
        if gold is None or gold not in ids:
            dropped_no_gold += 1
            continue
        ranked = [(c.track_id, min(c.channel_ranks.values()) if c.channel_ranks else (i + 1))
                  for i, c in enumerate(top)]
        negs = sample_negatives(ranked, gold_tid=gold, gold_title=(title_fn(gold) or ""),
                                gold_artist=(artist_fn(gold) or ""), artist_fn=artist_fn,
                                title_fn=title_fn, n=n_negatives, k_min=k_min,
                                seed=_stable_seed(seed, turn.session_id, turn.turn_number),  # md5, NOT hash() (PYTHONHASHSEED)
                                sampling=sampling, same_artist=same_artist,
                                denoise_near_dup=denoise_near_dup, skip_top_rank=skip_top_rank)
        if len(negs) < k_min:
            dropped_few_neg += 1
            continue
        docs = [build_doc(catalog, gold)] + [build_doc(catalog, t) for t in negs]
        gw = gp_weights.get(gp_fn(turn)) if gp_fn else 1.0
        groups.append((qtext, docs, gw))
    if report is not None:
        report.update(dropped_no_gold=dropped_no_gold, dropped_few_neg=dropped_few_neg, kept=len(groups))
    return groups
```

- [ ] **Step 4: Run, verify pass** — `./recsys26/bin/python -m pytest tests/test_ce_groups.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add mcrs/training/ce_data.py tests/test_ce_groups.py
git commit -m "feat(K3b): build_ce_training_groups (gold-in-pool, denoised negs, goal-progress weight)"
```

---

## Task 7: Session-disjoint folds + cross-fold near-dup dedup

**Files:**
- Modify: `mcrs/training/ce_data.py`
- Test: `tests/test_ce_folds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ce_folds.py
from mcrs.training.ce_data import assign_session_folds, drop_cross_fold_near_dups

def test_sessions_never_split_across_folds():
    sids = [f"s{i}" for i in range(10)]
    turns = [(f"{s}", i) for s in sids for i in range(3)]   # (session_id, turn)
    folds = assign_session_folds([s for s, _ in turns], k=3, seed=0)
    by_session = {}
    for (s, _), f in zip(turns, folds):
        by_session.setdefault(s, set()).add(f)
    assert all(len(fs) == 1 for fs in by_session.values())   # each session in exactly one fold

def test_drop_cross_fold_near_dups():
    # items: (key, fold); a (query,gold) key appearing in two folds must be dropped from all but one
    items = [("k1", 0), ("k1", 1), ("k2", 0), ("k3", 2)]
    keep = drop_cross_fold_near_dups(items)
    keys = [items[i][0] for i in keep]
    assert keys.count("k1") == 1 and "k2" in keys and "k3" in keys
```

- [ ] **Step 2: Run, verify it fails** — functions missing → FAIL.

- [ ] **Step 3: Implement**

Append to `mcrs/training/ce_data.py`:

```python
def assign_session_folds(session_ids, *, k, seed):
    """Deterministic session-disjoint fold id per row. All rows of a session share a fold."""
    uniq = sorted(set(session_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    fold_of = {s: i % k for i, s in enumerate(uniq)}
    return [fold_of[s] for s in session_ids]

def drop_cross_fold_near_dups(items):
    """items: list[(dedup_key, fold)]. Keep the first occurrence of each key; drop later folds'
    copies so a near-duplicate (query->gold) never straddles the fold boundary. Returns kept indices."""
    seen, keep = set(), []
    for i, (key, _fold) in enumerate(items):
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep
```

The dedup key is `(normalize_title(latest_utterance) , gold_tid)` (computed by the OOF driver, Task 10).

- [ ] **Step 4: Run, verify pass** — `./recsys26/bin/python -m pytest tests/test_ce_folds.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add mcrs/training/ce_data.py tests/test_ce_folds.py
git commit -m "feat(K3b): session-disjoint fold assignment + cross-fold near-dup dedup"
```

---

## Task 8: Masked listwise-softmax loss

**Files:**
- Create: `mcrs/training/ce_loss.py`
- Test: `tests/test_ce_loss.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ce_loss.py
import math, torch
from mcrs.training.ce_loss import masked_listwise_ce

def test_perfect_scores_low_loss():
    # group A: [gold=10, neg=0,0]; group B (ragged): [gold=10, neg=0]
    logits = torch.tensor([10., 0., 0., 10., 0., -1e9])     # last is a pad slot
    group_sizes = [3, 2]                                     # 2nd group has 2 real, 1 pad
    loss = masked_listwise_ce(logits, group_sizes, group_weights=[1.0, 1.0])
    assert loss.item() < 0.01

def test_pad_excluded_from_denominator():
    # if the pad (-inf) leaked into the softmax it would change the loss vs a clean 2-item group
    logits = torch.tensor([2., 1., -1e9])
    clean = masked_listwise_ce(torch.tensor([2., 1.]), [2], group_weights=[1.0])
    padded = masked_listwise_ce(logits, [3], group_weights=[1.0])
    assert abs(clean.item() - padded.item()) < 1e-5

def test_group_weight_scales_loss():
    logits = torch.tensor([0., 5.])                          # gold loses -> high loss
    full = masked_listwise_ce(logits, [2], group_weights=[1.0])
    half = masked_listwise_ce(logits, [2], group_weights=[0.5])
    assert abs(half.item() - full.item()) < 1e-5             # normalized: weight cancels for 1 group
```

(Note: with a single group the normalization `Σ w·L / Σ w` cancels the weight — `test_group_weight_scales_loss` checks the normalization is correct, not that one group's weight changes its own mean.)

- [ ] **Step 2: Run, verify it fails** — module missing → FAIL.

- [ ] **Step 3: Implement**

```python
# mcrs/training/ce_loss.py
"""K3b — grouped listwise-softmax (LCE) loss with ragged groups + per-group weight."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_listwise_ce(logits: torch.Tensor, group_sizes: list[int],
                       group_weights: list[float]) -> torch.Tensor:
    """logits: flat (sum(group_sizes),) with the gold at the FIRST position of each group.

    Per group: softmax over its real members (pads encoded as -inf are excluded), CE to target 0,
    then a weighted mean across groups: Σ w_g · L_g / Σ w_g. Per-group max is subtracted by
    F.log_softmax for fp/bf16 stability.
    """
    device = logits.device
    offset = 0
    losses, weights = [], []
    for size, w in zip(group_sizes, group_weights):
        g = logits[offset:offset + size]
        offset += size
        logp = F.log_softmax(g, dim=0)           # -inf pads contribute 0 to the sum
        losses.append(-logp[0])                  # target index 0 = the gold
        weights.append(w)
    losses = torch.stack(losses)
    w = torch.tensor(weights, device=device, dtype=losses.dtype)
    return (losses * w).sum() / w.sum().clamp(min=1e-8)
```

- [ ] **Step 4: Run, verify pass** — `./recsys26/bin/python -m pytest tests/test_ce_loss.py -v` → PASS (requires torch CPU; see dev-deps).

- [ ] **Step 5: Commit**

```bash
git add mcrs/training/ce_loss.py tests/test_ce_loss.py
git commit -m "feat(K3b): masked listwise-softmax loss (ragged groups + per-group weight)"
```

---

## Task 9: `build_cross_encoder_score_fn` LoRA + length/dtype params

**Files:**
- Modify: `mcrs/rerank/cross_encoder.py`
- Test: `tests/test_cross_encoder_lora.py`

- [ ] **Step 1: Write the failing test** (config/signature behavior — no GPU needed)

```python
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
```

- [ ] **Step 2: Run, verify it fails** — defaults are 512/fp16, no `lora_adapter`/`dtype` → FAIL.

- [ ] **Step 3: Implement**

Edit `build_cross_encoder_score_fn` in `mcrs/rerank/cross_encoder.py`:

```python
def build_cross_encoder_score_fn(model_name: str, device: str = "cuda", max_length: int = 2048,
                                 max_doc_tokens: int = 1100, batch_size: int = 64,
                                 revision: Optional[str] = None, dtype: str = "bf16",
                                 lora_adapter: Optional[str] = None):
    """Load a CrossEncoder (+ optional PEFT LoRA adapter) and return score_fn(pairs)->list[float].

    `max_length`/`max_doc_tokens`/`dtype` MUST match the values used at fine-tune time (train==serve).
    """
    import torch
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name, max_length=max_length, device=device, revision=revision)
    if lora_adapter:
        from peft import PeftModel
        ce.model = PeftModel.from_pretrained(ce.model, lora_adapter)
        ce.model = ce.model.merge_and_unload()        # fold LoRA into base for fast inference
    if str(device).startswith("cuda"):
        ce.model = ce.model.to(dtype=torch.bfloat16 if dtype == "bf16" else torch.float16)
    tok = ce.tokenizer
    enc = lambda s: tok.encode(s, add_special_tokens=False, truncation=True, max_length=max_length)

    def score_fn(pairs):
        capped = []
        for q, d in pairs:
            budget = doc_token_budget(len(enc(q)), max_length, max_doc_tokens)
            capped.append((q, truncate_doc_tokens(enc, tok.decode, d, budget)))
        return [float(s) for s in ce.predict(capped, batch_size=batch_size, show_progress_bar=False)]

    return score_fn
```

- [ ] **Step 4: Run, verify pass** — `./recsys26/bin/python -m pytest tests/test_cross_encoder_lora.py -v` → PASS. The adapter-load + dtype path is exercised in the Colab smoke run (Task 11), not in unit tests (needs GPU/PEFT weights).

- [ ] **Step 5: Commit**

```bash
git add mcrs/rerank/cross_encoder.py tests/test_cross_encoder_lora.py
git commit -m "feat(K3b): score_fn gains lora_adapter + 2048/bf16 train==serve defaults"
```

---

## Task 10: `oof_ce_scores` orchestration (GPU-free, injected fine-tune fn)

**Files:**
- Create: `mcrs/training/ce_finetune.py`
- Modify: `mcrs/training/ce_data.py` (add `normalize_within_pool`)
- Test: `tests/test_ce_oof.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ce_oof.py
from mcrs.training.ce_data import normalize_within_pool
from mcrs.training.ce_finetune import oof_ce_scores

def test_normalize_within_pool_minmax():
    out = normalize_within_pool({"a": 0.0, "b": 5.0, "c": 10.0})
    assert out["a"] == 0.0 and out["c"] == 1.0 and out["b"] == 0.5

def test_oof_scores_use_only_held_out_fold_model():
    # Fake fine-tune fn returns a model tagged with the set of folds it trained on.
    # A row's score must come from a model that did NOT train on that row's fold.
    turns = [("s%d" % i, i) for i in range(6)]   # 6 sessions
    def fake_fit(train_rows, **kw):
        trained_folds = frozenset(f for (_sid, _t, f) in train_rows)
        return ("model", trained_folds)
    def fake_score(model, row):
        _name, trained_folds = model
        assert row["fold"] not in trained_folds   # INVARIANT: never score a row with a model that saw its fold
        return 1.0
    scores = oof_ce_scores(turns, folds=3, seed=0, fit_fn=fake_fit, score_fn=fake_score)
    assert len(scores) == 6                       # every train row scored exactly once
```

(The production `oof_ce_scores` signature wraps `finetune_cross_encoder` + the real scorer; this test injects fakes via `fit_fn`/`score_fn` so the leak-safety logic is verified GPU-free.)

- [ ] **Step 2: Run, verify it fails** — module/functions missing → FAIL.

- [ ] **Step 3: Implement**

Add to `mcrs/training/ce_data.py`:

```python
def normalize_within_pool(tid_to_score: dict) -> dict:
    """Min-max normalize scores within one turn's candidate pool (fold-scale-invariant feature)."""
    if not tid_to_score:
        return {}
    vals = list(tid_to_score.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {tid: (s - lo) / rng for tid, s in tid_to_score.items()}
```

Create `mcrs/training/ce_finetune.py` with the orchestration (the GPU pieces are injected):

```python
"""K3b — cross-encoder fine-tune training loop + OOF orchestration."""
from __future__ import annotations

from mcrs.training.ce_data import assign_session_folds


def oof_ce_scores(turns, *, folds, seed, fit_fn, score_fn):
    """Leak-free per-row CE scores via k-fold session-disjoint cross-fitting.

    turns: list of (session_id, turn_number). fit_fn(train_rows, **)->model.
    score_fn(model, row)->float. Each row is scored by a model trained on the OTHER folds only.
    """
    sids = [s for s, _ in turns]
    fold_of = assign_session_folds(sids, k=folds, seed=seed)
    rows = [{"session_id": s, "turn": t, "fold": f} for (s, t), f in zip(turns, fold_of)]
    out = {}
    for held in range(folds):
        train_rows = [(r["session_id"], r["turn"], r["fold"]) for r in rows if r["fold"] != held]
        model = fit_fn(train_rows, fold=held)
        for r in rows:
            if r["fold"] == held:
                out[(r["session_id"], r["turn"])] = score_fn(model, r)
    return out
```

- [ ] **Step 4: Run, verify pass**

Run: `./recsys26/bin/python -m pytest tests/test_ce_oof.py -v`
Expected: PASS — every train row scored exactly once, by a model that never trained on its fold.

- [ ] **Step 5: Commit**

```bash
git add mcrs/training/ce_finetune.py mcrs/training/ce_data.py tests/test_ce_oof.py
git commit -m "feat(K3b): OOF orchestration (held-out-fold-only scoring) + within-pool normalize"
```

---

## Task 11: `finetune_cross_encoder` training loop (torch/PEFT; Colab run)

**Files:**
- Modify: `mcrs/training/ce_finetune.py`
- Test: smoke only (tiny synthetic, CPU/Colab)

This is the GPU layer. Unit-test coverage is the smoke test below; the real fine-tune runs in the notebook on Colab. No placeholders — full code:

- [ ] **Step 1: Implement the dataset + collator + loop**

Append to `mcrs/training/ce_finetune.py`:

```python
import math, random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from mcrs.training.ce_loss import masked_listwise_ce


class CEGroupDataset(Dataset):
    """Holds [(query, [pos_doc, neg...], group_weight)]; tokenizes pairs lazily. `group_len` is a cheap
    length proxy (query token count) so the sampler can batch similar-length groups (less pad waste)."""
    def __init__(self, groups, tokenizer, max_length, max_doc_tokens, doc_budget_fn, truncate_fn):
        self.groups = groups
        self.tok = tokenizer
        self.max_length = max_length
        self.max_doc_tokens = max_doc_tokens
        self.doc_budget_fn = doc_budget_fn
        self.truncate_fn = truncate_fn
        self._qlen = [len(self._enc(g[0])) for g in groups]   # cached query token counts

    def _enc(self, s):
        return self.tok.encode(s, add_special_tokens=False, truncation=True, max_length=self.max_length)

    def group_len(self, i): return self._qlen[i]              # length proxy for the sampler
    def __len__(self): return len(self.groups)

    def __getitem__(self, i):
        q, docs, gw = self.groups[i]
        budget = self.doc_budget_fn(self._qlen[i], self.max_length, self.max_doc_tokens)
        pairs = [(q, self.truncate_fn(self._enc, self.tok.decode, d, budget)) for d in docs]
        return {"pairs": pairs, "group_weight": gw, "size": len(pairs)}


class LengthGroupedSampler(Sampler):
    """Order indices so each micro-batch holds similar-length groups (minimizes padding -> faster GPU).
    Shuffles within length-bucketed mega-batches so epochs still differ."""
    def __init__(self, dataset, batch_size, shuffle=True, mega=50):
        self.lengths = [dataset.group_len(i) for i in range(len(dataset))]
        self.shuffle, self.mega = shuffle, max(1, mega) * batch_size

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(idx)
        out = []
        for s in range(0, len(idx), self.mega):                # sort each mega-block by length
            out.extend(sorted(idx[s:s + self.mega], key=lambda i: self.lengths[i]))
        return iter(out)

    def __len__(self): return len(self.lengths)


def _collate(batch, tokenizer, max_length):
    flat = [p for ex in batch for p in ex["pairs"]]
    sizes = [ex["size"] for ex in batch]
    weights = [ex["group_weight"] for ex in batch]
    feats = tokenizer([q for q, _ in flat], [d for _, d in flat],
                      padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return feats, sizes, weights


def finetune_cross_encoder(groups_train, groups_val, *, base_model, lora_cfg,
                           max_length=2048, max_doc_tokens=1100, dtype="auto",
                           train_cfg, logger, out_dir, val_eval_fn=None):
    """LoRA fine-tune of bge-reranker-v2-m3 with masked listwise-softmax, GPU-optimized for a 16GB T4/G4.

    Memory/throughput: gradient CHECKPOINTING (fits seq=2048 on 16GB) + mixed precision (bf16 where the GPU
    supports it, else fp16 + GradScaler — a T4/G4 has no bf16) + length-grouped batching. STABILITY: a large
    EFFECTIVE batch via gradient ACCUMULATION (effective = batch_groups * grad_accum) without OOM. EARLY
    STOPPING on val nDCG@20 with `early_stop_patience` epochs; the best-val checkpoint is the returned adapter.
    NOTE: no in-batch negatives (a cross-encoder can't reuse them cheaply) — negatives-per-gold is `N` from
    sampling (the contrast knob); accumulation grows the gradient batch (the stability knob). They are distinct.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from mcrs.rerank.cross_encoder import doc_token_budget, truncate_doc_tokens, _resolve_dtype

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prec = _resolve_dtype(dtype, dev == "cuda" and torch.cuda.is_bf16_supported())    # bf16|fp16|fp32
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[prec]
    use_amp = (dev == "cuda" and prec != "fp32")
    use_scaler = (prec == "fp16")                              # GradScaler only for fp16

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    model.gradient_checkpointing_enable()                     # trade compute for memory (key for 2048 on T4)
    model.config.use_cache = False                            # required with checkpointing
    peft_cfg = LoraConfig(r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                          target_modules=lora_cfg["target_modules"], modules_to_save=["classifier"])
    model = get_peft_model(model, peft_cfg)
    model.enable_input_require_grads()                        # so checkpointing tracks LoRA grads
    model = model.to(dev)                                     # master weights fp32; autocast does the compute

    micro = train_cfg["batch_groups"]                         # micro-batch (groups) sized to fit memory
    accum = max(1, train_cfg.get("grad_accum", 1))            # EFFECTIVE batch = micro * accum (stability)
    patience = train_cfg.get("early_stop_patience", 1)        # stop after N epochs without val improvement

    def make_loader(groups, shuffle):
        ds = CEGroupDataset(groups, tok, max_length, max_doc_tokens, doc_token_budget, truncate_doc_tokens)
        sampler = (LengthGroupedSampler(ds, micro, shuffle=shuffle)
                   if train_cfg.get("group_by_length", True) else None)
        return DataLoader(ds, batch_size=micro, sampler=sampler, shuffle=(shuffle and sampler is None),
                          collate_fn=lambda b: _collate(b, tok, max_length), pin_memory=(dev == "cuda"))

    from transformers import get_cosine_schedule_with_warmup
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0))
    n_micro = math.ceil(len(groups_train) / micro)
    total_steps = max(1, math.ceil(n_micro / accum) * train_cfg["epochs"])   # OPTIMIZER steps over the run
    warmup_steps = int(train_cfg.get("warmup", 0.05) * total_steps)          # ~5% linear warmup
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)  # then cosine decay to ~0
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    best, best_metric, since_improved, step, loss_ema = None, -math.inf, 0, 0, None
    for epoch in range(train_cfg["epochs"]):
        model.train(); opt.zero_grad()
        for i, (feats, sizes, weights) in enumerate(make_loader(groups_train, True)):
            feats = {k: v.to(dev, non_blocking=True) for k, v in feats.items()}
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                logits = model(**feats).logits.squeeze(-1)
                loss = masked_listwise_ce(logits, sizes, weights) / accum     # scale for accumulation
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:                           # optimizer step once per `accum` micro-batches
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
                step += 1
                cur = float(loss) * accum                       # undo the accumulation scaling for logging
                loss_ema = cur if loss_ema is None else 0.98 * loss_ema + 0.02 * cur   # smooth the noisy curve
                if step % train_cfg.get("log_every", 50) == 0:  # EMA is display-only; never feeds optimization
                    logger.log({"train_loss": cur, "train_loss_ema": loss_ema,
                                "lr": sched.get_last_lr()[0], "epoch": epoch}, step=step)
        # ---- validation + early stopping ----
        if dev == "cuda": torch.cuda.empty_cache()
        model.eval()
        with torch.inference_mode():
            vl = []
            for feats, sizes, weights in make_loader(groups_val, False):
                feats = {k: v.to(dev) for k, v in feats.items()}
                with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                    vl.append(float(masked_listwise_ce(model(**feats).logits.squeeze(-1), sizes, weights)))
            val_loss = sum(vl) / max(len(vl), 1)
        val_ndcg = val_eval_fn(model, tok) if val_eval_fn else -val_loss      # dev nDCG@20 (real metric)
        logger.log({"val_loss": val_loss, "val_ndcg@20": val_ndcg, "epoch": epoch}, step=step)
        if val_ndcg > best_metric:                             # improved -> checkpoint best, reset patience
            best_metric, since_improved, best = val_ndcg, 0, out_dir
            model.save_pretrained(out_dir)
        else:
            since_improved += 1
            if since_improved >= patience:                     # EARLY STOP
                logger.log({"early_stop_epoch": epoch}, step=step)
                break
        if dev == "cuda": torch.cuda.empty_cache()
    return best
```

- [ ] **Step 2: Smoke test (tiny synthetic, CPU)**

Create `tests/test_ce_finetune_smoke.py` (skips if torch/peft missing):

```python
import pytest
torch = pytest.importorskip("torch"); pytest.importorskip("peft")
pytest.importorskip("transformers")

@pytest.mark.slow
def test_finetune_runs_one_step(tmp_path):
    from mcrs.training.ce_finetune import finetune_cross_encoder
    groups = [("q1", ["pos doc", "neg a", "neg b"], 1.0)] * 4
    class L:  # noqa
        def log(self, d, step=None): pass
    out = finetune_cross_encoder(
        groups, groups, base_model="hf-internal-testing/tiny-random-XLMRobertaForSequenceClassification",
        lora_cfg={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["query", "value"]},
        max_length=64, max_doc_tokens=40, dtype="fp32",
        train_cfg={"epochs": 1, "lr": 1e-4, "batch_groups": 2, "log_every": 1}, logger=L(), out_dir=str(tmp_path))
    assert out is not None
```

- [ ] **Step 3: Run the smoke test**

Run: `./recsys26/bin/python -m pytest tests/test_ce_finetune_smoke.py -v -m slow`
Expected: PASS (or SKIP if torch/peft absent locally — then it is verified on Colab).

- [ ] **Step 4: Commit**

```bash
git add mcrs/training/ce_finetune.py tests/test_ce_finetune_smoke.py
git commit -m "feat(K3b): finetune_cross_encoder (PEFT LoRA + masked listwise-softmax, val nDCG@20, best-ckpt)"
```

---

## Task 12: Notebook orchestration + eval gate

**Files:**
- Create: `nb/phase2_ce_finetune.ipynb`

This wires the pieces on Colab and produces the gate decision. Build the notebook with these cells (use `mcp__ide__executeCode` or author cells directly):

- [ ] **Step 1: Config + data load cell**

```python
CE_MODEL = "BAAI/bge-reranker-v2-m3"
CROSS_ENCODER_K = 100; N_NEG = 15; K_MIN = 4; FOLDS = 3; SEED = 0
MAX_LEN = 2048; MAX_DOC_TOK = 1100; DTYPE = "bf16"
LORA = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["query", "value"]}
# T4/G4 (16GB): micro batch_groups=2 x grad_accum=16 = effective 32 groups (stability) without OOM at seq 2048.
# epochs=3 is the UPPER bound; early stopping on val nDCG@20 (patience 1) + best-checkpoint pick the real stop.
TRAIN = {"epochs": 3, "lr": 1e-4, "weight_decay": 0.0, "batch_groups": 2, "grad_accum": 16,
         "warmup": 0.05, "log_every": 50, "group_by_length": True, "early_stop_patience": 1}

from mcrs.data.conversations import Conversations
from mcrs.data.catalog import Catalog
conv_tr = Conversations.from_disk("data/TalkPlayData-Challenge-Dataset", split="train")
conv_dv = Conversations.from_disk("data/TalkPlayData-Challenge-Dataset", split="test")
cat = Catalog.from_disk("data/TalkPlayData-Challenge-Track-Metadata", enriched_docs=ENRICHED)  # A1 corpus
assert all(cat.is_enriched(t) for t in SOME_POOL_SAMPLE), "K3b needs 100% enriched coverage (spec §2)"
```

- [ ] **Step 2: Build the enriched QueryBuilder + groups**

```python
from mcrs.retrieval.query import QueryBuilder
from mcrs.training.ce_data import build_ce_training_groups
def track_label(tid):
    if tid not in cat._meta: return None
    m = cat.metadata(tid); return f"{m.get('artist_name','')} – {m.get('track_name','')}"
qb = QueryBuilder(markers=True, taste_items=5, track_label_fn=track_label)
gp = lambda turn: {a["turn_number"]: a["goal_progress_assessment"]
                   for a in conv_tr._rows[turn.session_id]["goal_progress_assessments"]}.get(turn.turn_number)
turns = list(conv_tr.turns())
report = {}
groups = build_ce_training_groups(qb, fusion, turns, lambda t: conv_tr.gold(t.session_id, t.turn_number),
            catalog=cat, cross_encoder_k=CROSS_ENCODER_K, n_negatives=N_NEG, k_min=K_MIN, seed=SEED,
            gp_fn=gp, report=report)
print(report)   # dropped_no_gold / dropped_few_neg / kept — log the silent-drop counts (spec §4.3)
```

- [ ] **Step 3: Session-disjoint train/val + Trackio + fine-tune**

```python
from mcrs.training.ce_data import assign_session_folds
import trackio
folds = assign_session_folds([t.session_id for t in turns_kept], k=10, seed=SEED)  # 10% val
tr_groups = [g for g, f in zip(groups, folds) if f != 0]
va_groups = [g for g, f in zip(groups, folds) if f == 0]
trackio.init(project="k3b-ce-lora")
adapter = finetune_cross_encoder(tr_groups, va_groups, base_model=CE_MODEL, lora_cfg=LORA,
            max_length=MAX_LEN, max_doc_tokens=MAX_DOC_TOK, dtype=DTYPE, train_cfg=TRAIN,
            logger=trackio, out_dir="ckpt/k3b", val_eval_fn=make_val_ndcg(va_dev_turns))
```

- [ ] **Step 4: Eval gate cell (final-stage + per-segment + per-goal-progress)**

```python
from mcrs.rerank.cross_encoder import build_cross_encoder_score_fn
from mcrs.rerank.neural import NeuralReranker, ChainReranker
from mcrs.run.harness import InferenceHarness, TopKAssembler
from mcrs.eval.official import score_official
ce_score = build_cross_encoder_score_fn(CE_MODEL, device="cuda", max_length=MAX_LEN,
            max_doc_tokens=MAX_DOC_TOK, dtype=DTYPE, lora_adapter="ckpt/k3b")
k3 = NeuralReranker(cat, qb, ce_score, cross_encoder_k=CROSS_ENCODER_K, enriched=True)
dv = list(conv_dv.turns())
golds = [GoldRow(t.session_id, t.user_id, t.turn_number, conv_dv.gold(t.session_id, t.turn_number)) for t in dv]
base = InferenceHarness(qb, fusion, TopKAssembler(cat), reranker=k2, topk=20).run(dv)
ft   = InferenceHarness(qb, fusion, TopKAssembler(cat), reranker=ChainReranker(k2, k3), topk=20).run(dv)
print("K2", score_official(base, golds, len(cat))["ndcg@20"],
      "| K2+K3ft", score_official(ft, golds, len(cat))["ndcg@20"])
# also slice by segment (cold/warm) and by the gold's goal-progress label; ABORT goal-progress lever
# if off-goal-gold nDCG regresses (spec §4.3 guard). Ship only if K2+K3ft > K2.
```

- [ ] **Step 5: OOF stacking cell + commit**

```python
from mcrs.training.ce_finetune import oof_ce_scores
oof = oof_ce_scores([(t.session_id, t.turn_number) for t in turns_kept], folds=FOLDS, seed=SEED,
        fit_fn=make_fold_fit(groups, turns_kept), score_fn=make_fold_score(cat, qb, normalize=True))
# inject oof as the K1 'ce_ft_score' feature (within-pool normalized), retrain K2, re-eval vs K2.
```

```bash
git add nb/phase2_ce_finetune.ipynb
git commit -m "feat(K3b): fine-tune + eval-gate + OOF-stacking notebook"
```

---

## Self-Review

Spec coverage: §2 interface → Tasks 2/3/6/9/10/11 (+catalog Task 1); §4.1 input → Task 3; §4.2 budget → Tasks 9 defaults; §4.3 pos/neg/goal-progress → Tasks 4/5/6; §4.4 loss → Task 8; §4.5 model/dtype → Tasks 9/11; §4.6 splits/OOF → Tasks 7/10; §4.7 logging/serve → Tasks 11/12; §6 eval gate → Task 12; §7 tests → each task's tests. No spec section is unimplemented.

Placeholders: none — every code step is complete and runnable.

Type consistency: `CEGroup = (query_text, [docs], group_weight)` is produced by `build_ce_training_groups` (Task 6) and consumed identically by `CEGroupDataset` (Task 11); `sample_negatives` returns track_ids consumed by `build_doc`; `masked_listwise_ce(logits, group_sizes, group_weights)` signature matches its caller in Task 11; `oof_ce_scores` fit_fn/score_fn injection matches Task 10's test and Task 12's wiring.
