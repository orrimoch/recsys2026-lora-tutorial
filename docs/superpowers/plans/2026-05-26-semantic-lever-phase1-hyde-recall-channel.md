# Semantic Lever — Phase 1: HyDE Recall Channel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a HyDE recall channel — Qwen2.5-7B turns the conversation into pseudo-track descriptions, which a qwen3 dense retriever matches against the catalog — as a 4th, opt-in channel of `wrrf_union_v1`, and measure its recall@100 lift on full dev.

**Architecture:** Reuse the existing CMQR pattern (generate N texts → inner retriever → RRF-fuse). A `HydeGenerator` (LLAMA_MODEL wrapper around Qwen2.5-7B, cached by conversation hash) emits pseudo-tracks; a `HydeQwen3Retriever` embeds them with the existing `DENSE_PRECOMPUTED` metadata-qwen3 retriever and RRF-fuses the per-doc lists. The factory exposes it as `hyde_qwen3` and adds it to `wrrf_union_v1` behind `extra_config["use_hyde"]` so the recall ablation is channel-on vs channel-off.

**Tech Stack:** Python, transformers (Qwen2.5-7B-Instruct + Qwen3-Embedding-0.6B), numpy, pytest. Heavy model/data steps run on Colab; all pure logic is unit-tested locally with stubs.

**Why this scope:** Full-dev decomposition `nDCG@20 = recall@100 × conditional-on-hit` (0.1425 = 0.4375 × 0.3257). Phase 1 attacks recall@100 only. Phase 2 (reranker features) is a separate plan, gated on the recall lift measured here. See `docs/superpowers/specs/2026-05-26-semantic-lever-design.md`.

**Test convention:** New tests go in `tests/`, importing `from mcrs...` like the existing suite. Run from the repo root: `python -m pytest tests/<file> -v`. If `mcrs` is not importable, prepend the path inside the test file: `import sys; sys.path.insert(0, "music-crs-baselines")` (mirror whatever the existing `tests/test_train_bi_encoder.py` does).

---

## File Structure

- Create `music-crs-baselines/mcrs/system_prompts/hyde_pseudo_track.txt` — the HyDE system prompt (asset).
- Create `music-crs-baselines/mcrs/query_rewriters/hyde.py` — `build_hyde_messages`, `parse_hyde_output`, `HydeGenerator` (generation + conversation-hash cache).
- Create `music-crs-baselines/mcrs/retrieval_modules/hyde_qwen3.py` — `rrf_fuse`, `HydeQwen3Retriever` (the retriever-interface channel).
- Modify `music-crs-baselines/mcrs/retrieval_modules/__init__.py` — add the `hyde_qwen3` branch and a `_wrrf_union_v1_specs(extra_config)` helper; add the gated 4th sub to `wrrf_union_v1`.
- Create `tests/test_hyde_generation.py`, `tests/test_hyde_retriever.py`, `tests/test_union_hyde_wiring.py`.
- Modify `colab/72_build_lgbm_features_train.ipynb` cell 7 — recall@100 ablation (channel on/off).

---

### Task 1: HyDE system prompt asset

**Files:**
- Create: `music-crs-baselines/mcrs/system_prompts/hyde_pseudo_track.txt`

- [ ] **Step 1: Write the prompt file**

```
You are a music search assistant. Read the conversation between a user and a
music recommender. Predict what the user wants to hear NEXT and describe it as
if writing catalog metadata for the ideal next track.

Output EXACTLY this format and nothing else:
INTENT: <one short line: genre, mood, era, tempo, activity/context>
1. <a plausible track description: genre, mood, instrumentation, vocal style, era — DO NOT invent real song or artist names>
2. <a second, slightly different plausible track description>
3. <a third plausible track description>

Rules:
- Describe musical ATTRIBUTES, never real track titles or artist names.
- Each numbered line is a self-contained description of one hypothetical track.
- Stay consistent with the conversation's vibe and any stated constraints.
```

- [ ] **Step 2: Verify it loads and is non-empty**

Run: `python -c "p='music-crs-baselines/mcrs/system_prompts/hyde_pseudo_track.txt'; t=open(p).read(); assert 'INTENT:' in t and t.strip(); print('ok', len(t))"`
Expected: `ok <len>`

- [ ] **Step 3: Commit**

```bash
git add music-crs-baselines/mcrs/system_prompts/hyde_pseudo_track.txt
git commit -m "feat(hyde): pseudo-track generation prompt"
```

---

### Task 2: Prompt builder + output parser (pure functions)

**Files:**
- Create: `music-crs-baselines/mcrs/query_rewriters/hyde.py` (functions only this task)
- Test: `tests/test_hyde_generation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hyde_generation.py
from mcrs.query_rewriters.hyde import build_hyde_messages, parse_hyde_output


def test_build_hyde_messages_has_system_and_user():
    msgs = build_hyde_messages("user: something upbeat", "SYS")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["role"] == "user"
    assert "something upbeat" in msgs[1]["content"]


def test_parse_hyde_output_extracts_intent_and_docs():
    text = (
        "INTENT: dreamy synth-pop for a late night drive\n"
        "1. A hazy mid-tempo synth-pop track, breathy female vocals, 80s analog pads\n"
        "2. Downtempo electronic, reverb-heavy guitar, nostalgic and melancholic\n"
        "3. Slow dream-pop, shoegaze textures, soft male vocals\n"
    )
    out = parse_hyde_output(text)
    assert out["intent_query"] == "dreamy synth-pop for a late night drive"
    assert len(out["hyde_docs"]) == 3
    assert out["hyde_docs"][0].startswith("A hazy mid-tempo")


def test_parse_hyde_output_tolerates_missing_intent_and_paren_numbering():
    out = parse_hyde_output("1) only one description here\n")
    assert out["hyde_docs"] == ["only one description here"]
    assert out["intent_query"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hyde_generation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcrs.query_rewriters.hyde'`

- [ ] **Step 3: Write minimal implementation**

```python
# music-crs-baselines/mcrs/query_rewriters/hyde.py
"""HyDE generation: conversation -> intent query + pseudo-track descriptions.

Reuses the project LM wrapper (LLAMA_MODEL, exposing .lm/.tokenizer/.device)
and caches outputs by a hash of the conversation so the 8K-turn dev pass and
re-runs never re-call the model.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional


def build_hyde_messages(conversation: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Conversation:\n{conversation.strip()}\n"},
    ]


def parse_hyde_output(text: str) -> dict:
    """Parse model output into {"intent_query": str, "hyde_docs": list[str]}."""
    intent = ""
    docs: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m_intent = re.match(r"(?i)^intent\s*:\s*(.+)$", line)
        if m_intent:
            intent = m_intent.group(1).strip()
            continue
        m_doc = re.match(r"^\d+[.)]\s*(.+)$", line)
        if m_doc:
            docs.append(m_doc.group(1).strip())
    return {"intent_query": intent, "hyde_docs": docs}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hyde_generation.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/query_rewriters/hyde.py tests/test_hyde_generation.py
git commit -m "feat(hyde): prompt builder + output parser"
```

---

### Task 3: HydeGenerator (generation + cache)

**Files:**
- Modify: `music-crs-baselines/mcrs/query_rewriters/hyde.py` (append the class)
- Test: `tests/test_hyde_generation.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hyde_generation.py  (append)
import json
from mcrs.query_rewriters.hyde import HydeGenerator


class _FakeLM:
    """Stub matching the LLAMA_MODEL surface HydeGenerator uses."""
    def __init__(self, canned: str):
        self.canned = canned
        self.calls = 0

        class _Tok:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "PROMPT"
        self.tokenizer = _Tok()
        self.lm = self            # _generate_one is monkeypatched in the test
        self.device = "cpu"


def test_generator_caches_and_parses(tmp_path, monkeypatch):
    prompt = tmp_path / "p.txt"
    prompt.write_text("SYS", encoding="utf-8")
    lm = _FakeLM("INTENT: x\n1. doc one\n2. doc two\n")
    gen = HydeGenerator(lm, str(prompt), cache_dir=str(tmp_path))

    # Stub the raw generation so the test needs no real model.
    monkeypatch.setattr(gen, "_generate_one", lambda conv: lm.canned)

    out1 = gen.generate_batch(["conv A"])
    assert out1[0]["hyde_docs"] == ["doc one", "doc two"]
    assert (tmp_path / "hyde").exists()

    # Second call for the same conversation must hit cache (no new generation).
    called = {"n": 0}
    monkeypatch.setattr(gen, "_generate_one",
                        lambda conv: (called.__setitem__("n", called["n"] + 1), "")[1])
    out2 = gen.generate_batch(["conv A"])
    assert out2[0]["hyde_docs"] == ["doc one", "doc two"]
    assert called["n"] == 0  # served from disk cache


def test_generator_fallback_when_no_docs(tmp_path, monkeypatch):
    prompt = tmp_path / "p.txt"; prompt.write_text("SYS", encoding="utf-8")
    gen = HydeGenerator(_FakeLM(""), str(prompt), cache_dir=str(tmp_path))
    monkeypatch.setattr(gen, "_generate_one", lambda conv: "garbage with no numbers")
    out = gen.generate_batch(["conv B"])
    assert out[0]["hyde_docs"] == ["conv B"]  # falls back to the conversation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hyde_generation.py -k generator -v`
Expected: FAIL — `ImportError: cannot import name 'HydeGenerator'`

- [ ] **Step 3: Write minimal implementation (append to hyde.py)**

```python
# music-crs-baselines/mcrs/query_rewriters/hyde.py  (append)
class HydeGenerator:
    def __init__(self, lm, system_prompt_path, cache_dir: str = "./cache",
                 n_docs: int = 3, max_new_tokens: int = 192):
        self.lm = lm
        self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        self.cache_root = Path(cache_dir) / "hyde"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.n_docs = int(n_docs)
        self.max_new_tokens = int(max_new_tokens)

    def _cache_path(self, conversation: str) -> Path:
        h = hashlib.sha1(conversation.encode("utf-8")).hexdigest()[:24]
        return self.cache_root / f"{h}.json"

    def _load_cached(self, conversation: str) -> Optional[dict]:
        cp = self._cache_path(conversation)
        if cp.exists():
            try:
                payload = json.loads(cp.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "hyde_docs" in payload:
                    return payload
            except (OSError, json.JSONDecodeError):
                pass
        return None

    def _save_cached(self, conversation: str, parsed: dict) -> None:
        self._cache_path(conversation).write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    def _generate_one(self, conversation: str) -> str:
        import torch
        messages = build_hyde_messages(conversation, self.system_prompt)
        tok, model = self.lm.tokenizer, self.lm.lm
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_text, return_tensors="pt").to(self.lm.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False)
        return tok.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    def generate_batch(self, conversations: list[str]) -> list[dict]:
        results: list[dict] = []
        for conv in conversations:
            cached = self._load_cached(conv)
            if cached is not None:
                results.append(cached)
                continue
            parsed = parse_hyde_output(self._generate_one(conv))
            if not parsed["hyde_docs"]:
                parsed["hyde_docs"] = [parsed["intent_query"] or conv]
            self._save_cached(conv, parsed)
            results.append(parsed)
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hyde_generation.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/query_rewriters/hyde.py tests/test_hyde_generation.py
git commit -m "feat(hyde): HydeGenerator with conversation-hash cache"
```

---

### Task 4: rrf_fuse + HydeQwen3Retriever

**Files:**
- Create: `music-crs-baselines/mcrs/retrieval_modules/hyde_qwen3.py`
- Test: `tests/test_hyde_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hyde_retriever.py
from mcrs.retrieval_modules.hyde_qwen3 import rrf_fuse, HydeQwen3Retriever


def test_rrf_fuse_combines_per_doc_lists():
    lists = [["a", "b", "c"], ["b", "a", "d"]]
    fused = rrf_fuse(lists, rrf_k=60, topk=3)
    assert fused[:2] == ["a", "b"]  # appear high in both
    assert len(fused) == 3


class _FakeGen:
    def generate_batch(self, conversations):
        return [{"intent_query": "i", "hyde_docs": [f"{c}-d1", f"{c}-d2"]}
                for c in conversations]


class _FakeInner:
    """Returns a deterministic ranked list keyed off the query string."""
    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None):
        return [[f"T:{q}", "shared", "x"] for q in queries]


def test_retriever_generates_then_fuses_per_query():
    r = HydeQwen3Retriever(_FakeGen(), _FakeInner(), topk_per_doc=3, rrf_k=60)
    out = r.batch_text_to_item_retrieval(["Q1", "Q2"], topk=5)
    assert len(out) == 2
    # Q1's fused list draws only from Q1's two docs (T:Q1-d1, T:Q1-d2, shared, x)
    assert "shared" in out[0]
    assert any(t.startswith("T:Q1") for t in out[0])
    assert not any(t.startswith("T:Q2") for t in out[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hyde_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcrs.retrieval_modules.hyde_qwen3'`

- [ ] **Step 3: Write minimal implementation**

```python
# music-crs-baselines/mcrs/retrieval_modules/hyde_qwen3.py
"""HyDE recall channel: conversation -> pseudo-tracks -> dense retrieval -> RRF.

Composes a HydeGenerator (text) with a dense inner retriever (DENSE_PRECOMPUTED
over metadata-qwen3) so pseudo-track descriptions match in the catalog's own
embedding space. Implements the standard retriever interface so the wRRF
factory can use it as a union channel.
"""
from __future__ import annotations


def rrf_fuse(ranked_lists: list[list[str]], rrf_k: int, topk: int) -> list[str]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, tid in enumerate(lst):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [tid for tid, _ in ordered[:topk]]


class HydeQwen3Retriever:
    def __init__(self, generator, inner_retriever,
                 topk_per_doc: int = 100, rrf_k: int = 60):
        self.generator = generator
        self.inner = inner_retriever
        self.topk_per_doc = int(topk_per_doc)
        self.rrf_k = int(rrf_k)

    def batch_text_to_item_retrieval(self, queries, topk, user_ids=None):
        gen = self.generator.generate_batch(queries)
        flat_docs: list[str] = []
        spans: list[tuple[int, int]] = []
        for g in gen:
            docs = g["hyde_docs"] or [g["intent_query"]]
            start = len(flat_docs)
            flat_docs.extend(docs)
            spans.append((start, len(flat_docs)))
        if not flat_docs:
            return [[] for _ in queries]
        hits = self.inner.batch_text_to_item_retrieval(flat_docs, topk=self.topk_per_doc)
        return [rrf_fuse(hits[s:e], self.rrf_k, topk) for (s, e) in spans]

    def text_to_item_retrieval(self, query, topk, user_id=None):
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hyde_retriever.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/hyde_qwen3.py tests/test_hyde_retriever.py
git commit -m "feat(hyde): HydeQwen3Retriever channel with RRF fusion"
```

---

### Task 5: Factory wiring (`hyde_qwen3` type + gated union sub)

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/__init__.py`
- Test: `tests/test_union_hyde_wiring.py`

The union sub-spec list is built behind a helper so we can test the gating
without loading any model. The full `hyde_qwen3` instantiation (loads Qwen2.5-7B
+ the dense inner) is integration, verified on Colab in Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_union_hyde_wiring.py
from mcrs.retrieval_modules import _wrrf_union_v1_specs


def test_union_specs_default_has_three_channels():
    specs = _wrrf_union_v1_specs({})
    types = [s["type"] for s in specs]
    assert types == ["bm25", "dense_metadata_qwen3", "same_artist"]


def test_union_specs_add_hyde_when_enabled():
    specs = _wrrf_union_v1_specs({"use_hyde": True, "w_hyde": 0.8})
    types = [s["type"] for s in specs]
    assert "hyde_qwen3" in types
    hyde = next(s for s in specs if s["type"] == "hyde_qwen3")
    assert hyde["weight"] == 0.8
    assert hyde["topk_internal"] == 100
    assert hyde["extra_config"]["hyde_model"] == "Qwen/Qwen2.5-7B-Instruct"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_union_hyde_wiring.py -v`
Expected: FAIL — `ImportError: cannot import name '_wrrf_union_v1_specs'`

- [ ] **Step 3: Implement the helper + branch + gated sub**

In `music-crs-baselines/mcrs/retrieval_modules/__init__.py`, add the helper near the top (module scope):

```python
def _wrrf_union_v1_specs(extra_config: dict) -> list[dict]:
    """Sub-retriever specs for wrrf_union_v1. HyDE is opt-in via use_hyde."""
    ec = extra_config or {}
    specs = [
        {"type": "bm25", "topk_internal": 100,
         "weight": float(ec.get("w_bm25", 1.0))},
        {"type": "dense_metadata_qwen3", "topk_internal": 100,
         "weight": float(ec.get("w_qwen", 0.7))},
        {"type": "same_artist", "topk_internal": 100,
         "weight": float(ec.get("w_artist", 1.0))},
    ]
    if ec.get("use_hyde"):
        specs.append({
            "type": "hyde_qwen3", "topk_internal": 100,
            "weight": float(ec.get("w_hyde", 1.0)),
            "extra_config": {
                "hyde_model": ec.get("hyde_model", "Qwen/Qwen2.5-7B-Instruct"),
                "n_docs": int(ec.get("hyde_n_docs", 3)),
                "topk_per_doc": int(ec.get("hyde_topk_per_doc", 100)),
            },
        })
    return specs
```

Replace the inline `sub_specs=[...]` inside the existing `wrrf_union_v1` branch with `sub_specs=_wrrf_union_v1_specs(extra_config)` (keep the corpus_types/dense_metadata_qwen3 corpus override exactly as the current branch sets it; only the list construction moves into the helper).

Add the `hyde_qwen3` construction branch (loads the LM + dense inner lazily at build time):

```python
    elif retrieval_type == "hyde_qwen3":
        from ..lm_modules.llama import LLAMA_MODEL
        from ..query_rewriters.hyde import HydeGenerator
        from .hyde_qwen3 import HydeQwen3Retriever
        import os
        ec = extra_config or {}
        lm = LLAMA_MODEL(model_name=ec.get("hyde_model", "Qwen/Qwen2.5-7B-Instruct"))
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "system_prompts", "hyde_pseudo_track.txt")
        generator = HydeGenerator(
            lm, prompt_path, cache_dir=cache_dir, n_docs=int(ec.get("n_docs", 3)))
        inner = load_retrieval_module(
            "dense_metadata_qwen3", dataset_name, track_split_types,
            corpus_types, cache_dir, extra_config={})
        return HydeQwen3Retriever(
            generator, inner, topk_per_doc=int(ec.get("topk_per_doc", 100)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_union_hyde_wiring.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m pytest tests/test_hyde_generation.py tests/test_hyde_retriever.py tests/test_union_hyde_wiring.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: Commit**

```bash
git add music-crs-baselines/mcrs/retrieval_modules/__init__.py tests/test_union_hyde_wiring.py
git commit -m "feat(hyde): factory branch + gated union channel (use_hyde)"
```

---

### Task 6: Colab recall ablation (integration, runs on GPU)

This is the validation gate — not a pytest. Add a cell to `colab/72_build_lgbm_features_train.ipynb` (after cell 7) that builds the union with `use_hyde` off vs on and compares recall, reusing the in-memory `queries`/`golds`/`played`/`user_ids` from cell 7.

**Files:**
- Modify: `colab/72_build_lgbm_features_train.ipynb` (new cell after cell 7)

- [ ] **Step 1: Add the ablation cell**

```python
# 8) Phase-1 recall ablation: union without vs with the HyDE channel.
# Reuses queries/golds/played/user_ids built in cell 7.
import numpy as np
from mcrs.retrieval_modules import load_retrieval_module

def recall_at(cands, k):
    return float(np.mean([1.0 if g in c[:k] else 0.0 for c, g in zip(cands, golds)]))

base = load_retrieval_module('wrrf_union_v1', ITEM_DB, ['all_tracks'], CORPUS,
                             CACHE_DIR, extra_config={})
hyde = load_retrieval_module('wrrf_union_v1', ITEM_DB, ['all_tracks'], CORPUS,
                             CACHE_DIR, extra_config={'use_hyde': True, 'w_hyde': 1.0})

ctx = [{'history_tids': p} for p in played]
cb = base.batch_text_to_item_retrieval(queries, topk=100, user_ids=user_ids, batch_context=ctx)
ch = hyde.batch_text_to_item_retrieval(queries, topk=100, user_ids=user_ids, batch_context=ctx)

print('=== Phase-1 recall ablation (n=' + str(len(golds)) + ', FULL dev) ===')
print('  union (3-chan)   : recall@20=' + str(round(recall_at(cb,20),4)) +
      ' @100=' + str(round(recall_at(cb,100),4)))
print('  union + HyDE     : recall@20=' + str(round(recall_at(ch,20),4)) +
      ' @100=' + str(round(recall_at(ch,100),4)) + '   (G1 gate 0.46)')
print('  delta recall@100 :', round(recall_at(ch,100) - recall_at(cb,100), 4))
```

- [ ] **Step 2: Smoke on a subsample first**

Set `N_EVAL = 1500` in cell 7, re-run cells 7 then 8. Expected: the cell runs end-to-end, HyDE cache populates under `CACHE_DIR/hyde/`, and it prints both recall lines. Confirms the LLM + channel wiring work before the full pass.

- [ ] **Step 3: Full-dev run**

Set `N_EVAL = None`, re-run cells 7 then 8.
Expected/target: `union + HyDE` recall@100 ≥ baseline + 0.03 (and clears 0.46). Decision: if the lift is real, proceed to Phase 2 (reranker features); if flat, fall back to a stronger embedding model for the HyDE channel (spec risk note) before abandoning.

- [ ] **Step 4: Commit the notebook**

```bash
git add colab/72_build_lgbm_features_train.ipynb
git commit -m "eval(hyde): Phase-1 recall ablation cell (union +/- HyDE)"
```

---

## Self-Review

**Spec coverage:**
- HyDE generation (Component 1) → Tasks 1–3. ✓
- Recall channel (Component 2) → Task 4 + factory branch in Task 5. ✓
- Wire into wrrf_union_v1, ensemble-first, opt-in → Task 5 (`use_hyde` gate). ✓
- Caching by conversation hash → Task 3 (`HydeGenerator._cache_path`). ✓
- Validation: recall@100 channel-on vs off, full dev, ≥ +0.03 → Task 6. ✓
- Reuse CMQR/dense/LM infra → `DENSE_PRECOMPUTED` inner (Task 5), `LLAMA_MODEL` wrapper (Task 5). ✓
- Phase 2 (reranker features) → intentionally out of scope; separate plan, gated on Task 6.
- CMQR-rewrite baseline (spec "cheap comparison") → deferred; only run if HyDE recall is flat (keeps Phase 1 focused on one channel). Noted here so it isn't lost.

**Placeholder scan:** No TBD/TODO; every code step has complete code and exact run commands.

**Type consistency:** `HydeGenerator.generate_batch` returns `list[{"intent_query","hyde_docs"}]`, consumed by `HydeQwen3Retriever.batch_text_to_item_retrieval` (reads `g["hyde_docs"]`/`g["intent_query"]`). Factory builds `HydeGenerator(lm, prompt_path, cache_dir, n_docs)` and `HydeQwen3Retriever(generator, inner, topk_per_doc)` — signatures match Tasks 3 and 4. `_wrrf_union_v1_specs` keys (`type/topk_internal/weight/extra_config`) match what `rrf.py` reads. Consistent.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-semantic-lever-phase1-hyde-recall-channel.md`.
