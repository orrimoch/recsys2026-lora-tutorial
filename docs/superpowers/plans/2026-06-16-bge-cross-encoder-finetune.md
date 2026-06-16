# BGE Cross-Encoder Fine-Tune (Path A) Implementation Plan — v2 (parity-fixed)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `BAAI/bge-reranker-v2-m3` as a plain text cross-encoder on the *exact current serve distribution* so it funnels the recall pool better than the zero-shot OOD bge (which failed the 9c gate) and the lite-LLM stage-1.

**Architecture:** `AutoModelForSequenceClassification` (num_labels=1) on the bge backbone, trained with listwise softmax (LCE). The defining principle of v2: **one canonical union (`UNION_EC` below) is used identically for mining negatives, for the nb90 gate, and for serve config 224** — eliminating the three-way train/gate/serve skew the v1 review found. The fine-tuned model is a drop-in for serve via `bge_reranker.py`'s `model_path`.

**Tech Stack:** PyTorch + HF `transformers`, `datasets`, the repo's `mcrs` package (`wrrf_union_v1`, `build_retrieval_query`, `build_user_dialog`, `bge_reranker`), Colab GPU, `pytest`.

---

## v1 review outcome → v2 fixes (all four bugs)

| v1 bug (verified) | v2 fix |
|---|---|
| **BLOCKER-1** mining/gate/serve were 3 different pools | One `UNION_EC` (below) used in builder + gate + config 224. Builder & gate pass **real per-turn history ctx** (`history_tids`, `user_dialog`, `colbert_query`), not an empty stub |
| **BLOCKER-3** CE needs compact query but serve feeds the reranker the full `raw_*` query | Phase 2 adds a `compact_queries` side-channel in `crs_baseline` so the bge stage gets the compact query (gate already uses compact → now serve matches) |
| **MAJOR-1** reranker doc corpus == retrieval corpus (`crs_baseline.py:409/415`) | Phase 2 adds a dedicated `reranker_corpus_types` param so the CE gets `…|tag_list` without altering the dense retrieval channel |
| **MAJOR-2** train `max_length=320` vs serve hard-coded 256 | Task 1 makes `bge_reranker` `max_length` a param; train + gate + serve all use **512** |

**Confirmed sound by review:** the `listwise_softmax_loss`/`ce_group_metrics` reuse is correct; no train/test leakage (mining walks `split="train"`, gate evals `split="test"`).

**Strategic caveat (acknowledged, user chose to build):** DEV recall has not transferred to Blind all campaign; the wall is partly a knowledge gap a metadata CE shares. The gate (Task 7) is the kill-switch.

---

## Decisions locked

```python
# THE one canonical union for mining + gate + serve. 219-lineage, PG off (user-confirmed),
# clap off (fusion-noise finding project_fusion_sweep_clap_noise_2026_06_16). bm25 +
# dense_metadata_qwen3_instruct(w0.7) + same_artist are always-on union defaults.
UNION_EC = {
    "use_sasrec": True, "w_sasrec": 1.0,
    "use_colbert": True, "w_colbert": 1.0,
    "colbert_index_name": "colbert-music-v1", "colbert_model": None,
    "colbert_q_len": 96, "colbert_compact_query": True,
    "use_clap_text": False,        # fusion-noise; flip to True only to match config 205 exactly
    "use_propose_ground": False,   # PG off in practice (YAMLs 205-222 are stale)
}
RETRIEVAL_QUERY_MODE = "raw_enriched"   # union main query (215/219 serve); colbert routes compact internally
CE_QUERY_MODE        = "compact_colbert"  # the (query,doc) pair query — forced by CE token budget; matches 9c
DOC_CORPUS  = ["track_name", "artist_name", "album_name", "tag_list"]   # CE doc text
MAX_LENGTH  = 512   # bge tokenizer cap — same at train, gate, serve
POOL_SIZE   = 300   # union top-K for negative mining (deep-pool headroom from gate-A)
```

**Infra dependency:** the ColBERT PLAID index `colbert-music-v1` and the CLAP/SASRec artifacts must be present in the Colab cache (the nb82/nb90 bootstrap provides them). Mining loads the full union — hours of A100 time.

---

## File Structure

- **Modify** `music-crs-baselines/mcrs/rerankers/bge_reranker.py` — (a) extract `build_tid_text_map(metadata_dict, corpus_types)`; (b) add `max_length` constructor param (default 256, back-compat). 
- **Create** `scripts/build_bge_ce_training_data.py` — walk train convs → `(compact_query, gold, neg_texts)` JSONL, negs from `UNION_EC` top-300 with real per-turn ctx.
- **Create** `scripts/train_bge_cross_encoder.py` — plain-CE trainer; reuses pure fns from `train_cross_encoder.py`.
- **Create** `tests/test_bge_reranker_text_helper.py`, `tests/test_bge_ce_training_data.py`, `tests/test_train_bge_cross_encoder.py`.
- **Create** `colab/91_train_bge_cross_encoder.ipynb` — Colab build+train orchestrator.
- **Modify** `colab/90_colbert_dev_experiments.ipynb` cell `9c` — rebuild the deep pool with `UNION_EC`, add `BGE_MODEL_PATH` + `MAX_LENGTH` knobs.
- **(Phase 2)** Modify `crs_baseline.py` + `rerankers/__init__.py` (reranker_corpus_types + compact side-channel); create `config/224-*.yaml`.

---

## Phase 1 — Build, train, gate

### Task 1: bge_reranker — shared text helper + configurable max_length

**Files:**
- Modify: `music-crs-baselines/mcrs/rerankers/bge_reranker.py`
- Test: `tests/test_bge_reranker_text_helper.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bge_reranker_text_helper.py
from mcrs.rerankers.bge_reranker import build_tid_text_map

def test_build_tid_text_map_pipe_joins_and_flattens_tags():
    md = {"t1": {"track_name": "Song A", "artist_name": "Artist X",
                 "album_name": "Alb", "tag_list": ["indie", "melancholic"]},
          "t2": {"track_name": "Song B", "artist_name": "Artist Y",
                 "album_name": None, "tag_list": []}}
    out = build_tid_text_map(md, ["track_name", "artist_name", "album_name", "tag_list"])
    assert out["t1"] == "Song A | Artist X | Alb | indie, melancholic"
    assert out["t2"] == "Song B | Artist Y"   # empty/None dropped
```

- [ ] **Step 2: Run to verify fail**

Run: `cd music-crs-baselines && python -m pytest ../tests/test_bge_reranker_text_helper.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_tid_text_map'`

- [ ] **Step 3: Add the helper + max_length param**

Add the module-level helper (mirrors the existing inline logic):

```python
def build_tid_text_map(metadata_dict: dict, corpus_types: list[str]) -> dict[str, str]:
    """Pipe-joined CE doc text: "name | artist | album | tag, tag". List fields
    comma-joined; empty/None dropped. SINGLE source of truth shared by serve
    (BGE_RERANKER) and the CE training-data builder so (query,doc) is identical."""
    out: dict[str, str] = {}
    for tid, row in metadata_dict.items():
        parts: list[str] = []
        for f in corpus_types:
            v = row.get(f)
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v if x is not None)
            if v:
                parts.append(str(v))
        out[tid] = " | ".join(parts)
    return out
```

In `BGE_RERANKER.__init__`, add `max_length: int = 256` to the signature and store `self.max_length = max_length`. In `_score_batch`, replace the hard-coded `max_length=256` with `max_length=self.max_length`. In `_load_or_build_tid_text`, replace the inline `for row in concat:` build with:

```python
        metadata_dict = {row["track_id"]: row for row in concat}
        tid_to_text = build_tid_text_map(metadata_dict, corpus_types)
```

- [ ] **Step 4: Run to verify pass + no regression**

Run: `cd music-crs-baselines && python -m pytest ../tests/test_bge_reranker_text_helper.py ../tests/test_bge_reranker_override.py -v`
Expected: PASS (text output unchanged; max_length default 256 unchanged)

- [ ] **Step 5: Thread max_length through the factory**

In `mcrs/rerankers/__init__.py` `load_reranker_module`, add `bge_max_length: int = 256` to the signature and pass `max_length=bge_max_length` into the `BGE_RERANKER(...)` constructor call (the `if reranker_type == "bge_reranker_v2_m3":` branch).

- [ ] **Step 6: Run the reranker factory tests + commit**

Run: `cd music-crs-baselines && python -m pytest ../tests/test_rerankers.py ../tests/test_bge_reranker_override.py -v`
Expected: PASS

```bash
git add music-crs-baselines/mcrs/rerankers/bge_reranker.py music-crs-baselines/mcrs/rerankers/__init__.py tests/test_bge_reranker_text_helper.py
git commit -m "feat(bge_reranker): build_tid_text_map helper + configurable max_length (CE parity)"
```

---

### Task 2: Builder pure functions

**Files:**
- Create: `scripts/build_bge_ce_training_data.py` (helpers only)
- Test: `tests/test_bge_ce_training_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bge_ce_training_data.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "music-crs-baselines"))
from build_bge_ce_training_data import select_ce_negatives, build_ce_row

def test_select_ce_negatives_drops_gold_and_caps():
    assert select_ce_negatives("g", ["g","n1","n2","n3","n4"], 3) == ["n1","n2","n3"]

def test_select_ce_negatives_gold_absent_keeps_all_capped():
    assert select_ce_negatives("g", ["n1","n2","n3"], 10) == ["n1","n2","n3"]

def test_build_ce_row_uses_text_map_and_keeps_index_alignment():
    text_map = {"g":"gold txt","n1":"neg1 txt","n2":"neg2 txt"}
    row = build_ce_row(query="goal: chill\nput on something mellow", gold_tid="g",
                       neg_tids=["n1","nX","n2"], text_map=text_map, user_id="u1", session_id="s1")
    assert row["pos"] == "gold txt"
    assert row["neg"] == ["neg1 txt","neg2 txt"]   # nX (absent) dropped, order kept
    assert row["neg_tids"] == ["n1","n2"] and row["pos_tid"] == "g" and row["user_id"] == "u1"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_bge_ce_training_data.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the pure helpers** (header + functions)

```python
# scripts/build_bge_ce_training_data.py  (top of file)
"""Plain bge cross-encoder training-data builder (Path A v2).
Walks HF train conversations, mines hard negatives from THE canonical serve union
(UNION_EC: sasrec+colbert+bm25+dense+same_artist, clap off, PG off) with real
per-turn history ctx, and writes the EXACT compact query + tag-enriched pipe doc
text the serve bge_reranker uses. One JSONL row per music turn."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines")); sys.path.insert(0, str(REPO_ROOT / "scripts"))


def select_ce_negatives(gold_tid: str, candidate_tids: list[str], max_negatives: int) -> list[str]:
    """Pool minus the gold, hardest-first (pool rank order), capped."""
    return [t for t in candidate_tids if t != gold_tid][:max_negatives]


def build_ce_row(query: str, gold_tid: str, neg_tids: list[str], text_map: dict[str, str],
                 user_id: Optional[str], session_id: Optional[str]) -> dict[str, Any]:
    """One JSONL triple; negs absent from text_map dropped from BOTH neg & neg_tids."""
    kept = [t for t in neg_tids if t in text_map]
    return {"query": query, "pos": text_map[gold_tid], "neg": [text_map[t] for t in kept],
            "pos_tid": gold_tid, "neg_tids": kept, "user_id": user_id, "session_id": session_id}
```

- [ ] **Step 4: Run to verify pass + commit**

Run: `python -m pytest tests/test_bge_ce_training_data.py -v`  → Expected: 3 passed

```bash
git add scripts/build_bge_ce_training_data.py tests/test_bge_ce_training_data.py
git commit -m "feat(ce-build): pure helpers for plain bge CE training-data builder"
```

---

### Task 3: Builder `main()` — canonical-union mining with real per-turn ctx

**Files:** Modify `scripts/build_bge_ce_training_data.py` (add `main()`).

> No unit test (loads the full union + GPU). Validated by the Task 5 smoke run.

- [ ] **Step 1: Write `main()`** — note the dual query (raw for union, compact for the CE pair + colbert) and the real per-turn ctx that fixes BLOCKER-1:

```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-conv-hf", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--track-meta-hf", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default="../experiments/cache")
    p.add_argument("--pool-size", type=int, default=300)
    p.add_argument("--max-negatives", type=int, default=63)
    p.add_argument("--doc-corpus", default="track_name,artist_name,album_name,tag_list")
    p.add_argument("--retrieval-query-mode", default="raw_enriched")
    p.add_argument("--ce-query-mode", default="compact_colbert")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args()

    import pandas as pd
    from datasets import load_dataset
    from mcrs.retrieval_modules import load_retrieval_module
    from mcrs.crs_baseline import build_retrieval_query
    from mcrs.retrieval_modules.sasrec_model import build_user_dialog
    from mcrs.rerankers.bge_reranker import build_tid_text_map
    from mcrs.db_item.music_catalog import MusicCatalogDB

    doc_corpus = [c.strip() for c in args.doc_corpus.split(",") if c.strip()]
    base_corpus = ["track_name", "artist_name", "album_name"]

    # 1) catalog -> full metadata rows -> CE doc text (shared helper = serve parity)
    item_db = MusicCatalogDB(args.track_meta_hf, ["all_tracks"], base_corpus)
    text_map = build_tid_text_map(item_db.metadata_dict, doc_corpus)

    # 2) THE canonical serve union (UNION_EC). Must match nb90 gate + config 224.
    union = load_retrieval_module(
        "wrrf_union_v1", args.track_meta_hf, ["all_tracks"], base_corpus, args.cache_dir,
        extra_config={"use_sasrec": True, "w_sasrec": 1.0, "use_colbert": True, "w_colbert": 1.0,
                      "colbert_index_name": "colbert-music-v1", "colbert_model": None,
                      "colbert_q_len": 96, "colbert_compact_query": True,
                      "use_clap_text": False, "use_propose_ground": False})

    # 3) walk conversations -> per-music-turn rows with REAL history (fixes BLOCKER-1)
    conv = load_dataset(args.train_conv_hf, split="train")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    buf_raw, buf_cmp, buf_ctx, buf_uid, buf_gold, buf_sess = [], [], [], [], [], []

    def flush():
        nonlocal n_written
        if not buf_raw:
            return
        cands = union.batch_text_to_item_retrieval(
            buf_raw, topk=args.pool_size, user_ids=buf_uid, batch_context=buf_ctx)
        with open(args.output, "a") as f:
            for j in range(len(buf_raw)):
                gold = buf_gold[j]
                if gold not in text_map:
                    continue
                negs = select_ce_negatives(gold, list(cands[j]), args.max_negatives)
                row = build_ce_row(buf_cmp[j], gold, negs, text_map, buf_uid[j], buf_sess[j])
                if len(row["neg"]) < 2:
                    continue
                f.write(json.dumps(row) + "\n"); n_written += 1
        buf_raw.clear(); buf_cmp.clear(); buf_ctx.clear(); buf_uid.clear(); buf_gold.clear(); buf_sess.clear()

    n_rows = 0
    for sess in conv:
        df = pd.DataFrame(sess["conversations"])
        goal = (sess.get("conversation_goal") or {}).get("listener_goal", "") or ""
        up = sess.get("user_profile"); uid = sess.get("user_id"); sid = sess.get("session_id")
        for _, music in df[df["role"] == "music"].iterrows():
            tn = int(music["turn_number"])
            prior = df[(df["turn_number"] < tn) | ((df["turn_number"] == tn) & (df["role"] == "user"))]
            prior_turns = [{"role": ("assistant" if t["role"] == "music" else t["role"]),
                            "content": (item_db.id_to_metadata(t["content"]) if t["role"] == "music"
                                        else t["content"])} for _, t in prior.iterrows()]
            played = list(df[(df["role"] == "music") & (df["turn_number"] < tn)]["content"])
            raw_q = build_retrieval_query(prior_turns, mode=args.retrieval_query_mode,
                                          goal_text=goal, user_profile=up)
            cmp_q = build_retrieval_query(prior_turns, mode=args.ce_query_mode,
                                          goal_text=goal, user_profile=up)
            buf_raw.append(raw_q); buf_cmp.append(cmp_q)
            buf_ctx.append({"history_tids": played, "user_dialog": build_user_dialog(prior.to_dict("records")),
                            "colbert_query": cmp_q})
            buf_uid.append(uid); buf_gold.append(music["content"]); buf_sess.append(sid)
            if len(buf_raw) >= args.batch_size:
                flush()
            n_rows += 1
            if args.max_rows and n_rows >= args.max_rows:
                flush(); print(f"[ce-build] DONE (max-rows) -> {args.output} ({n_written})", file=sys.stderr); return
    flush()
    print(f"[ce-build] DONE -> {args.output} ({n_written} rows from {n_rows} turns)", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Import-lint**

Run: `cd music-crs-baselines && python -c "import sys; sys.path.insert(0,'../scripts'); import build_bge_ce_training_data"`
Expected: no error

- [ ] **Step 3: Commit**

```bash
git add scripts/build_bge_ce_training_data.py
git commit -m "feat(ce-build): canonical-union mining main() (real per-turn ctx, dual query, PG/clap off)"
```

---

### Task 4: Plain-CE trainer

**Files:** Create `scripts/train_bge_cross_encoder.py`; Test `tests/test_train_bge_cross_encoder.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_bge_cross_encoder.py
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_bge_cross_encoder import CEJsonlDataset, flatten_groups

def test_flatten_groups_pos_first_then_negs():
    pairs, is_pos = flatten_groups([{"query":"q","pos":"P","neg":["A","B"]}])
    assert pairs == [("q","P"),("q","A"),("q","B")] and is_pos == [True,False,False]

def test_dataset_samples_n_negatives(tmp_path):
    fp = tmp_path/"d.jsonl"
    fp.write_text(json.dumps({"query":"q","pos":"P","pos_tid":"g","neg":["A","B","C","D"],
                              "neg_tids":["a","b","c","d"],"user_id":"u"})+"\n")
    row = CEJsonlDataset(str(fp), n_negatives=2, seed=0)[0]
    assert row["pos"]=="P" and len(row["neg"])==2 and set(row["neg"]).issubset({"A","B","C","D"})
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_train_bge_cross_encoder.py -v` → Expected: FAIL (module not found)

- [ ] **Step 3: Write the trainer**

```python
# scripts/train_bge_cross_encoder.py
"""Path A: fine-tune BAAI/bge-reranker-v2-m3 as a PLAIN text cross-encoder
(AutoModelForSequenceClassification, num_labels=1) with listwise softmax (LCE).
Reuses pure fns from train_cross_encoder.py. Loads drop-in at serve via bge_reranker."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_cross_encoder import listwise_softmax_loss, ce_group_metrics, resolve_max_norm


def flatten_groups(rows):
    """Rows -> flat (query,doc) pairs + is_positive mask, pos at each group start."""
    pairs, is_pos = [], []
    for r in rows:
        pairs.append((r["query"], r["pos"])); is_pos.append(True)
        for n in r["neg"]:
            pairs.append((r["query"], n)); is_pos.append(False)
    return pairs, is_pos


class CEJsonlDataset:
    def __init__(self, path, n_negatives=15, seed=42, split="all", val_fraction=0.0, split_key="user_id"):
        self.rows = [json.loads(l) for l in open(path) if l.strip()]
        if val_fraction > 0:
            keyed = sorted({str(r.get(split_key)) for r in self.rows})
            rng = random.Random(seed); rng.shuffle(keyed)
            val_keys = set(keyed[:int(len(keyed) * val_fraction)])
            self.rows = [r for r in self.rows if (str(r.get(split_key)) in val_keys) == (split == "val")]
        self.n_negatives = n_negatives; self.rng = random.Random(seed)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; negs = r["neg"]
        if len(negs) > self.n_negatives: negs = self.rng.sample(negs, self.n_negatives)
        return {"query": r["query"], "pos": r["pos"], "neg": negs}


def score_batch(model, tokenizer, rows, max_length, device):
    import torch
    pairs, is_pos = flatten_groups(rows)
    enc = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                    truncation=True, max_length=max_length, return_tensors="pt").to(device)
    logits = model(**enc).logits.view(-1)
    return logits, torch.tensor(is_pos, dtype=torch.bool, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3"); ap.add_argument("--hub-repo", default="")
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--n-negatives", type=int, default=15)
    ap.add_argument("--max-length", type=int, default=512); ap.add_argument("--loss-temperature", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=25.0); ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)

    tr = CEJsonlDataset(args.triples, args.n_negatives, args.seed, "train", args.val_fraction)
    va = CEJsonlDataset(args.triples, args.n_negatives, args.seed, "val", args.val_fraction) if args.val_fraction > 0 else None
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); use_amp = device == "cuda"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.train(); step = 0
    for epoch in range(args.epochs):
        for rows in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, is_pos = score_batch(model, tok, rows, args.max_length, device)
                loss = listwise_softmax_loss(logits, is_pos, args.loss_temperature)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), resolve_max_norm(args.max_grad_norm)))
            opt.step(); opt.zero_grad(); step += 1
            if step % args.log_every == 0:
                t1, nd = ce_group_metrics(logits.detach().float(), is_pos)
                print(f"[ce] ep{epoch} step{step} loss={loss.item():.4f} top1={t1:.3f} ndcg={nd:.4f} gn={gn:.1f}", file=sys.stderr)
        if va is not None:
            model.eval(); vt1 = vnd = vn = 0.0
            with torch.no_grad():
                for vr in DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x):
                    vl, vp = score_batch(model, tok, vr, args.max_length, device)
                    g = int(vp.sum().item()); a, b = ce_group_metrics(vl.float(), vp)
                    vt1 += a * g; vnd += b * g; vn += g
            print(f"[ce] EPOCH {epoch} VAL top1={vt1/max(vn,1):.3f} ndcg={vnd/max(vn,1):.4f}", file=sys.stderr)
            model.train()

    model.save_pretrained(args.output_dir); tok.save_pretrained(args.output_dir)
    print(f"[ce] saved -> {args.output_dir}", file=sys.stderr)
    if args.hub_repo:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(args.hub_repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.hub_repo, repo_type="model")
        print(f"[ce] pushed -> {args.hub_repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the stub-model scoring test**

```python
def test_score_batch_runs_on_stub():
    import torch
    from train_bge_cross_encoder import score_batch
    class _Tok:
        def __call__(self, q, d, **k):
            class E(dict):
                def to(self, *_): return self
            return E(input_ids=torch.zeros(len(q),4,dtype=torch.long),
                     attention_mask=torch.ones(len(q),4,dtype=torch.long))
    class _Out:
        def __init__(self,n): self.logits = torch.arange(n,dtype=torch.float).view(-1,1)
    class _Model:
        def __call__(self, **k): return _Out(k["input_ids"].shape[0])
    logits, is_pos = score_batch(_Model(), _Tok(), [{"query":"q","pos":"P","neg":["A","B"]}], 8, "cpu")
    assert logits.shape[0] == 3 and is_pos.tolist() == [True,False,False]
```

- [ ] **Step 5: Run to verify pass + commit**

Run: `python -m pytest tests/test_train_bge_cross_encoder.py -v` → Expected: 3 passed

```bash
git add scripts/train_bge_cross_encoder.py tests/test_train_bge_cross_encoder.py
git commit -m "feat(ce-train): plain bge cross-encoder trainer (LCE, max_length 512)"
```

---

### Task 5: Colab orchestrator notebook `colab/91_train_bge_cross_encoder.ipynb`

- [ ] **Step 1:** Bootstrap cell — copy the nb90 bootstrap verbatim (clone `recall-union-lgbm`, mount Drive, deps, ensure `colbert-music-v1` index + artifacts in cache). `GEMINI_API_KEY` NOT needed (PG off).
- [ ] **Step 2:** Smoke build — `python scripts/build_bge_ce_training_data.py --output /content/drive/MyDrive/recsys2026/ce_smoke.jsonl --cache-dir experiments/cache --max-rows 64`. Inspect line 1: compact query, pipe+tag pos/neg.
- [ ] **Step 3:** Full build — same without `--max-rows`. Heavy (full union over train set, hours on A100). Drive output survives disconnect.
- [ ] **Step 4:** Train — `python scripts/train_bge_cross_encoder.py --triples .../ce_train.jsonl --output-dir .../bge_ce_ft --hub-repo OrRim123/recsys2026-bge-ce-music-v1 --epochs 2`. Expect loss ↓ from ~ln(16)=2.77, VAL top1/ndcg ↑.
- [ ] **Step 5:** Commit the notebook.

```bash
git add colab/91_train_bge_cross_encoder.ipynb
git commit -m "feat(nb91): Colab build+train orchestrator for bge cross-encoder"
```

---

### Task 6: Make the nb90 9c gate serve-faithful + gate the FT model

**Files:** Modify `colab/90_colbert_dev_experiments.ipynb` cell `9c` (idx 20).

- [ ] **Step 1: Add knobs** next to the existing 9c knobs:

```python
BGE_MODEL_PATH = None   # None=OOD; 'OrRim123/recsys2026-bge-ce-music-v1' = the fine-tuned CE
MAX_LENGTH     = 512    # must equal training max_length
```

- [ ] **Step 2: Rebuild the deep pool with the canonical UNION_EC** (replaces the reduced `sas` pool). Insert before the `deep = ...` line:

```python
union_ce = load_retrieval_module('wrrf_union_v1', ITEM_DB, ['all_tracks'], CORPUS, CACHE_DIR,
    extra_config={'use_sasrec': True, 'w_sasrec': 1.0, 'use_colbert': True, 'w_colbert': 1.0,
                  'colbert_index_name': 'colbert-music-v1', 'colbert_model': None,
                  'colbert_q_len': 96, 'colbert_compact_query': True,
                  'use_clap_text': False, 'use_propose_ground': False})
ctx_ce = [dict(ctx[i], colbert_query=queries_compact[i]) for i in sub]   # route compact to colbert
deep = union_ce.batch_text_to_item_retrieval([queries[i] for i in sub], topk=max(POOL_DEPTHS),
                                             user_ids=uid, batch_context=ctx_ce)
```

(Delete the old `deep = sas.batch_text_to_item_retrieval(qf, ...)` line.)

- [ ] **Step 3: Pass model_path + max_length to the bge loader:**

```python
bge = load_reranker_module('bge_reranker_v2_m3', ITEM_DB, ['all_tracks'], _bge_corpus, CACHE_DIR,
                           model_path=BGE_MODEL_PATH, bge_max_length=MAX_LENGTH)
```

- [ ] **Step 4: Validate JSON + commit**

```bash
python3 -c "import json,ast; nb=json.load(open('colab/90_colbert_dev_experiments.ipynb')); ast.parse(''.join(nb['cells'][20]['source'])); print('ok')"
git add colab/90_colbert_dev_experiments.ipynb
git commit -m "feat(nb90 9c): serve-faithful pool (canonical UNION_EC) + BGE_MODEL_PATH/MAX_LENGTH knobs"
```

---

### Task 7: Run + gate decision (human-run on Colab)

- [ ] **Step 1:** nb91 → build (smoke then full) → train → push.
- [ ] **Step 2:** nb90: `BGE_MODEL_PATH='OrRim123/recsys2026-bge-ce-music-v1'`, run cells `1,2,3,5,20`. ALSO run once with `BGE_MODEL_PATH=None` (OOD baseline on the SAME corrected pool — the true control).
- [ ] **Step 3: Gate (pre-registered):**
  - **PASS** → FT-bge retention @100→24 **>> 81.5%** (lite-LLM) AND final recall@20 **> 0.43** → Phase 2.
  - **FAIL** (flat vs OOD-bge on the corrected pool, or below the LLM) → **bank 0.50**.
- [ ] **Step 4:** Record result in memory `project_reranker_ceiling_resume_2026_06_16`.

---

## Phase 2 — Serve wiring (GATED on Task 7 PASS)

### Task 8: `reranker_corpus_types` (MAJOR-1 fix)

- [ ] In `crs_baseline.py`, add a `reranker_corpus_types` constructor param (default `None` → falls back to `self.corpus_types`). Pass it as the `corpus_types` arg in the `load_reranker_module(...)` call at line ~415 instead of `self.corpus_types`. Leave the retrieval call (line 409) on `self.corpus_types`. Add `reranker_corpus_types` to `load_reranker_module`'s signature, forwarded to `BGE_RERANKER(corpus_types=...)`.
- [ ] Test `tests/test_reranker_corpus_override.py`: construct with `reranker_corpus_types=[...,'tag_list']`, assert the bge reranker's `corpus_types` includes `tag_list` while the retrieval module's does not.
- [ ] `git commit -m "feat(crs): reranker_corpus_types — CE tag docs without altering retrieval"`

### Task 9: compact-query side-channel (BLOCKER-3 fix)

- [ ] In `crs_baseline.py` near line 750/941, build a `compact_queries` list (`mode='compact_colbert'`) alongside `retrieval_inputs`, and pass it to `reranker.rerank(...)` as a `reranker_queries=` kwarg. In the two-stage reranker, stage-1 (bge) consumes `reranker_queries`; stage-2 (LLM) keeps the full `retrieval_inputs`. Other rerankers accept-and-ignore the new kwarg (interface parity, like the existing side-channels).
- [ ] Test: assert the bge stage receives the compact query, the LLM stage the full query.
- [ ] `git commit -m "feat(crs): compact-query side-channel for the bge CE stage-1"`

### Task 10: config 224 + Blind

- [ ] Create `config/224-union-sasrec-colbert-noclap-nopg-bgeCE-stage1-llm-stage2-v5kto-blindA.yaml`: clone 219, `retrieval_topk: 300`, reranker = two-stage `{stage1: bge_reranker_v2_m3, model_path: OrRim123/recsys2026-bge-ce-music-v1, reranker_corpus_types: [...,tag_list], bge_max_length: 512, k1: <shortlist>}` → `{stage2: llm_listwise, gemini-2.5-flash, k2: 20}`.
- [ ] **Run full `pytest tests/`** before committing (config-222 lesson: a wrapper-signature crash a skipped test would have caught).
- [ ] Responder-silent Blind nDCG read; compare to the 0.29–0.33 band. Ship only if it clears noise.

### Task 11 (cleanup, user-requested): fix stale PG YAMLs

- [ ] Set `use_propose_ground: false` (+ remove `w_propose_ground`/`pg_*`) in configs 205–222, since PG is off in practice. Commit separately: `chore(config): set PG off to match live serve (was stale-on)`. Do NOT bundle with the CE work.

---

## Self-Review

- **Spec coverage:** all four review bugs mapped to fixes (table up top); train=gate=serve union via `UNION_EC` (Tasks 3, 6, 10); tags-in-docs without retrieval drift (Task 8); compact query at serve (Task 9); max_length matched (Task 1). Gate-first kill-switch (Task 7). PG cleanup (Task 11). ✅
- **Placeholder scan:** every code step has full code; commands have expected output. The `<shortlist>` in Task 10 is a value chosen from Task 7's best `(pool,k2)` — explicitly gated on the gate output, not a placeholder. ✅
- **Type consistency:** `build_tid_text_map(metadata_dict, corpus_types)` identical in Task 1 & 3; `select_ce_negatives`/`build_ce_row`/`flatten_groups`/`CEJsonlDataset`/`score_batch` match across tasks+tests; `bge_max_length` kwarg name consistent (Task 1 factory, Task 6 gate, Task 10 config); reused `listwise_softmax_loss`/`ce_group_metrics`/`resolve_max_norm` verified present in `train_cross_encoder.py`. ✅

**Known risk (unchanged):** Phase-1 mining runs the full union over the train set — hours of A100. `--max-rows 50000` subsamples for a faster v1 (biases negatives slightly). And the strategic caveat: even a clean gate PASS has historically not transferred to Blind — Task 7 is the honest kill-switch, Task 10's Blind read the final one.
