# RecSys Challenge 2026 — Music-CRS — Master Plan

> **Goal:** **nDCG@20 ≥ 0.55 on Blind B** (primary, hard target — leader-class territory), with **LLM-as-a-Judge** response quality as the strong secondary track. Top-10 leaderboard follows from hitting this.
> **Status:** Clean re-architecture after mid-dev issues. Prior approach plateaued well below target (≈ **0.1933** blind, vs. official `LLaMA-1B + BM25` baseline 0.0815); we rebuild around a clean **hybrid-retrieval → rerank → filter → respond** spine engineered specifically to clear 0.55.
>
> **What 0.55 actually demands (drives every priority below).** With a single gold track per turn, `nDCG@20 = recall@20 × E[1/log2(rank+1) | hit]`. So 0.55 is only reachable as a product of *both* a high recall ceiling *and* excellent top-rank precision — e.g. **recall@20 ≈ 0.72 with mean hit-rank ≈ 2** (`E[·]≈0.77`), or **recall@20 ≈ 0.80 at mean hit-rank ≈ 2.7** (`E[·]≈0.69`). Implications: (1) recall@20 must reach **~0.75+** (recall@200 far higher) — the recall wall is the make-or-break lever; (2) the reranker must routinely place the gold in the **top 1–3**, not just top-20; (3) marginal levers that don't move recall@20 *or* hit-rank are deprioritized regardless of how clever they are.
> **Owner:** Or · **Last updated:** 2026-06-16 · **Source of truth for metrics:** the official `music-crs-evaluator` repo.

---

## 0. How to use this document

This plan is written for execution by Claude Code on the local repo, with train/inference on HF Colab notebooks connected to the repo/branch, and models saved/loaded from the HF Hub. We fine-tune and ship **open-weight** models (via the Hub); **external LLM APIs (e.g. Gemini-lite) are permitted by the competition rules** and used — behind cost/cache gates — for catalog/doc enrichment, query refinement, and the response (see §6.2). Every phase has: an objective, exact deliverables, configs, tests, a code-review gate, and a "definition of done". Work fail-fast: the cheapest baseline that touches the whole pipeline first, then iterate on the highest-ROI lever.

**Repo starting point (branch `fresh-start`, as of 2026-06-16).** This is a deliberate clean reset. What's on disk:
- **`music-crs-baselines/`** — *pristine* official baseline (re-cloned from `nlp4musa/music-crs-baselines`, unmodified). The clean foundation: `mcrs/retrieval_modules/{bm25,bert}.py` (the only two retrievers it ships — **no fusion/RRF, no dense-precomputed**), `mcrs/{crs_baseline.py,db_item/music_catalog.py,db_user/user_profile.py,lm_modules/llama.py}`, runnable `run_inference_devset.py` / `run_inference_blindset.py`, `config/` (4 official YAMLs), `lowerbound/{popularity,random_sample}.py`, and the official **`tips/`** notes (`add_reranker.md`, `improve_item_representation.md`, `use_genrec_semantic_ids.md`). RRF fusion + all extra channels are reimplemented in our own `mcrs/` package (see §6.3).
- **`music-crs-evaluator/`** — *pristine* official scorer: `evaluate_devset.py`, `make_ground_truth.py`, `metrics/{metrics_recsys.py,metrics_diversity.py}`. **This is the only local source of truth for metrics.**
- **`data/`** — official `TalkPlayData-Challenge-*` datasets on disk (gitignored); re-fetch via `download_data.py` / `download_data.sh`.
- Env: `requirements.txt`, `setup_venv.sh` (venv `recsys26`, Python 3.10). New work lands in fresh `nb/` notebooks + a rebuilt `mcrs/` (see §6); prior-work code is reintroduced only behind an ablation gate.

**Non-negotiable rules (read before any code):**
1. Retrieve candidates from the **entire** track catalog (`track_split_types: ["all_tracks"]`). Never subset/filter the catalog universe at retrieval time — doing so invalidates the submission. (Post-retrieval *reranking/pruning to the top-20* is allowed and expected; that is not catalog subsetting.)
2. **No leakage, ever.** A turn-`t` prediction may only use utterances `1..t`, the user profile, and listening history available *up to that point*. Never use the turn's own gold track or any future turn. (See §11.)
3. **Train on Train, select on Development, sanity-check on Blind A, never tune on Blind B.** Use the official evaluator as the *only* local scorer so local == leaderboard.
4. Every code change: **unit tests + code review** before merge (§14).
5. Stay inside the competition rules: open-weight models for anything we train and ship; external LLM APIs are allowed and used cost-gated (§6.2, §15).

---

## 1. TL;DR strategy

The task is **next-track prediction conditioned on a multi-turn dialogue + user profile + listening history**, where each turn has exactly **one** ground-truth track. That single-relevant-item structure means:

- **Recall@K of the candidate pool is the ceiling.** If the gold track isn't in our top-K candidates, no reranker can recover it. → Invest first in *high-recall hybrid retrieval over all 50k tracks*.
- **Then ranking precision is the multiplier.** Moving the gold item from rank ~15 to ~3 is a large nDCG@20 jump (nDCG@20 = `1/log2(rank+1)` for a single hit). → A **gradient-boosted ranker (LambdaMART/LightGBM)** that fuses all signals is the single highest-ROI component — cheap, CPU-only, robust, no GPU.
- **Diversity is complementary and lower-weight.** Don't trade nDCG for it; harvest it "for free" from tail ranks (positions 11–20 rarely hold the gold item) and from genuine per-user personalization.
- **Response quality (Gemini judge + Distinct-2) is a separable second track.** Generate grounded, personalized, varied explanations with Gemini *lite*; it doesn't touch retrieval scoring.

**The spine (minimal, robust, high-performance):**

```
EDA → [Retrieval: BM25(enriched) ⊕ Dense-text(BGE/E5) ⊕ Content-kNN(history) ⊕ CF(user emb)]
     → weighted-RRF fuse → top-K (sized by recall probe; §7.3.1)
     → [Rerank: LightGBM LambdaMART over fused features  (+ optional cross-encoder/ColBERT)]
     → [Filter: dedup, uniqueness, history rule (A/B), tail-diversify]
     → top-20  →  [Respond: Gemini-lite, personalized explanation]
     → submission JSON
```

Pre-computed track (multimodal audio/content) and user (CF) embeddings are provided — we exploit them directly to avoid training expensive encoders.

---

## 2. Competition facts (anchored to official sources)

### Task
Conversational Music Recommendation. Two outputs per session-turn: (a) a **ranked list of top-20 track_ids** from the catalog; (b) a **natural-language response** justifying the picks.

### Metrics & exact computation
The final score combines several dimensions; **every listed dimension contributes** to the ranking, with **recommendation quality (nDCG@20) as the primary anchor** and **LLM-as-a-Judge** the heaviest response dimension.

| Dimension | Definition | Notes for us |
|---|---|---|
| **nDCG@20** *(primary)* | `nDCG@k = DCG@k / IDCG@k`; the official `metrics_recsys.get_ndcg` implements the **binary** form `DCG@k = Σ_{i=1..k} rel_i/log2(i+1)`, `rel_i ∈ {0,1}` (identical to the `(2^rel−1)/log2` gain form for a single binary gold). **Single** gold per turn. Aggregation (verified against `evaluate_devset.py`): **turn-bucket mean → mean across turn buckets — NO session-level average.** | Single relevant item ⇒ nDCG@20 = `1/log2(rank+1)` if gold in top-20 else 0. Recall@20 is the ceiling; rank is the multiplier. Also tracked at k=1,10. **Recall is our internal diagnostic/ceiling — the official scorer computes nDCG only (recall/hit/mrr/map are disabled in the evaluator), so our recall gates are not leaderboard numbers.** |
| **LLM-as-a-Judge** *(heavy, response)* | Gemini judge scores **Personalization** + **Explanation Quality** of the text response only, independent of rec accuracy. Prompt is *not* published. | Optimize genuine grounding (profile + retrieved tracks + dialogue), not a guessed judge prompt. |
| **Catalog Diversity** *(complementary)* | unique recommended tracks ÷ total catalog size (0–1). | Don't over-optimize. Popularity baseline = 0.0004 (collapse); Random = 0.965. |
| **Lexical Diversity** *(complementary)* | **Distinct-2** = unique bigrams ÷ total bigrams across responses. | Vary phrasing; avoid templated boilerplate. |

**Aggregation:** retrieval metrics per session-turn → averaged within turn → macro-averaged across turns; diversity computed globally over the full prediction file. **Replicate this exactly locally using the official evaluator — don't reimplement.** Source of truth in-repo: `music-crs-evaluator/metrics/metrics_recsys.py` (nDCG/recall), `music-crs-evaluator/metrics/metrics_diversity.py` (catalog + Distinct-2), `music-crs-evaluator/evaluate_devset.py` (the scoring entrypoint), `music-crs-evaluator/make_ground_truth.py` (builds dev gold from the dataset). Read these first to pin exact formulas/averaging; our metric-parity test (§14) asserts our scores == this code on a fixture.

### Data: TalkPlayData-Challenge (verify live on HF before coding)
*On disk under `data/` (gitignored): `TalkPlayData-Challenge-{Dataset, Track-Metadata, Track-Embeddings, User-Metadata, User-Embeddings, Blind-A}`. Re-fetch with `download_data.py` / `download_data.sh`. Catalog accessor: `music-crs-baselines/mcrs/db_item/music_catalog.py`; user accessor: `mcrs/db_user/user_profile.py`. Baseline reference configs to clone the run-config shape from: `music-crs-baselines/config/llama1b_{bm25,bert}_{devset,blindset_A}.yaml`.*
- **Conversations:** multi-turn dialogues, **avg ~8 turns**; fields `session_id` (`{user_id}__{date}`), `user_id`, `turn_number` (1–8), utterance text, plus **conversation goal** and **goal-progress** assessments. Dev split ≈ **1k sessions**.
- **Track metadata:** **⚠ size discrepancy** — main site says ">1M tracks", evaluator/baseline says ~50.4k; **on-disk `all_tracks` verified = 47,071 tracks** (`dataset_info.json`), i.e. ≈47k — **not 1M**. Brute-force dense over the full catalog is therefore trivial (one matmul; no ANN). P0 still confirms the Blind universe uses the same `all_tracks`. **Fields are List[str]** (`track_name`, `artist_name`, `album_name`, `tag_list`, `artist_id`, `album_id`, `ISRC`) + scalar `popularity`, `release_date`, `duration` — doc builders must join the lists (see F1). 6 track-embedding modalities incl. `cf-bpr`.
- **User profiles:** ≈ **9.09k users**: `user_id`, age, gender, country, listening-history `track_id` list.
- **Pre-extracted embeddings:** multimodal **track** embeddings (audio/content) + **user** embeddings (collaborative filtering). Use directly.

### Splits
**Train**, **Development** (public, has local ground truth via `make_ground_truth.py`), **Blind A** (interim leaderboard, CodaBench), **Blind B** (final, released ~15–23 Jun — *verify*; final leaderboard computed on B).

### Submission JSON (strict)
List of objects: `session_id` (str), `user_id` (str), `turn_number` (int 1–8), `predicted_track_ids` (ordered, **unique**, ≤20, valid catalog IDs), `predicted_response` (str, may be empty). One entry per **every** session×turn. `json.dump(..., ensure_ascii=False)`.

### Timeline (⚠ sources disagree — verify on CodaBench)
Blind B release **15 Jun (baseline repo)** vs **23 Jun (main site)**; challenge ends **30 Jun**; final leaderboard **6 Jul**; code upload **9 Jul**; paper **20 Jul**. **We have ~2 weeks — front-load the high-ROI levers.**

### Baseline numbers (devset) to beat
Random nDCG@20 0.0001 · Popularity 0.0024 · **LLaMA-1B + BM25 0.0815**. Our current best ≈ **0.1933** (blind). Top-10 target: push materially above this via recall + reranking.

---

## 3. Insights that drive the design

1. **Single gold item ⇒ retrieval recall is king, then rank precision.** Budget effort accordingly.
2. **50k catalog ⇒ full-catalog dense retrieval is a single matrix multiply on GPU** (≈ seconds). No FAISS/ANN approximation, which also removes an approximation/leak surface. If the catalog is truly ~1M, add FAISS-IVF/HNSW then.
3. **Embeddings are a gift.** Provided track (audio/content) + user (CF) embeddings let us get strong personalization and content similarity with *zero* encoder training. Train only the cheap fusion layer (GBDT).
4. **Conversation carries explicit + latent intent.** Explicit: artist/genre/era/mood mentions, "more like X", "something upbeat". Latent: goal field, trajectory across turns. Extract both (rules + LLM query refinement).
5. **Response track is fully decoupled** from retrieval scoring — parallelize its development; it only needs the final top-20 + context.
6. **Diversity tradeoff:** personalization naturally spreads catalog coverage; tail-rank diversification is nearly free for nDCG@20.

---

## 4. Success criteria / definition of done

- **Primary:** **nDCG@20 ≥ 0.55** on Blind (dev as the selection proxy). Intermediate gates that make 0.55 plausible: **recall@20 ≥ 0.75** and **recall@200 ≥ 0.90** after fusion (Phase 1); reranked **mean hit-rank ≤ ~2.5** and devset nDCG@20 ≥ 0.45 before levers, ≥ 0.55 after (Phase 2+). If a phase can't move recall@20 or hit-rank toward these, it doesn't ship.
- **Pipeline DoD:** local nDCG (official evaluator) within rounding of leaderboard on Blind A; submission passes the validation checklist every time; full run reproducible from a single config + notebook.
- **Per-phase DoD:** stated at the end of each phase below. A phase is "done" only with tests green + code review approved + results logged to memory (§18).

---

## 5. Phase 0 — Data understanding & EDA (do this *before* any modeling)

**Objective:** know the data cold; catch leakage traps; size every design choice (context length, candidate K, cold-user share) to the *actual* data.

**Steps & questions to answer (notebook `nb/phase0_eda.ipynb`, config `config/eda.yaml`):**
1. **Catalog size & integrity:** exact track count; confirm 50k vs 1M; embedding dims (track, user); any `track_id` in conversations missing from metadata/embeddings; duplicate tracks.
2. **Ground-truth structure:** confirm **one gold track per turn**; how it's defined (the track played/selected at turn `t`?); whether gold can equal a track already in the user's history (decides the "filter history" A/B). Inspect `make_ground_truth.py` output directly.
3. **Conversation anatomy:** turn-count distribution (confirm 1–8); utterance token-length distribution per turn and **cumulative context length** (drives encoder truncation policy, §8); language(s); presence/format of the **conversation goal** and **goal-progress** fields; how much explicit metadata (artist/genre/era) appears in utterances vs. pure mood/semantic intent.
4. **User profiles & cold/warm split:** history-length distribution → define cold vs warm thresholds (§10); share of users with empty/degenerate CF embedding; demographic coverage (country/age/gender).
5. **Track metadata quality:** tag vocabulary & coverage; release-date range; popularity distribution (long tail?); missing fields.
6. **Embedding sanity:** are provided track text vs audio embeddings? norm distribution; do nearest-neighbors of a track make musical sense (artist/tag coherence)? does user-CF·track-emb produce sensible personalized neighbors?
7. **Recall ceiling probe (critical):** for the dev gold tracks, measure recall@{50,100,200,500} of each candidate generator alone (BM25, dense-text, content-kNN-from-history, CF) and of RRF fusion. **This tells us K and which retrievers matter** before we build anything heavy.
8. **Leakage audit:** verify Train/Dev/Blind session disjointness; confirm no future-turn fields leak into per-turn inputs.

**Deliverables:** `reports/eda.md` (findings + decisions: K, context-length cap, cold/warm thresholds, which retrievers to keep), recall-ceiling table, plots.
**Tests:** schema/asserts on row counts, ID join coverage, "single gold per turn" assertion, no-leak split check.
**DoD:** every Phase-1+ design parameter is justified by an EDA number, recorded in memory.

---

## 6. Architecture overview

Two-stage retrieve-then-rank, plus a decoupled responder — mirrors and extends the official baseline (`mcrs/retrieval_modules`, `mcrs/lm_modules`, `db_user`, `db_item`).

```
                       ┌─────────────── per session-turn t ───────────────┐
 user profile ─┐       │  context = utterances[1..t] + goal + history≤t    │
 history≤t ────┼──────▶│                                                    │
 utterances≤t ─┘       │  (A) Query construction:                          │
                       │      rules + LLM query-refinement (Gemini-lite)    │
                       │  (B) Candidate generation over ALL tracks:         │
                       │      BM25(enriched docs) | Dense-text(BGE/E5)      │
                       │      | Content-kNN(history) | CF(user emb)         │
                       │  (C) RRF fuse → top-K (§7.3)                       │
                       │  (D) Rerank: LightGBM LambdaMART (feature fusion)  │
                       │      (+optional BGE-reranker/ColBERT cross-enc)    │
                       │  (E) Filter: dedup/unique, history A/B, tail-MMR   │
                       │  (F) top-20 ranked track_ids                       │
                       │  (G) Respond: Gemini-lite explanation              │
                       └────────────────────────────────────────────────────┘
                                          │
                                          ▼
                          submission JSON (strict schema)
```

Repo layout — **build the clean spine on the pristine baseline; reintroduce prior-work code only behind an ablation gate.** Target structure (new code under `mcrs/` extending the pristine baseline package; do not edit the `music-crs-{baselines,evaluator}/` pristine copies):
```
music-crs-baselines/   PRISTINE — foundation: mcrs/{retrieval_modules/{bm25,bert}(only these),
                       crs_baseline,db_item/music_catalog,db_user/user_profile,lm_modules/llama},
                       run_inference_{devset,blindset}.py, config/, lowerbound/, tips/
music-crs-evaluator/   PRISTINE — scorer: evaluate_devset.py, make_ground_truth.py, metrics/
mcrs/  (new, ours)     retrieval_modules/  rerank_modules/  filter_modules/
                       lm_modules/(query_refine, responder)  enrich/  # extends baseline
config/                *.yaml per experiment   (shape from baseline config/*.yaml)
nb/                    one self-contained notebook per phase + shared inference/experiments (see §6.1)
                       phase0_eda.ipynb  phase1_retrieval.ipynb  phase2_rerank.ipynb
                       phase4_responder.ipynb  inference_blindA.ipynb  experiments.ipynb
exp/                   inference/{devset,blindA,blindB}/  scores/  ground_truth/
tests/  reports/  cache/  data/(gitignored)   lora_adapters/(gitignored; canonical copy → HF Hub)
```

### 6.1 Notebook architecture (one self-contained notebook per phase)

Each phase that trains a model gets **its own self-contained notebook** — clone repo/branch, install, train, checkpoint, save, eval, all in one runnable artifact. No cross-notebook hidden state. GPU runs on Colab; the **HF Hub is the source of truth for every trained artifact** (Colab disk is ephemeral). All foundation-model fine-tuning uses **LoRA/PEFT** (small adapters, fast, cheap, Hub-loadable); the LightGBM reranker follows the same train→save→eval discipline on CPU (no LoRA — it's not a foundation model).

**Per-phase training notebook — canonical skeleton (`nb/phaseN_<name>.ipynb`):**
1. **Setup** — **mount Google Drive** (every notebook: set `HF_HOME` to a Drive path so the dataset/model cache persists, and write all outputs/checkpoints/reports under a Drive `outputs/` dir — Colab disk is ephemeral), clone repo@branch, `pip install` deps, auth to HF Hub via **Colab Secrets** (`userdata.get('HF_TOKEN')` → `huggingface_hub.login`; Gemini via `userdata.get('GEMINI_API_KEY')` when the notebook needs the responder/enrichment) — **never hardcode keys**, set seeds, read `config/<exp_id>.yaml`. Load datasets from HF (`load_dataset`) and pass them to the F1 loaders' row constructors — don't depend on a local `load_from_disk` layout.
2. **Data** — load from `data/` / HF datasets; build the phase's train/val split (session-disjoint, causal, no leakage per §11); assert schema.
3. **Model + LoRA** — load the open base foundation model from the Hub; attach a **LoRA/PEFT** adapter (rank/alpha/dropout/target-modules in config); print trainable-param count.
4. **Train** — TRL/PEFT trainer with **checkpointing**: `save_steps`/`save_total_limit`, **resumable** from the latest checkpoint (Colab can disconnect), eval-on-steps, early stopping; checkpoints to Drive/`lora_adapters/` and mirrored to the Hub.
5. **Save (end of train)** — push the **final adapter + tokenizer + config** to a versioned HF Hub repo (`recsys2026-<phase>-<exp_id>`); optionally a merged model. Record the commit hash/revision.
6. **Eval** — score on dev with the **official evaluator** (`music-crs-evaluator`) for the phase's anchor metric (recall@K / nDCG@20 / proxy-judge); write `exp/scores/<exp_id>.json` + a `reports/experiments.md` row; assert against the phase DoD gate (§4).
7. **Done criteria** — adapter on Hub + metrics logged + gate check printed.

**Shared notebooks:**
- **`nb/inference_blindA.ipynb`** — the submission path: load the chosen trained adapter(s) **by Hub revision**, run the full `retrieve → rerank → filter → respond` pipeline over **Blind A**, write the strict submission JSON, validate it via `mcrs/run/harness.py` (`validate_submission`) + the notebook blind guards, and (optionally) the Blind-A score tracker. One config in → one validated `prediction.json` out. (A Blind B variant is a one-line dataset swap at the end.)
- **`nb/experiments.ipynb`** — scratch/ablation space: sweep configs, compare encoders/features/levers, quick recall-ceiling probes. Promising settings get promoted into a phase notebook's `config/<exp_id>.yaml`; this notebook is **not** a source of canonical artifacts.

*(`phase0_eda.ipynb` is analysis-only — no training; `phase3_filter` is rule/config logic exercised inside the experiments + inference notebooks rather than a trainer. When building these, lean on the `huggingface-llm-trainer` skill for the TRL/PEFT + Hub-saving patterns.)*

### 6.2 LLM augmentation across the pipeline (cross-cutting)

LLMs (open-source on Colab, or a hosted lite model — competition rules permit external APIs) are a **first-class tool at every stage**, applied to the **catalog, the query, and the documents** — not just the final response. Each use ships behind the same discipline: **ablation-gated** (keep only on a devset recall@K / nDCG@20 / judge win), **cached** by content hash, **cost-capped**, and **causal** (per-turn LLM inputs use only utterances ≤t). Offline catalog/doc passes are one-time over ~50k tracks; per-turn passes are cached per (session, turn).

| Stage | LLM applied to | What it does | Existing code to port |
|---|---|---|---|
| **Catalog / docs** (offline, cached) | the catalog & track docs | doc2query expansion; descriptive blurbs / inferred mood-genre tags for sparse tracks; metadata normalization → richer BM25 + dense docs (closes the conversational↔metadata vocabulary gap — a big recall lever) | `mcrs/enrich/doc2query.py` + `nb/a1_enrich_catalog.ipynb` |
| **Query** (per-turn, cached) | the dialogue | structured intent extraction (`{positive/negative attrs, seed_artists, mood, genre, era}`); query rewrite/expansion; **HyDE** pseudo-doc as a dense query; artist/genre hypothesis; multi-query | recoverable from old git branches (`recall-union-lgbm`, `exp/*`) |
| **Retrieval** (channel) | query → seed items | LLM proposes seed tracks/artists that become a retrieval channel (propose-ground); HyDE doc feeds the dense channel | recoverable from old git branches (`recall-union-lgbm`, `exp/*`) |
| **Reranking** | (query, candidate) | LLM listwise/pointwise reranking of the top-K; LLM relevance score folded in as a **GBDT feature** (stacking, §9.2) rather than replacing the GBDT | recoverable from old git branches (`stage-b-cross-encoder`, `exp/*`) |
| **Filtering** | candidate set + intent | intent-consistency / constraint checks (era/explicit/language/"not too slow"); LLM-assisted dedup of near-duplicate tracks — within the retrieved set only (never re-opens the catalog, §11) | new (small), gated |
| **Response** | top-20 + profile | grounded, personalized, varied explanation (the LLM-judge track) | `mcrs/lm/responder.py` |

**Guardrails:** prompt-injection-safe (sanitize track names/utterances before templating); train/serve use the **same** LLM revision + prompt + truncation (alignment rule, §8); the LLM never sees a future turn or the gold track; and an LLM lever that doesn't move recall@K, hit-rank, or the judge score in ablation is dropped (it adds latency/cost otherwise). See §7.1 (query), §9.2 (rerank), §11 (filter), §12 (catalog/doc levers), §13 (response) for the per-stage detail.

### 6.3 Asset reuse map — pipeline component → existing code → status

> **Reuse policy:** prior-work modules are working *implementations* — treat each as a fast head-start to re-evaluate fresh against the rebuilt spine, **not** as settled results (we're not bound by prior outcomes). **Reuse** = clean, generic infra worth keeping. **Port behind gate** = adopt if it earns an ablation win here and now. **Rebuild** = re-architect fresh. The pristine baseline is always the starting foundation. Every component gets a clean, current ablation — nothing is pre-judged.

| Component (plan §) | Existing code | Status |
|---|---|---|
| Scorer / ground truth (§2, §14) | `music-crs-evaluator/{evaluate_devset,make_ground_truth}.py`, `metrics/*` | **Reuse verbatim** (source of truth) |
| Inference harness (§14, §18) | `music-crs-baselines/run_inference_{devset,blindset}.py`, `crs_baseline.py` | **Reuse / extend** as the run spine |
| BM25 sparse (§7.2.1) | pristine `…/retrieval_modules/bm25.py`; doc-format helpers recoverable from old git branches (`recall-union-lgbm`) | **Reuse** (BM25); port doc-format helpers |
| Dense-text (§7.2.2) | pristine `…/retrieval_modules/bert.py`; dense-encoder variants recoverable from old git branches (`recall-union-lgbm`) | **Port behind gate** (pick encoder per §7 ablation) |
| Content-kNN / history (§7.2.3) | `mcrs/retrieval/related_artist.py`; session-history / session-CF / same-artist variants recoverable from old git branches (`recall-union-lgbm`) | **Port behind gate** |
| CF retriever (§7.2.4) | `cf_bpr` (user-emb · track-emb) recoverable from old git branches (`recall-union-lgbm`) | **Port behind gate** |
| Audio/CLAP channels (§7) | CLAP recall/similarity/text channels recoverable from old git branches (`recall-union-lgbm`) | **Port behind gate** |
| RRF fusion (§7.3) | weighted RRF (**not in pristine baseline**) recoverable from old git branches (`recall-union-lgbm`) | **Port + verify** (math checked correct; size `topk_internal`/K per §7.3.1) |
| LightGBM LambdaMART rerank (§9.1) | LGBM reranker + relevance scorer recoverable from old git branches (`recall-union-lgbm`) | **Port behind gate** (primary reranker) |
| Cross-encoder / ColBERT rerank (§9.2) | ColBERT training/index → `mcrs/training/colbert_{data,finetune,index}.py` + `nb/phase2_colbert_finetune.ipynb`; cross-encoder / late-interaction variants recoverable from old git branches (`stage-b-cross-encoder`) | **Port behind gate** (this is where prior work went deep) |
| Listwise / two-stage rerank (§9.2) | listwise / two-stage / chain rerankers recoverable from old git branches (`stage-b-cross-encoder`) | **Port behind gate** |
| Sequential model (SASRec) (§7/§9 feature) | SASRec model/seq + training recoverable from old git branches (`fresh-model`, `exp/*`) | **Port behind gate** |
| Query refinement (§7.1, §12) | query rewriters (structured-query / intent-state / HyDE / CMQR / propose-ground / …) recoverable from old git branches (`recall-union-lgbm`, `exp/*`) | **Port behind gate** |
| Catalog enrichment / doc2query (§12) | `mcrs/enrich/doc2query.py` + `nb/a1_enrich_catalog.ipynb` (Gemini doc2query, resumable) | **Port behind gate** |
| Catalog embedding (§7, §15) | embed-catalog pass folded into `nb/a1_enrich_catalog.ipynb` | **Reuse / port** (one-time, cached to Hub) |
| Responder (§13) | `mcrs/lm/responder.py`; pristine `lm_modules/llama.py` | **Port behind gate** (the LLM-judge lever) |
| Submission validation (§11, §14) | `mcrs/run/harness.py` (`validate_submission`) + notebook blind guards | **Reuse** (schema/precheck guard) |
| Score tracking (§18) | Blind-A score tracker recoverable from old git branches (`recall-union-lgbm`) | **Reuse** |
| Semantic-ID / generative retrieval (§12) | SID generator recoverable from old git branches (`exp/*`); baseline `tips/use_genrec_semantic_ids.md` | **Research lever** — pursue if it helps clear 0.55 |
| Lowerbounds (sanity) | pristine `lowerbound/{popularity,random_sample}.py` | **Reuse** (baseline floors) |

---

## 7. Phase 1 — Retrieval (recall-oriented candidate generation)

**Objective:** maximize recall@K of the gold track over the **full catalog**, cheaply and robustly. This phase produces the biggest nDCG gains.

### 7.1 Query construction (per turn, causal)
- **Rule-based core (free, fast):** concatenate utterances `1..t` with recency weighting (most-recent turn boosted); pull explicit entities (artist/genre/era/mood terms) via metadata-vocabulary matching; append the conversation **goal** field.
- **LLM query refinement (Gemini-lite, optional, cached):** turn messy multi-turn dialogue into a clean structured query: `{positive_attrs, negative_attrs (e.g. "not too slow"), seed_artists, mood, genre, era}`. Few-shot + CoT prompt (prompt-design discipline in §13). Cache per (session,turn). Only enable if EDA/ablation shows lift — fail-fast without it first.

### 7.2 Candidate generators (each over `all_tracks`)
1. **BM25 (sparse)** over **enriched** track docs: `track_name + artist_name + album_name + tags + release_date(+ doc2query expansions, §12)`. Matches the baseline `corpus_types`. Strong for explicit artist/genre/title intent. *(Ref: Robertson & Zaragoza 2009.)*
2. **Dense-text bi-encoder:** embed enriched track doc and the constructed query with a strong open encoder. **Default: BGE-large-en-v1.5** (or `bge-m3` if dialogues are multilingual); alternative **E5-large-v2 / GTE-large**. Respect `query:`/`passage:` prefixes (E5) and the 512-token cap (§8). Brute-force cosine over 50k = one matmul. *(Refs: Xiao et al. BGE 2023; Wang et al. E5 2022.)*
3. **Content-kNN from history (provided track embeddings):** mean/recency-weighted pool of the user's recent track embeddings → nearest catalog tracks. Captures "more of what they've been playing". Warm-user signal.
4. **CF retriever (provided user embedding):** `user_emb · track_emb` top-N. Personalization prior. Degrades for cold users → gated by history length.

### 7.3 Fusion — (weighted) Reciprocal Rank Fusion
`score(d) = Σ_r w_r / (k + rank_r(d))`, `k≈60`, rank 1-indexed, a doc absent from sub-`r` contributes 0. Rank-based ⇒ robust, no score-scale calibration, no training. With `w_r ≡ 1` this is vanilla RRF; **weights `w_r` are the one real fusion knob** (a strong channel whose gold sits at rank ~30 gets swamped by channels stacking the top — so under-weighting a strong channel silently caps recall). *(Ref: Cormack et al. 2009. Impl: weighted RRF (`RRF_MODEL` / `fuse_per_sub`, caches per-sub rankings so the weight sweep re-fuses cheaply) recoverable from old git branches (`recall-union-lgbm`). Note: the official baseline ships NO fusion — this code is prior-work-only and must be re-verified, not assumed correct.)*

**Fusion-correctness requirements (must all hold or recall numbers lie):**
1. **One shared ID space.** Every channel must emit *canonical* catalog `track_id`s over the **same** universe (`all_tracks`) before fusion. *(Verified: `bm25.py` returns canonical metadata keys, and `fuse_per_sub` does not re-normalize — it relies on channels being canonical. `strip_track_id_prefix` in `colbert_late.py` is a doc-**text** helper, not an id normalizer, so don't lean on it for ids.)* If two channels disagree on a track's id, RRF double-counts or misses it — **add a fusion-time test asserting every per-sub output ⊆ catalog id set**; any channel that derives/prefixes ids must canonicalize before returning.
2. **Drop noise channels.** A channel that doesn't add *unique* recall injects false top-rank items that displace golds — RRF is not free to add to. Keep a channel only if it lifts fused recall@K in the ablation (§7.4); otherwise remove it (or down-weight to ~0).
3. **Weights tuned, not guessed.** Sweep `w_r` (and `k`) on dev *recall@K* via the cached re-fusion path; lock the chosen weights in `config/<exp_id>.yaml` so train and serve fuse identically. Segment-aware weights (cold vs warm) only if the ablation shows a per-segment win.
4. **Determinism.** Fixed sub-orderings ⇒ stable tie-break; assert reproducibility.

### 7.3.1 Candidate-pool top-K — size it correctly (two-tier)
There are **two distinct K's**, and getting them wrong is a silent recall ceiling:
- **`topk_internal` (per-channel pull depth)** — how many results each sub returns *before* fusion (`RRF_MODEL` default is **60**, which is too shallow for our target). A gold ranked 61–500 in a channel is dropped before fusion ever sees it. **Set `topk_internal ≥ fusion-K` (target 300–500)**, sized from the per-channel recall-at-depth probe (§5.7).
- **Fusion pool K (the reranker's candidate set)** — set **K = the smallest K where fused recall@K ≥ 0.90** (the §7 DoD), measured by the probe — *not* a hard-coded 200. Likely **300–500** given the 0.55 target; LightGBM (CPU) handles a large pool fine.
- **Two-tier hand-off to rerank (§9):** the **GBDT** reranks the full large pool (K≈300–500, cheap); the **optional cross-encoder/ColBERT** only re-scores the GBDT's **top ~100–200** (cost scales with K). So: wide for recall, narrow for the expensive precision stage — never let the cross-encoder budget shrink the recall pool.

**DoD for K:** chosen `topk_internal`, fusion-K, and cross-encoder-K are each justified by a probe number and recorded in config; `topk_internal ≥ fusion-K ≥ cross-encoder-K`; raising K further yields < +0.5% recall (pool saturated).

### 7.4 Experiments (log each to memory)
- E1: BM25-only (reproduce/upgrade baseline) → confirm pipeline + local==leaderboard.
- E2: Dense-text only (BGE vs E5 vs GTE).
- E3: + Content-kNN; E4: + CF; E5: RRF all. Pick the subset that maximizes recall@K with fewest parts.

**Tests:** retrieval returns valid catalog IDs only; uniqueness; recall@K parity vs the EDA probe; no-leak (turn-t query uses only ≤t); RRF determinism.
**DoD:** fused **recall@20 ≥ 0.75** and **recall@200 ≥ 0.90** on devset (the ceiling a 0.55 nDCG@20 needs — if recall@200 falls short, no reranker can recover it, so this gate blocks Phase 2 until met: add channels/enrichment until cleared). Decision logged on which retrievers to keep.

---

## 8. Sequence / context-length discipline (cross-cutting, but decide here)

- **Encoders** (BGE/E5/GTE) cap at **512 tokens**. Track docs are short (safe). The **dialogue context grows with turns** — measure cumulative length in EDA (§5.3). Policy: keep the **most recent turns verbatim**, compress older turns (recency window or a one-line LLM summary), always keep the goal field. Never silently truncate the *latest* utterance (it carries the active intent).
- **ColBERT / cross-encoder** (if used, §9 rerank): respect its own max length; truncate the *document* side, preserve the query side.
- **Gemini responder/refiner:** lite models have large context, but keep prompts tight for cost (§15). 
- Document the chosen caps in `config/*.yaml` so train and inference use **identical** truncation (alignment rule).

---

## 9. Phase 2 — Reranking (precision-oriented; the nDCG multiplier)

**Objective:** reorder the K candidates so the gold track lands as high as possible (ideally rank 1–3).

### 9.1 Primary reranker — LightGBM LambdaMART (learning-to-rank)
Cheap, CPU, fast, robust, fuses heterogeneous signals — best ROI. *(Refs: Burges 2010 LambdaMART; Ke et al. 2017 LightGBM. Official "tips": `music-crs-baselines/tips/add_reranker.md`. Existing impl to port (LGBM reranker + relevance scorer) recoverable from old git branches (`recall-union-lgbm`).)*

**Group:** one group per (session, turn). **Label:** 1 for the gold track, 0 otherwise. **Negatives:** hard negatives = the other K−1 retrieved candidates (in-distribution, the realistic confusables). **Objective:** `lambdarank`, eval `ndcg@20`.

**Features per (turn, candidate)** — all causal:
- Retrieval signals: BM25 score, dense-text cosine, content-kNN sim, CF score, RRF score, and each retriever's *rank*.
- Content match: tag overlap (query↔track), artist match (was artist mentioned?), genre/era/mood match, title/lexical overlap.
- Personalization: candidate↔history-pool similarity; candidate popularity; user-CF·candidate.
- Context: turn_number, history length (cold/warm), goal-type one-hot, query length.
- Track priors: popularity (log), recency.

**Anti-overfit:** session-level CV (never split a session across folds), early stopping on a Train-internal holdout (keep Dev clean for final selection), shallow trees + L1/L2 + feature/bagging subsample; watch train-vs-val nDCG gap (overfit) and absolute level (underfit). Log feature importances; prune dead features.

### 9.2 Optional neural reranker (only if GBDT plateaus and time allows)
- **Cross-encoder:** `BAAI/bge-reranker-v2-m3` scoring (dialogue-context, enriched-track-doc) pairs on the top-K. Strongest single semantic reranker; GPU but only K≈100–200 pairs/turn. *(Ref: BGE-reranker.)*
- **ColBERT late-interaction:** token-level MaxSim between query and enriched doc — middle ground between bi- and cross-encoder, good when intent is phrase-level. *(Ref: Khattab & Zaharia 2020.)*
- **LoRA option:** fine-tune the cross-encoder (or a query bi-encoder head) with **LoRA** on Train to save GPU/time vs full fine-tune; tiny adapter, fast, HF-Hub-loadable. Gate behind a clear ablation win.
- **Stacking:** feed the cross-encoder score back in as a feature to the GBDT rather than replacing it — usually the best of both.

**Tests:** group construction has exactly one positive per group; no candidate leaks the label; feature functions are pure & causal; rerank improves devset nDCG@20 vs RRF order; deterministic given seed.
**DoD:** reranked devset **nDCG@20 ≥ 0.45** with **mean hit-rank ≤ ~2.5** (clearly beats RRF-only); on track to ≥ 0.55 after the §12 advanced levers. CV gap controlled; importances sane.

---

## 10. Cold vs warm users

- **Define** thresholds from EDA history-length distribution (e.g. cold = 0–N history tracks / degenerate CF embedding).
- **Warm:** full pipeline; CF + content-kNN carry strong personalization.
- **Cold:** down-weight/disable CF & history-kNN (avoid degenerate-embedding noise); lean on **conversation intent** (BM25 + dense-text) + **demographic popularity priors** (country/age-conditioned popular tracks as a soft prior, never as a catalog filter). Add an `is_cold` feature so the GBDT learns the regime switch itself.
- Track nDCG@20 **separately for cold vs warm** in every experiment to catch a segment regressing.

---

## 11. Phase 3 — Filtering (within retrieved candidates only)

**Objective:** clean, rule-correct, schema-valid top-20 — *without* touching the catalog universe.

- **Dedup & uniqueness** (hard requirement); ensure exactly 20 filled (backfill from next-best candidates if short).
- **History rule (A/B):** test removing tracks already in the user's history. In music, replays are real, so this may *hurt* — decide empirically per EDA §5.2 + ablation, possibly conditioned on the gold-can-be-in-history finding.
- **Tail diversification (free diversity):** apply light **MMR** only to positions ~11–20 (where the gold item is unlikely) to lift catalog-diversity without risking top-rank nDCG. Top 1–10 stay purely relevance-ranked. *(Ref: Carbonell & Goldstein 1998.)* Gate behind "nDCG@20 not harmed".
- **Validity guard:** every `track_id` ∈ catalog.

**Tests:** output always 20 unique valid IDs; tail-MMR never reorders top-10; history-filter toggle covered; nDCG@20 non-regression assertion.
**DoD:** submission validator passes; diversity improves or holds with zero nDCG@20 loss.

---

## 12. Advanced levers (query expansion · doc2query · catalog enrichment)

All **one-time, cached** (compute once over 50k tracks, reuse everywhere → cost control):
- **Catalog enrichment (Gemini-lite):** for sparse tracks, generate a short descriptive blurb / inferred mood-genre tags from name+artist+album → enrich BM25 docs & dense docs. *(Official "tips": `music-crs-baselines/tips/improve_item_representation.md`. Existing impl: `mcrs/enrich/doc2query.py` + `nb/a1_enrich_catalog.ipynb`.)*
- **doc2query / docTTTTTquery (Gemini-lite or a T5):** generate likely user queries a track answers; append to its BM25 doc to close the vocabulary gap between conversational language and metadata. Big sparse-recall lever. *(Ref: Nogueira & Lin 2019; same `enrich_*` scripts cover this.)*
- **Query expansion / refinement (Gemini-lite):** §7.1 — structured intent extraction; pseudo-relevance feedback from top content-neighbors. *(Existing impls to port behind gate, recoverable from old git branches (`recall-union-lgbm`, `exp/*`).)*
- **Generative retrieval / Semantic IDs (research, only if core plateaus):** RQ-VAE codebook over track embeddings + small LM to generate IDs. *(Official "tips": `music-crs-baselines/tips/use_genrec_semantic_ids.md`; existing scaffold to build on (SID generator) recoverable from old git branches (`exp/*`).)* High effort; pursue if it helps clear 0.55.

Each lever ships behind an **ablation gate**: keep only if it lifts devset nDCG@20 (or the judge dimension) at acceptable cost.

---

## 13. Phase 4 — Recommendation assembly & Response generation (LLM-as-Judge + Distinct-2)

**Objective:** maximize Personalization + Explanation Quality (Gemini judge) and Distinct-2, decoupled from retrieval.

- **Model:** **Gemini lite** (e.g. `gemini-2.5-flash-lite` / `2.0-flash-lite` — **verify current model id**), via the Gemini API key. Batched, cached.
- **Inputs (grounding):** user profile (age/gender/country), conversation goal, last user utterance + brief dialogue gist, and the **top recommended tracks** (names/artists/tags). Grounding in *actual* retrieved tracks + user specifics is what the Personalization dimension rewards.
- **Prompt design:** system prompt with **domain knowledge** (music-curator persona), explicit **instructions** (justify picks via the user's stated mood/genre/history; be specific; one coherent paragraph), **few-shot** examples spanning cold/warm + different goals, light **CoT** ("first note the user's intent, then map each pick"). Keep responses **varied** (anti-template) to help Distinct-2 — reference concrete track/artist/mood details rather than reusable filler.
- **Caution:** the judge prompt is secret — optimize *genuine* quality, do not overfit to a guessed rubric. Hold a small human eyeball check + a proxy judge (your own Gemini call with a reasonable rubric) for iteration, clearly labeled as a proxy.
- **Cost control:** only generate for required session-turns; cache; short outputs; lite model. (§15)

**Tests:** prompt-builder is deterministic & injection-safe (track names sanitized); response non-empty & within length; proxy-judge harness runs; Distinct-2 computed via official metric.
**DoD:** responses are grounded & varied; proxy Personalization/Explanation up vs. baseline; Distinct-2 ≥ baseline 0.2558.

---

## 14. Engineering: tests, code review, git, agents

- **Git:** one feature branch per phase/experiment off the working branch; small PRs. Use the repo tools; never commit to main directly.
- **Tests (every change):** in `tests/` — (1) **metric-parity** test that our local scorer == official `music-crs-evaluator` on a fixture; (2) **no-leak** test (turn-t inputs ⊆ ≤t); (3) **submission-schema** validator (fields, ≤20, unique, valid IDs, all session×turns present, `ensure_ascii=False`); (4) retrieval recall test; (5) rerank group/label test; (6) filter uniqueness/tail-MMR test. CI runs them on push.
- **Code review:** Claude Code review subagent on each PR; human (Or) approves. No merge without green tests + review.
- **Parallel agents (when in Claude Code):** parallelize independent work — e.g. one agent runs the BM25/dense retriever ablations while another builds the responder prompts and a third writes tests. Keep shared artifacts (enriched docs, embeddings cache) as the sync point.
- **Determinism:** fixed seeds; pinned model revisions from HF Hub; configs capture every knob.

---

## 15. Cost & time budget

- **GPU:** embedding the 50k catalog once (minutes) + brute-force dense retrieval (seconds/turn batched) + optional cross-encoder on K≤200 pairs. GBDT is **CPU**. → small GPU footprint; do heavy steps once and cache embeddings to HF Hub.
- **Gemini:** **enrichment/doc2query = one-time over ~50k tracks** (the big batch — start with the **free tier**, batch, cache, and only enrich sparse tracks if full-catalog is too costly); **responses = ~8k session-turns** (lite, short, cached). **Query-refinement** only if it earns its keep. Estimate cost per 1k calls before scaling; set a hard budget cap and a dry-run token count.
- **Caching everywhere:** enriched docs, doc2query, embeddings, LLM query refinements, responses — keyed by content hash; never recompute.

---

## 16. Risk register / failure modes

| Risk | Symptom | Mitigation |
|---|---|---|
| Catalog is actually ~1M | dense brute-force too slow; recall probe slow | Add FAISS IVF/HNSW; chunk embedding; confirm size in EDA first. |
| Local ≠ leaderboard | devset nDCG diverges from Blind A | Use official evaluator verbatim; replicate macro-avg order exactly; reconcile on first Blind A submit. |
| Leakage | suspiciously high devset, low blind | No-leak tests; causal feature audit; session-disjoint CV. |
| Overfit reranker | big train↔val gap | session CV, early stop on Train holdout, regularization, keep Dev clean. |
| Diversity collapse | catalog_diversity → ~0 (popularity trap) | personalization + tail-MMR; monitor diversity each run. |
| Gemini cost/limits | quota/$$ spike | one-time cached enrichment, lite model, batch, budget cap, free tier first. |
| History-filter wrong call | nDCG drops | A/B both ways, decide on data. |
| Blind B timing surprise | miss deadline | front-load Week-1 spine; have a submittable system by day 3–4. |

---

## 17. Milestone timeline (~2 weeks; adjust to verified dates)

- **Day 1–2 — Phase 0:** EDA + recall-ceiling probe + catalog-size & gold-structure confirmation + leakage audit. Decide K, context caps, cold/warm thresholds.
- **Day 3–4 — Phase 1+3 spine:** BM25→RRF hybrid + filter + trivial responder → **first valid Blind A submission** (sanity: local==leaderboard). Lock the harness.
- **Day 5–7 — Phase 2:** LightGBM LambdaMART reranker + features → devset nDCG@20 ≥ 0.45, mean hit-rank ≤ ~2.5; cold/warm tracking; ablations logged.
- **Day 8–10 — §12 advanced levers (the push to 0.55):** doc2query + catalog enrichment + query refinement (gated, to raise recall@20); cross-encoder/ColBERT rerank stacked into the GBDT (to sharpen top-1–3); optional LoRA fine-tune if it earns its gate. Parallel: Phase 4 responder prompt-engineering for the judge + Distinct-2.
- **Day 11–13 — Hardening & ensemble:** best retriever subset + reranker + responder; tail-diversify; full reproducibility; frugal Blind A confirmations.
- **Day 14 — Blind B final:** run inference on Blind B, validate schema, submit, archive config + models to HF Hub.

(Reserve Blind A submissions — they're limited; confirm, don't grid-search on the leaderboard.)

---

## 18. Experiment tracking & memory

- **Per experiment:** a `config/<exp_id>.yaml` (every knob), a row in `reports/experiments.md` (exp_id, hypothesis, components, devset nDCG@1/10/20, cold/warm nDCG, diversity, Distinct-2, cost, decision: keep/drop), and the score JSON in `exp/scores/`.
- **Notebooks (see §6.1):** one self-contained LoRA fine-tuning notebook per training phase (`nb/phaseN_<name>.ipynb` — train + checkpoint + save adapter to Hub + eval), a dedicated `nb/inference_blindA.ipynb` (load adapters by Hub revision → pipeline → submission JSON → precheck), and `nb/experiments.ipynb` (ablations/sweeps). Offline dev in Claude Code; train/infer on Colab connected to repo+branch; **trained artifacts live on the HF Hub** (versioned by revision), never only on Colab disk.
- **Memory:** after each phase, persist key results & conclusions (what worked, recall ceilings, chosen K, encoder choice, history-filter decision, prompt that helped the judge) so future iterations build on settled facts.

---

## 19. References

- Robertson & Zaragoza (2009) — *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Karpukhin et al. (2020) — *Dense Passage Retrieval (DPR).*
- Xiao et al. (2023) — *BGE / C-Pack* embeddings (incl. `bge-m3`, `bge-reranker-v2`).
- Wang et al. (2022) — *E5: Text Embeddings by Weakly-Supervised Contrastive Pretraining.*
- Khattab & Zaharia (2020) — *ColBERT: Efficient late-interaction retrieval.*
- Nogueira & Lin (2019) — *doc2query / docTTTTTquery* document expansion.
- Nogueira et al. (2020) — *monoT5* sequence-to-sequence reranking.
- Cormack et al. (2009) — *Reciprocal Rank Fusion.*
- Burges (2010) — *From RankNet to LambdaRank to LambdaMART.*
- Ke et al. (2017) — *LightGBM.*
- Carbonell & Goldstein (1998) — *Maximal Marginal Relevance (MMR).*
- Järvelin & Kekäläinen (2002) — *Cumulated Gain-based Evaluation (nDCG).*
- Official (external): `nlp4musa/music-crs-baselines`, `nlp4musa/music-crs-evaluator`, challenge site `recsyschallenge.com/2026`, dataset collection `talkpl-ai/talkplay-data-challenge` (HF).

**In-repo references (branch `fresh-start`):**
- Scorer & ground truth: `music-crs-evaluator/{evaluate_devset.py, make_ground_truth.py, metrics/{metrics_recsys.py, metrics_diversity.py}}`.
- Baseline foundation: `music-crs-baselines/{run_inference_devset.py, run_inference_blindset.py, mcrs/, config/, lowerbound/}`.
- Official tips: `music-crs-baselines/tips/{improve_item_representation.md, add_reranker.md, use_genrec_semantic_ids.md}`.
- Prior-work implementations to mine/port (§6.3 asset map): ported code lives under `mcrs/` (e.g. `mcrs/enrich/doc2query.py`, `mcrs/lm/responder.py`, `mcrs/retrieval/related_artist.py`, `mcrs/training/colbert_*.py`, `mcrs/run/harness.py`); the rest (extra retrieval/rerank/query-rewrite modules, SASRec & ColBERT scaffolds, score tracker, the `82_*`/`90_*` ColBERT notebooks) is recoverable from old git branches (`recall-union-lgbm`, `stage-b-cross-encoder`, `fresh-model`, `exp/*`).
- Data & env: `data/TalkPlayData-Challenge-*` (gitignored), `download_data.py` / `download_data.sh`, `requirements.txt`, `setup_venv.sh`.

---

### Appendix A — First-PR checklist (Phase 0 → first submission)
- [ ] Verify catalog size, embedding dims, ID join coverage on live HF data.
- [ ] Confirm single-gold-per-turn + whether gold ∈ history.
- [ ] Recall@{50,100,200,500} probe per retriever + RRF (decide K).
- [ ] Context-length distribution → truncation policy in config.
- [ ] Cold/warm thresholds.
- [ ] Metric-parity test green vs official evaluator.
- [ ] BM25→RRF→filter→trivial-responder produces a schema-valid devset JSON; local nDCG@20 reproduces baseline; one Blind A submit to confirm local==leaderboard.