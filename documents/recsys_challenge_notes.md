# RecSys Challenge 2026 — Challenge Notes

**Source of truth for the Music Conversational Recommendation task.** Distilled from `music-crs-baselines/` + `music-crs-evaluator/` READMEs and code. File references are to those directories unless noted.

---

## 1. The Task

### Goal — *dynamic* music recommendations, delivered agent-style
This is **not** static list-building. The system must act like a **conversational agent** that produces **dynamic recommendations** whose ranking is shaped by:
- **Retrieved-item ranking** from the RecSys module
- **Nuanced user preferences** extracted from dialog + profile
- **NLU** over the turn and conversation history
- **Exploration through dialogue** — asking, offering, refining
- **LLM-generated signals / rationales** that justify and personalize the picks

> "Music-CRS focuses on the evolving landscape of music discovery, where static recommendation lists are being replaced by dynamic, conversational interactions. As users increasingly interact with AI through natural language, there is a critical need for systems that can seamlessly integrate Natural Language Understanding (NLU) with high-precision Recommender Systems (RecSys). This challenge aims to push the boundaries of how AI understands nuanced user preferences, explores musical tastes through dialogue, and provides contextually relevant track recommendations." — `music-crs-baselines/readme.md:3`

### Main task
**The system must understand the user's preferences** — from (a) the conversation history (previous turns in the current session) and (b) the user's profile — and return a relevant ranked list of tracks plus a grounded natural-language response.

### Per-turn I/O
Per turn the system outputs:
1. **A ranked list of up to 20 track IDs** — the "recommendation"
2. **A natural-language response** — explaining / personalizing the recommendation

Evaluated on **8 turns per session** (macro-averaged).

---

## 1.5 Baseline System Architecture (2-Stage Pipeline)

From `music-crs-baselines/readme.md:29–42`. The official baseline is a **two-stage pipeline**:

1. **RecSys** — retrieves candidate tracks matching user preferences
2. **LLM** — generates a natural-language response explaining the recommendations

### Core components

| Component | Description | Module |
|---|---|---|
| **LLM** | Generates natural language responses (baseline: Llama-3.2-1B-Instruct) | `mcrs/lm_modules/` |
| **RecSys** | Retrieves relevant tracks via BM25 (sparse) or BERT (dense) | `mcrs/retrieval_modules/` |
| **User DB** | Stores user profiles (`user_id, age_group, gender, country_name`) | `mcrs/db_user/user_profile.py` |
| **Item DB** | Track metadata (`track_name, artist_name, album_name, release_date, tag_list, …`) | `mcrs/db_item/music_catalog.py` |

Additional module folders exist but are **empty stubs** in the baseline (opportunities to fill):
- `mcrs/rerankers/` — no reranker abstraction yet (tip #1 below)
- `mcrs/query_rewriters/` — no query rewrite stage yet
- `mcrs/embedders/` — custom embedders

### Pipeline per turn (`mcrs/crs_baseline.py`)
1. Compose system prompt from `system_prompts/` (roleplay + response_generation ± personalization with user profile)
2. Concatenate full chat history + current turn → retrieval query
3. RecSys returns **top-20** track IDs
4. LLM generates response conditioned on system prompt + chat history + **top-1** track metadata (hard-coded in baseline)

---

## 2. Datasets (HuggingFace, `talkpl-ai/`)

| Dataset | Rows | What it contains |
|---|---|---|
| `TalkPlayData-Challenge-Dataset` | 1k sessions × 8 turns | Multi-turn conversations; roles: `user`, `music`, `system` |
| `TalkPlayData-Challenge-Track-Metadata` | **50.4k tracks** | `track_id, track_name, artist_name, album_name, release_date, tag_list, …` |
| `TalkPlayData-Challenge-User-Metadata` | 9.09k users | `user_id, age_group, gender, country_name` |
| `TalkPlayData-Challenge-Track-Embeddings` | 50.4k | Pre-computed track embeddings (can be empty rows — needs imputation) |
| `TalkPlayData-Challenge-User-Embeddings` | 9.09k | Pre-computed user embeddings |
| `TalkPlayData-Challenge-Blind-A` | released Apr 10 2026 | First blind test set (evaluated via CodaBench) |
| `TalkPlayData-Challenge-Blind-B` | released Jun 15 2026 | Second blind test set |

### Keys that link the datasets
- **`session_id`** — format `"{user_id}__{YYYY-MM-DD}"`
- **`user_id`** — links conversations to user metadata + embeddings
- **`turn_number`** — 1..8
- **`track_id`** — UUID; links conversations to track metadata + embeddings

### Conversation turn structure
```python
turn = {"role": "user" | "music" | "system", "content": "..."}
```
- `role="user"`: what the human said (text)
- `role="music"`: ground truth track (`content` = `track_id`)
- `role="system"`: ground truth response text

---

## 3. Splits & CRITICAL Constraints

### Splits
- Conversations: `train` / `test`
- Items: `all_tracks` only (no pre-split subsets for inference)
- Users: `all_users`

### ⚠️ HARD RULE — `track_split_types` must equal `["all_tracks"]`

Must always retrieve from the full catalog. **Filtering the item pool by split invalidates the submission.**

> "During inference, the recommender must always retrieve candidates from the entire track catalogue. Do not filter, subset, or restrict tracks using `track_split_types` or any other mechanism… If you do not use `all_tracks`, evaluation may be considered invalid." — `music-crs-baselines/readme.md:70–82`

CLAUDE.md enforces the same rule at project level.

### Blind evaluation
- Blind-A / Blind-B scored only server-side via **CodaBench leaderboard**. Server holds additional metrics to prevent leakage (`music-crs-evaluator/readme.md:51–59`).
- Locally we can only evaluate the **devset**.

---

## 4. Ground Truth Format

Generated by `music-crs-evaluator/make_ground_truth.py`. Schema:

```json
{
  "session_id": "69137__2020-02-08",
  "user_id": "69137",
  "turn_number": 1,
  "ground_truth_track_id": "715f8aff-7c99-46b8-8f9d-6d1aa1ae0372"
}
```

- One entry per `(session × turn 1..8)` → ~8 000 entries for 1 k dev sessions.
- **Exactly one ground-truth track per turn** (recall denominator is always 1).
- Saved at `music-crs-evaluator/exp/ground_truth/devset.json`. **Do not modify** (CLAUDE.md rule).

---

## 5. Submission / Prediction Format

### Required JSON schema
```json
[
  {
    "session_id": "69137__2020-02-08",
    "user_id": "69137",
    "turn_number": 1,
    "predicted_track_ids": ["715f8aff…", "73562c63…", "…"],
    "predicted_response": "Here are some songs you might enjoy…"
  }
]
```

### Rules
- `predicted_track_ids`: **ordered** (most relevant first), up to **20 tracks**, **no duplicates**
- All track IDs must exist in `TalkPlayData-Challenge-Track-Metadata`
- One entry per `(session × turn)`; **all 8 turns required** or evaluation fails (`music-crs-evaluator/readme.md:219`)
- Empty `predicted_response` allowed (Random/Popularity baselines submit `""`)
- Save JSON with `ensure_ascii=False` (needed for non-ASCII track/artist names)

### File locations
- Dev: `music-crs-evaluator/exp/inference/devset/{tid}.json`
- Blind: `music-crs-evaluator/exp/inference/blindset_A/{tid}.json`

### CodaBench packaging
Zip must contain **`prediction.json` (singular)** at the zip root. The server reads from `/app/input/res/prediction.json`. Any other layout rejected.

---

## 6. Evaluation Metrics

Defined in `music-crs-evaluator/metrics/`.

### Retrieval (from `metrics_recsys.py`)
- **nDCG@1, nDCG@10, nDCG@20** — the primary scoring metrics.
- Formula: `DCG@k = Σ (2^rel_i − 1) / log₂(i+1)`, `nDCG@k = DCG@k / IDCG@k`
- `rel_i = 1` if predicted track at position *i* is the ground truth, else 0. Since each turn has a single GT track, `IDCG@k = 1` whenever `k ≥ 1`.
- **Macro-averaging**: first average across turns within a session, then across all sessions (`evaluate_devset.py:65–66`). Not micro-averaged.

### Diversity (from `metrics_diversity.py`)
- **`catalog_diversity`** = `|unique recommended tracks| / |catalog|` (50.4k)
- **`lexical_diversity`** = Distinct-2 bigram ratio across all `predicted_response` strings

### LLM-as-Judge — **Gemini, blind-set only** ⚠️
Source: `documents/RecSys2026_links.html:200–228` (official challenge website).

- Blind-set `predicted_response` strings are scored by a **Google Gemini** model used as an automatic judge.
- Two text-only dimensions (independent of recommendation accuracy):
  - **Personalization** — does the response reflect user context / preferences?
  - **Explanation Quality** — is the justification coherent, grounded, useful?
- **Prompt is not disclosed** to preserve blind-evaluation integrity.
- **Every listed dimension contributes to the final score** (exact aggregation weights also not published).
- Gemini ≠ Gemma. Gemini is Google's closed, API-only frontier LLM family. Gemma is their open-weights smaller family. We cannot run the judge locally; we can only approximate it (e.g., via Gemini API, or with a local proxy judge like Qwen — accepting a distribution gap).

**Implications for design**:
1. Response quality matters on Blind — we can't score it locally via nDCG alone.
2. Personalization + explanation quality are the knobs; optimizing purely for retrieval accuracy leaves points on the table.
3. Self-play / DPO loops using our own LLM-as-Judge (e.g. Qwen judging Qwen) will be **misaligned with the real judge**. Calibrate against Gemini API samples before trusting the signal (see `documents/research/recent_papers_ideas.md §6 — Judging the Judges`).

### Validation enforcement
- `metrics_recsys.py:127–130` raises `ValueError` on duplicate predictions or duplicate GT.
- `IDCG` computation caps at `min(len(gold), k)` — since `len(gold)=1`, effectively always 1.

### Baseline leaderboard (devset)

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog Div | Lexical Div |
|---|---|---|---|---|---|
| Random | 0.0000 | 0.0001 | 0.0001 | 0.9652 | 0.0000 |
| Popularity | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.0000 |
| LLaMA-1B + BM25 | 0.0098 | **0.0627** | 0.0815 | 0.3795 | 0.2558 |

*(Source: `music-crs-evaluator/readme.md:179–183`)*

---

## 7. Provided Baselines

All under `music-crs-baselines/`, runnable from `run_baselines.sh`:

| Baseline | Config | Notes |
|---|---|---|
| **Random** | `lowerbound/random_sample.py` | 20 random tracks; empty response |
| **Popularity** | `lowerbound/popularity.py` | Top-20 train-popular tracks (same for every turn); empty response |
| **LLaMA-1B + BM25** | `config/llama1b_bm25_devset.yaml` | Sparse retrieval over metadata fields |
| **LLaMA-1B + BERT** | `config/llama1b_bert_devset.yaml` | Dense retrieval (BERT) |
| **Dense (Qwen3)** | `config/005-dense-qwen3-metadata.yaml`, `config/006-dense-metadata-instruct.yaml` | Precomputed embeddings + Qwen3 query encoder |
| **RRF hybrid** | `config/007-rrf-bm25-dense-v1.yaml` | Reciprocal Rank Fusion (BM25 + dense) |
| **Weighted RRF** | `config/009-wrrf-bm25-dense-v1.yaml`, `010-wrrf-bm25-dense-lyrics-v1.yaml` | Weighted fusion across streams |
| **Sequential rerank** | `config/008-bm25-then-dense-rerank.yaml` | BM25 → dense rerank |

---

## 8. Config Knobs (YAML)

Common fields across configs in `music-crs-baselines/config/`:

| Field | Type | Purpose | Example |
|---|---|---|---|
| `lm_type` | str \| null | LLM for response (null = retrieval-only) | `"meta-llama/Llama-3.2-1B-Instruct"` |
| `retrieval_type` | str | Retrieval backend key | `"bm25"`, `"bert"`, `"dense_precomputed"`, `"rrf_bm25_dense_metadata_v1"` |
| `test_dataset_name` | str | HF conversational dataset | `"talkpl-ai/TalkPlayData-Challenge-Dataset"` |
| `item_db_name` | str | HF track metadata dataset | `"talkpl-ai/TalkPlayData-Challenge-Track-Metadata"` |
| `user_db_name` | str | HF user profile dataset | `"talkpl-ai/TalkPlayData-Challenge-User-Metadata"` |
| `track_split_types` | list[str] | **MUST be `["all_tracks"]`** | `["all_tracks"]` |
| `user_split_types` | list[str] | User subset | `["all_users"]` |
| `corpus_types` | list[str] | Track fields used for retrieval + stringification | `["track_name","artist_name","album_name","release_date","tag_list"]` |
| `cache_dir` | str | Local artifact caching | `"./cache"`, `"../../experiments/cache"` |
| `device` | str | Compute device | `"cuda"`, `"mps"`, `"cpu"` |
| `attn_implementation` | str | Attention backend | `"flash_attention_2"`, `"eager"` |

---

## 9. Official Tips (`music-crs-baselines/tips/`)

Three tip docs ship with the baseline. All three are reproduced below **in full** so this note is self-contained.

### Tip 1 — Add Reranker Module (`tips/add_reranker.md`)
> Refine initial retrieval results with a second-stage ranker.

**Option A: Embedding-based reranking**
- Use user embeddings for personalization
  - Compute user profile from listening history
  - Score candidates by user-item similarity
- Cross-modal reranking: combine multiple signals (text relevance + audio similarity + user preference)

**Option B: LLM-based reranking**
- Use LLM to judge relevance of top-k candidates
- Prompt: `"Rank these tracks by relevance to: {user_query}"`
- Models suggested: **Llama-3-8B, Qwen-7B, or specialized rankers**

**Suggested integration** (from the tip doc):
```python
# Add to CRS pipeline after retrieval
retrieval_items = self.retrieval.text_to_item_retrieval(query, topk=100)

# Rerank top candidates
if self.reranker:
    retrieval_items = self.reranker.rerank(
        query=query,
        candidates=retrieval_items[:50],
        user_profile=user_profile,
        topk=20
    )
```

**Pattern**: retrieve top-100 → rerank to top-20. Hooks into the currently-empty `mcrs/rerankers/` module.

**Resources**: `talkpl-ai/TalkPlayData-2-User-Embeddings`, `TalkPlayData-2-Track-Metadata`, `TalkPlayData-2-Track-Embeddings`.

---

### Tip 2 — Improve Item Representation (`tips/improve_item_representation.md`)

**2.1 Add more track information**

*Option A — more text fields*: the baseline only uses `track_name, artist_name, album_name`. Add: **genre tags, mood labels, release year, popularity scores**. Edit `corpus_types` in the config to include `tag_list`.

*Option B — audio features*: instead of text only, use the **actual sound**. Try **CLAP** (text+audio aligned model). Helps find songs that *sound* similar, not just described similarly.

Implementation hints from the tip doc:
- Change `_stringify_metadata()` to include more fields
- Add code to extract audio features
- Combine text and audio together

**2.2 Use a better retrieval model**

Replace the basic BM25 / BERT with stronger text encoders:

- **Qwen2.5-Embedding** — multilingual
- **Contriever** — strong zero-shot retrieval
- **E5** / **BGE** — currently SOTA for text embeddings
- **ColBERT** — late-interaction, finer-grained matching

**Resources**: `talkpl-ai/TalkPlayData-2-Track-Metadata`, `talkpl-ai/TalkPlayData-2-Track-Embeddings`.

---

### Tip 3 — Generative Retrieval / Semantic IDs (`tips/use_genrec_semantic_ids.md`)

> Replace embedding similarity with **end-to-end generation**: instead of retrieve-then-generate, directly generate track identifiers.

**Semantic IDs approach**:
- Assign **hierarchical semantic IDs** to tracks (e.g., `jazz/smooth/piano/0042`)
- Train an LLM to generate relevant track IDs given the user query
- A single model replaces both retrieval and generation stages

**Benefits**:
- Unified architecture
- Can model complex user intent
- Leverages LLM reasoning capabilities

**Implementation steps** (from the tip doc):
1. Create a semantic ID system for tracks
2. Fine-tune an LLM to generate track IDs
3. Optionally use collaborative filtering for ID assignments

**Concrete methods** for this path: see `documents/research/recent_papers_ideas.md §3` — TIGER (foundational), Text2Tracks (Spotify, music analogue), GRID (practitioner's handbook), LETTER (CF + text SIDs), LIGER (hybrid gen+dense), LC-Rec (LLaMA/Qwen recipe), IDGenRec (textual IDs), Joint-SIDs (for search+rec dual use).

**Resources**: `talkpl-ai/TalkPlayData-2-Track-Metadata`, `talkpl-ai/TalkPlayData-2-Track-Embeddings`.

---

## 10. Gotchas & Hidden Rules

1. **Top-1 goes to the LLM** — baseline pipeline passes only the top-1 retrieved track into the response prompt (`crs_baseline.py:123`). Hard-coded; pipeline-level decision.
2. **Empty track embeddings exist** — some rows in the pre-computed embedding datasets are empty. Impute (artist → category → global mean), don't drop.
3. **Macro-averaging means uneven-turn robustness matters** — a bad early turn hurts just as much as a bad late turn.
4. **Single GT per turn** — means nDCG@k is effectively Hit@k divided by a log-discount; the metric rewards pushing the one correct track toward rank 1.
5. **`ensure_ascii=False`** when writing `prediction.json` — track names contain non-ASCII characters (accents, CJK, etc.).
6. **All 8 turns required per session** — missing entries fail evaluation.
7. **Do NOT modify** `music-crs-evaluator/exp/ground_truth/` (project rule).
8. **Do NOT commit large artifacts** — model weights, cached indices, datasets. Use `data/`, `cache/`, `experiments/` (all git-ignored).
9. **Blind evaluation is one-shot-ish** — you only learn Blind leaderboard scores; iterate on devset locally first (per submission prep protocol: train-only during iteration, train+dev for the final Blind submission).
10. **Blind response quality is judged by Gemini (not Gemma, not us)** — nDCG alone doesn't predict Blind rank. Optimize for Personalization + Explanation Quality too. See §6.

---

## 11. Commands Reference

```bash
# Inference (from music-crs-baselines/)
python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 16

# Evaluation (from music-crs-evaluator/)
python make_ground_truth.py
python evaluate_devset.py --tid llama1b_bm25_devset
```

---

## 12. Where to Look for More

- Task + baseline overview: `music-crs-baselines/readme.md`
- Evaluation details + submission format: `music-crs-evaluator/readme.md`
- Per-paper research summaries + build order: `documents/research/recent_papers_ideas.md`
- HF dataset pages: `https://huggingface.co/talkpl-ai/`
- CodaBench leaderboard (submissions): see challenge site
