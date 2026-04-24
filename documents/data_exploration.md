# TalkPlay Challenge — Data Exploration

Snapshot of every dataset under `data/`: what exists, what's structured vs unstructured, where the gaps are. Task/metric definitions live in [`recsys_challenge_notes.md`](./recsys_challenge_notes.md); this doc is pure data inventory.

All numbers below were produced by [`analysis/explore_data.py`](./analysis/explore_data.py) — rerun it to refresh.

---

## 1. At a glance

| Dataset | Splits (rows) | Primary key | Kind | Size |
|---|---|---|---|---|
| TalkPlayData-Challenge-Dataset | train (15,199), test (1,000) | `session_id` | sessions + conversations | 230 MB |
| TalkPlayData-Challenge-Blind-A | test (80) | `session_id` | held-out leaderboard sessions (truncated) | 600 KB |
| TalkPlayData-Challenge-Track-Metadata | all_tracks (47,071), test_tracks (7,405) | `track_id` | track catalog + tags | 43 MB |
| TalkPlayData-Challenge-Track-Embeddings | all_tracks (47,071), test_tracks (7,405) | `track_id` | 6 precomputed embeddings | 1.9 GB |
| TalkPlayData-Challenge-User-Metadata | all_users (8,772) | `user_id` | demographics | 720 KB |
| TalkPlayData-Challenge-User-Embeddings | train (8,591), test_warm (371), test_cold (129) | `user_id` | user CF embedding | 10 MB |

All datasets are HuggingFace Arrow format; load with `datasets.load_from_disk(path)`.

### Structured vs unstructured

| Modality | Where |
|---|---|
| Structured (categorical/numeric) | `user_profile.*`, `conversation_goal.*`, track `popularity` / `duration` / `release_date`, user demographics |
| Unstructured text | `conversations[].content`, `conversations[].thought`, `conversation_goal.listener_goal`, track `track_name` / `artist_name` / `album_name` / `tag_list` |
| Dense / continuous | 6 track embedding columns + 1 user embedding column |
| Semi-structured (list-valued) | every track metadata field except `popularity`/`duration`/`release_date` (see §3), `conversations[]`, `goal_progress_assessments[]` |

---

## 2. Sessions & Conversations

`TalkPlayData-Challenge-Dataset` (train/test) + `TalkPlayData-Challenge-Blind-A` (test only) share the same schema. Each row = one session.

### Schema

| Column | Type | Notes |
|---|---|---|
| `session_id` | string | unique per row |
| `user_id` | string | joins to User-Metadata / User-Embeddings |
| `session_date` | string (YYYY-MM-DD) | never null in these splits |
| `user_profile` | struct | embedded snapshot — see below |
| `conversation_goal` | struct{category, listener_goal, specificity} | |
| `conversations` | list<struct{content, role, thought, turn_number}> | chat log |
| `goal_progress_assessments` | list<struct{goal_progress_assessment, turn_number}> | per-turn assessment text |

`user_profile` struct fields: `age` (int), `age_group`, `country_code`, `country_name`, `gender`, `preferred_language`, `preferred_musical_culture`, `user_id`, `user_split`.

### Turn structure (CRITICAL)

Every full session in train/test has **exactly 24 entries** in `conversations[]` — one `user` + one `music` + one `assistant` per turn × 8 turns. Blind-A is **truncated** (1–22 entries, median 11.5) — you only see a prefix and must continue.

| Split | rows | conv len p0 / p50 / p100 | gpa len p0 / p50 / p100 |
|---|---|---|---|
| Dataset/train | 15,199 | 24 / 24 / 24 | 8 / 8 / 8 |
| Dataset/test | 1,000 | 24 / 24 / 24 | 8 / 8 / 8 |
| Blind-A/test | 80 | 1 / 11.5 / 22 | 1 / 4.5 / 8 |

Role distribution is always perfectly balanced (e.g. train: 121,592 each of `user` / `music` / `assistant`). In Blind-A, counts reflect truncation (360 `user`, 280 `music`, 280 `assistant`).

### `user_profile` distributions

| Field | train | test | blind-A |
|---|---|---|---|
| `age` | 1..71 (median 22, mean 23.6) | 6..62 (med 23, mean 24.7) | 14..42 (med 23, mean 24.1) |
| `age_group` top | 20s 57%, 10s 28%, 30s 12% | similar | similar |
| `gender` | male 65.5% / female 34.5% | male 62% / female 38% | male 72% / female 28% |
| `country_code` top | US, BR, PL, UK, DE, RU | same (different ordering) | US, PL, RU, BR, NO, FR |
| `preferred_language` | **always "English"** — no signal | same | same |
| `preferred_musical_culture` | 600+ distinct values, long tail (top: Western, American, Anglo-American, …) | long tail | long tail |
| `user_split` | `train_warm` only | `test_warm` 800 / `test_cold` 200 | `challenge` only |

No nulls in any `user_profile` field across any split.

### `conversation_goal` distributions

| Field | Values |
|---|---|
| `category` | 11 codes (A–K); train top: H, B, K, J, G, E |
| `listener_goal` | free text, very diverse (6,000+ distinct in train) — describes the user's intent |
| `specificity` | 4 codes: **LH 34%, HL 31%, LL 23%, HH 11%** (train) — two-letter code for (listener, track) specificity |

### Conversation content / thought

Char-length percentiles (train; test/Blind-A similar):

| Field | p0 | p50 | p95 | p100 | empty |
|---|---|---|---|---|---|
| `content` | 7 | 190 | 376 | 6,110 | 0 |
| `thought` | 42 | 497 | 1,299 | 18,003 | **~37%** (136,223 / 364,776 turns) |

The `thought` field is the assistant/music agent's inner monologue — **empty on every `user` turn** (user has no "thought"), which is why the empty-rate is ~⅓. Non-empty thoughts are long (~500 chars median).

Sample train turn:
```
role='user'
content="I want to discover some new artists. Do you have anything that's a bit intense or dramatic?"
turn_number=1
```

### Temporal split (important)

| Split | `session_date` range |
|---|---|
| Dataset/train | 2006-05-28 .. 2018-12-31 |
| Dataset/test | 2019-01-02 .. 2020-03-20 |
| Blind-A/test | 2008-12-20 .. 2018-12-28 |

Train/test is a time-based split. Blind-A dates overlap with train — the split is by `user_id`, not date.

---

## 3. Track Metadata

### Schema

| Column | Type | Notes |
|---|---|---|
| `track_id` | string | primary key, 47,071 unique |
| `ISRC` | list<string> | 1,334 / 47,071 empty (~2.8%); usually len 1 |
| `track_name` | list<string> | len 1 for >99%, max 3 (multi-title tracks) |
| `artist_name` | list<string> | median 1, **max 31** (compilations / "Various Artists") |
| `album_name` | list<string> | median 1, max 2 |
| `tag_list` | list<string> | median **17**, max **105**, 87 empty |
| `popularity` | float64 | 0–93, median 38, mean 35.8, std 20.6 |
| `release_date` | string | 644 / 47,071 empty; includes `0000-01-01` sentinel (unknown) |
| `duration` | int64 (ms) | min **0** (some zero-duration rows — data quality), median 227,000, max 2,558,287 (~42 min) |
| `artist_id` | list<string> (UUIDs) | median 1, max 33 |
| `album_id` | list<string> (UUIDs) | median 2, max 10 |

### Why so many list-valued fields?

Tracks are deduplicated by audio but metadata is aggregated across all occurrences of that audio across releases / compilations. A track re-released on a greatest-hits album will have **two `album_id`s**, same `track_name`. Real-world "Various Artists" compilations drive the long tail of `artist_name` (max 31 entries).

**Practical rule**: for single-value access take `[0]`, but be aware the list may contain legitimately distinct values.

### Data quality flags

- `duration == 0` exists — filter or impute before using duration in features.
- `release_date` uses `0000-01-01` or empty string for unknown — treat as missing.
- `tag_list` is rich but very noisy (user-generated: `'goeiepoep'`, `'top quality'`, …) — stopword-like tokens require cleaning.

`test_tracks` is a **subset** of `all_tracks` (schema identical, 7,405 of the 47,071). Retrieval must always operate on the full `all_tracks` catalog per the challenge rules.

---

## 4. Track Embeddings

6 precomputed embedding columns per track. Aligned 1-to-1 with Track-Metadata by `track_id` (symmetric diff = 0 for both splits).

| Column | Dim | Source | Normalized? | Norm p5 / p50 / p95 | Empty rows (/47,071) |
|---|---|---|---|---|---|
| `audio-laion_clap` | **512** | LAION-CLAP audio encoder | **L2 (unit norms)** | 1.000 / 1.000 / 1.000 | 492 (1.05%) |
| `image-siglip2` | **768** | SigLIP-2 on album art | no | 8.84 / 10.08 / 11.74 | 586 (1.24%) |
| `cf-bpr` | **128** | BPR collaborative filtering | no | 0.024 / 0.028 / 0.032 | 616 (1.31%) |
| `attributes-qwen3_embedding_0.6b` | 1024 | Qwen3-Embedding-0.6B on text attributes | no | 84.1 / 90.7 / 96.0 | 492 (1.05%) |
| `lyrics-qwen3_embedding_0.6b` | 1024 | Qwen3-Embedding-0.6B on lyrics | no | 94.5 / 108.2 / 117.8 | 492 (1.05%) |
| `metadata-qwen3_embedding_0.6b` | 1024 | Qwen3-Embedding-0.6B on metadata text | no | 89.1 / 97.8 / 108.4 | 492 (1.05%) |

### Gotchas

- **Dimensions are not uniform.** `audio`, `image`, `cf-bpr` have *different* dims from the three Qwen3 variants — concatenation / stacking needs dim-aware code.
- **Only `audio-laion_clap` is L2-normalized.** All others must be normalized before cosine similarity (see `music-crs-baselines/mcrs/retrieval_modules/dense_precomputed.py:165`).
- **Empty embeddings are stored as empty lists**, not zero vectors. On a 5,000-row sample of non-empty rows, zero all-zero vectors were found — so `norm == 0` is effectively a proxy only for empties.
- **The same ~492 tracks are missing their audio / qwen3 embeddings** (likely the same source rows; CF and image have slightly different missing sets).
- Imputation contract already in repo: artist-mean → global-mean, then L2-normalize — `music-crs-baselines/mcrs/retrieval_modules/dense_precomputed.py:106–171`. Use it; don't re-invent.

`test_tracks` empty-embedding counts match `all_tracks` exactly for every column — the test-tracks split contains the *same empties*, because it's a projection of all_tracks.

---

## 5. User Metadata

One row per user, 8,772 users. Clean — no nulls in any column.

| Column | Type | Summary |
|---|---|---|
| `user_id` | string | unique |
| `age` | int | 1..71, median 23, mean 24.1 |
| `age_group` | string | 20s 63%, 10s 21%, 30s 12%, 40s 2.5%, 50s 0.7%, 60+ 0.1% |
| `country_code` | string | top: US 18%, BR 11%, PL 10%, UK 7.5%, RU 6%, DE 6% |
| `country_name` | string | paired with `country_code` |
| `gender` | string | male 72%, female 28% (no other values) |

`age=1` exists (48 users have age ≤ 5) — likely data-entry artifacts rather than real toddlers.

---

## 6. User Embeddings

Single column: `cf-bpr` (128-dim, same as track `cf-bpr`).

| Split | Rows | Empty `cf-bpr` rows | Norm p5 / p50 / p95 |
|---|---|---|---|
| train | 8,591 | 0 | 0.028 / 0.038 / 0.060 |
| test_warm | 371 | 0 | 0.038 / 0.065 / 0.106 |
| test_cold | 129 | **104 (80.6%)** | 0.028 / 0.041 / 0.092 (on the 25 non-empty) |

### The warm/cold distinction

- **Cold users**: no (or empty) CF embedding — must be served via content / text signals only.
- **Warm users**: have a CF embedding. And critically: **every test_warm user is also present in the train user-embedding split** (intersection = 371 / 371). So the CF model was trained on warm users' history; at test time we simply look up their embedding.
- `test_cold ∩ user_emb_train = 0` ✓ — cold users are truly unseen by CF.
- 25 / 129 cold users have a non-empty `cf-bpr`; these are cold by the *session* definition (unseen in the test sessions' training signal) but still have some pre-existing embedding. Don't assume empty == cold uniformly.

---

## 7. Cross-dataset relationships

```
 User-Metadata (8,772) ──┐
                         │ user_id
 User-Embeddings ────────┤
  ├─ train    (8,591) ───┤       ┌── TalkPlayData-Challenge-Dataset
  ├─ test_warm  (371) ───┼─────► │    ├─ train (15,199 sessions, 8,591 users)
  └─ test_cold  (129) ───┘       │    └─ test  (1,000 sessions, 500 users)
                                 │
                                 └── TalkPlayData-Challenge-Blind-A
                                       └─ test (80 sessions, 54 users)

 conversations[].content references tracks by NAME (free text) — no track_id FK.

 Track-Metadata (47,071) ←─ track_id ─→ Track-Embeddings (47,071)
   └─ test_tracks (7,405) is a SUBSET
```

### Overlap facts

| Claim | Holds? |
|---|---|
| Every session `user_id` appears in User-Metadata | ✓ (0 missing) |
| `sessions_train.user_ids == user_emb_train.user_ids` | ✓ (both 8,591, full overlap) |
| `sessions_test.user_ids == user_emb_test_warm ∪ user_emb_test_cold` | ✓ (371 + 129 = 500) |
| `user_emb_test_warm ⊆ user_emb_train` | ✓ (371 / 371) |
| `user_emb_test_cold ∩ user_emb_train = ∅` | ✓ |
| Track-Metadata IDs == Track-Embeddings IDs (both splits) | ✓ (symmetric diff 0) |
| `test_tracks ⊆ all_tracks` | ✓ |
| User-Metadata has 21 users who appear in no session | quirk — dangling demographics, safe to ignore |

### Blind-A users

Blind-A has 54 users, 80 sessions. Of those 54:
- 23 overlap with `sessions_test` users (and all 23 are `test_warm`)
- 0 overlap with `test_cold`
- **31 are net-new** — not in any session split

So Blind-A is a mix of "warm users we've seen in test" and "brand new users" — expect cold-start to matter.

---

## 8. Data quality notes (actionable)

| Issue | Where | Handling |
|---|---|---|
| Empty embedding lists | Track-Embeddings (~492–616 rows per column) | Impute via artist-mean → global-mean → L2-normalize. Existing code: `dense_precomputed.py:106–171`. |
| Cold-user empty cf-bpr | User-Embeddings test_cold (104 / 129) | No CF signal — fall back to content/text retrieval. |
| `preferred_language` is constant | all session splits | Drop as a feature. |
| `thought` is empty on user turns (~37%) | `conversations[]` | Expected; don't treat as missing. |
| `release_date` sentinels | Track-Metadata (644 empty + `0000-01-01` entries) | Treat as missing, not as year 0. |
| `duration == 0` | Track-Metadata | Filter/impute before using duration. |
| `tag_list` noise | Track-Metadata (user-generated) | Heavy cleaning / filtering required before use. |
| List-valued metadata (`artist_name` max 31) | Track-Metadata | `[0]` is usually safe; be aware of compilations. |
| User-Metadata extras | 21 users with no sessions | Ignore. |
| Child-age outliers | User-Metadata (`age ≤ 5`) | Likely bad data; filter before age-conditioned modeling. |

---

## How to refresh this doc

```bash
python documents/analysis/explore_data.py > /tmp/recsys_explore.txt
# then update the tables above from the printed sections
```
