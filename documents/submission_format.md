# Submission Format — TalkPlayData RecSys Challenge 2026

Consolidated from `music-crs-evaluator/readme.md` (§Inference JSON Format, §Validation Checklist) and `music-crs-baselines/run_inference_blindset.py`.

## 1. Where submissions go

- **Leaderboard host:** [CodaBench](https://www.codabench.org/) (per evaluator readme §Overview).
- **Blind-set scoring is server-side only.** The local `music-crs-evaluator` repo explicitly does **not** support Blind A / Blind B evaluation — it only evaluates the dev set. Full blind scoring uses additional server-side metrics (LLM-judge) that are withheld to prevent leakage.
- Two blind phases:
  - **Blind A** — submission system opens **Apr 15, 2026**.
  - **Blind B** — submission system opens **Jun 15, 2026** (final, determines winners).

## 2. File format — predictions

One JSON file per evaluation dataset. Saved locally at:

```
exp/inference/<eval_dataset>/<tid>.json
```

Examples:
- `exp/inference/devset/my_model_devset.json`
- `exp/inference/blindset_A/my_model_blindA.json`
- `exp/inference/blindset_B/my_model_blindB.json`

The file is a **JSON array**, one entry per `(session, turn)` pair:

```json
[
  {
    "session_id": "69137__2020-02-08",
    "user_id": "69137",
    "turn_number": 1,
    "predicted_track_ids": [
      "715f8aff-7c99-46b8-8f9d-6d1aa1ae0372",
      "73562c63-02e3-4278-baf3-aeb3252f8b33",
      "4302b6cf-afe4-45d9-ab72-bd477086d838",
      "f20c5819-a312-4a6d-9ad1-46deccb4ff2f"
    ],
    "predicted_response": "Here are some songs you might enjoy."
  }
]
```

### Required fields

| Field | Type | Constraints |
|---|---|---|
| `session_id` | string | Format `{user_id}__{date}`. |
| `user_id` | string | Must match the session. |
| `turn_number` | int | Range 1–8. |
| `predicted_track_ids` | list[string] | **Ordered** by relevance (most relevant first), **up to 20**, **unique** within the list, must resolve to Challenge-Track-Metadata IDs. |
| `predicted_response` | string | Generated natural-language reply. Can be `""` but the field must be present. Affects Lexical Diversity + LLM-Judge. |

### Coverage requirement

- One entry **per session × per turn (1–8)**. Missing entries = missing predictions for those turns, which tank nDCG for those rows.
- Use `track_split_types=["all_tracks"]`. Any other split invalidates the submission (per baselines readme §Track Split Types).

### Serialization

```python
with open(path, "w", encoding="utf-8") as f:
    json.dump(inference_results, f, ensure_ascii=False)
```

`ensure_ascii=False` is mandatory — Unicode in titles/artists/responses must round-trip.

## 3. Packaging for CodaBench (confirmed 2026-04-19)

CodaBench competition URL: https://www.codabench.org/competitions/15786/

**Required layout — verified via actual scorer error:**

- Upload a `.zip` containing a single file at the archive root named exactly `prediction.json` (singular, no `s`).
- CodaBench extracts submissions into `res/`, so the scorer reads it at `/app/input/res/prediction.json`.
- Any other name fails with:
  ```
  FileNotFoundError: prediction.json not found at /app/input/res/prediction.json.
  Please rename your submission file to prediction.json
  ```

Known-bad layouts that have been rejected:

| Layout | Inner path | Result |
|---|---|---|
| Flat, original tid filename | `naive_bm25_blindset_A.json` | Rejected — wrong filename. |
| Structured readme path | `exp/inference/blindset_A/naive_bm25_blindset_A.json` | Rejected — wrong filename + wrong layout. |
| Flat, plural | `predictions.json` | Rejected — singular required. |

**Correct build (produces `prediction.zip`):**

```bash
# From the blindset output directory:
cp <tid>.json /tmp/prediction.json
( cd /tmp && zip prediction.zip prediction.json )
mv /tmp/prediction.zip .
# verify:
unzip -l prediction.zip   # must show exactly: prediction.json at root
```

- Do **not** include model weights, caches, or dataset files in the predictions zip. Predictions zips are data-only.
- The `<tid>.json` on disk can keep its descriptive name for local tracking; only the copy inside the zip must be `prediction.json`.

## 4. Code upload (separate deliverable)

Per the official timeline (evaluator readme §Timeline):

- **Jul 9, 2026** — "Upload code of the final predictions." This is a **second, separate** submission from the leaderboard JSON. Purpose: reproducibility of the final leaderboard score. The channel (GitHub link, zip to organizers, EasyChair attachment) is defined by the organizers, not this repo.

Expect to provide:
- Source code used for the final Blind-B run.
- Config YAMLs, entrypoint commands, and environment pin (requirements/lockfile).
- A README describing how to reproduce the final JSON from raw data.
- Any custom model weights or a pointer to where they're hosted.

## 5. Validation checklist (run before every submission)

- [ ] JSON saved at `exp/inference/<eval_dataset>/<tid>.json`.
- [ ] All 5 required fields present in every entry.
- [ ] Coverage: every `(session_id, turn_number)` in the blind set has an entry; turns span 1–8.
- [ ] `predicted_track_ids` length ≤ 20, no duplicates within the list, ordered by relevance.
- [ ] All track IDs exist in TalkPlayData-Challenge-Track-Metadata.
- [ ] `predicted_response` is a string (may be empty) — not null, not missing.
- [ ] `json.dump(..., ensure_ascii=False)` used.
- [ ] Inference used `track_split_types=["all_tracks"]`.
- [ ] File loads back via `json.load` without error and matches expected session count.
- [ ] (If zipping) archive contains exactly the JSON at root, no extra files.

## 6. Quick sanity script

```python
import json, collections
with open("exp/inference/blindset_A/my_model.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

assert isinstance(preds, list) and preds, "empty or non-array"
seen = collections.Counter()
for row in preds:
    for k in ("session_id", "user_id", "turn_number", "predicted_track_ids", "predicted_response"):
        assert k in row, f"missing {k}"
    assert isinstance(row["turn_number"], int) and 1 <= row["turn_number"] <= 8
    tids = row["predicted_track_ids"]
    assert isinstance(tids, list) and len(tids) <= 20 and len(set(tids)) == len(tids)
    assert isinstance(row["predicted_response"], str)
    seen[(row["session_id"], row["turn_number"])] += 1

dups = [k for k, c in seen.items() if c > 1]
assert not dups, f"duplicate (session, turn) entries: {dups[:5]}"
print(f"OK — {len(preds)} entries, {len({s for s, _ in seen})} sessions")
```

Run this against the generated file before zipping or uploading.

## 7. Open items to confirm on CodaBench

- ~~Accepted upload format: raw `.json` vs `.zip`.~~ Confirmed: `.zip`.
- ~~If zip: required inner filename and folder layout.~~ Confirmed: `prediction.json` at archive root (server maps it to `/app/input/res/prediction.json`).
- Per-day / per-phase submission quota.
- Whether separate uploads are needed for Blind A vs Blind B, or the same submission slot switches over on Jun 15.
- Whether `predicted_response` is scored for every turn or only a sampled subset (affects LLM-judge variance).

## 8. References

- `music-crs-evaluator/readme.md` §Inference JSON Format, §Validation Checklist, §Timeline.
- `music-crs-baselines/readme.md` §Track Split Types.
- `music-crs-baselines/run_inference_blindset.py` — canonical implementation of the output shape.
- `evaluation_framework.md` — how the submitted JSON is scored.
