# D1 — Inference & Submission Harness (Blind A/B)

> Delivery. Orchestrates the full causal spine per session-turn and emits the strict `prediction.json`.
> The place "local == leaderboard" is verified before any Blind submission. Notebook
> `nb/inference_blindA.ipynb` (§6.1). See `000_INDEX.md`.

## 1. Purpose
Run every module end-to-end over a split (devset / Blind A / Blind B), produce a schema-valid submission, and confirm the local official score reproduces the leaderboard — reproducibly from one config + notebook.

## 2. Interface / contract
Lives in `mcrs/run/harness.py` + `nb/inference_blindA.ipynb`.

```python
def run_inference(split: str, cfg: "RunConfig") -> list[SubmissionRow]: ...
    # loads artifacts by Hub revision, runs the spine per turn, returns SubmissionRows
def write_submission(rows: list[SubmissionRow], path: str) -> None: ...  # strict JSON, ensure_ascii=False
def precheck(path: str, catalog: "Catalog") -> None: ...                 # mcrs/run/harness.py validate_submission + nb blind guards
```

**Wiring (the ordered spine D1 calls per turn, all causal):**
`F1.Conversations.turns → R1 (build Query) → [R2 refine, gated] → channels R3/R4/R5/R6 → R7 fuse (→ Candidate pool) → K1 features → K2 rerank (→ K3 stack/re-score, gated) → L1 filter (→ ≤20 ids) → S1 respond → assemble SubmissionRow`. Then `write_submission` → `precheck` → (dev only) F3 `score_official`.

## 3. Dependencies
Every module (F1–S1) + F3 (`score_official`). Reuse/extend pristine `music-crs-baselines/run_inference_devset.py` / `run_inference_blindset.py` / `mcrs/crs_baseline.py` as the run spine; `mcrs/run/harness.py` (`validate_submission`) + the notebook's blind guards (validation). A per-submission score log is recoverable from the old git branches (recall-union-lgbm, stage-b-cross-encoder, fresh-model, exp/*). Loads trained artifacts (K2 model; K3/S1/SASRec LoRA adapters) **by HF Hub revision**.

## 4. Design & logic
- **Causal orchestration:** for each (session, turn) build the `TurnContext` (≤t) via F1 and run the spine in the fixed order above; gold is never read during inference.
- **Artifacts by revision:** every trained component is loaded by a pinned Hub revision recorded in `RunConfig`; no "latest" floating tags.
- **Config hash:** D1 computes and records a hash of the **entire resolved config** (channel weights, K's, model revisions, prompts, truncation, history-rule) into the run output + `reports/experiments.md`, so a submission is fully reproducible and train==serve is auditable.
- **Strict output (plan §2, rule #1):** `prediction.json` (singular) — one entry per **every** session×turn, `predicted_track_ids` ordered/unique/≤20/valid catalog ids, `predicted_response` (may be ""), `json.dump(ensure_ascii=False)`. Retrieval is over the **full catalog** (`all_tracks`); no catalog subsetting.
- **Local == leaderboard:** on devset, run F3 `score_official` and reconcile against the leaderboard on the first Blind A submit; the official aggregation (turn-bucket macro, no session avg — F3) is the only selection number.
- **Blind A vs B:** Blind B is a one-line dataset swap at the end (same harness, same config); Blind-A submissions are limited — **confirm, don't grid-search on the leaderboard**.
- **Cold-start / missing data:** a turn with no usable channel output still emits 20 via L1 backfill + a (possibly popularity-prior) fill; never crash, never emit < required rows.

## 5. Reuse
Extend pristine `run_inference_{devset,blindset}.py` + `crs_baseline.py` (the run spine); reuse `mcrs/run/harness.py` (`validate_submission`) + the notebook's blind guards (a per-submission score log is recoverable from the old git branches). **Reuse / extend.**

## 6. Eval & acceptance gate
End-to-end **schema-valid** `prediction.json` (precheck passes: ≤20, unique, valid ids, all session×turns present, `ensure_ascii=False`); on devset, F3 `score_official` runs and the value **reproduces the Blind A leaderboard** within reconciliation tolerance on the first submit; the full run is reproducible from one config + notebook (config hash recorded).

## 7. Tests
- Schema validator (the §6 checks) green on a generated devset submission; a malformed fixture fails.
- Spine wiring: a tiny end-to-end smoke (a few sessions) runs all stages and produces valid rows.
- Causal/no-leak at the harness level (F3 hooks): no future turn / gold enters any stage.
- Determinism: same config + revisions + seed → byte-identical submission.
- Coverage: every (session, turn) in the split has exactly one output row.
- `local == official`: D1's devset score equals F3 `score_official` on the same file.

## 8. Failure modes & guards
- **Catalog subsetting** (rule #1 violation) → retrieval always over `all_tracks`; validity guard in L1; assert at write.
- **Missing (session,turn) row / dup ids** → coverage + uniqueness asserts before submit (official scorer raises otherwise).
- **Floating model tags** → pin Hub revisions; record in config hash.
- **local ≠ leaderboard** → use F3 official aggregation verbatim; reconcile on first Blind A; pinned evaluator commit.
- **Wasted Blind-A submissions** → gate submissions behind a devset improvement + a confirm-only policy.
- **Colab disconnect mid-run** → checkpoint per-batch outputs; resumable.

## 9. Config knobs
`run.split`, `run.artifacts.{k2_revision, k3_adapter, responder_revision, sasrec_adapter, ...}`, `run.output_path`, `run.precheck` (true), `run.score_official` (dev only), plus the full pipeline config (R1/R2/channels/R7/K1/K2/K3/L1/S1) — D1 hashes the resolved whole. Defaults/types from F2 loader.

## 10. Definition of Done & review checklist
- [ ] `run_inference` runs the full ordered spine causally; loads artifacts by Hub revision.
- [ ] `prediction.json` strict-schema valid (precheck green); coverage = every session×turn.
- [ ] Devset F3 `score_official` runs; reconciled with Blind A on first submit (local == leaderboard).
- [ ] Config hash recorded to the run output + `reports/experiments.md`; run reproducible from one config.
- [ ] Determinism + coverage + no-leak tests green; Blind B = one-line swap verified.
- [ ] Code review approved; full-catalog retrieval (rule #1) asserted.

## 11. Build order & dependencies
**Built last** — orchestrates all modules. Depends on: every module (F1–S1) + F3. A minimal version (`F1+F2+F3 → R1+R3 → R7 → L1 → trivial responder`) is the **first valid Blind A submission** (plan §17 day 3–4) and is brought up as soon as that subset exists; the full spine lands as K2/K3/S1 mature. **Blocks:** the actual leaderboard submissions.
