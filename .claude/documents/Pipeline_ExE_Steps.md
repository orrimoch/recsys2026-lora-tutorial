# Pipeline execution steps (chronological)

End-to-end run order: enriched catalog → ColBERT → K2 → (K3b) → Blind-A submission. All notebooks run
on Colab GPU. Run each notebook top-to-bottom unless noted. Artifacts persist to Drive at
`OUT = /content/drive/MyDrive/recsys2026/outputs`, so steps can run in separate sessions.

Time/cost are ROUGH estimates (full data; not yet measured on this codebase). Cost basis (Colab
pay-as-you-go): G4/T4-class ≈ $0.18/hr, L4 ≈ $0.50/hr, A100 ≈ $1.30/hr. "G4" below = the standard
T4/L4 tier; pick A100 for the heavy training steps if you want them to finish fast.

## Common setup (every notebook)
- Colab links below open the GitHub copy (`orrimoch/recsys2026-lora-tutorial`, branch `fresh-start`) —
  they reflect the LATEST notebooks only after the branch is pushed. Colab "Open from GitHub" needs the
  repo public, or sign in to Colab with the GitHub account that can read it.
- Runtime: Colab GPU. Colab Secrets (userdata): `HF_TOKEN` (all); `GEMINI_API_KEY` (Step 5 responder only).
- Cells 1–4 of every nb are identical boilerplate: mount Drive, HF login, `git clone --branch fresh-start`,
  `pip install` (includes `pip uninstall -y torchao` — keep it). The clone cell `git pull`s, so a new
  session picks up the latest code.

## Dependency order (why this sequence)
ColBERT → K2 → K3b. K2 trains on the 8-channel spine that INCLUDES the ColBERT channel (ColBERT must
exist first); K3b loads K2 as a prerequisite. Blind serve needs only ColBERT + K2 (K3b is not chained
at serve — it's the distillation teacher / a future reranker). `D_LEN=512` + `EXPANSION_FIRST=True` are
identical across finetune / rerank / blindA / dev_experiments so they all share ONE PLAID index.

---

## Step 0 — Enriched catalog  ·  `nb/a1_enrich_catalog.ipynb`  ·  ALREADY DONE — SKIP
- Colab: https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-start/nb/a1_enrich_catalog.ipynb
- A valid `OUT/catalog_enriched_*.parquet` already exists, so do not re-run A1. (For reference: it's a
  Gemini doc2query enrich, needs `GEMINI_API_KEY`, ~CPU/API, resumable.) Every later nb auto-loads the
  newest `catalog_enriched_*.parquet`. Only re-run if you intentionally rebuild the catalog.
- Time/cost: n/a (skipped).

## Step 1 — Fine-tune ColBERT  ·  `nb/phase2_colbert_finetune.ipynb`  ·  GPU
- Colab: https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-start/nb/phase2_colbert_finetune.ipynb
- Output: checkpoint `OUT/colbert/music-colbert-v1` + PLAID index under `OUT/colbert_plaid_ft/`.
- Config (cell 6): `D_LEN=512`, `K_NEGS=25`, `EXPANSION_FIRST=True`, `USE_DISTILLATION=False` (default),
  `FALSE_NEG_DROP_QUANTILE=0.0`, `TRAIN_SESSIONS=0` (all train).
- Cells: run all. Cell 14 = train (LoRA + dev-recall selection + early stop). Cells 16–18 = G3 recall gate.
- Gate (G3): ColBERT beats dense @200, or fused lift ≥ +0.01 @200, or unique ≥ 0.02.
- Heaviest step: encodes 47k docs @ D_LEN=512 for the index + LoRA train. On T4, D_LEN=512 can be slow/OOM —
  prefer A100. Time/cost — G4/T4: ~4–8 h (~$0.7–1.5); A100: ~1.5–3 h (~$2–4).

## Step 2 — Train K2 (LGBM reranker)  ·  `nb/phase2_rerank.ipynb`  ·  GPU
- Colab: https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-start/nb/phase2_rerank.ipynb
- Output: `OUT/k2_lgbm.txt` (+ `.features.json`). The ONLY notebook that trains K2.
- Prereq: Step 1 ColBERT checkpoint (asserted). Config (cell 6): `USE_COLBERT=True`, `COLBERT_D_LEN=512`
  (matches Step 1), `CE_STACK=True`, `CROSS_ENCODER_K=50` (frozen-CE FEATURE depth — not K3b's 200).
- Cells: run all. Cell 10 is the slow part: the frozen bge-reranker scores train+dev pools (cached to
  Drive, so a re-run is fast). LGBM fit itself is fast (CPU). Cell 12 prints fusion-only vs +K2 nDCG@20.
- Gate: +K2 nDCG@20 clearly beats fusion-only.
- Time/cost (the frozen-CE pass dominates) — G4/T4: ~2–4 h (~$0.4–0.7); A100: ~0.5–1.5 h (~$0.7–2).

## Step 3 — Fine-tune K3b cross-encoder  ·  `nb/phase2_ce_finetune.ipynb`  ·  GPU  ·  OPTIONAL for submit
- Colab: https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-start/nb/phase2_ce_finetune.ipynb
- Output: adapter `OUT/ckpt_k3b`. Used as the ColBERT distillation TEACHER / a future final-stage
  reranker. NOT chained in blind serve, so SKIP for a plain submission; run it only to enable
  distillation (then re-run Step 1) or to gate K3b on its own.
- Prereq: Step 2 K2 (loads `k2_lgbm.txt`, asserted). Config: `CROSS_ENCODER_K=200`, `N_NEG=30`,
  `FALSE_NEG_DROP_QUANTILE=0.0`.
- Cells: run all. Cell 6 = train; cell 7 = K2-vs-K2+K3b dev gate. Gate: K2+K3b > K2 and the
  `DOES_NOT_MOVE_TOWARD_GOAL` slice not regressed.
- Time/cost — TRAIN_SUBSET=4000 @ K=200: G4/T4 ~3–6 h (~$0.5–1.1); A100 ~1–2 h (~$1.3–2.6).
  Full data (15k @ K=500): A100 ~4–6 h (~$5–8) — A100 strongly preferred.

## Step 4 — Dev-confirm  ·  `nb/phase3_blindA_submission.ipynb` with `BLIND=False`  ·  GPU
- Colab: https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-start/nb/phase3_blindA_submission.ipynb
- Purpose: run the full serve spine on the dev FINAL turns (Blind-A-comparable) and read official nDCG@20
  BEFORE spending a slot. Prereq: Steps 1–2.
- Config: `BLIND=False`, `RESPONDER='stub'`, `D_LEN=512`, `K2_MODEL_PATH=OUT/k2_lgbm.txt`. Run all cells.
- Watch: PLAID index prints "loaded" (reuses Step 1's index, no rebuild); `fusion channels` includes
  `colbert`; K2 feature count matches (the `rerank()` guard raises on any train/serve mismatch). Read
  `DEV-CONFIRM official` nDCG@20.
- Time/cost — G4/T4: ~30–60 min (~$0.1–0.2); A100: ~15–30 min (~$0.3–0.7).

## Step 5 — Submit Blind-A  ·  `nb/phase3_blindA_submission.ipynb` with `BLIND=True`  ·  GPU
- Colab: https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-start/nb/phase3_blindA_submission.ipynb  (same notebook as Step 4; just flip `BLIND=True`)
- Purpose: 80 Blind-A targets → Gemini responder → CodaBench zip (`prediction.json` at zip root). Prereq:
  a passing Step 4. For the composite, retrain the winners on train+dev first.
- Env: `GEMINI_API_KEY` required (responder; the nb preflights it before GPU work). Config: `BLIND=True`,
  `RESPONDER='gemini'`. Blind guards assert 80 rows / one-per-session / keys==targets / non-empty before zipping.
- Output: `OUT/../recsys2026_submissions/<date>-blind-...zip` → upload to CodaBench.
- Time/cost (tiny pipeline; responder is API) — G4/T4: ~10–20 min (~$0.05); A100: ~5–10 min (~$0.2).
  Responder API (Gemini flash, 80 rows): a few minutes, ~negligible $.

---

## Minimal path to a submission (distillation OFF, K3b skipped)
Step 1 → Step 2 → Step 4 → Step 5. Total GPU ≈ G4/T4 ~7–13 h (~$1.3–2.4) or A100 ~2.5–5 h (~$3–7).
Adding Step 3 (K3b) ≈ +3–6 h (T4) / +1–2 h (A100), only needed for the distillation upgrade.

## Colab pre-flight (one-time, from the audit/review)
- K2: first fit shows no LightGBM error about `eval_at` (both early-stop and plain branches).
- ColBERT index: Step 4 reports "loaded" not "built" → blindA reuses Step 1's index (D_LEN/sig aligned).
- Feature parity: print `k2.feature_names_`; confirm `rank_inv__colbert` present and the full ordered list
  matches what blindA builds (else the guard raises — retrain K2).

## Optional: enable distillation (ColBERT KD from the K3b teacher) — bootstrap the cycle
Default OFF. ColBERT↔K3b is circular, so bootstrap:
1. Run Steps 1–3 once with defaults (contrastive ColBERT, K2, K3b teacher adapter).
2. In `phase2_colbert_finetune` set `USE_DISTILLATION=True` (teacher = `OUT/ckpt_k3b`); re-run Step 1 →
   distilled ColBERT v2 (triple cache key includes `_kd`, so it rebuilds).
3. Re-run Step 2 (K2 on v2 ColBERT), then Steps 4–5.
   VERIFY-ON-COLAB first: the PyLate KD dataset/collator API for the installed pylate version (loss switch
   is wired; exact KD data format is version-specific). Gate KD vs contrastive on dev nDCG@20.
