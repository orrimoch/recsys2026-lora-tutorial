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
- Prereq: Step 1 ColBERT checkpoint (asserted). Config (cell 7): `USE_COLBERT=True`, `COLBERT_D_LEN=512`
  (matches Step 1), `CE_STACK=True`, `CROSS_ENCODER_K=50` (frozen-CE FEATURE depth — not K3b's 200),
  `TRAIN_SESSIONS=7000` (was 3000 — re-verifying data saturation under the new spine + the `_norm` fix;
  K2 historically saturates at 3000, so 7000 may add ~0 at ~2x the CE-pass cost — drop back to 3000 if equal).
- Cells: run all. The slow part: the frozen bge-reranker scores train+dev pools (chunked, cached +
  RESUMABLE on Drive, so a re-run / disconnect is cheap). LGBM fit itself is fast (CPU).
- Gate: read the `GATE final-turn` nDCG@20 line (Blind-A scores the final turn only — the all-turns
  number printed beside it is a secondary diagnostic, NOT the gate). +K2 must clearly beat fusion-only.
- Time/cost (frozen-CE pass dominates) — at `TRAIN_SESSIONS=7000`: G4/T4 ~2–2.5 h, A100 ~1–1.5 h;
  at 3000 ~half that (~same nDCG).

## Step 3 — Fine-tune K3b cross-encoder  ·  `nb/phase2_ce_finetune.ipynb`  ·  GPU  ·  OPTIONAL for submit
- Colab: https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-start/nb/phase2_ce_finetune.ipynb
- STATUS (2026-06-20): the conversion-diagnosis probe CONFIRMED that CHAINING a cross-encoder after K2
  at serve REGRESSES (frozen CE: −0.075 @K=50 / −0.154 @K=100 recall@20; worse with depth). So K3b is
  NO LONGER a serve reranker — it is being repurposed as an OOF FEATURE for K2 (`ce_ft_score`, stacked
  like the frozen `ce_score`; K2 stays in control). The old "K2 vs K2+K3b chained gate" is SUPERSEDED.
- Selection is now K2-FREE: the in-training checkpoint metric scores the CE ALONE over the fused pool
  (`NeuralReranker`, not `ChainReranker(k2, …)`). This unblocks the fine-tune (no production-K2
  reconstruction needed) AND fixes the in-sample-K2 selection leak (Tier-3 #1).
- Output: adapter `OUT/ckpt_k3b`. Also still usable as the ColBERT distillation TEACHER.
- Config: `CROSS_ENCODER_K=200`, `N_NEG=30`, `FALSE_NEG_DROP_QUANTILE=0.0`, `TRAIN_SUBSET=4000`.
- RESTRUCTURE IN PROGRESS (Step 2a→2b): K2-free selection done; the cheap feature-utility validation
  (2a, 1 fine-tune) and — only if it passes — the OOF cross-fit + K2 retrain (2b) are being wired.
  Run mechanics firm up as those land.
- Time/cost — `TRAIN_SUBSET=4000` @ K=200: G4/T4 ~3–6 h, A100 ~1–2 h. The 2b OOF feature multiplies
  the fine-tune cost by (fold count + 1).

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

## Diagnosis (slot-free, no submission)  ·  `nb/phase3_conversion_diagnosis.ipynb`
- Colab: https://colab.research.google.com/github/orrimoch/recsys2026-lora-tutorial/blob/fresh-start/nb/phase3_conversion_diagnosis.ipynb
- Reconstructs the production K2 serve spine (mirrors Step 4) on the dev FINAL-turn proxy and decomposes
  the nDCG@20 gap into recall_loss (gold never reached the top-500 pool → ColBERT) vs ranking_loss (gold
  in pool, not top-20 → K2/CE). Also runs a frozen-CE re-scoring probe and a gold-rank histogram.
- Latest read (2026-06-20): RANKING-bound (recall@500=0.68, +K2 recall@20=0.29, conversion=0.43);
  16% of turns sit at K2-rank 21–50 (cheap headroom); CE chain regresses (→ K3b as a feature, not a
  chain). Final-turn proxy is 100% WARM, so cold-segment recall is wasted for the leaderboard.
- Use it after any K2/ColBERT change to attribute the movement (recall_loss vs ranking_loss). No slot.

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
