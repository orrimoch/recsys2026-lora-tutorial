# Training pipelines — leak & correctness audit

Adversarial audit of every model with training (K2 LGBM, K3b cross-encoder, ColBERT) plus the shared
eval/data surface, run before any nDCG@20 tuning. Four parallel read-only auditors covered: label/gold
leak into features, train/dev contamination, selection-set leak, train==serve parity, negative
sampling, metric correctness, and cross-session artifact leakage.

## Verdict

No CRITICAL data leak in any trained model or in the shared eval/data code. Past dev/gate numbers are
trustworthy — they are not inflated by contamination. The historical K3b val-fold leak (val nDCG once
scored K2 on its own train fold) is verified fixed: selection now runs on session-disjoint fold-0, the
gate on the clean test split. Everything found is correctness hardening, train/serve transfer risk
(would deflate, not inflate), or model quality.

## Findings and status

Severity, where, issue, and what was done. "Fixed" = applied in this batch (commit pending).

| ID | Sev | File | Issue | Status |
|----|-----|------|-------|--------|
| X1 | HIGH | `nb/phase2_rerank.ipynb` | Canonical K2 trained on 7 channels (no ColBERT); blind serve fuses 8 (with ColBERT) and loads this K2 → `rerank()` feature-spec guard (`lgbm.py:129`) would reject it / train-serve skew. | Fixed — added flag-gated ColBERT PLAID channel + `per_channel_query_builders` routing to training, frozen-CE dev pools, and dev eval (mirrors the blind spine). |
| C1 | HIGH | `mcrs/retrieval/colbert_channel.py`, `mcrs/config.py` | Production `colbert_doc_text` serve default `expansion_first=False` diverges from training (`True`) → serve doc-text skew, deflates live recall vs gate. | Fixed — `expansion_first` threaded through `from_catalog`/`from_pylate` (default True) + new `config.retrieval.colbert.expansion_first`. |
| C2 | HIGH | `nb/phase2_colbert_finetune.ipynb` | Gate PLAID index keyed on checkpoint only, not the enriched-doc version/`D_LEN`/`EXPANSION_FIRST`; a stale-but-valid index passes the smoke self-heal → wrong recipe reported as the gate. | Fixed — added `_doc_sig` to the gate index name (mirrors the phase3 notebooks). |
| K1 | HIGH | `mcrs/rerank/lgbm.py:47` | `_cap` reused one fixed seed across all groups → positionally-correlated, biased negative subsample. | Fixed — per-group seed `random.Random(f"{seed}|{salt}")` with `salt=(session_id,turn_number)`. |
| K2 | HIGH | `nb/phase2_rerank.ipynb` | `dense_cos` `_qcache` keyed only on `(session_id,turn_number)`, shared train+dev (safe today, unguarded). | Fixed — `_qcache.clear()` before the dev pass. |
| M-ce1 | MED | `mcrs/training/ce_finetune.py` + nb | Val-callback scoring and production `build_cross_encoder_score_fn` are two code paths kept aligned by hand. | Open — factor into one shared helper. |
| M-ce2 | MED | `mcrs/rerank/neural.py` | Train==serve `max_doc_chars` parity relies on an implicit default of 2000. | Open — pin a shared constant / assert. |
| M-cb1 | MED | `mcrs/training/colbert_data.py:32` | Hard-neg mining excludes the exact gold but not same-recording/near-dup variants → noisy contrast. | Open — drop ISRC/title-key matches if the catalog isn't recording-deduped. |
| M-cb2 | MED | `mcrs/training/colbert_finetune.py` | Selection metric (rerank a 100-pool) ≠ G3 gate (full-catalog) → can pick a suboptimal checkpoint (deflation, not leak). | Open — accept as cheap proxy or validate the selected ckpt on full-catalog. |
| M-sh1 | MED | `mcrs/run/harness.py:95` | `validate_submission` tolerates extra prediction rows (scoring itself is safe). | Open — also flag `extra = seen - expected`. |
| L-sh2 | LOW | `mcrs/data/ids.py` | `track_id:` prefix stripping must be a no-op vs the official grader, else blind submission must emit raw-form ids (leaderboard transfer risk, not a dev-number issue). | Open — one-line assert in the submission nb. |
| misc | LOW | several | `k_min` dead param in `sample_negatives`; `release_year=0.0` sentinel; goal-progress feature must stay out of the submission (serve-unavailable); catalog_diversity not comparable final-turn vs all-turns. | Noted — no action now. |

## Verified clean (checked, no leak)

- No gold/label in any K2 feature; `dense_cos`/`ce_score` use no gold; `ce_score` min-max normalized within-pool only.
- Train (`dsd['train']`) and dev (`dsd['test']`) session-disjoint; K2 early-stop val carved from train only, session-disjoint; dev eval over `conv_dv` only.
- Frozen CE genuinely has no adapter (leak-free stack, no OOF needed); scored over train+dev pools with the same `-1.0` out-of-pool sentinel.
- K3b: gold always at loss index 0, never sampled as a negative, near-dup denoise never drops the gold; masked listwise loss correct; adapter-attach asserted (never serves base silently).
- ColBERT: selection `[0:SEL]` vs gate `[SEL:GATE]` disjoint; triples from train only; gold never a hard negative; history strictly in-session past turns; ids canonical on both sides of every recall match.
- Shared: nDCG@20 delegates to the official metric (single-gold idcg correct); segmentation uses past in-session history only; `build_artist_cooc` built on the train split only; fusion routing hard-fails on a bad `per_channel` key; `build_rerank_groups` routes per-channel queries identically to serve.

## Post-fix ML review (adversarial)

A second adversarial reviewer checked the fix batch: verdict SHIP, no CRITICAL/HIGH. All five fixes
verified correct (train==serve parity exact, gold never dropped, deterministic-yet-decorrelated
sampling, doc-sig captures the doc-vector inputs, qcache clear correctly positioned). Two cheap
guards were added in response:
- `mcrs/rerank/train.py` — `build_rerank_groups` now raises on a routing key that matches no channel
  `query_key` (mirrors `InferenceHarness`), so a typo'd ColBERT key can't silently train K2 on a
  full-vs-focused-skewed pool. Covered by `test_build_groups_raises_on_unmatched_routing_key`.
- `nb/phase2_rerank.ipynb` — `assert _enr` in the `USE_COLBERT` block (the trainer requires an
  enriched parquet, matching the enriched-trained ColBERT checkpoint).

Colab pre-flight before trusting the rebuilt K2 number: (1) run phase2_colbert_finetune first and
confirm the checkpoint lands at `COLBERT_OUT_DIR`; (2) confirm the ColBERT checkpoint loads with only
pylate+peft; (3) print `k2.feature_names_` and confirm `rank_inv__colbert` is present and the full
ordered list equals what phase3 builds; (4) confirm the PLAID index reports "built" then "loaded" on
re-run (doc-sig cache behaves); (5) tie out fusion-only nDCG@20 vs the finetune gate's fused number.

## Verification of this fix batch

- `pytest tests/` → 419 passed, 1 skipped. The only 4 failures are `ModuleNotFoundError: lightgbm` (Colab-only dep, not installed in local dev; they fail identically without these changes).
- ColBERT channel suite: 19 passed (covers the C1 `expansion_first` path).
- Notebook wiring checked against real signatures: `InferenceHarness(... per_channel_query_builders=)`, `build_or_load_plaid`, `colbert_retrieve`, `build_rerank_groups(... per_channel_query_builders=)`, `ColBERTChannel.from_catalog(... expansion_first=)`.
- The phase2_rerank ColBERT changes run on GPU/Colab only (PLAID); not executed here — covered by the pending ML review + a Colab smoke run before relying on the K2 number.

## Next

The open MEDIUM/LOW items are quality/robustness, safe to batch later. The HIGH batch above unblocks
trustworthy K2 numbers on the real spine; proceed to the nDCG levers in `NDCG_Improvement_Plan.md`
only after a Colab dev-confirm of the rebuilt K2.
