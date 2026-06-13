# Loop Experiments Log — Pre-Registration Journal

Pre-registration journal for the autonomous research loop (see RESEARCH_CHARTER.md).
NOTE: distinct from `experiments_log.md` (the pre-existing W1–W8 gate decision log).

Each experiment is pre-registered BELOW (hypothesis + gate + decision rule) BEFORE the run.
After results, fill the `--- run ---` block. INCONCLUSIVE is a first-class verdict.
Template:

## EXP-NNN — <lever> — YYYY-MM-DD
Hypothesis:   <falsifiable claim>
Lever:        recall | reranker | responder
Change:       <config NNN / code diff summary>
Pre-registered gate:  <metric> on <Tier-1 signal> must beat <baseline> by <delta>
Baseline:     <honest number + source>
Decision rule: PASS -> keep/promote; FAIL -> revert;
               INCONCLUSIVE (|Δ| < <noise band>) -> no change, counts toward switch rule
Smoke:        pytest <paths> green before handoff
--- run ---
Result:       <pasted RESULTS_JSON>
Verdict:      PASS | FAIL | INCONCLUSIVE
Local→Blind:  local Δ <>, blind Δ <>   (only if submitted)
Memory:       <file written/updated>

---

## BUILD NOTES / open refinements (2026-06-13)
- nb74 emitter reports a RETRIEVAL-ONLY composite (cat_div/lex_div = 0; it is the retrieval
  gate). The loop MUST judge nb74 on `ndcg@20` (turn-1), not its `composite` field.
- nb80 emitter records BLIND (Tier-2) metrics the human pastes from the CodaBench result page.
  It is NOT the Tier-1 LLM signal — blind scores can't be computed locally.
- FOLLOW-UP (before the first responder iteration): wire a RESULTS_JSON emitter onto the DEV
  responder + OFFLINE Gemini judge path (gemini_judge_responses.py — likely nb79/nb75) so the
  Tier-1 LLM-axis reward is captured locally and leak-free. The plan wired nb74+nb80 only.

## EXP-000 — example (do not run) — 2026-06-13
Hypothesis:   Gemini responder on config-203 track_ids (nDCG 0.30, no Q*) lifts composite to ~0.47.
Lever:        responder
Change:       config 205 = 203 recall track_ids + gemini_responder.py
Pre-registered gate:  blindA composite must beat 0.44 (config 204) by > 0.05 (noise band)
Baseline:     config 204 composite 0.44 (benchmarks.md / memory project_blind_a_state_responder_lever)
Decision rule: PASS -> promote to blind candidate; FAIL -> revert; INCONCLUSIVE (|Δ|<0.05) -> next hypothesis
Smoke:        n/a (example)
--- run ---
Result:       (not run)
Verdict:      (n/a)

---

## EXP-001 — responder base swap (203 tracks + Gemini) — 2026-06-13
Hypothesis:   Putting config-203 Blind-A track_ids (Blind nDCG 0.30, no intent_state Q*) under
              the SAME Gemini best-of-3 responder (LLM ~4.2) lifts Blind composite from 0.44
              (config 204, nDCG 0.24) to ~0.47. Both axes are independently Blind-grounded
              (203 nDCG 0.30 from its 0.37 submission; Gemini LLM ~4.2 from the 204 submission),
              so this banks the best-known config to the leaderboard.
Lever:        responder base / recall (drop intent_state Q* by using the 203 recall stack)
Change:       config-203 Blind-A track_ids + nb80 Gemini responder swap (TOP_N=1, BEST_OF=1,
              gemini-2.5-pro, plain prompt, STRUCTURED_PERSONALITY=False). NO code change —
              track_ids untouched, only predicted_response regenerated.
              BEST_OF=1 chosen for fast iterations AND apples-to-apples: config 204's 0.44
              baseline was itself produced with bo1 (zip ...204-full-gemini-bo1.zip), so this
              isolates the recall/nDCG lever (203's 0.30 vs 204's 0.24), not the responder.
              (TOP_N=1 is the v5-kto-winning setting; TOP_N=3 was a speculative dilution risk.)
Pre-registered gate:  Blind-A composite (CodaBench) ≥ 0.44 (current best, config 204); target ≈ 0.47.
Baseline:     config 204 = Blind 0.44 (nDCG 0.24 / Cat 0.03 / Lex 0.79 / LLM 4.2);
              config 203 = Blind 0.37 (nDCG 0.30 / LLM 2.85 v5-kto).
              Source: memory project_blind_a_state_responder_lever_2026_06_11 + 203/204 yaml headers.
Decision rule: PASS if composite ≥ 0.46 → new best (update benchmarks.md + memory).
              INCONCLUSIVE if 0.42–0.46 (within ±0.05 noise of 0.44) → still adopt the 203 base
              (its nDCG edge is Blind-proven) but flag the LLM axis didn't separate; counts
              toward the switch rule.
              FAIL if < 0.42 → responder regressed on 203 tracks; investigate before next submit.
De-risk:      SMOKE pass WAIVED by operator (2026-06-13) — full 80-row run directly. Risk accepted:
              known-good Gemini responder + per-row API-failure fallback keeps the original response.
Budget:       1 of 3 weekly Blind slots (0/3 used as of 2026-06-13).
Smoke:        n/a (no code change; harness pytest suite already green, 17/17).
Reviews:      code-review N/A (no diff); RecSys-researcher review REQUIRED on the verdict before banking.
--- run ---
Result:       {"exp":"EXP-001","config":203,"ndcg@20":0.30,"cat_div":0.03,"lex_div":0.78,"llm_judge":4.15,"composite":0.4673,"n_sessions":80,"gate":"blindA"}
Verdict:      PASS (per gate, 0.4673 ≥ 0.46) → NEW BEST-KNOWN (was 0.44), NOT a statistically
              confirmed win. Gain +0.025 over config 204 is ~1σ on 80 sessions (SE≈0.026, p≈0.17,
              ~22% by noise) → inside the ±0.05 band. nDCG 0.30 vs 204's 0.24 is DIRECTIONALLY
              consistent with the Q*-harms-nDCG hypothesis (two independent Blind measurements,
              pre-predicted direction) — supported, NOT proven at 80 sessions. LLM held 4.2→4.15
              (within noise; responder is NOT isolated — 203 tracks yield their own LLM). Both axes
              moved: recall +~0.03, LLM −~0.004.
Local→Blind:  projected ~0.47, blind 0.4673. Tight — BUT the nDCG projection is an algebraic identity
              for unchanged track_ids (not a calibration win); do not over-read one data point.
Reviews:      RecSys-researcher = APPROVE-WITH-CORRECTIONS (framing above is the corrected version).
              No code diff → code-review N/A.
Decision:     ADOPT "config-203 track_ids + Gemini-pro bo1" as new benchmark (best-known 0.4673).
              Budget spent: 1/3 weekly. Next lever = in-pool SASRec reranker (dev gate, turn-1
              STRATIFIED per review) — see EXP-002 plan.
Memory:       [[project_autonomous_research_loop_2026_06_13]] + benchmarks.md + submissions_log.md updated.

---

## EXP-002 — in-pool SASRec reranker (DEV gate, no Blind slot) — 2026-06-13
Hypothesis:   Fine-tuning SASRec on the in-pool ranking objective (softmax-CE over the SASRec-free
              recall pool top-100, goal-ful context) converts a meaningful share of the documented
              18.2% recall→ranking gap, lifting dev nDCG@20 above the lgbm_relev baseline 0.1652.
Lever:        reranker (in-pool SASRec ranking). Attacks the conversion gap, NOT the 42% recall wall.
Change:       RUN the already-built lever (no code diff): nb74 cell 53 #12c-train (warm-start
              sasrec_v1, topk=100, SMOKE→full → sasrec_inpool_v1) → cell 54 #12c-inpool dev gate.
Pre-registered gate (PRIMARY): in-pool SASRec-alone OVERALL dev nDCG@20 (nb74 cell 54, test split,
              goal-ful 3-way parity) vs lgbm_relev=0.1652 and recall-only=0.1498.
Baseline:     lgbm_relev dev nDCG@20 = 0.1652; recall-only = 0.1498 (SASRec_Improved_Plan.md §7).
Decision rule (STAGED):
              FAIL  if overall < 0.1652 → in-pool objective didn't convert; bank the negative,
                    switch lever (do NOT pursue turn-1/3-way).
              INCONCLUSIVE if 0.1652 ≤ overall < 0.1752 (gain < +0.01) → marginal; only worth the
                    LGBM+inpool-feature variant (c) if it specifically helps, else shelve.
              PASS  if overall ≥ 0.1752 → promising; BUT before any Blind promote, run the FOLLOW-UP
                    turn-1-stratified 3-way (SASRec-alone vs LGBM-alone vs LGBM+inpool-feat, plan
                    §10.2): SASRec is COLD on turn-1 (empty sequence) and Blind is turn-uniform
                    (~1/8 turn-1), so SHIP only the variant that does NOT regress turn-1 vs LGBM.
                    Promote requires the crs_baseline goal-ful serve-parity change (plan §2).
Budget:       0 Blind slots (DEV/Tier-1 only). Blind confirmation only AFTER a dev PASS + the
              turn-1 check + the serve-parity code change.
Smoke:        cell 53 SMOKE=True (300 sessions / 1 epoch) to validate shapes + gold-in-pool% BEFORE
              the full --epochs 3 fine-tune.
Reviews:      code-review N/A (running built code, no diff). RecSys-researcher REQUIRED on the
              dev-gate verdict before banking. Tier-3 watch: dev is a cold-Blind-biased proxy — a
              dev nDCG gain must pass the turn-1 robustness check before we trust Blind transfer.
--- run ---
Result:       in-pool SASRec-ALONE OVERALL dev nDCG@20: smoke(300 sess/1ep)=0.0922; FULL(71470
              trainable/121592 = 58.8% gold-in-pool, 3 epochs, loss 3.309→3.184→3.105)=0.0885.
              vs lgbm_relev 0.1652, recall-only 0.1498.
Verdict:      FAIL (0.0885 < 0.1652; below even recall-only 0.1498). Training-INVARIANT (40× data /
              3× epochs: 0.0922→0.0885 with loss monotone-down) → ceiling is STRUCTURAL, not undertraining.
              CONFOUND (RecSys review, structurally derivable not post-hoc): gate ranks the SERVE pool
              (union WITH SASRec, use_sasrec=True) but the model trained on the SASRec-FREE pool. The
              inpool model IS a SASRec dual-encoder → it over-scores SASRec-retrieved items, displacing
              the conversion-gap golds (which SASRec did NOT retrieve) below rank 20. Model has real
              signal (≈2.1× random) but can't beat the wRRF prior it discards.
Reviews:      RecSys-researcher = SIGN-OFF on FAIL banking with the confound framing. It recommended
              EXP-003 (option c: LGBM + inpool-score-as-feature). Claude OVERRULES jumping there:
              inpool_score on TRAIN is in-sample (model trained on train) = the documented sasrec_rank_inv
              leak trap (+internal/−dev) unless OOF cross-fit; and the conversion-gap bucket ceiling is
              small (~+0.01). Option (c) DEFERRED (only with proper OOF).
Decision:     BANK FAIL; in-pool-ALONE closed. NEXT = ONE cheap decisive diagnostic: re-gate on the
              MATCHED SASRec-free pool (cell 54 use_sasrec=False) — does the model rank its training-
              matched pool well? PASS-on-matched → confound confirmed, in-pool salvageable via a
              pool-parity retrain (parked, leak-aware). FLAT(~0.09)-on-matched → in-pool truly dead →
              pivot to the 42% new-artist WALL (propose-ground variant) = the bigger nDCG bucket.
Diagnostic:   Re-gated on the TRAIN-matched (SASRec-free) pool: inpool = 0.1093 vs same-pool
              recall-only prior = 0.1449 (inpool −0.036 BELOW the prior even on its own pool).
              CONCLUSION: confound was real but PARTIAL (matching pool gained only +0.021:
              0.0885→0.1093); the dominant cause is the model is a genuinely WEAK ranker — a single
              SASRec dual-encoder loses to the multi-signal wRRF fusion (BM25+dense+same_artist) it
              replaces, even under ideal matched-pool conditions. Pool-parity retrain ceiling ≈ 0.11
              << 0.1652 → NOT worth it. IN-POOL RERANKING CLOSED. Option (c) also closed (inpool
              score loses to the prior → redundant as an LGBM feature + leak-risky).
Next lever:   PIVOT to the 42% new-artist recall WALL (the bigger nDCG bucket) — to be pre-registered
              as EXP-003. Cheapest first probe: attributes-qwen3 orthogonal dense channel (training-free,
              cold-query-safe). Stronger option: a goal-seeded generate→retrieve propose-ground variant.
Memory:       [[project_autonomous_research_loop_2026_06_13]] + [[project_ndcg_campaign_status_2026_06_08]].

---

## EXP-003 — attributes-qwen3 2nd dense channel (DEV recall gate, no Blind slot) — 2026-06-13
Hypothesis:   Adding the orthogonal, precomputed attributes-qwen3 dense channel (use_attributes=True)
              to union+SASRec lifts TURN-1 recall@100 (the cold/Blind proxy) beyond noise — reaching
              new-artist wall golds the metadata-only content channel misses. attributes-qwen3 is
              verified orthogonal to metadata-qwen3 (0.62 cosine, 3.6% top-10 overlap). Training-free,
              config-only; channel + embeddings + tests already exist (report §3.1 "START HERE", un-run).
Lever:        recall (orthogonal content channel). nb74 cell 7 (#4-attr), NO code diff.
Pre-registered gate: nb74 cell 7 — TURN-1 recall@100 with use_attributes=True (best of w∈{0.3,0.4,0.7})
              vs the union+SASRec baseline turn-1 recall@100 (printed by the same cell, by-turn table).
Baseline:     union+SASRec recall@100 overall ≈ 0.4961; turn-1 baseline read from the cell-7 output.
Decision rule: PASS if best-w turn-1 recall@100 lifts ≥ +0.01 AND consistent across ≥2 of the 3 weights
              (robust, not a single-weight fluke) → proceed to the nDCG-CONVERSION check (campaign trap:
              recall-up-but-Blind-nDCG-flat, so a recall PASS is necessary NOT sufficient). Conversion
              test = Blind confirm (203-stack + use_attributes track_ids + Gemini responder; cheap @10/day).
              INCONCLUSIVE if lift < +0.01 (within noise) → orthogonal vectors don't add turn-1 reach with
              the EXISTING query → query is the bottleneck → next = structured content query (#3.1b cell 10)
              or goal+culture enrichment (#3.2 cell 8).
              FAIL if turn-1 recall DROPS at all weights → attributes injects RRF noise → don't use.
Budget:       0 Blind (DEV gate). Blind conversion test only after a dev PASS.
Reviews:      code-review N/A (config-only built cell). RecSys-researcher REQUIRED on the verdict — is
              the turn-1 lift beyond N(turn-1) sampling noise, and likely to CONVERT vs the documented
              recall-up/Blind-flat pattern?
--- run ---
Result:       (pending human run: nb74 cells 1→3→4→7)
Verdict:      (pending)
