# Benchmarks — RecSys 2026 Music CRS

Blind-A composite = 0.50·nDCG@20 + 0.10·CatDiv + 0.10·LexDiv + 0.30·((LLM−1)/4).
Blind set = 80 sessions × 1 turn; single-sample SE ≈ 0.026 (±~0.05 noise band). A higher
composite from a single submission is "best-known", NOT a statistically confirmed win.

## Blind-A — best-known

| date | config | composite | nDCG@20 | CatDiv | LexDiv | LLM | notes |
|---|---|---|---|---|---|---|---|
| 2026-06-13 | 205 (re-submit): flash-rank@2048 + Gemini-pro bo1 | **0.50** | 0.33 | 0.03 | 0.78 | 4.35 | best-known; REPLICATION of 205 (identical config) — only LLM moved 4.25→4.35 = judge variance (+0.0075 composite); confirms composite reproduces at 0.49–0.50 |
| 2026-06-13 | EXP-006: 205 = 203 tracks + flash-rank@2048 + Gemini-pro bo1 | 0.49 | 0.33 | 0.03 | 0.78 | 4.25 | flash listwise ranker (only change vs EXP-001); nDCG 0.30→0.33 |
| 2026-06-13 | EXP-001: 203 tracks + Gemini-pro bo1 | 0.4673 | 0.30 | 0.03 | 0.78 | 4.15 | +0.025 over 204 (~1σ, within noise) |
| 2026-06-13 | 207: e5-replace w1.5 + k=100 + Gemini-flash-LITE | 0.41 | 0.28 | 0.03 | 0.77 | 3.55 | REGRESSION −0.08; two unvalidated changes bundled (lite responder −0.0525, e5/k100 retrieval −0.025) |
| 2026-06-11 | 204: full stack + Gemini bo1 | 0.44 | 0.24 | 0.03 | 0.79 | 4.2 | intent_state Q* hurt nDCG (0.30→0.24) |
| 2026-06-11 | 203: ColBERT+CLAP + v5-kto | 0.37 | 0.30 | 0.03 | 0.79 | 2.85 | strong recall, weak responder |

**Current best-known: config 205 (203 tracks + flash-rank@2048 + Gemini-pro bo1) = 0.50.** Two
independent Blind draws of the IDENTICAL config read 0.49 and 0.50 — the composite reproduces at
0.49–0.50. CAVEAT: this is two draws of one config; nDCG 0.33 is the SAME track_ids both times
(one effective retrieval observation, not two), so the composite point estimate is reproduced but
the retrieval component is n=1. The 0.49→0.50 delta is pure LLM-judge variance (4.25→4.35,
identical Gemini-pro responder) = +0.0075 composite, well inside the ±0.05 band — NOT a real gain.
Implication: a real LLM-axis win must clear ≈ +0.5 LLM points (≈ +0.0375 composite) to be
Blind-confirmable in isolation; +0.10 LLM is noise.
