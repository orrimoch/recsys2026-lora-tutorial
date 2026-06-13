# Benchmarks — RecSys 2026 Music CRS

Blind-A composite = 0.50·nDCG@20 + 0.10·CatDiv + 0.10·LexDiv + 0.30·((LLM−1)/4).
Blind set = 80 sessions × 1 turn; single-sample SE ≈ 0.026 (±~0.05 noise band). A higher
composite from a single submission is "best-known", NOT a statistically confirmed win.

## Blind-A — best-known

| date | config | composite | nDCG@20 | CatDiv | LexDiv | LLM | notes |
|---|---|---|---|---|---|---|---|
| 2026-06-13 | EXP-001: 203 tracks + Gemini-pro bo1 | **0.4673** | 0.30 | 0.03 | 0.78 | 4.15 | new best-known; +0.025 over 204 (~1σ, within noise) |
| 2026-06-11 | 204: full stack + Gemini bo1 | 0.44 | 0.24 | 0.03 | 0.79 | 4.2 | prior best; intent_state Q* hurt nDCG (0.30→0.24) |
| 2026-06-11 | 203: ColBERT+CLAP + v5-kto | 0.37 | 0.30 | 0.03 | 0.79 | 2.85 | strong recall, weak responder |

**Current best-known: EXP-001 (203 tracks + Gemini-pro bo1) = 0.4673.** The 203-vs-204 gain is
attributed to recall (203's nDCG 0.30 vs 204's 0.24, Q*-free) — directionally consistent across
two independent Blind measurements, not statistically confirmed at 80 sessions.
