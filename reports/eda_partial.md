# P0 — Partial Recall-Ceiling (local CPU validation run)

> **Scope:** validates the probe stack end-to-end on real dev data with the CPU-only channels.
> **Excludes the dense-text channel (R4)** — it needs a real query encoder (Colab/GPU); added in
> `nb/phase0_eda.ipynb` for the full ceiling. Also: `content_knn` here pools **cf-bpr** vectors
> (cheap proxy), not the proper `metadata-qwen3` content embeddings — so content is undersold.
> Treat these as a **lower bound**, not the final ceiling.

## Setup
- 150 dev sessions = **1,200 turns** (cold=300 / warm=900), `cold_threshold=1`.
- Pull depth 200; canonical-id check: **golds_in_catalog = 1200/1200** (every gold resolves).
- Channels: BM25 (R3), content-kNN on cf-bpr (R5), CF on cf-bpr (R5), same-artist (R5), weighted RRF k=60 (R7).

## Recall-ceiling (partial)
| channel | r@20 | r@100 | r@200 | unique-recall |
|---|---|---|---|---|
| same_artist | 0.197 | 0.366 | **0.388** | 0.087 |
| content_knn (cf-bpr proxy) | 0.122 | 0.227 | 0.269 | 0.028 |
| bm25 | 0.124 | 0.226 | 0.253 | 0.061 |
| cf | 0.019 | 0.070 | 0.100 | 0.029 |
| **FUSED** | **0.247** | **0.399** | **0.456** | — |

FUSED r@200 by segment: cold = 0.437, warm = 0.462.

## Reading
- **Fused r@200 ≈ 0.46 without dense-text** — far below the §7 gate (recall@200 ≥ 0.90). The gap is the recall wall; closing it needs the **dense-text channel + catalog enrichment (A1) + extension channels (R6)**. This matches the plan's central thesis.
- **same_artist is the strongest single channel** (r@200 0.388, top unique 0.087) — session artist continuity is a dominant signal.
- **cf alone is weak** (r@200 0.100) — collaborative prior is a minor contributor at this depth; content/text channels carry recall.
- cold vs warm fused recall are close (~0.44 vs ~0.46) at this stage.

## Next (full probe — `nb/phase0_eda.ipynb`, Colab)
Add the dense-text channel (BGE/qwen3 encoder), use `metadata-qwen3` for content-kNN, run full dev at depth 500, answer the remaining §5 EDA questions (gold∈history rate, context-length cap, cold/warm threshold sweep, leakage audit), and emit the DESIGN PARAMETERS block (fusion_k, topk_internal, channel keep-list) per `20_P0_*`.
