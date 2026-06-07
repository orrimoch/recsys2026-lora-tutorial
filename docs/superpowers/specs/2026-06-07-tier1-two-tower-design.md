# Tier-1 #3.3 — intent→content two-tower recall channel

Date: 2026-06-07
Status: approved (design)

## Context

The recall ceiling (the ~43% all-channel-miss, ~99% new-artist wall) is a
content/intent→item problem the reranker cannot touch. The union has one
undertrained generic dense channel; nothing has trained a content matcher on the
(intent→gold) pairs or fused the multiple frozen catalog modalities. A learned
two-tower over frozen multimodal item embeddings + a learned query head raises
the recall ceiling for never-seen artists.

## Decisions (approved)

- Item modalities: ALL FIVE — metadata + attributes + lyrics (Qwen3, 1024 each)
  + audio-laion_clap (512) + image-siglip2 (768). Per-modality LayerNorm (already
  in ItemFusion) handles scale; per-modality dropout guards overfitting.
- Reuse `ItemFusion` (sasrec_model) as the item tower; reuse train_sasrec loaders;
  reuse the carve_temporal_selection_set holdout; mirror the sasrec_seq channel.
- Opt-in union channel `use_two_tower` (default off).

## Architecture

- `mcrs/retrieval_modules/two_tower_model.py`
  - `TwoTowerModel(item_modality_dims, q_in_dim=1024, d=256, temperature=0.07)`
    - item tower = `ItemFusion(item_modality_dims, d)` (reused)
    - query tower = LN -> Linear -> LN -> GELU -> Dropout -> Linear (q_in_dim -> d)
    - `encode_item(feats) -> L2-normalized (B,d)`; `encode_query(q_emb) -> (B,d)`
    - `score(q, items) -> cosine / temperature`
  - `info_nce_loss(model, q_emb, pos_feats, extra_neg_feats=None)` — in-batch
    negatives (+ optional appended hard negatives); CE with diagonal targets.
- `mcrs/retrieval_modules/two_tower_channel.py`
  - `TwoTowerRetriever(model, item_repr, track_ids, query_encode)` — encode query
    (frozen Qwen3) -> project -> cosine over the precomputed item-repr matrix ->
    top-K. Standard retriever interface (mirrors SasrecRetriever).
- `scripts/train_two_tower.py` (Colab GPU)
  - causal (query→gold) pairs from the train split (query = raw_enriched text at
    turn t via build_retrieval_query; gold = the music track); time-based holdout
    via carve_temporal_selection_set (latest ~15% held out for internal dev).
  - query encoded by frozen Qwen3-Embedding-0.6B (instruct); item feats = the 5
    frozen modalities (reuse train_sasrec item-feature loader).
  - negatives: in-batch + same-artist-different-track hard negatives.
  - exclude played tracks from candidates AND negatives; per-modality dropout.
  - saves model state + precomputed item-repr matrix (like sasrec_v1).
- `_wrrf_union_v1_specs`: `if ec.get("use_two_tower")` -> append
  {type:"two_tower", weight: w_two_tower default 0.7, extra_config:{model_dir}}.
- `load_retrieval_module` branch `two_tower`: load model + item_repr + query
  encoder -> TwoTowerRetriever.

## Leak-free recipe

TIME-BASED holdout ONLY (random splits leak future artists). Causal queries
(turn k uses turns < k via prior_turns). Exclude played tracks. Hard negatives
from a frozen snapshot, never the live model. Gate on time-based-dev recall@100 +
turn-1; treat in-sample contrastive loss as untrusted. If the cosine ever becomes
an LGBM feature it MUST be generated out-of-fold (concat_oof_features path).

## Regression safety (must not break Tier 0 / prior Tier-1)

- All opt-in/new files; `use_two_tower` default off -> config 194 + base union +
  prior files bit-identical.
- Full test suite after change: prior tests green; only the 4 known pre-existing
  unrelated failures remain.
- Tier-end QA re-runs prior-tier guard tests.

## Tests (TDD, CPU, no GPU/data)

- TwoTowerModel: encode_item/encode_query return (B,d) L2-normalized; score shape.
- info_nce_loss decreases on a fixed overfit batch (mirror test_sasrec_model).
- TwoTowerRetriever: with a fake model + tiny item_repr, returns top-K by cosine;
  batch alignment preserved.
- union gating: two_tower off by default; appended at w_two_tower when on.

## Validation gate (Colab, no reranker retrain)

Train on Colab; build the two_tower channel; report recall@100 turn-1 on the
time-based dev vs baseline (and on the new-artist subset specifically). Proceed to
fold into the combined feature-rebuild + retrain ONLY if turn-1 / new-artist
recall lifts beyond noise. Kill criterion: standalone recall@100 on the new-artist
subset does not beat the raw metadata-qwen3 channel after the first training run.

## Non-goals

No reranker retrain here. No OOF LGBM feature yet. No serve-config change.
