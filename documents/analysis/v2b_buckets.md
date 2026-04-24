# Bucket analysis — tid=004 (post-v2b champion ~= v2a)

Source: `experiments/runs/004-bm25-imputed/buckets.json` ·
**overall nDCG@10 = 0.0752** · 8000 queries

## Breakdown

| bucket | n | ndcg@10 | note |
|---|---|---|---|
| q0_userwarm_trackcold | 2,320 | 0.0849 | **strong** — BM25+tags wins on rare tracks |
| q0_usercold_trackcold | 832 | **0.0906** | **strongest** — text matching doesn't need user history |
| q4_userwarm_trackwarm | 1,344 | 0.0774 | top-pop with warm user; around average |
| q2_userwarm_trackwarm | 796 | 0.0637 | mid-pop warm user; below average |
| q1_userwarm_trackwarm | 843 | 0.0666 | low-pop warm user |
| q3_userwarm_trackwarm | 593 | 0.0667 | high-pop warm user |
| q2_usercold_trackwarm | 300 | 0.0750 | |
| q3_usercold_trackwarm | 201 | 0.0660 | |
| q1_usercold_trackwarm | 289 | 0.0584 | **weak** |
| q4_usercold_trackwarm | 482 | **0.0550** | **weakest large bucket** |

## Observations

1. **BM25 + tags does remarkably well on cold tracks**
   (pop-quintile 0, i.e., tracks never seen in train conversations):
   0.085–0.091 nDCG@10 for queries whose gold is a cold track. Any
   follow-up should avoid breaking this — purely CF-based methods would
   likely regress here.

2. **Weakest large cell**: `q4_usercold_trackwarm` (482 queries at 0.055).
   The failure mode: cold user (no personalization signal), popular gold
   track (many candidates), text query alone is not enough to pick the
   right warm track. This is exactly the use-case v4 (RRF BM25+dense)
   and v5 (adding cf-bpr as a stream) target — dense semantic matching
   helps disambiguate among popular candidates, even when the user has
   no history.

3. **Warm-user warm-track** (q1–q3) sits at 0.064–0.067, worse than
   average. These users have train-split history so a collaborative
   signal (cf-bpr, user-embeddings) could personalize the shortlist
   without needing text ambiguity resolution.

## Hypothesis for post-v4 backlog

After v4/v5 land, look specifically at the `q4_usercold_trackwarm`
bucket in the new bucket analyses. If v4 helps there by > 0.01, v5's
cf-bpr addition is likely marginal for cold users. If v4 helps little
there, consider:

- **User-embedding augmentation** for warm users: use
  Challenge-User-Embeddings as an extra RRF stream. Current backlog has
  cf-bpr (item collaborative filter) but not user-embedding direct
  retrieval. Potential new v5.5.
- **Popularity prior at rerank** (v12) specifically as a tie-breaker
  for `q4_userwarm_trackwarm` where top-pop gold is among many
  candidates — the popularity smoothing would bias toward more
  likely-to-be-gold choices.
- **Cold-user prompt enrichment** (related to v8 HyDE): pseudo-query
  expansion may help cold users more than warm users since there's no
  history to anchor the request.
