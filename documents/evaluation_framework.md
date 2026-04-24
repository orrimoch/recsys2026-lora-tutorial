# Evaluation Framework — TalkPlayData RecSys Challenge 2026

Multi-dimensional evaluation assessing both *what* the system recommends and *how* it recommends it.

## 1. Composite Score

```
Score = 0.50 × nDCG@20
      + 0.10 × CatalogDiversity
      + 0.10 × LexicalDiversity
      + 0.30 × LLM-Judge
```

| Dimension | Weight | What it measures | How it is computed | Role |
|---|---|---|---|---|
| **nDCG@20** | 0.50 | Ranking quality of the recommended tracks. | Computed from the ranked list of predicted tracks against the ground-truth relevant item. Higher-ranked correct recommendations receive more credit. | Primary recommendation metric. |
| **Catalog Diversity** | 0.10 | How broadly a system covers the music catalog. | Number of unique recommended tracks across all predictions divided by the total catalog size. | Complementary diversity indicator. |
| **Lexical Diversity** | 0.10 | How varied the generated language is. | Distinct-2: unique bigrams divided by total bigrams across generated responses. | Complementary response-generation indicator. |
| **LLM-as-a-Judge** | 0.30 | Quality of the generated explanation. | Blind-set responses are judged by a Gemini model used as an automatic judge. Evaluates two text-only dimensions: Personalization and Explanation Quality, independent of recommendation accuracy. Judge family is disclosed; the evaluation prompt is not published. | Blind-set response-quality evaluation. |

## 2. Metric Definitions

### 2.1 nDCG@k (Normalized Discounted Cumulative Gain)

Measures ranking quality by comparing the predicted ranking to the ideal ranking.

```
nDCG@k = DCG@k / IDCG@k
DCG@k  = Σ_{i=1..k}  1[pred_i ∈ gold] / log2(i + 1)
```

- Each conversation turn has **exactly one** ground-truth track.
- Reported at **nDCG@1**, **nDCG@10**, and **nDCG@20**.
- **Primary metric: nDCG@20**.
- Results are **macro-averaged across sessions**.

### 2.2 Catalog Diversity

```
CatalogDiversity = |unique recommended tracks| / |catalog|
```

Measures whether the system explores the full music catalog or concentrates on popular items.

### 2.3 Lexical Diversity (Distinct-2)

```
LexicalDiversity = |unique bigrams across all responses| / |total bigrams across all responses|
```

Encourages varied, non-repetitive natural-language generation.

### 2.4 LLM-as-a-Judge

- Judge model family: **Gemini** (specific prompt undisclosed).
- Scores a sampled subset of sessions on multiple dimensions, each on a **1–5 integer scale**.
- Dimensions evaluate written response independently of recommendation accuracy (Personalization + Explanation Quality).
- Dimensions are **min-max normalized to [0, 1]** with fixed bounds (min=1, max=5) before composite aggregation:

```
score_norm = (score - 1) / (5 - 1) = (score - 1) / 4
```

- Weight in composite: **0.30**.

## 3. Leaderboard Phases

| Phase | Dataset | Window | Metrics |
|---|---|---|---|
| **Blind A** | TalkPlayData-Challenge-Blind-A | Apr 10 – Jun 25 | All metrics |
| **Blind B** | TalkPlayData-Challenge-Blind-B | Jun 10 – Jun 25 | All metrics (**Final**) |

## 4. Modeling Implications

- **nDCG@20 dominates (0.50)** → ranking quality of top-20 retrieval is the single highest-leverage lever. Invest in retrieval + reranking before polishing text.
- **Catalog Diversity (0.10)** penalizes popularity collapse. A system that always returns the same head-heavy top-20 will cap this term. Mitigate with: MMR, per-session novelty penalty, popularity debiasing, or sampling from long-tail.
- **Lexical Diversity via Distinct-2 (0.10)** rewards bigram variety. Template-heavy explanations (e.g. "I recommend X because you liked Y") will score low. Rotate phrasings, vary connective tokens, avoid boilerplate openers.
- **LLM-Judge (0.30)** is independent of whether the recommendation is correct — a wrong track with a personalized, well-written explanation can still earn judge points. Do not couple explanation generation too tightly to the ranker's top-1.
- **Combined weight of text-side terms (0.10 + 0.30 = 0.40)** ≈ the ranking term's weight. Ignoring generation quality leaves ~40% of the score on the table.
- **Normalization asymmetry**: nDCG is already in [0,1]; diversity terms are ratios in [0,1]; LLM-judge is renormalized from 1–5. All four terms are on the same scale at aggregation time.

## 5. Optimization Checklist

- [ ] Log `nDCG@1 / @10 / @20` every run; treat **nDCG@20** as the champion metric.
- [ ] Track unique-track count across the full prediction set — not per session — for Catalog Diversity.
- [ ] Compute Distinct-2 on the **full** response corpus (cross-session bigram pool), not per response.
- [ ] Build a local proxy judge (e.g. GPT-4-class or Claude) since the Gemini prompt is hidden — calibrate but don't over-fit.
- [ ] Before submitting, sanity-check the composite on dev: `0.50·nDCG@20 + 0.10·CatDiv + 0.10·LexDiv + 0.30·judge_proxy`.
- [ ] Watch for **popularity collapse** — a +0.01 nDCG@20 win from pure pop-bias can cost more than it gains once diversity terms are counted.

## 6. Open Questions

- Is Catalog Diversity computed over the union of top-20 across all sessions, or top-1? (Assumed: top-20 union.)
- Is Distinct-2 tokenized per response then pooled, or computed on a concatenated corpus? (Assumed: pooled bigrams across responses.)
- How large is the LLM-judge sample? (Affects variance of the 0.30 term.)
- Are ties in nDCG broken deterministically, or does rank order within tied scores matter?
