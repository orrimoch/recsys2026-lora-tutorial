# EDA — missing tag_list / release_date in Challenge-Track-Metadata

**Context:** backlog item v2b (tid=003) hypothesises that artist-level median
imputation recovers ~5% of tracks that would otherwise be unmatched in the
text-retrieval path. Before writing any code, check the real missingness.

## Dataset

- `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` split `all_tracks`
- 47,071 tracks · 9,255 distinct artists · mean 5.09 tracks/artist,
  median 1 track/artist (long-tail; most artists have one track each)

## Missingness

| Field | Empty count | % of catalog |
|---|---|---|
| `tag_list` (empty `list[str]`) | **87** | **0.19%** |
| `release_date` (None / empty string) | **644** | **1.37%** |

Neither comes close to the "~5%" in the v2b hypothesis. The hypothesis was
drafted before looking at the data; the EDA corrects it.

## Imputability (same-artist fallback)

| Field | Missing | Artist has ≥1 non-empty sibling | Share of missing |
|---|---|---|---|
| `tag_list` | 87 | **60** | 69.0% |
| `release_date` | 644 | **0** | 0.0% |

**Release_date is effectively non-imputable from this catalogue.** When a
track has no `release_date`, the artist's other tracks don't have one either
— those artists appear to be entirely undated in the metadata. There is no
signal to borrow from.

**Tag_list is partially imputable** but the addressable set is ~60 tracks
(0.13% of the catalogue).

## Tag distribution (non-empty `tag_list`)

- min = 1, median = 17, mean = 33.6, max = 105 tags/track

The catalogue has strong tag coverage where tags exist.

## Expected impact on nDCG@10

Upper bound: even if every one of the 60 imputable tracks becomes the gold
match for a devset query it wouldn't have matched before, at ~8000 devset
queries the max swing is `≤ 60 / 8000 ≈ 0.0075`. Realistically the gain is
an order of magnitude smaller (most of those queries wouldn't have the
track as gold; imputed tags are a weak signal vs. the artist's stronger
signals already in the corpus; many of the 60 tracks are from artists the
retriever already surfaces via their other tagged tracks).

**Best-case expected Δ ≈ 0.0003; threshold is 0.01479; no prerequisite
pass** → v2b is expected to be rejected.

## Imputation rule (if we proceed)

- **`tag_list`**: for a track with empty `tag_list`, take the union of
  tags across all other tracks by the same artist (de-duplicated). Do not
  cap the length — BM25 handles long documents fine. Apply before building
  the BM25 index.
- **`release_date`**: skip. Non-imputable from this catalogue.

## Recommendation

Run a minimal v2b with `tag_list` imputation only. Confirm the EDA's
prediction empirically (a 2-minute retrieval-only run). Expect rejection.
The real value here is the EDA itself, which calibrates the hypothesis
and saves us from chasing similar imputation ideas without checking the
numbers first.
