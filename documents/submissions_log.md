# RecSys 2026 — Blind submissions log

`scripts/validate_prediction.py --check-budget` parses this file for the
weekly submission cap (3/week per plan §2.6). Each accepted Blind-A or
Blind-B submission MUST append a row here with the actual upload date.

Format: any line containing `[blindA]` or `[blindB]` and a `YYYY-MM-DD`
date is counted. Recommended row shape:

    YYYY-MM-DD [blindA|blindB] tid=<config-id> stage=<W4|W5|W6|W7> score=<gemini> ndcg20=<retrieval>

The validator's `_DATE_RE` matches the first `YYYY-MM-DD` token on the
line, so put the date first if there could be ambiguity.

## Submissions

2026-06-13 [blindA] tid=EXP-001-203-gemini-bo1 stage=loop score=4.15 ndcg20=0.30 composite=0.4673
2026-06-13 [blindA] tid=EXP-006-205-flashrank2048-gemini-bo1 stage=loop score=4.25 ndcg20=0.33 composite=0.49

<!-- Example row format (do NOT include a real date — the validator's
_DATE_RE does not understand HTML comments and would count it against
the weekly cap):
YYYY-MM-DD [blindA] tid=301-pilot-blindset-A stage=W6-pilot score=3.20 ndcg20=0.21
-->
2026-06-13 [blindA] tid=EXP-010-207-e5replace-w1.5-k100-litegemini stage=loop score=3.55 ndcg20=0.28 composite=0.41
2026-06-13 [blindA] tid=205-resubmit-flashrank2048-gemini-pro-bo1 stage=loop score=4.35 ndcg20=0.33 composite=0.50
