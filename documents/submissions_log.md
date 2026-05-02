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

(none yet — log starts post deep-review fix wave 2026-05-02)

<!-- Example row (uncomment and fill on each upload):
2026-05-09 [blindA] tid=301-pilot-blindset-A stage=W6-pilot score=3.20 ndcg20=0.21
-->
