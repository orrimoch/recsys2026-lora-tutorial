# Blind responses — scored archive

Append-only. One entry per scored blind submission: all 80 (query, predicted_response, blind_score) triples. Source for Judge-behavior-notes mining (J-A1) since we don't run a local LLM judge.

Per `recsys_challenge_plan.md` §2.5.4.

## Schema (per blind submission)

```markdown
## Exp {exp_id} — Blind-{A|B} — YYYY-MM-DD — composite_blind={x.xx}, LLM={y.yy}

### Rows

| # | query (truncated) | predicted_track_ids[0] | predicted_response (full) | turn_num | user_profile (age/country/gender) | session_id |
|---|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... | ... |
| ... |

### Post-score inspection (manual / J-A1)

- **Worst-5 rows**: rows {..., ..., ..., ..., ...} — common failure pattern: {observation}
- **Best-5 rows**: rows {..., ..., ..., ..., ...} — common success pattern: {observation}
- **Hypotheses added to `agent_memory.md` Judge-behavior-notes**: {list}
- **Hypotheses validated / falsified vs prior registry**: {list}
```

---

*(Empty — first entry after Wave 3 Blind-A submission scores.)*
