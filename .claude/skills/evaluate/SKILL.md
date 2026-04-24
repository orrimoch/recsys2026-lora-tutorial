---
name: evaluate
description: Run evaluation on an experiment and compare against baselines
disable-model-invocation: true
---

Evaluate an experiment's results and compare against baselines.

Usage: /evaluate <tid>

Steps:
1. Run `python evaluate_devset.py --tid <tid>` from `music-crs-evaluator/`
2. Read the scores from `exp/scores/devset/<tid>.json`
3. Compare against baseline results in CLAUDE.md
4. Append results to `documents/experiments/experiment-log.md`
5. Report: scores, delta vs baselines, observations

$ARGUMENTS is the experiment tid to evaluate.
