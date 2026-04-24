---
name: experiment
description: Set up and run a new experiment — creates config, implements changes, logs results
disable-model-invocation: true
---

Set up and run a new experiment on the CRS pipeline.

Usage: /experiment <description of what to try>

Steps:
1. Read `.claude/agents/experimenter.md` for the full workflow
2. Read the most relevant existing config YAML as a starting point
3. Create a new config YAML in `music-crs-baselines/config/`
4. Implement any new modules or changes needed
5. Test locally with a small batch if possible
6. Log the experiment setup to `documents/experiments/experiment-log.md`
7. Report: what was changed, how to run it, expected impact

$ARGUMENTS is a description of the experiment to run.
