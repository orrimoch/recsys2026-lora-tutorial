---
name: review
description: Review recent code changes for correctness and pipeline compatibility
disable-model-invocation: true
---

Review code changes for correctness, performance, and pipeline compatibility.

Usage: /review

Steps:
1. Read `.claude/agents/code-reviewer.md` for the review checklist
2. Check git diff for recent changes
3. Validate module interfaces, output format, constraints
4. Report: issues found, suggestions, approve/request changes
