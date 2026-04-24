---
name: research
description: Research a topic relevant to the challenge — papers, techniques, implementations
disable-model-invocation: true
---

Research a topic and save findings for future reference.

Usage: /research <topic>

Steps:
1. Read `.claude/agents/researcher.md` for the research workflow
2. Search for relevant papers, repos, and blog posts
3. Summarize findings with relevance to our pipeline
4. Save report to `documents/research/<topic-name>.md`
5. Report: key findings, recommended next steps

$ARGUMENTS is the topic to research.
