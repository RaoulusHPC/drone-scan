---
name: story
description: Start work on a Jira story of project "DRN" (key KAN) (e.g. /story KAN-12) - reads the story, plans, and follows the team workflow.
---
Work on Jira story $ARGUMENTS following docs/projektmanagement.md section 6:

1. Read the story via the Atlassian MCP server (summary, description, acceptance criteria, links/blockers). If blockers are open, stop and report.
2. Read the relevant sections of docs/architektur.md and related ADRs.
3. Enter plan mode: list files to touch, tests per acceptance criterion, VRAM/license impact, risks.
4. Post the plan as a comment on the Jira story and wait for the user's approval.
5. After approval: create branch feat/<KEY>-<slug>, write tests first, implement, run `make check` (and `make test-gpu` for GPU code).
6. Run subagent `reviewer` (plus `license-auditor`, `vram-guard`, `flight-safety` where relevant) and fix BLOCKERs.
7. Open the PR titled "<KEY>: <summary>", move the story to "In Review", comment with PR link and measured values (runtime, peak VRAM).
Never set the story to Done yourself.
