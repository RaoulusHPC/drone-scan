---
name: retro
description: Capture sprint retrospective learnings and turn them into concise rules in CLAUDE.md (/retro).
---
1. Ask the user for: what went well, what went badly, one improvement (or read notes given in $ARGUMENTS).
2. Derive at most 3 concrete, checkable rules for Claude Code.
3. Append them to the "Lessons Learned" section of CLAUDE.md in the format "YYYY-MM-DD – rule – reason (KAN-key)". Keep CLAUDE.md short: merge or remove outdated rules instead of growing endlessly.
4. Optionally create Jira tasks (label `retro`) for improvements that need work.
