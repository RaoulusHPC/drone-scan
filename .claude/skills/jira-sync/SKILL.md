---
name: jira-sync
description: Create or update the Jira backlog (project "DRN", key KAN) from backlog/backlog.yaml via the Atlassian MCP server (/jira-sync). Idempotent via backlog/jira-map.json.
---
Sync backlog/backlog.yaml to Jira (site b-f-u.atlassian.net, project key KAN). $ARGUMENTS may restrict to ids (e.g. "EP-01 US-001").

1. Read backlog/backlog.yaml and backlog/jira-map.json (create `{}` if missing). Items already in the map are only updated, never duplicated.
2. Show the user a summary (number of epics/stories to create or update) and ask for confirmation before the first write.
3. Create epics first (issue type "Epic", summary "<EP-id> <title>", labels [<ms>, "epic"]). Record keys in jira-map.json after EACH create (so a crash never causes duplicates).
4. Create stories/tasks with: issue type from `type`, summary "<US-id> <title>", parent = epic key, priority (Must→Highest, Should→High, Could→Medium), story points field ("Story point estimate" in team-managed projects; check field metadata once), labels = [ms] + labels, description = story text + acceptance criteria as checklist + context (same format as tools/backlog_export.py jira_description).
5. Add issue links "Blocks" for every dependency (dep blocks story).
6. Run `python3 tools/backlog_export.py` so docs/user-stories.md shows the Jira keys; commit on a branch `chore/jira-sync` and open a PR.
7. Report created/updated/failed items. Never delete or transition issues.
