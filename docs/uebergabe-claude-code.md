# Übergabe an Claude Code (lokal, Entwicklungsrechner mit GTX 1080)

## Einmalig
```bash
git clone git@github.com:RaoulusHPC/drone-scan.git && cd drone-scan
claude                 # Claude Code im Repo starten (liest CLAUDE.md automatisch)
/mcp                   # Atlassian-MCP authentifizieren (b-f-u.atlassian.net)
```

## 1. Backlog in Jira (erledigt am 21.11.2026)
Alle 12 Epics (KAN-4 … KAN-15) und 58 Stories (KAN-16 … KAN-73) sind angelegt, inkl. 78 „Blocks“-Verknüpfungen. Zuordnung lokal ↔ Jira: `backlog/jira-map.json` (US-n = KAN-(n+15)).
Story Points stehen in der Beschreibung (das Projekt hat kein Story-Point-Feld). Meilensteine sind als Labels `m0`–`m5` gesetzt; Jira-Releases noch nicht angelegt.
Bei Backlog-Änderungen: `backlog/backlog.yaml` anpassen → `/jira-sync` (aktualisiert nur, legt keine Duplikate an).

## 2. Sprint 0 starten
In Jira einen Sprint „S0 23.11.–04.12.2026“ anlegen und alle Issues mit Label `m0` hineinziehen (JQL: `project = KAN AND labels = m0`). Dann in Claude Code:
```
/story KAN-22   # US-007 Jira- und Claude-Code-Workflow prüfen
/story KAN-16   # US-001 Repo-Grundgerüst
```
Empfohlene Reihenfolge Sprint 0: KAN-22 (US-007) → KAN-16 (US-001) → KAN-20 (US-005) → KAN-17 (US-002) → KAN-21 (US-006) → KAN-23 (US-008) → KAN-18 (US-003) → KAN-19 (US-004, Spike auf der GTX 1080).

## Start-Prompt (falls du lieber frei formulierst)
> Lies CLAUDE.md, docs/architektur.md (inkl. ADR-0010 zur GTX 1080) und docs/projektmanagement.md. Beginne dann Sprint 0 mit KAN-22 und danach KAN-16 (US-001) gemäß dem Ablauf in /story: Plan zuerst als Jira-Kommentar, auf meine Freigabe warten.

## Was Claude Code nicht darf (technisch abgesichert)
- Befehle an echte Drohnen (Hook `.claude/hooks/block_real_flight.py`)
- Force-Push, Löschen von `workspace/`
- Stories auf *Done* setzen (macht der PO nach dem Merge)
