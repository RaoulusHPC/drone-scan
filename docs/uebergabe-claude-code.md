# Übergabe an Claude Code (lokal, Entwicklungsrechner mit GTX 1080)

## Einmalig
```bash
git clone git@github.com:RaoulusHPC/drone-scan.git && cd drone-scan
claude                 # Claude Code im Repo starten (liest CLAUDE.md automatisch)
/mcp                   # Atlassian-MCP authentifizieren (b-f-u.atlassian.net)
```

## 1. Backlog nach Jira übertragen
```
/jira-sync
```
Legt 12 Epics + 58 Stories im Jira-Projekt „DRN“ (Schlüssel `KAN`) an, verknüpft Abhängigkeiten, schreibt `backlog/jira-map.json` und aktualisiert `docs/user-stories.md` mit den Jira-Keys (als PR).

## 2. Sprint 0 starten
In Jira einen Sprint „S0 23.11.–04.12.2026“ anlegen und die M0-Stories (Label `m0`) hineinziehen. Dann in Claude Code:
```
/story KAN-<Key von US-001>
```
Empfohlene Reihenfolge Sprint 0: US-007 → US-001 → US-005 → US-002 → US-006 → US-008 → US-003 → US-004 (Spike auf der GTX 1080).

## Start-Prompt (falls du lieber frei formulierst)
> Lies CLAUDE.md, docs/architektur.md (inkl. ADR-0010 zur GTX 1080) und docs/projektmanagement.md. Führe dann `/jira-sync` aus. Danach beginne Sprint 0 mit US-001 gemäß dem Ablauf in /story: Plan zuerst als Jira-Kommentar, auf meine Freigabe warten.

## Was Claude Code nicht darf (technisch abgesichert)
- Befehle an echte Drohnen (Hook `.claude/hooks/block_real_flight.py`)
- Force-Push, Löschen von `workspace/`
- Stories auf *Done* setzen (macht der PO nach dem Merge)
