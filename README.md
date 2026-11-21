# drone-scan (Arbeitstitel)

Open-Source-Pipeline: Drohnen- oder Smartphone-Video → 3D-Modell von Innenräumen (Gaussian Splatting) → **Gegenstandssuche** und **virtuelle Begehung**. Läuft lokal auf einem Laptop mit RTX 2060 (6 GB), Bedienung per Android-Browser.

**Status:** Projektaufsetzung (Sprint 0). Noch kein lauffähiger Code. Der erste Schritt ist Epic EP-01 im Jira-Projekt „DRN“ (Schlüssel `KAN`).

## Dokumente

| Dokument | Inhalt |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Regeln und Arbeitsweise für Claude Code |
| [docs/architektur.md](docs/architektur.md) | Architektur nach arc42 |
| [docs/adr/](docs/adr/) | Architekturentscheidungen |
| [docs/projektmanagement.md](docs/projektmanagement.md) | Vorgehen, Meilensteine, Sprints, Jira, DoR/DoD |
| [docs/user-stories.md](docs/user-stories.md) | Backlog (generiert aus `backlog/backlog.yaml`) |
| [docs/uebergabe-claude-code.md](docs/uebergabe-claude-code.md) | Start mit Claude Code, Jira-Sync |
| [docs/drohnen-recherche.md](docs/drohnen-recherche.md) | Drohnenvergleich und Empfehlung |

## Loslegen mit Claude Code

Siehe [docs/uebergabe-claude-code.md](docs/uebergabe-claude-code.md): `claude` starten → `/mcp` → `/story KAN-22` (Jira-Backlog ist bereits angelegt: KAN-4 … KAN-73).

Backlog ändern: `backlog/backlog.yaml` bearbeiten → `python3 tools/backlog_export.py` (validiert und erzeugt Markdown + Jira-CSV) → `/jira-sync`.

## Lizenz

Copyright © 2026 Raoul Schönhof

drone-scan ist freie Software unter der **GNU Affero General Public License v3.0 oder später** (`AGPL-3.0-or-later`), siehe [LICENSE](LICENSE) und [ADR-0011](docs/adr/0011-projektlizenz-agpl.md). Wer eine veränderte Version weitergibt **oder als Netzdienst betreibt**, muss den Nutzern den Quellcode dieser Version zugänglich machen. Externe Beiträge werden nur mit Contributor License Agreement (CLA) angenommen. Für eine kommerzielle Lizenz außerhalb der AGPL wende dich an den Rechteinhaber.

Fremde Abhängigkeiten behalten ihre eigenen Lizenzen (Lizenzpolitik: [ADR-0007](docs/adr/0007-lizenzpolitik.md)).
