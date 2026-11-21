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

## Voraussetzungen (GPU-Container)

Die Pipeline läuft in zwei Containern: `worker` (PyTorch 2.14 + CUDA 12.6 + gsplat) und `colmap` (COLMAP 4.2.1). Beide sind für Pascal (`sm_61`, z. B. GTX 1060) und Turing (`sm_75`, z. B. RTX 2060) gebaut.

1. **Ubuntu** 22.04/24.04 mit **NVIDIA-Treiber ≥ 560** (unterstützt CUDA 12.6). Prüfen mit `nvidia-smi`.
2. **Docker Engine** mit **Compose-Plugin** (`docker compose version`).
3. **NVIDIA Container Toolkit**, eingerichtet mit `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. Prüfen mit `docker info | grep -i nvidia`.

**Prüfbefehl:**

```bash
docker compose build                 # einmalig, dauert eine Weile (COLMAP und gsplat werden kompiliert)
make docker-check                    # GPU im Container sichtbar? erwartet z. B. "(6, 1)" (GTX 1060) oder "(7, 5)" (RTX 2060)
```

Standardmäßig nutzen die Container die GPU mit Index 0. Bei mehreren GPUs lässt sie sich mit `DRONESCAN_GPU_DEVICE=<index> docker compose …` wählen. Auf dem Entwicklungsrechner ist die RTX 2070 SUPER reserviert und wird nur mit Zustimmung des POs genutzt.

## Loslegen mit Claude Code

Siehe [docs/uebergabe-claude-code.md](docs/uebergabe-claude-code.md): `claude` starten → `/mcp` → `/story KAN-22` (Jira-Backlog ist bereits angelegt: KAN-4 … KAN-73).

Backlog ändern: `backlog/backlog.yaml` bearbeiten → `python3 tools/backlog_export.py` (validiert und erzeugt Markdown + Jira-CSV) → `/jira-sync`.

## Lizenz

Copyright © 2026 Raoul Schönhof

drone-scan ist freie Software unter der **GNU Affero General Public License v3.0 oder später** (`AGPL-3.0-or-later`), siehe [LICENSE](LICENSE) und [ADR-0011](docs/adr/0011-projektlizenz-agpl.md). Wer eine veränderte Version weitergibt **oder als Netzdienst betreibt**, muss den Nutzern den Quellcode dieser Version zugänglich machen. Externe Beiträge werden nur mit Contributor License Agreement (CLA) angenommen; bis ein CLA vorliegt, werden keine externen Beiträge angenommen. Für eine kommerzielle Lizenz außerhalb der AGPL wende dich an den Rechteinhaber.

Fremde Abhängigkeiten behalten ihre eigenen Lizenzen (Lizenzpolitik: [ADR-0007](docs/adr/0007-lizenzpolitik.md)).
