# CLAUDE.md – drone-scan

Lokale Open-Source-Pipeline: **Drohnen- oder Smartphone-Video → 3D-Modell eines Innenraums** (Gaussian Splatting). Darauf bauen auf: **Gegenstandssuche** („Wo ist mein Schlüssel?“) und **virtuelle Begehung** (Verkaufsgespräche). Annahme: Während des Flugs sind keine Personen oder Tiere im Gebäude.

Details: `docs/architektur.md` (arc42) · `docs/projektmanagement.md` · `docs/user-stories.md` · `docs/adr/` · Backlog in **Jira, Projekt „DRN“ (Schlüssel `KAN`)**.

## Harte Regeln (nie verletzen)

1. **Lizenzen (ADR-0007, ADR-0012):** Open Source, bevorzugt MIT/BSD/Apache-2.0. GPL/AGPL nur als separater Prozess oder Container (OpenMVS, ArduPilot), nie in `src/` importieren.
   **Non-commercial/research-only ist erlaubt** (Non-Profit, Images nur lokal), wird aber im Lizenz-Gate als `NC` markiert und muss vor jeder Weitergabe an Dritte oder kommerziellen Nutzung neu geprüft werden. Proprietäre NVIDIA-GPU-Laufzeit (CUDA, cuDNN, cuBLAS …) ist erlaubt. **Docker-Images nur lokal bauen, nie in eine Registry pushen.** Das Repo ist öffentlich: **NC-Code und NC-Gewichte nie ins Repo** (nicht vendoren, kopieren, als Submodule einbinden). Nur lokal aus Upstream bauen (Dockerfile) bzw. per `drone-scan models pull` laden.
   **Kein Freibrief für Stack-Wechsel:** gsplat bleibt; Inria-3DGS, DUSt3R/MASt3R oder SuperPoint statt der etablierten Komponenten nur mit ADR und PO-Freigabe. Bei Modellen zählt die Lizenz der **Gewichte**. Jede neue Abhängigkeit oder jedes Modell → `licenses/allowlist.yaml` bzw. `models.lock` prüfen; unklare Lizenzen bleiben verboten. Im Zweifel Subagent `license-auditor` fragen und den PO informieren.
2. **VRAM (alle Werte in GiB):** Zielhardware RTX 2060, **6 GiB**. Budget pro Prozess: **≤ 5,5 GiB gesamt** (inkl. CUDA-Kontext). Jeder GPU-Code braucht ein Profil `low_vram_6gb` und muss mit `DRONESCAN_VRAM_LIMIT_GB=5.0` laufen (PyTorch-Allocator-Grenze, erzwungen über `torch.cuda.set_per_process_memory_fraction`; der CUDA-Kontext kommt hinzu). GPU-Jobs laufen nie parallel.
3. **Offline & Datenschutz:** Zur Laufzeit keine Netzwerkzugriffe. Einzige Ausnahme ist `drone-scan models pull` (einmaliger, explizit gestarteter Modell-Download). Keine echten Wohnungsaufnahmen oder Gesichter in Git, Tests oder Logs. Testdaten nur aus `data/samples/` (eigene, freigegebene Aufnahmen).
4. **Flugsicherheit:** Du sendest **niemals** Befehle an echte Drohnen. Flugcode wird ausschließlich gegen ArduPilot-SITL getestet. Echte Flüge startet nur der Mensch im eigenen Terminal (`--real-flight` plus Bestätigung). Ein Hook blockiert solche Aufrufe.
5. **Keine großen Dateien in Git:** Videos, Frames und Modelle liegen in `workspace/` bzw. `data/` (gitignored). Beispiel-Clips über 20 MB nur per Git LFS.

## Hardware (ADR-0010)

- **Du läufst auf dem Entwicklungsrechner mit GTX 1060 6GB** (Pascal, `sm_61`, steuert auch den Bildschirm, ca. 1 GiB belegt). Ziel der Nutzer: RTX 2060 (6 GiB, `sm_75`).
- **Die RTX 2070 SUPER im selben Rechner ist für ein anderes Projekt reserviert:** Nutzung nur nach vorheriger Zustimmung des POs für den konkreten Lauf **und** nur, wenn sie gerade frei ist (`nvidia-smi`). GPU-Läufe immer auf die GTX 1060 festlegen (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`; `make test-gpu` macht das).
- PyTorch nur **< 2.15 mit CUDA 12.6** (`https://download.pytorch.org/whl/cu126`). Neuere CUDA-Builds unterstützen Pascal nicht. Nie auf cu128/cu13x „upgraden“.
- CUDA-Erweiterungen/COLMAP immer für `6.1;7.5` bauen. Auf Pascal `fp32` statt fp16.
- GPU-Tests immer mit `DRONESCAN_VRAM_LIMIT_GB=5.0`. Laufzeiten auf der 1060 sind nicht die Zielwerte.

## Stack & Befehle

Python 3.12 + **uv** · FastAPI · Pydantic · Typer · pytest · ruff · mypy · TypeScript + Vite (Viewer) · Docker Compose + NVIDIA Container Toolkit.

```bash
make setup        # uv sync, pnpm install, pre-commit, docker images bauen
make check        # ruff + mypy + pytest -m "not gpu and not bench" + Lizenz-Gate  ← vor jedem PR
make test-gpu     # pytest -m gpu auf der GTX 1060, Allocator-Limit 5.0 GiB
make smoke        # Pipeline auf data/samples/kurz.mp4 (< 2 min)
make bench        # Benchmarks auf Referenzräumen (langsam, nicht in CI)
make viewer-dev   # Vite-Dev-Server für viewer/
docker compose up # server + worker lokal
drone-scan run --scan workspace/<id> --profile low_vram_6gb   # komplette Pipeline
```

> Status: Das Repo ist im Aufbau. Diese Befehle werden in Epic EP-01 (Stories „Repo-Grundgerüst“, „CI“, „Docker“) genau so angelegt. Gibt es einen Befehl noch nicht, lege ihn passend an statt Alternativen zu erfinden.

## Struktur

```text
src/dronescan/{common,capture,ingest,sfm,recon,search,privacy,export,server}/   # Python-Pakete
flight/            # ArduPilot/MAVLink, SITL-Welten (ab M5)
viewer/            # PWA (TypeScript)
docker/            # Dockerfiles (worker, colmap, sitl)
licenses/          # allowlist.yaml, denylist.yaml
models.lock        # Modellgewichte: name, version, sha256, lizenz, quelle
data/samples/      # kleine Testclips (LFS)
workspace/         # Scans (gitignored)
docs/              # arc42, ADRs, Fehlercodes
backlog/           # backlog.yaml (Quelle für Jira-Import)
```

## Stufenvertrag (jede Pipeline-Stufe)

- Idempotenter CLI-Befehl `drone-scan <stufe> --scan <dir> [--profile] [--force]`.
- Liest nur definierte Eingabeartefakte und schreibt nach `<scan>/<stufe>/` plus `manifest.json` (Input-Hashes, Parameter, Tool-Versionen, `peak_vram_mb`, `warnings`, `hints`).
- Exitcodes: `0` ok · `10` Eingabe fehlt · `20` Qualität unzureichend (mit `hints` in Nutzersprache) · `30` OOM · `1` sonst.
- Ist das Manifest gültig, wird die Stufe übersprungen (Cache). `--force` erzwingt den Neulauf.

## Code-Konventionen

- Code, Bezeichner, Commits und Kommentare auf **Englisch**. Doku und UI-Texte auf **Deutsch** (UI i18n-fähig).
- Typannotationen überall, `mypy --strict` für `src/`. Datenobjekte als Pydantic-Modelle, kein globaler Zustand.
- Konfiguration nur über `common.config` (Pydantic-Settings + YAML-Profile), keine Magic Numbers in Stufen.
- Logging über `structlog`, strukturiert. Keine Bildinhalte oder Pfade mit Personennamen loggen.
- Fehlermeldungen an Nutzer: Ursache plus konkrete Abhilfe. Neue Codes in `docs/fehlercodes.md` eintragen.
- Schwere native Tools (COLMAP, OpenMVS) werden über `subprocess` bzw. Container aufgerufen und in einer Adapterklasse gekapselt.

## Tests

- Akzeptanzkriterien der Story → zuerst Tests schreiben (pytest, Given/When/Then im Docstring).
- Marker: `gpu` (braucht CUDA), `bench` (langsam), `sitl` (Flugsimulation). CPU-Tests müssen ohne GPU laufen.
- Golden-Data: `data/samples/kurz.mp4` plus Erwartungswerte in `tests/golden/`. Schwellwerte statt exakter Pixelwerte.
- Viewer: Vitest (Unit) und Playwright (E2E).

## Arbeitsweise mit Jira & Stories

- Jira-Zugriff über den Atlassian-MCP-Server (`.mcp.json`). **Starte jede Story mit `/story KAN-<nr>`.**
- Ablauf: Story lesen → Plan Mode → Plan als Jira-Kommentar → Freigabe abwarten → Branch `feat/KAN-<nr>-<slug>` → Tests → Implementierung → `make check` → Review durch Subagent `reviewer` → PR „KAN-<nr>: <Titel>“ → Jira-Status *In Review* (in der deutschen Jira-UI: *Wird überprüft*) + PR-Link → Merge.
- Commits: Conventional Commits mit Key, z. B. `feat(sfm): add AprilTag scale alignment (KAN-27)`.
- **Self-Merge:** PRs selbst mergen, wenn alles davon zutrifft: `make check` grün, `reviewer` ohne BLOCKER, konfliktfrei, keine Änderung an `flight/`, `CLAUDE.md`, `.claude/hooks/`, `.claude/settings.json`, `.claude/agents/`, `.claude/skills/` oder `licenses/` (Details: `docs/projektmanagement.md` § 6). Sonst mergt der PO.
- Jira-Status *Done* (*Fertig*) setzt nach dem Merge die **PM-Session** (vom PO delegiert), nicht die Entwickler-Session.
- Nicht ohne Rückfrage: Stories löschen, Prioritäten ändern.
- **PM-Session:** Eine separate Claude-Session arbeitet als Projektmanager. Sie überwacht den Fortschritt, pflegt Jira/GitHub und Sprints und bündelt PO-Fragen. Ihre Anweisungen kommen als Nachricht in die Session oder als Story-Kommentar mit Präfix `[PM]`. Vor jeder Story die `[PM]`-Kommentare lesen.
- Architekturrelevante Entscheidung? → ADR-Entwurf mit `/adr` und PO fragen, statt still zu entscheiden.
- Unklare Akzeptanzkriterien → nachfragen statt raten.

## Subagents & Skills

- Agents (`.claude/agents/`): `reviewer`, `license-auditor`, `vram-guard`, `flight-safety`.
- Skills (`.claude/skills/`): `/story`, `/adr`, `/retro`, `/jira-sync` (Backlog → Jira). Übergabe-Anleitung: `docs/uebergabe-claude-code.md`.

## Domänenwissen (kurz)

- Gute Aufnahme: Licht an, ≤ 0,5 m/s, 1–1,5 m Wandabstand, Türdurchgänge doppelt, 2–4 AprilTags (tag36h11, bekannte Kantenlänge) pro Wohnung.
- Typische Fehlerbilder: weiße Wände, Spiegel und Glas → wenig Features. Bewegungsunschärfe. Mehrere Akkuflüge → Merge über Türbereiche/Tags.
- Splats sind für Optik da, Messungen kommen aus SfM + AprilTags (Maßstab) bzw. dem Mesh.

## Lessons Learned

<!-- Wird nach jeder Retro ergänzt (/retro). Format: Datum – Regel – Anlass (KAN-Key) -->
