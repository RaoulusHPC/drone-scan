# User Stories – drone-scan

> Generiert aus `backlog/backlog.yaml` mit `tools/backlog_export.py` – **nicht von Hand bearbeiten**.
> Führendes System für Status und Sprints ist **Jira (Projekt „DRN“ (Schlüssel `KAN`))**. Priorität: MoSCoW → Jira (Must=Highest, Should=High, Could=Medium).

**Umfang:** 12 Epics · 58 Stories/Tasks · 277 Story Points

## Personas

| ID | Persona | Beschreibung |
|---|---|---|
| P1 | Suchende (Privatnutzer) | Möchte verlegte Gegenstände schnell finden, kein Technikwissen. |
| P2 | Präsentierende (Makler/Verkäufer) | Will Immobilien virtuell zeigen, braucht ansprechende, datenschutzkonforme Ergebnisse. |
| P3 | Betrachtende (Interessent) | Öffnet eine Begehung per Link auf Smartphone/Laptop, ohne Installation. |
| P4 | Pilot | Fliegt die Drohne (oft identisch mit P1/P2), will sicher und einfach gute Aufnahmen machen. |
| P5 | Entwickler (PO + Claude Code) | Braucht reproduzierbare, testbare, lizenzsaubere Grundlagen. |

## Übersicht nach Meilenstein

| Meilenstein | Stories | SP | Must | Should | Could |
|---|---|---|---|---|---|
| M0 Fundament & Spikes | 8 | 27 | 8 | 0 | 0 |
| M1 MVP „Video → Begehung“ | 16 | 72 | 12 | 4 | 0 |
| M2 Gegenstandssuche | 6 | 28 | 4 | 1 | 1 |
| M3 Drohnen-Erfassung | 9 | 41 | 3 | 3 | 3 |
| M4 Präsentationsreife | 12 | 56 | 3 | 5 | 4 |
| M5 Assistierter Flug | 7 | 53 | 1 | 3 | 3 |

## Übersicht nach Epic

| Epic | Titel | Stories | SP |
|---|---|---|---|
| EP-01 | Fundament & Entwicklungsumgebung | 7 | 24 |
| EP-02 | Erfassung & Import | 5 | 17 |
| EP-03 | Frame-Aufbereitung & Qualitätsprüfung | 3 | 11 |
| EP-04 | Rekonstruktion: Kameraposen, Maßstab, Zusammenführung | 5 | 31 |
| EP-05 | Virtuelle Begehung (Splat & Viewer) | 5 | 29 |
| EP-06 | Gegenstandssuche | 6 | 31 |
| EP-07 | Präsentation & Export für Verkaufsgespräche | 4 | 23 |
| EP-08 | Datenschutz & Datenhaltung | 4 | 14 |
| EP-09 | Web-App, Jobs & Bedienung | 5 | 21 |
| EP-10 | Flugunterstützung & Autonomie | 6 | 45 |
| EP-11 | Qualitätssicherung & Compliance | 5 | 16 |
| EP-12 | Hardware-Evaluierung & Erfassungsleitfaden | 3 | 15 |

## EP-01 – Fundament & Entwicklungsumgebung

### US-001: Repo-Grundgerüst mit uv, Paketstruktur und Makefile

`Task` · **Must** · 3 SP · M0 Fundament & Spikes · Labels: infra, claude-code · Abhängig von: –

Als **Entwickler (PO + Claude Code)** möchte ich ein Repository mit der in CLAUDE.md beschriebenen Struktur, uv-Projekt und Make-Zielen, damit Claude Code und ich von Anfang an mit identischen Befehlen arbeiten.

**Akzeptanzkriterien**

- [ ] Gegeben ein frischer Clone, wenn `make setup` läuft, dann sind Python-Umgebung (3.12, uv.lock) und pre-commit eingerichtet.
- [ ] `make check` läuft fehlerfrei (ruff, mypy --strict, pytest mit einem Platzhaltertest).
- [ ] Paketstruktur `src/dronescan/{common,capture,ingest,sfm,recon,search,privacy,export,server}` existiert mit Typer-CLI `drone-scan --help`.
- [ ] `.gitignore` schließt `workspace/`, `data/` (außer `data/samples/`) und Modellgewichte aus.

### US-002: CI-Pipeline (CPU) mit Lint, Typen, Tests

`Task` · **Must** · 2 SP · M0 Fundament & Spikes · Labels: infra, claude-code · Abhängig von: US-001

Als **Entwickler (PO + Claude Code)** möchte ich dass jeder Push automatisch geprüft wird, damit Fehler und Regelverstöße früh auffallen.

**Akzeptanzkriterien**

- [ ] CI (GitHub Actions oder Bitbucket Pipelines) führt `make check` bei jedem Push und PR aus.
- [ ] Ein fehlschlagender Test oder Typfehler blockiert den Merge (Branch-Schutz dokumentiert).
- [ ] Laufzeit der CI < 10 min.

### US-003: Docker-Images für Worker (CUDA) und COLMAP

`Task` · **Must** · 5 SP · M0 Fundament & Spikes · Labels: infra · Abhängig von: US-001

Als **Entwickler (PO + Claude Code)** möchte ich reproduzierbare Container für GPU-Pipeline und COLMAP, damit die Software auf jedem Ubuntu-Laptop mit NVIDIA-GPU gleich läuft.

**Akzeptanzkriterien**

- [ ] `docker compose build` erzeugt Images `worker` (CUDA 12.6, PyTorch < 2.15 cu126, gsplat gebaut für sm_61 und sm_75) und `colmap` (CUDA-Architekturen 61;75).
- [ ] `docker compose run worker python -c 'import torch; print(torch.cuda.get_device_capability())'` läuft auf der GTX 1080 (6, 1) ohne Kompatibilitätswarnung.
- [ ] README beschreibt Voraussetzungen (Treiber, NVIDIA Container Toolkit) inkl. Prüfbefehl.

### US-004: Spike: COLMAP + gsplat auf GTX 1080 (6-GB-Grenze) mit Smartphone-Video

`Task` · **Must** · 5 SP · M0 Fundament & Spikes · Labels: spike, sfm, recon · Abhängig von: US-003

Als **Entwickler (PO + Claude Code)** möchte ich nachweisen, dass ein Raum aus einem Handyvideo auf der GTX 1080 im 6-GB-Budget der RTX 2060 rekonstruierbar ist, damit das Hauptrisiko (R-01, R-03) vor der Umsetzung bewertet ist.

**Akzeptanzkriterien**

- [ ] 3 Referenzräume (hell/texturiert, weiße Wände, Küche/Bad mit Spiegel) mit dem Smartphone aufgenommen; Verarbeitung auf der GTX 1080 mit 5,5-GB-Grenze.
- [ ] gsplat-Training auf Pascal (sm_61, fp32) nachgewiesen oder Fallback dokumentiert (ADR-0010, Risiko R-13).
- [ ] Je Raum dokumentiert: registrierte Frames (%), Laufzeit je Stufe, Peak-VRAM, subjektive Qualität (Screenshots).
- [ ] Ergebnisbericht `docs/spikes/spike-01.md` mit Empfehlung für Profilparameter `low_vram_6gb`.

### US-005: Konfigurationssystem mit Profilen und VRAM-Limit

`Task` · **Must** · 3 SP · M0 Fundament & Spikes · Labels: infra, claude-code · Abhängig von: US-001

Als **Entwickler (PO + Claude Code)** möchte ich zentrale Konfiguration mit Profilen `low_vram_6gb` und `default_8gb`, damit alle Stufen dieselben Grenzwerte nutzen und der Zielrechner simulierbar ist.

**Akzeptanzkriterien**

- [ ] Pydantic-Settings laden YAML-Profile; Werte per Umgebungsvariable überschreibbar.
- [ ] `DRONESCAN_VRAM_LIMIT_GB` setzt `torch.cuda.set_per_process_memory_fraction` beim Start jeder GPU-Stufe.
- [ ] Unit-Tests für Laden, Überschreiben und ungültige Werte.

### US-006: Stufen-Framework: Manifest, Caching, Exitcodes

`Task` · **Must** · 5 SP · M0 Fundament & Spikes · Labels: infra, claude-code · Abhängig von: US-005

Als **Entwickler (PO + Claude Code)** möchte ich eine gemeinsame Basisklasse für Pipeline-Stufen, damit jede Stufe idempotent, nachvollziehbar und cachebar ist.

**Akzeptanzkriterien**

- [ ] Basisklasse schreibt `manifest.json` (Input-Hashes, Parameter, Tool-Versionen, Zeit, peak_vram_mb, warnings, hints).
- [ ] Bei unveränderten Eingaben und Parametern wird die Stufe übersprungen; `--force` erzwingt Neulauf.
- [ ] Exitcodes 0/10/20/30/1 sind implementiert und getestet; OOM wird als 30 erkannt.

### US-007: Jira- und Claude-Code-Workflow einrichten

`Task` · **Must** · 1 SP · M0 Fundament & Spikes · Labels: infra · Abhängig von: –

Als **Entwickler (PO + Claude Code)** möchte ich Atlassian-MCP, Skills und Subagents funktionsfähig, damit Stories mit `/story KAN-xx` durchgängig bearbeitet werden.

**Akzeptanzkriterien**

- [ ] `.mcp.json` verbunden, `/mcp` authentifiziert; Claude Code kann eine KAN-Story lesen und kommentieren.
- [ ] Hook `block_real_flight.py` blockiert Testbefehl mit `--real-flight` nachweislich (Exit 2).
- [ ] Board-Spalten und Labels gemäß docs/projektmanagement.md angelegt.


## EP-02 – Erfassung & Import

### US-013: Scan anlegen und Videos importieren (CLI)

`Story` · **Must** · 3 SP · M1 MVP „Video → Begehung“ · Labels: capture, claude-code · Abhängig von: US-006

Als **Pilot** möchte ich einen Scan anlegen und ein oder mehrere Videos (je Akkuflug) hinzufügen, damit alle Aufnahmen eines Gebäudes zusammen verarbeitet werden.

**Akzeptanzkriterien**

- [ ] `drone-scan scan create <name>` legt `workspace/<id>/scan.yaml` an.
- [ ] `drone-scan flight add <scan> <video>` kopiert das Video, liest Metadaten (ffprobe: Auflösung, fps, Dauer, Codec, Rotation) und vergibt eine Flug-ID.
- [ ] Nicht unterstützte Formate oder defekte Dateien → Exit 10 mit verständlicher Meldung.

### US-014: Kameraprofil und Intrinsik-Schätzung

`Story` · **Should** · 3 SP · M1 MVP „Video → Begehung“ · Labels: capture, sfm · Abhängig von: US-013

Als **Pilot** möchte ich dass das System meine Kamera (Drohne/Handy/Action-Cam) passend behandelt, damit die Rekonstruktion auch bei Weitwinkel/Fisheye gelingt.

**Akzeptanzkriterien**

- [ ] Kameraprofile (YAML) für Smartphone, DJI Neo 2, Tello, RunCam Thumb 2 mit Kameramodell (PINHOLE/OPENCV/OPENCV_FISHEYE).
- [ ] Ohne Profil: automatische Schätzung durch COLMAP, Ergebnis wird im Scan gespeichert.
- [ ] Optionaler Kalibrierbefehl mit Schachbrett-/ChArUco-Video.

### US-015: Upload über Smartphone-Browser

`Story` · **Must** · 5 SP · M1 MVP „Video → Begehung“ · Labels: server, viewer · Abhängig von: US-013, US-037

Als **Pilot** möchte ich Videos direkt vom Android-Smartphone hochladen, damit ich keinen Kartenleser und keine Kommandozeile brauche.

**Akzeptanzkriterien**

- [ ] Upload großer Dateien (≥ 4 GB) per Chunked Upload mit Fortschrittsanzeige und Fortsetzen nach Abbruch.
- [ ] Nach dem Upload erscheint der Flug im Scan mit Metadaten.

### US-016: Telemetrie-Import (optional)

`Story` · **Could** · 3 SP · M3 Drohnen-Erfassung · Labels: capture · Abhängig von: US-013

Als **Pilot** möchte ich vorhandene Flugtelemetrie (CSV, ArduPilot-Log, SRT-Untertitel) mit dem Video verknüpfen, damit Höhe/Lage die Rekonstruktion stützen.

**Akzeptanzkriterien**

- [ ] Import von DJI-SRT (falls vorhanden) und ArduPilot-DataFlash-Log; Zeitsynchronisation über Startzeit/Offset.
- [ ] Telemetrie ist optional; die Pipeline funktioniert ohne.

### US-017: Tello-Adapter (Aufnahme per Python)

`Story` · **Could** · 3 SP · M3 Drohnen-Erfassung · Labels: capture, hardware · Abhängig von: US-013

Als **Pilot** möchte ich mit einer Tello per Laptop ein Video aufzeichnen, damit ich günstig mit einer programmierbaren Drohne testen kann.

**Akzeptanzkriterien**

- [ ] Adapter auf Basis djitellopy speichert Videostream lokal als MP4 mit Zeitstempeln.
- [ ] Steuerung bleibt manuell (App/Controller); der Adapter sendet keine Flugbefehle.
- [ ] Tests mit Mock-Stream (keine echte Drohne in der CI).


## EP-03 – Frame-Aufbereitung & Qualitätsprüfung

### US-018: Frame-Extraktion mit Schärfe- und Duplikatfilter

`Story` · **Must** · 5 SP · M1 MVP „Video → Begehung“ · Labels: ingest, claude-code · Abhängig von: US-013

Als **Entwickler (PO + Claude Code)** möchte ich aus Videos automatisch scharfe, nicht redundante Keyframes, damit SfM schneller und robuster läuft.

**Akzeptanzkriterien**

- [ ] Dekodierung via ffmpeg (Rotation korrekt); Schärfe per Laplace-Varianz, adaptive Schwelle je Video.
- [ ] Duplikatfilter über Bildähnlichkeit/optischen Fluss; Zielabstand konfigurierbar (z. B. 2–4 Keyframes/s).
- [ ] `frames.parquet` enthält je Frame Zeitstempel, Schärfe, Auswahlgrund.

### US-019: Qualitätsreport mit Handlungsempfehlungen

`Story` · **Must** · 3 SP · M1 MVP „Video → Begehung“ · Labels: ingest, viewer · Abhängig von: US-018

Als **Pilot** möchte ich nach dem Upload erfahren, ob meine Aufnahme gut genug ist, damit ich bei Bedarf sofort nachfliegen kann.

**Akzeptanzkriterien**

- [ ] `quality.json` bewertet Schärfe, Belichtung, Bewegungsgeschwindigkeit, Abdeckung (grob).
- [ ] Bei ungenügender Qualität: Exit 20 und ≥ 1 konkreter Hinweis (z. B. „Licht einschalten“, „langsamer fliegen“) (Q-9).
- [ ] Report im Viewer verständlich dargestellt (Ampel).

### US-020: Belichtungs- und Weißabgleichsnormalisierung

`Story` · **Could** · 3 SP · M4 Präsentationsreife · Labels: ingest · Abhängig von: US-018

Als **Präsentierende (Makler/Verkäufer)** möchte ich gleichmäßige Farben in der Begehung, damit die Präsentation professionell wirkt.

**Akzeptanzkriterien**

- [ ] Optionale Normalisierung von Belichtung/Weißabgleich zwischen Flügen; vorher/nachher-Vergleich im Bericht.
- [ ] Kein negativer Einfluss auf die Registrierungsquote (Benchmark).


## EP-04 – Rekonstruktion: Kameraposen, Maßstab, Zusammenführung

### US-021: Kameraposen mit COLMAP (Stufe `sfm`)

`Story` · **Must** · 5 SP · M1 MVP „Video → Begehung“ · Labels: sfm, claude-code · Abhängig von: US-018, US-003

Als **Entwickler (PO + Claude Code)** möchte ich aus Keyframes Kameraposen und eine dünne Punktwolke, damit Splat-Training und Suche darauf aufbauen können.

**Akzeptanzkriterien**

- [ ] COLMAP-Adapter (Container) mit Feature-Extraktion, sequenziellem Matching + Loop-Detection, Mapper.
- [ ] Registrierungsquote und mittlerer Reprojektionsfehler im Manifest; < 80 % registriert → Exit 20 mit Hinweisen.
- [ ] Läuft im Profil `low_vram_6gb` (Bildkante begrenzt).

### US-022: AprilTag-Maßstab und Schwerkraftausrichtung

`Story` · **Must** · 5 SP · M3 Drohnen-Erfassung · Labels: sfm · Abhängig von: US-021

Als **Präsentierende (Makler/Verkäufer)** möchte ich dass das Modell in echten Metern vorliegt und gerade steht, damit Maße und Grundriss stimmen.

**Akzeptanzkriterien**

- [ ] Erkennung von AprilTags (tag36h11) mit konfigurierbarer Kantenlänge; Triangulation über mehrere Frames.
- [ ] Maßstabsfehler ≤ 3 % bei 2 Tags im Raum (Q-4), Boden horizontal ausgerichtet.
- [ ] Druckvorlage (PDF) für Tags im Repo.

### US-023: Mehrere Flüge zu einem Modell zusammenführen

`Story` · **Must** · 8 SP · M3 Drohnen-Erfassung · Labels: sfm · Abhängig von: US-021, US-022

Als **Pilot** möchte ich dass mehrere Akkuflüge automatisch zu einem Gebäude zusammengesetzt werden, damit auch größere Wohnungen erfasst werden können.

**Akzeptanzkriterien**

- [ ] Flüge werden gemeinsam bzw. nacheinander registriert; gemeinsame Bereiche/Tags verbinden die Teilmodelle.
- [ ] Nicht verbindbare Flüge werden gemeldet mit Hinweis „Brückenflug durch Tür X“.
- [ ] Test mit 3 Flügen einer Referenzwohnung.

### US-024: Alternative Feature-Matcher für texturarme Räume

`Story` · **Should** · 5 SP · M3 Drohnen-Erfassung · Labels: sfm, compliance · Abhängig von: US-021, US-012

Als **Entwickler (PO + Claude Code)** möchte ich einen lizenzsauberen Fallback-Matcher (z. B. ALIKED + LightGlue), damit weiße Wände und schwierige Räume häufiger gelingen.

**Akzeptanzkriterien**

- [ ] Lizenzprüfung durch `license-auditor` dokumentiert (keine SuperPoint-Gewichte).
- [ ] Automatischer Fallback, wenn Registrierungsquote < 80 %; Vergleich im Benchmark.

### US-025: Raumaufteilung (Chunking) für große Gebäude

`Story` · **Should** · 8 SP · M3 Drohnen-Erfassung · Labels: recon, sfm · Abhängig von: US-026, US-023

Als **Präsentierende (Makler/Verkäufer)** möchte ich auch Häuser mit vielen Räumen auf 6 GB verarbeiten, damit die Hardwaregrenze kein Ausschlusskriterium ist.

**Akzeptanzkriterien**

- [ ] Aufteilung der Szene in räumliche Abschnitte mit Überlappung; Training je Abschnitt im VRAM-Budget.
- [ ] Zusammengesetzte Szene im Viewer ohne sichtbare Nähte an Abschnittsgrenzen (subjektiver Test + Screenshot).


## EP-05 – Virtuelle Begehung (Splat & Viewer)

### US-026: Gaussian-Splatting-Training im 6-GB-Profil (Stufe `recon.splat`)

`Story` · **Must** · 8 SP · M1 MVP „Video → Begehung“ · Labels: recon, claude-code · Abhängig von: US-021, US-005

Als **Präsentierende (Makler/Verkäufer)** möchte ich aus den Kameraposen eine fotorealistische 3D-Szene, damit Interessenten die Räume virtuell begehen können.

**Akzeptanzkriterien**

- [ ] Training mit gsplat; Profil `low_vram_6gb` hält Peak ≤ 5,5 GB (Q-2).
- [ ] Validierungsframes (z. B. jedes 8.) → PSNR/SSIM in `metrics.json`.
- [ ] OOM → automatische Wiederholung mit kleinerem Profil (einmal).
- [ ] Referenzwohnung 60 m² in ≤ 60 min Gesamtlaufzeit (Q-1, Messung dokumentiert).

### US-027: Szenenkomprimierung für Smartphone-Wiedergabe

`Story` · **Must** · 5 SP · M1 MVP „Video → Begehung“ · Labels: recon, viewer · Abhängig von: US-026

Als **Betrachtende (Interessent)** möchte ich dass die Begehung auf meinem Android-Handy flüssig läuft, damit ich sie ohne leistungsstarken Rechner ansehen kann.

**Akzeptanzkriterien**

- [ ] Export als komprimierte Szene (z. B. SPZ) mit Begrenzung auf ≤ 1,5 M Gaussians (Pruning).
- [ ] ≥ 30 fps auf einem Mittelklasse-Android-Gerät (Q-7), gemessen und dokumentiert.

### US-028: PWA-Viewer: Begehung mit Touch-Navigation

`Story` · **Must** · 8 SP · M1 MVP „Video → Begehung“ · Labels: viewer, claude-code · Abhängig von: US-027

Als **Betrachtende (Interessent)** möchte ich die Szene im Browser mit Touch-Gesten erkunden, damit ich mich intuitiv durch die Räume bewegen kann.

**Akzeptanzkriterien**

- [ ] Splat-Rendering (three.js-basiert) mit Modi Orbit und Begehung (Augenhöhe, Bodenkollision grob).
- [ ] Touch: Wischen = Blick, Zwei-Finger = Bewegung; Desktop: WASD + Maus.
- [ ] Ladeanzeige und Fallback-Meldung bei fehlender WebGL2-Unterstützung.

### US-029: Raumliste und Sprungpunkte

`Story` · **Should** · 3 SP · M4 Präsentationsreife · Labels: viewer · Abhängig von: US-028

Als **Betrachtende (Interessent)** möchte ich direkt zu „Küche“ oder „Bad“ springen, damit ich mich schnell orientiere.

**Akzeptanzkriterien**

- [ ] Präsentierende können Sprungpunkte (Position + Blickrichtung + Name) setzen.
- [ ] Raumliste im Viewer; Antippen fliegt animiert zum Punkt.

### US-030: Mesh-Erzeugung (optional) für Messungen

`Story` · **Could** · 5 SP · M4 Präsentationsreife · Labels: recon · Abhängig von: US-026, US-022

Als **Präsentierende (Makler/Verkäufer)** möchte ich ein grobes Mesh der Räume, damit Maße und Grundriss abgeleitet werden können.

**Akzeptanzkriterien**

- [ ] Mesh aus gerenderten Tiefen (TSDF, Open3D) oder OpenMVS als separatem Container-Werkzeug.
- [ ] Export als glTF 2.0 (GLB); Laufzeit und VRAM dokumentiert.


## EP-06 – Gegenstandssuche

### US-031: Suchindex: Detektionen und Bild-Embeddings je Keyframe

`Story` · **Must** · 5 SP · M2 Gegenstandssuche · Labels: search, claude-code · Abhängig von: US-021, US-009

Als **Suchende (Privatnutzer)** möchte ich dass alle Aufnahmen nach dem Scan durchsuchbar sind, damit ich später schnell Gegenstände finden kann.

**Akzeptanzkriterien**

- [ ] OpenCLIP-Embeddings je Keyframe in FAISS-Index; Modell aus `models.lock` (Lizenz geprüft).
- [ ] Indexierung läuft im VRAM-Budget (Batchgröße aus Profil) und ist Teil der Pipeline.
- [ ] Index-Größe und Laufzeit im Manifest.

### US-032: Textsuche mit Open-Vocabulary-Detektion

`Story` · **Must** · 5 SP · M2 Gegenstandssuche · Labels: search · Abhängig von: US-031

Als **Suchende (Privatnutzer)** möchte ich per Freitext („Schlüsselbund“, „rote Brille“) suchen, damit ich nicht in Hunderten Bildern selbst suchen muss.

**Akzeptanzkriterien**

- [ ] Text → Top-k-Frames (FAISS) → Open-Vocabulary-Detektor (OWLv2 oder Grounding DINO) → Boxen mit Score.
- [ ] Antwort < 3 s nach Indexierung (Q-6) auf dem Zielrechner.
- [ ] Deutsche und englische Suchbegriffe (Übersetzungstabelle bzw. mehrsprachiges Modell).

### US-033: 3D-Lokalisierung der Treffer (Triangulation)

`Story` · **Must** · 5 SP · M2 Gegenstandssuche · Labels: search · Abhängig von: US-032, US-026

Als **Suchende (Privatnutzer)** möchte ich sehen, WO im Gebäude der Gegenstand liegt, damit ich direkt hingehen kann.

**Akzeptanzkriterien**

- [ ] Treffer aus ≥ 2 Frames werden trianguliert; Einzeltreffer über gerenderte Tiefe positioniert.
- [ ] Ergebnis: 3D-Position, Raum (falls bekannt), Konfidenz, 2–4 Belegbilder mit Box.
- [ ] Benchmark: Top-3-Trefferquote ≥ 70 % auf Testset mit 20 platzierten Gegenständen (Q-5).

### US-034: Suchergebnis im Viewer anzeigen

`Story` · **Must** · 5 SP · M2 Gegenstandssuche · Labels: viewer, search · Abhängig von: US-033, US-028

Als **Suchende (Privatnutzer)** möchte ich Treffer als Marker in der 3D-Szene und als Bildliste, damit ich das Ergebnis schnell überprüfen kann.

**Akzeptanzkriterien**

- [ ] Suchfeld im Viewer; Ergebnisliste mit Vorschaubildern und Konfidenz.
- [ ] Antippen fliegt zum 3D-Marker; Belegbild ist einblendbar.
- [ ] „Kein Treffer“ mit Tipp (andere Formulierung, Raum nachscannen).

### US-035: Präzise Segmentierung der Treffer (optional)

`Story` · **Could** · 3 SP · M2 Gegenstandssuche · Labels: search · Abhängig von: US-034

Als **Suchende (Privatnutzer)** möchte ich den gefundenen Gegenstand hervorgehoben sehen, damit ich ihn auf dem Bild sofort erkenne.

**Akzeptanzkriterien**

- [ ] SAM 2 (Apache-2.0) erzeugt Masken für Top-Treffer; VRAM im Budget.
- [ ] Maske im Belegbild und als Hervorhebung im Viewer.

### US-036: Vergleich zweier Scans (Was hat sich verändert?)

`Story` · **Could** · 8 SP · M4 Präsentationsreife · Labels: search, sfm · Abhängig von: US-033, US-022

Als **Suchende (Privatnutzer)** möchte ich einen neuen Scan mit einem älteren vergleichen, damit ich sehe, wo Gegenstände hinzugekommen oder verschwunden sind.

**Akzeptanzkriterien**

- [ ] Registrierung zweier Scans desselben Gebäudes (AprilTags/Features).
- [ ] Liste der Änderungsregionen mit Vorher/Nachher-Bildern.


## EP-07 – Präsentation & Export für Verkaufsgespräche

### US-042: Teilbares Offline-Viewer-Bundle

`Story` · **Must** · 5 SP · M4 Präsentationsreife · Labels: export, viewer · Abhängig von: US-027, US-046

Als **Präsentierende (Makler/Verkäufer)** möchte ich eine Begehung als Paket/Link an Interessenten weitergeben, damit sie ohne meinen Laptop angesehen werden kann.

**Akzeptanzkriterien**

- [ ] Export als statischer Ordner (HTML/JS + komprimierte Szene), lauffähig auf jedem statischen Webspace oder lokal.
- [ ] Größe ≤ 150 MB für 60 m²; Ladezeit auf 4G dokumentiert.
- [ ] Anonymisierung ist vor Export standardmäßig aktiv (US-046).

### US-043: Präsentationsmodus mit geführter Tour

`Story` · **Should** · 5 SP · M4 Präsentationsreife · Labels: viewer · Abhängig von: US-029

Als **Präsentierende (Makler/Verkäufer)** möchte ich eine Tour mit Stationen und Texten festlegen, damit ich im Verkaufsgespräch strukturiert führen kann.

**Akzeptanzkriterien**

- [ ] Stationen aus Sprungpunkten (US-029) mit Titel, Kurztext, Reihenfolge.
- [ ] Abspielmodus mit Weiter/Zurück; funktioniert auch im Export-Bundle.

### US-044: Grundriss-Skizze mit ungefähren Flächen

`Story` · **Could** · 8 SP · M4 Präsentationsreife · Labels: recon, export · Abhängig von: US-030, US-022

Als **Präsentierende (Makler/Verkäufer)** möchte ich einen groben 2D-Grundriss mit Flächenangaben, damit Interessenten die Raumaufteilung verstehen.

**Akzeptanzkriterien**

- [ ] Horizontalschnitt aus Mesh/Punktwolke → Polygon je Raum, SVG-Export.
- [ ] Flächenabweichung ≤ 10 % gegenüber Referenzmessung; deutlicher Hinweis „nicht maßstabsverbindlich“.

### US-045: Standardformate exportieren (glTF, PLY/SPZ, E57)

`Story` · **Should** · 5 SP · M4 Präsentationsreife · Labels: export · Abhängig von: US-026

Als **Präsentierende (Makler/Verkäufer)** möchte ich Ergebnisse in gängigen Formaten exportieren, damit Partner sie in eigenen Werkzeugen weiterverwenden.

**Akzeptanzkriterien**

- [ ] Splat als PLY und SPZ; Mesh als glTF 2.0; Punktwolke als E57.
- [ ] Exporter für glTF `KHR_gaussian_splatting` als austauschbares Modul (aktiv, sobald Extension ratifiziert).
- [ ] Validierung mit glTF-Validator bzw. Referenz-Reader im Test.


## EP-08 – Datenschutz & Datenhaltung

### US-046: Automatische Anonymisierung (Gesichter, Bildschirme, Dokumente)

`Story` · **Must** · 8 SP · M4 Präsentationsreife · Labels: privacy · Abhängig von: US-018, US-009

Als **Präsentierende (Makler/Verkäufer)** möchte ich dass Fotos von Personen, Bildschirme und Dokumente unkenntlich gemacht werden, damit ich Begehungen DSGVO-konform teilen kann.

**Akzeptanzkriterien**

- [ ] Detektion von Gesichtern (auch auf Fotos an der Wand), Bildschirmen und Textflächen in Keyframes; Weichzeichnung vor dem Training im Präsentationsmodus.
- [ ] Nutzer kann Bereiche manuell nachmarkieren; Modelle lizenzgeprüft.
- [ ] Bericht: Anzahl geschwärzter Regionen; Stichprobentest auf Referenzdaten.

### US-047: Einwilligung und Metadaten pro Scan

`Story` · **Should** · 2 SP · M4 Präsentationsreife · Labels: privacy · Abhängig von: US-040

Als **Präsentierende (Makler/Verkäufer)** möchte ich dokumentieren, dass der Eigentümer der Aufnahme zugestimmt hat, damit ich rechtlich abgesichert bin.

**Akzeptanzkriterien**

- [ ] Feld für Einwilligung (wer, wann, Zweck) in `scan.yaml`; Export nur mit gesetzter Einwilligung (abschaltbar für private Nutzung).
- [ ] Exporte tragen Metadaten: Aufnahmedatum, Hinweis „nicht maßstabsverbindlich“.

### US-048: Vollständiges Löschen eines Scans

`Story` · **Must** · 2 SP · M4 Präsentationsreife · Labels: privacy, server · Abhängig von: US-040

Als **Präsentierende (Makler/Verkäufer)** möchte ich einen Scan inklusive aller Ableitungen endgültig löschen, damit ich Löschpflichten erfüllen kann.

**Akzeptanzkriterien**

- [ ] Löscht Videos, Frames, Modelle, Index, Cache und Job-Historie; Löschprotokoll ohne Inhalte.
- [ ] Test stellt sicher, dass keine Dateien des Scans verbleiben.

### US-049: Speicherbereinigung von Zwischenständen

`Story` · **Could** · 2 SP · M3 Drohnen-Erfassung · Labels: server · Abhängig von: US-040

Als **Pilot** möchte ich Zwischenergebnisse automatisch aufräumen, damit mein Laptop-Speicher nicht vollläuft.

**Akzeptanzkriterien**

- [ ] Befehl/Schalter entfernt Frames und Zwischenstufen nach Abschluss, Ergebnisse bleiben.
- [ ] Anzeige des freigegebenen Speichers.


## EP-09 – Web-App, Jobs & Bedienung

### US-037: Lokaler API-Server mit Job-Queue

`Story` · **Must** · 8 SP · M1 MVP „Video → Begehung“ · Labels: server, claude-code · Abhängig von: US-006

Als **Entwickler (PO + Claude Code)** möchte ich einen Server, der Scans verwaltet und Pipeline-Jobs ausführt, damit die Bedienung per Browser möglich wird.

**Akzeptanzkriterien**

- [ ] FastAPI mit OpenAPI-3.1-Spezifikation; Endpunkte für Scans, Flüge, Jobs, Ergebnisse.
- [ ] Job-Queue (SQLite) mit GPU-Concurrency = 1; Abbruch und Wiederaufnahme von Jobs.
- [ ] Fortschritt per Server-Sent Events (Stufe, Prozent, Warnungen, Hinweise).

### US-038: Ein-Klick-Pipeline im Browser

`Story` · **Must** · 5 SP · M1 MVP „Video → Begehung“ · Labels: viewer, server · Abhängig von: US-037, US-026, US-028

Als **Pilot** möchte ich nach dem Upload nur auf „Verarbeiten“ tippen, damit ich ohne Technikwissen zur Begehung komme.

**Akzeptanzkriterien**

- [ ] Startet ingest → sfm → splat → Suchindex; Fortschritt mit geschätzter Restzeit.
- [ ] Bei Exit 20/30 verständliche Meldung mit nächstem Schritt.
- [ ] Am Ende Button „Begehung öffnen“.

### US-039: PWA installierbar im LAN (HTTPS)

`Story` · **Should** · 3 SP · M1 MVP „Video → Begehung“ · Labels: viewer, infra · Abhängig von: US-037

Als **Pilot** möchte ich die App auf meinem Android-Startbildschirm, damit ich sie wie eine App nutzen kann.

**Akzeptanzkriterien**

- [ ] Anleitung + Skript für lokales Zertifikat (z. B. mkcert); Manifest + Service Worker.
- [ ] QR-Code im Terminal/UI zum Öffnen der Adresse auf dem Handy.

### US-040: Scan-Übersicht und Verwaltung

`Story` · **Should** · 3 SP · M1 MVP „Video → Begehung“ · Labels: viewer, server · Abhängig von: US-037

Als **Präsentierende (Makler/Verkäufer)** möchte ich alle Scans mit Status, Datum und Vorschaubild sehen, damit ich Objekte und Kunden auseinanderhalte.

**Akzeptanzkriterien**

- [ ] Liste mit Name, Datum, Status, Speicherbedarf; Umbenennen, Notizen.
- [ ] Löschen (siehe US-048) mit Bestätigung.

### US-041: Fehlercode-Katalog und verständliche Meldungen

`Story` · **Should** · 2 SP · M1 MVP „Video → Begehung“ · Labels: server, viewer · Abhängig von: US-006

Als **Pilot** möchte ich bei Problemen eine klare Ursache und Lösung, damit ich nicht hängen bleibe.

**Akzeptanzkriterien**

- [ ] `docs/fehlercodes.md` mit allen Codes, Ursachen, Abhilfen; UI nutzt dieselben Texte.
- [ ] Jede Stufe meldet Fehler nur über Codes aus dem Katalog (Test prüft Vollständigkeit).


## EP-10 – Flugunterstützung & Autonomie

### US-053: SITL-Umgebung: ArduPilot + Gazebo mit Innenraumwelt

`Task` · **Should** · 8 SP · M5 Assistierter Flug · Labels: flight · Abhängig von: US-003

Als **Entwickler (PO + Claude Code)** möchte ich eine simulierte Wohnung mit simulierter Drohne, damit Flugcode ohne Risiko entwickelt und in der CI getestet werden kann.

**Akzeptanzkriterien**

- [ ] Container `sitl` startet ArduPilot Copter SITL + Gazebo mit Innenraumwelt und optischem Fluss/Abstandssensor.
- [ ] pytest-Marker `sitl`; Beispieltest: Start, 1 m Schweben, Landung.
- [ ] Alle Endpunkte 127.0.0.1; Hook blockiert Nicht-SITL-Ziele.

### US-054: Missionsplanung: Rasterflug aus Raumumriss

`Story` · **Should** · 8 SP · M5 Assistierter Flug · Labels: flight · Abhängig von: US-053

Als **Pilot** möchte ich aus einem grob gezeichneten Raumumriss eine Flugbahn erzeugen, damit die Drohne den Raum mit ausreichender Überlappung abfliegt.

**Akzeptanzkriterien**

- [ ] Eingabe: Polygon + Deckenhöhe; Ausgabe: Wegpunkte mit Blickrichtung, Abstand zu Wänden ≥ 1 m, ≤ 0,5 m/s.
- [ ] Simulierte Abdeckung ≥ 90 % der Wandfläche (SITL/Renderer).
- [ ] Mission als MAVLink-Missionsdatei exportierbar (QGroundControl-kompatibel).

### US-055: Sicherheitsfunktionen: Geofence, Failsafes, Not-Aus

`Story` · **Must** · 5 SP · M5 Assistierter Flug · Labels: flight · Abhängig von: US-053

Als **Pilot** möchte ich dass die Drohne bei Problemen sicher reagiert, damit weder Drohne noch Einrichtung beschädigt werden.

**Akzeptanzkriterien**

- [ ] Parameter-Set für Geofence (Raumgrenzen), Batterie-Failsafe (Landen), Link-Verlust (Landen), Max-Geschwindigkeit.
- [ ] Jeder Failsafe in SITL getestet; Subagent `flight-safety` ohne BLOCKER.
- [ ] Not-Aus-Schalter auf Fernsteuerung dokumentiert und getestet (Hardware, vom Menschen).

### US-056: Assistierter Flug auf Hardware (manuell freigegeben)

`Story` · **Could** · 8 SP · M5 Assistierter Flug · Labels: flight, hardware · Abhängig von: US-052, US-054, US-055

Als **Pilot** möchte ich eine geplante Mission im echten Raum fliegen lassen, die ich jederzeit übernehmen kann, damit Aufnahmen gleichmäßiger und reproduzierbarer werden.

**Akzeptanzkriterien**

- [ ] Start nur über menschlich gestarteten Befehl `--real-flight` mit interaktiver Bestätigung (nie durch Claude Code).
- [ ] Mission in einem Raum vollständig geflogen, Video automatisch dem Scan zugeordnet.
- [ ] Pilot kann jederzeit per Moduswechsel übernehmen; Flugprotokoll gespeichert.

### US-057: Live-Vorschau der Abdeckung während des Flugs

`Story` · **Could** · 8 SP · M5 Assistierter Flug · Labels: flight, viewer · Abhängig von: US-053

Als **Pilot** möchte ich noch während des Flugs sehen, welche Bereiche fehlen, damit ich nicht erst nach der Auswertung nachfliegen muss.

**Akzeptanzkriterien**

- [ ] Grober Live-Abdeckungsplan auf dem Smartphone (aus Stream oder Telemetrie), Latenz ≤ 2 s.
- [ ] Funktioniert mit SITL und mindestens einer Hardware-Konfiguration.

### US-058: Autonome Raumerkundung (Exploration) – Forschungsstory

`Story` · **Could** · 8 SP · M5 Assistierter Flug · Labels: flight, spike · Abhängig von: US-054, US-055

Als **Pilot** möchte ich dass die Drohne unbekannte Räume selbstständig erkundet, damit kein Umriss vorgezeichnet werden muss.

**Akzeptanzkriterien**

- [ ] Frontier-basierte Exploration in SITL mit Kollisionsvermeidung; Lizenz der genutzten VIO/SLAM-Komponenten geprüft (GPL nur als separater Prozess).
- [ ] Bewertung Machbarkeit auf Billig-Hardware als Bericht (Go/No-Go).


## EP-11 – Qualitätssicherung & Compliance

### US-008: Lizenz-Gate in der CI

`Story` · **Must** · 3 SP · M0 Fundament & Spikes · Labels: compliance, claude-code · Abhängig von: US-002

Als **Entwickler (PO + Claude Code)** möchte ich dass nicht erlaubte Lizenzen automatisch erkannt werden, damit das Produkt kommerziell nutzbar bleibt.

**Akzeptanzkriterien**

- [ ] `licenses/allowlist.yaml` und `denylist.yaml` (u. a. Inria-3DGS, DUSt3R, MASt3R, SuperPoint-Gewichte) existieren.
- [ ] CI prüft Python- und npm-Abhängigkeiten sowie `models.lock`; eine NC-Lizenz lässt den Build fehlschlagen (Q-8).
- [ ] Copyleft-Pakete sind nur in Container-Images ohne Import aus `src/` erlaubt; Verstoß wird gemeldet.

### US-009: Modell-Registry `models.lock` und `drone-scan models pull`

`Story` · **Must** · 3 SP · M1 MVP „Video → Begehung“ · Labels: compliance · Abhängig von: US-008

Als **Entwickler (PO + Claude Code)** möchte ich Modellgewichte versioniert, geprüft und einmalig herunterladen, damit die Laufzeit offline bleibt und Lizenzen nachvollziehbar sind.

**Akzeptanzkriterien**

- [ ] `models.lock` enthält je Modell Name, Version, Quelle, SHA-256, Code- und Gewichtelizenz.
- [ ] `drone-scan models pull` lädt nur gelistete Modelle und prüft die Hashes; Abweichung führt zu Abbruch.
- [ ] Zur Laufzeit erfolgt kein Netzwerkzugriff (Test mit blockiertem Netz).

### US-010: SBOM pro Release

`Story` · **Should** · 2 SP · M4 Präsentationsreife · Labels: compliance · Abhängig von: US-008, US-009

Als **Präsentierende (Makler/Verkäufer)** möchte ich eine Software-Stückliste zu jedem Release, damit ich Kunden und Partnern die Lizenzkonformität belegen kann.

**Akzeptanzkriterien**

- [ ] Release-Job erzeugt SBOM im Format SPDX 2.3 (JSON) und CycloneDX.
- [ ] SBOM enthält auch Modellgewichte aus `models.lock` und Container-Basisimages.

### US-011: Golden-Data-Smoketest und GPU-Testlauf mit VRAM-Grenze

`Story` · **Must** · 3 SP · M1 MVP „Video → Begehung“ · Labels: infra, claude-code · Abhängig von: US-006, US-021

Als **Entwickler (PO + Claude Code)** möchte ich einen schnellen End-to-End-Test auf einem Kurzclip, damit Regressionen in der Pipeline sofort sichtbar werden.

**Akzeptanzkriterien**

- [ ] `make smoke` verarbeitet `data/samples/kurz.mp4` in < 2 min zu Posen + Mini-Splat.
- [ ] `make test-gpu` läuft mit 5,5-GB-Grenze; Überschreitung lässt den Test fehlschlagen.
- [ ] Schwellwerte (registrierte Frames, PSNR) in `tests/golden/` versioniert.

### US-012: Benchmark-Suite auf Referenzräumen

`Story` · **Should** · 5 SP · M2 Gegenstandssuche · Labels: infra · Abhängig von: US-011

Als **Entwickler (PO + Claude Code)** möchte ich Qualitätsmetriken (Q-1 bis Q-7) reproduzierbar messen, damit Fortschritt und Regressionen objektiv bewertbar sind.

**Akzeptanzkriterien**

- [ ] `make bench` misst Laufzeit, Peak-VRAM, registrierte Frames, Maßstabsfehler, PSNR/SSIM, Suchtreffer Top-3.
- [ ] Ergebnisse als CSV/Markdown-Bericht mit Vergleich zum letzten Lauf.


## EP-12 – Hardware-Evaluierung & Erfassungsleitfaden

### US-050: Drohnen-Evaluierung: Videoqualität im Innenraum

`Task` · **Must** · 5 SP · M3 Drohnen-Erfassung · Labels: hardware, spike · Abhängig von: US-026, US-012

Als **Entwickler (PO + Claude Code)** möchte ich 2–3 Drohnen/Kameras im selben Referenzraum vergleichen, damit die Empfehlung für Nutzer auf Messwerten beruht.

**Akzeptanzkriterien**

- [ ] Kandidaten laut docs/drohnen-recherche.md (z. B. DJI Neo 2, gebrauchte Tello, Cinewhoop + Action-Cam) im selben Raum geflogen.
- [ ] Je Gerät: registrierte Frames, PSNR, Bewegungsunschärfe, Flugzeit, Handhabung – als Bericht.
- [ ] Kameraprofile (US-014) für die getesteten Geräte angelegt.

### US-051: Erfassungsleitfaden im Viewer (Checkliste vor dem Flug)

`Story` · **Should** · 2 SP · M3 Drohnen-Erfassung · Labels: viewer, hardware · Abhängig von: US-050

Als **Pilot** möchte ich vor dem Flug eine kurze Checkliste und Flugmuster, damit meine Aufnahmen beim ersten Versuch gut werden.

**Akzeptanzkriterien**

- [ ] Checkliste: Licht, Türen, Tags, Propellerschutz, Akku; Flugmuster-Grafik (Raum umrunden, Türen doppelt).
- [ ] Inhalte basieren auf Ergebnissen aus US-004 und US-050.

### US-052: Aufbau offene Flugplattform (ArduPilot-Cinewhoop)

`Task` · **Should** · 8 SP · M5 Assistierter Flug · Labels: hardware, flight · Abhängig von: US-050

Als **Entwickler (PO + Claude Code)** möchte ich eine eigene Indoor-Drohne mit ArduPilot, optischem Fluss und Aufnahmekamera, damit assistierter Flug mit offener Firmware möglich wird.

**Akzeptanzkriterien**

- [ ] Stückliste (≤ 400 €) und Bauanleitung in `docs/hardware/`; FC mit ArduPilot-Unterstützung und ausreichend Flash für Non-GPS-Navigation.
- [ ] Optical-Flow/ToF-Sensor (MatekSys 3901-L0X oder MicoAir MTF-01/02) eingerichtet und kalibriert.
- [ ] Positionshaltung (Loiter/FlowHold) indoor 60 s mit Drift ≤ 30 cm (manuell überwacht, Propellerschutz).
