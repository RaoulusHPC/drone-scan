# Projektmanagement – drone-scan

> Stand: 20.11.2026 · Werkzeug: **Jira** (Projekt „DRN“ (Schlüssel `KAN`), Scrum, team-managed) · Verwandte Dokumente: [Architektur](architektur.md) · [User Stories](user-stories.md)

---

## 1. Projektsteckbrief

| | |
|---|---|
| **Ziel** | Aus Drohnenvideos (≤ 500 €-Drohne) lokal auf einem Laptop (RTX 2060) 3D-Modelle von Innenräumen erzeugen, für (1) Gegenstandssuche und (2) virtuelle Begehung. Ausschließlich Open Source. |
| **Team** | Product Owner (Gründer) · Claude Code als Implementierungspartner · optional später Pilotkunden/Tester |
| **Vorgehen** | Scrum-light mit 2-Wochen-Sprints, Steuerung in Jira |
| **Laufzeit (Plan)** | Sprint 0 ab 23.11.2026, MVP (M1) Ende Januar 2027, Ausbaustufe Flug (M5) bis Juni 2027 |
| **Budget (Sachmittel)** | Entwicklung ≈ 100–250 €, optionale offene Flugplattform ≈ 250–400 € (siehe Abschnitt 8) |
| **Nicht-Ziele** | Rechtsverbindliche Flächenberechnung, Echtzeit-SLAM im MVP, Außenflüge, Cloud-Dienst, native App |

---

## 2. Vorgehensmodell

**Scrum-light für ein Ein-Personen-Team mit KI-Unterstützung:**

| Element | Ausgestaltung |
|---|---|
| Sprintlänge | 2 Wochen (Mo–Fr der Folgewoche) |
| Sprintplanung (≤ 45 min) | Sprintziel formulieren, Stories aus dem priorisierten Backlog ziehen, nur Stories mit „Definition of Ready“ |
| Daily (5 min, schriftlich) | Kurze Notiz im Jira-Board: Was ist erledigt, was blockiert? Claude Code kann den Status per Atlassian-MCP vorschlagen |
| Sprint Review (30 min) | **Demo mit echtem Scan** (Video → Ergebnis). Ein Bildschirmvideo pro Sprint dokumentiert den Fortschritt |
| Retrospektive (15 min) | 1 Verbesserung pro Sprint. **Lessons Learned fließen in `CLAUDE.md` zurück** |
| Backlog Refinement | Laufend, spätestens 2 Tage vor Sprintende: Akzeptanzkriterien schärfen, schätzen |

**Rollen**

- **Product Owner (du):** priorisiert, nimmt ab und setzt Stories auf *Done*. Reviewt und mergt PRs, die nicht unter die Self-Merge-Regel fallen (§ 6). Freigabe echter Flüge erfolgt nur durch dich.
- **Claude Code:** plant (Plan Mode), implementiert, testet, schreibt Doku-Updates, schlägt ADRs vor und aktualisiert Jira-Status.
- **Subagents (in `.claude/agents/`):** spezialisierte Prüfer (Lizenz, VRAM, Review, Flugsicherheit).

---

## 3. Meilensteine & Releaseplan

| MS | Name | Ergebnis („Done“ heißt …) | Sprints | Zieldatum* |
|---|---|---|---|---|
| **M0** | Fundament & Spikes | Repo, CI, Docker, Lizenz-Gate, Jira-Workflow. Spike-Ergebnisse: COLMAP + gsplat auf 6 GB, Drohnenvideo-Qualität | S0 | 04.12.2026 |
| **M1** | **MVP „Video → Begehung“** | Smartphone- oder Drohnenvideo hochladen → begehbare Szene im Android-Browser, Qualitätsreport | S1–S3 | 29.01.2027 |
| **M2** | „Wo ist mein Schlüssel?“ | Textsuche mit 3D-Markern und Belegbildern, Q-5 erfüllt | S4–S5 | 26.02.2027 |
| **M3** | Drohnen-Erfassung | Manueller Drohnenflug, mehrere Flüge zusammenführen, AprilTag-Maßstab, Drohnen-Evaluierung abgeschlossen | S6–S7 | 26.03.2027 |
| **M4** | Präsentationsreife | Begehung für Verkaufsgespräche: Export-Bundle, Raumliste, Anonymisierung, Grundriss (optional) | S8–S9 | 23.04.2027 |
| **M5** | Assistierter Flug | ArduPilot in SITL → Hardware: Positionshaltung indoor, Rasterflug-Mission, Failsafes | S10–S13 | 18.06.2027 |

\* Annahmen: Start 23.11.2026, Pause 21.12.2026–03.01.2027. Die Teamkapazität wird nach Sprint 1 kalibriert, die Daten werden danach angepasst.

**Sprintkalender (Plan)**

| Sprint | Zeitraum | Sprintziel |
|---|---|---|
| S0 | 23.11.–04.12.2026 | Fundament steht, Machbarkeit auf 6 GB nachgewiesen |
| S1 | 07.12.–18.12.2026 | Video → Keyframes → COLMAP-Posen per CLI |
| — | 21.12.2026–03.01.2027 | Pause |
| S2 | 04.01.–15.01.2027 | Splat-Training im 6-GB-Profil, Qualitätsreport |
| S3 | 18.01.–29.01.2027 | Server + PWA-Viewer, Upload bis Begehung → **M1** |
| S4 | 01.02.–12.02.2027 | Suchindex (Detektionen, Embeddings) |
| S5 | 15.02.–26.02.2027 | Textsuche → 3D-Marker im Viewer → **M2** |
| S6 | 01.03.–12.03.2027 | AprilTag-Maßstab, Multi-Flight-Merge |
| S7 | 15.03.–26.03.2027 | Drohnen-Evaluierung, Capture Guidance → **M3** |
| S8 | 29.03.–09.04.2027 | Export-Bundle, Anonymisierung, Raumliste |
| S9 | 12.04.–23.04.2027 | Grundriss, UX-Feinschliff, Pilottest → **M4** |
| S10–S13 | 26.04.–18.06.2027 | SITL, Hardware-Aufbau, Positionshaltung, Missionen → **M5** |

---

## 4. Jira-Konfiguration (Projekt „DRN“ (Schlüssel `KAN`))

**Projekt:** Scrum, team-managed, Name „Dronenprojekt“, Schlüssel `KAN` (Projektname „DRN“; ein späterer Schlüsselwechsel auf `DRN` ist in Jira möglich, alte Keys leiten weiter).

| Jira-Element | Verwendung |
|---|---|
| **Epic** | 12 Epics (EP-01 … EP-12), siehe [User Stories](user-stories.md) |
| **Story** | Nutzerwert mit Akzeptanzkriterien (Given/When/Then) |
| **Task** | Technische Arbeit ohne direkten Nutzerwert (z. B. Spikes) |
| **Subtask** | Optional. Claude Code legt Subtasks aus dem Plan an, wenn eine Story > 5 SP hat |
| **Bug** | Fehler mit Reproduktionsschritten, Scan-ID, Manifest-Auszug |
| **Story Points** | Fibonacci 1, 2, 3, 5, 8. Stories > 8 SP werden geteilt |
| **Priority** | Abbildung MoSCoW: Must → *Highest*, Should → *High*, Could → *Medium*, Won't (jetzt) → *Low* |
| **Labels** | `m0` … `m5` (Meilenstein), Komponenten-Labels `capture`, `ingest`, `sfm`, `recon`, `search`, `viewer`, `server`, `privacy`, `export`, `flight`, `infra`, `compliance`, `hardware`; `spike`, `claude-code` |
| **Releases / Versionen** | `M0-Fundament`, `M1-MVP`, `M2-Suche`, `M3-Drohne`, `M4-Präsentation`, `M5-Flug` |
| **Board-Spalten** | Backlog → Ready → In Progress → In Review (PR offen) → Done |
| **Jira-Status (deutsche UI)** | Board „KAN board“ (ID 2, team-managed). Spalte → Status: Backlog/To Do → *Zu erledigen*, Ready → *Bereit* (Status-ID **10008**, JQL-Name `"Ready"`; in Filtern bevorzugt die ID verwenden), In Progress → *In Arbeit*, In Review → *Wird überprüft*, Done → *Fertig*. Transitions immer über den **Zielstatus** (ID) auswählen, nicht über den Transitions- oder Spaltennamen |
| **Verknüpfungen** | „blocks / is blocked by“ für die Abhängigkeiten aus dem Backlog |

**Nützliche JQL-Filter**

```text
Aktueller Sprint:       project = KAN AND sprint in openSprints() ORDER BY rank
Bereit für Claude Code: project = KAN AND status = 10008 AND labels = claude-code ORDER BY priority DESC, rank
Meilenstein M1 offen:   project = KAN AND labels = m1 AND statusCategory != Done
Blockiert:              project = KAN AND issueLinkType = "is blocked by" AND statusCategory != Done
Lizenz-Themen:          project = KAN AND labels = compliance
```

**Anbindung Claude Code ↔ Jira:** Atlassian Rovo MCP Server, projektweit in `.mcp.json` hinterlegt (`https://mcp.atlassian.com/v2/mcp`), einmalig mit `/mcp` in Claude Code authentifizieren. Danach kann Claude Code Stories lesen, Status setzen und Kommentare (PR-Link, Testergebnis) schreiben.

---

## 5. Definition of Ready / Definition of Done

**Definition of Ready (Story darf in den Sprint)**
- [ ] Nutzerwert im Format „Als … möchte ich … damit …“
- [ ] Akzeptanzkriterien testbar (Given/When/Then), inkl. Fehlerfall
- [ ] Geschätzt (≤ 8 SP), Abhängigkeiten erledigt oder im selben Sprint
- [ ] Betroffene Bausteine und Artefakte benannt, bei Bedarf ADR verlinkt
- [ ] Testdaten vorhanden (Beispielvideo, Erwartungswerte)

**Definition of Done**
- [ ] Alle Akzeptanzkriterien durch automatische Tests abgedeckt (GPU-Tests markiert)
- [ ] `make check` grün (ruff, mypy, pytest CPU, Lizenz-Gate)
- [ ] Bei GPU-Code: `make test-gpu` grün **mit Allocator-Grenze 5,0 GiB**, Peak-VRAM (gesamt ≤ 5,5 GiB) im PR notiert
- [ ] Doku aktualisiert (Architektur/ADR/README/Fehlercodes, sofern betroffen)
- [ ] PR mit Story-Key im Titel, gemergt: durch Claude Code gemäß Self-Merge-Regel (§ 6), sonst vom PO reviewt und gemergt
- [ ] Jira-Story auf *Done*, Kommentar mit PR-Link und Demo-Hinweis

---

## 6. Arbeitsablauf mit Claude Code (pro Story)

```text
1. Jira:        Story KAN-xx ist "Bereit" (Spalte Ready, PO)
2. Claude Code: /story KAN-xx  → liest Story via MCP, Plan Mode, Plan als Jira-Kommentar
3. PO:          Plan freigeben oder korrigieren
4. Claude Code: Branch feat/KAN-xx-kurzname, Tests zuerst (Akzeptanzkriterien), Implementierung
5. Claude Code: make check (+ make test-gpu), Subagent "reviewer" + ggf. "license-auditor"/"vram-guard"
6. Claude Code: PR "KAN-xx: <Titel>", Story → "In Review", Kommentar mit PR-Link
7. Merge:       Claude Code selbst, wenn die Self-Merge-Regel erfüllt ist – sonst PO (Review, Merge)
   PO:          Story → "Done"
8. Retro-Fund:  Neue Regel? → CLAUDE.md ergänzen (/retro)
```

- **Commits:** Conventional Commits mit Story-Key, z. B. `feat(sfm): add AprilTag scale alignment (KAN-27)`.
- **Branches:** `feat/KAN-xx-…`, `fix/KAN-xx-…`, `spike/KAN-xx-…`.
- **Self-Merge-Regel (seit 21.11.2026, PO-Entscheidung):** Claude Code mergt einen PR selbst (Merge-Commit), wenn **alle** Bedingungen erfüllt sind:
  1. `make check` grün, sobald vorhanden auch die CI. Bei GPU-Code zusätzlich `make test-gpu` (5,0-GiB-Grenze, GTX 1060).
  2. Subagent `reviewer` meldet keine BLOCKER. Je nach Änderung gilt das auch für `license-auditor` und `vram-guard`.
  3. Der PR ist konfliktfrei (`mergeable_state = clean`).
  4. Der PR ändert **nichts** unter `flight/`, an `CLAUDE.md`, `.claude/hooks/`, `.claude/settings.json`, `.claude/agents/`, `.claude/skills/` oder `licenses/`. Solche PRs mergt immer der PO, weil sie Flugsicherheit, Lizenzpolitik, die Schutzmechanismen von Claude Code oder dessen eigene Regeln (einschließlich dieser Self-Merge-Regel) betreffen.

  Den Jira-Status *Done* setzt weiterhin der PO. Nach dem Merge kommentiert Claude Code den Merge-Commit in der Story.
- **Kontextpflege:** Eine Story pro Claude-Code-Sitzung. Bei langen Sitzungen kurz zusammenfassen bzw. neu starten.

---

## 7. Risikomanagement

Das Risikoregister steht in der [Architektur, Abschnitt 11](architektur.md#11-risiken-und-technische-schulden). Es wird **in jeder Retro** überprüft, und neue Risiken werden in Jira als Task mit Label `risk` erfasst.

**Top-3-Risiken mit Frühindikatoren**

| Risiko | Frühindikator | Gegenmaßnahme ausgelöst wenn … |
|---|---|---|
| R-01 SfM scheitert indoor | < 80 % registrierte Keyframes im Spike | … in 2 von 3 Referenzräumen → Matcher-Alternative, AprilTag-Pflicht |
| R-03 VRAM reicht nicht | Peak > 5,5 GiB im Profil `low_vram_6gb` | … bei Referenzwohnung → Chunking vorziehen (Story aus M3 nach M1) |
| R-05 Autonomie zu teuer/komplex | SITL-Positionshaltung instabil nach 2 Sprints | … → M5 auf „assistiert“ begrenzen, manueller Flug bleibt Standard |

---

## 8. Budget & Beschaffung

| Phase | Posten | ca. Kosten | Zeitpunkt |
|---|---|---|---|
| M0–M2 | Smartphone (vorhanden) als Videoquelle, AprilTags (Ausdruck) | 0 € | sofort |
| M3 | **Entwicklungsdrohne** (Empfehlung siehe [Drohnen-Recherche](drohnen-recherche.md)) | ≈ 100–240 € | Sprint 5 bestellen |
| M3 | Ersatzakkus, Propellerschutz, microSD | ≈ 30–60 € | mit Drohne |
| M5 | Offene Plattform: 3,5″-Cinewhoop mit ArduPilot-fähigem FC + Optical-Flow/ToF-Sensor + Action-Cam, Funk | ≈ 250–400 € | Sprint 9 bestellen |
| laufend | Software | 0 € (Open Source) | – |
| optional | Haftpflichtversicherung (Drohne), Rechtsberatung DSGVO sowie Lizenz/CLA/Quellcode-Angebot (ADR-0011) vor Pilotkunden | nach Angebot | vor M4 |

---

## 9. Kennzahlen (KPIs)

| KPI | Quelle | Ziel |
|---|---|---|
| Scan-Erfolgsquote (begehbare Szene ohne manuellen Eingriff) | `reports/summary.json` | ≥ 80 % ab M3 |
| Verarbeitungszeit pro 60 m² | Manifeste | ≤ 60 min |
| Peak-VRAM | Manifeste | ≤ 5,5 GiB |
| Treffer Top-3 Gegenstandssuche | Benchmark-Set | ≥ 70 % |
| Sprint-Zuverlässigkeit (erledigte / geplante SP) | Jira | 70–90 % |
| Lizenzverstöße im Main-Branch | CI | 0 |

---

## 10. Kommunikation & Dokumentation

- **Single Source of Truth:** Jira (Arbeitsstand) und Repository (`docs/`, ADRs). Das claude.ai-Projekt „Dronenprojekt“ hält Kopien der Kerndokumente für Gespräche außerhalb von Claude Code.
- **Pro Meilenstein:** Demo-Video, kurzer Release-Bericht (Was geht? Messwerte? Bekannte Grenzen?), Jira-Version auf *Released* setzen.
- **Entscheidungen:** als ADR im Repo, im Jira-Ticket verlinkt.
