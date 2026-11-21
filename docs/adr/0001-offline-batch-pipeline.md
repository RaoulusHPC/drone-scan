# ADR-0001: Offline-Batch-Pipeline statt Echtzeit-SLAM im MVP

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Die Zielhardware (RTX 2060, 6 GB) und Billigdrohnen ohne Onboard-Rechner erlauben keine robuste Echtzeit-Rekonstruktion. Die Anwendungsfälle (Suche, Begehung) brauchen keine Live-Karte.

## Entscheidung
Die Verarbeitung erfolgt nach dem Flug als Batch-Pipeline (ingest → sfm → recon → search). Echtzeit-Funktionen (Live-Vorschau, VIO) kommen frühestens in M5.

## Konsequenzen
+ Einfach, testbar, reproduzierbar, wenig VRAM
- Fehlende Abdeckung fällt erst nach dem Flug auf → Qualitätsreport mit Nachflug-Empfehlung
