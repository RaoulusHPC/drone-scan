# ADR-0007: Lizenzpolitik

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Das Projekt soll ausschließlich Open Source nutzen und kommerziell verwertbar bleiben.

## Entscheidung
Erlaubt: MIT, BSD-2/3, Apache-2.0, ISC, MPL-2.0, LGPL (dynamisch), CC-BY-4.0 (Daten). Copyleft (GPL/AGPL) nur als separates Werkzeug bzw. separater Prozess ohne Einbindung in eigenen Code. Verboten: jede Nutzungsbeschränkung (NC, Research-only, „non-commercial“), unklare Lizenzen. Für Modelle zählt die Lizenz der Gewichte. Durchsetzung über CI-Lizenz-Gate, Denylist, models.lock, SBOM (SPDX/CycloneDX).

## Konsequenzen
+ Rechtssicherheit für spätere Kommerzialisierung
- Einige SOTA-Modelle sind ausgeschlossen
