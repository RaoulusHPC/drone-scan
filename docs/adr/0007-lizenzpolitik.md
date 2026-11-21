# ADR-0007: Lizenzpolitik

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Das Projekt soll ausschließlich Open Source nutzen und kommerziell verwertbar bleiben.

## Entscheidung
Erlaubt: MIT, BSD-2/3, Apache-2.0, ISC, MPL-2.0, PSF-2.0 (Nachtrag 2026-11-21, PO), LGPL (dynamisch), CC-BY-4.0 (Daten). Copyleft (GPL/AGPL) nur als separates Werkzeug bzw. separater Prozess ohne Einbindung in eigenen Code. Verboten: jede Nutzungsbeschränkung (NC, Research-only, „non-commercial“), unklare Lizenzen. Für Modelle zählt die Lizenz der Gewichte. Durchsetzung über CI-Lizenz-Gate, Denylist, models.lock, SBOM (SPDX/CycloneDX).

**Umsetzung (KAN-23):**
- **Erlaubt:** `licenses/allowlist.yaml` (SPDX-IDs, `data_licenses` für Modelle/Daten, geprüfte `overrides`).
- **Verboten:** `licenses/denylist.yaml` (NC-/Research-Muster, Copyleft-IDs, verbotene Pakete und Modelle, `container_only_modules`).
- **Prüfung:** `make license-gate` als Teil von `make check` und der CI. Sie umfasst installierte Python-Pakete (Laufzeit- und Dev-Scope aus `uv.lock`), `viewer/node_modules`, `models.lock` und die Importe in `src/`.
- **Copyleft in Dev-Werkzeugen:** erzeugt nur eine Warnung, in der Laufzeit ist es ein Fehler.
- **Änderungen:** Änderungen an `licenses/` mergt immer der PO.

## Konsequenzen
+ Rechtssicherheit für spätere Kommerzialisierung
- Einige SOTA-Modelle sind ausgeschlossen
