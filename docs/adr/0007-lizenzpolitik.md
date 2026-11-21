# ADR-0007: Lizenzpolitik

- **Status:** angenommen; teilweise abgelöst durch ADR-0012 (NC/research-only erlaubt, NVIDIA-Laufzeit, Images nur lokal)
- **Datum:** 2026-11-20

## Kontext
Das Projekt soll ausschließlich Open Source nutzen und kommerziell verwertbar bleiben.

## Entscheidung
Erlaubt: MIT, BSD-2/3, Apache-2.0, ISC, MPL-2.0, PSF-2.0 (Nachtrag 2026-11-21, PO), LGPL (dynamisch), CC-BY-4.0 (Daten). Copyleft (GPL/AGPL) nur als separates Werkzeug bzw. separater Prozess ohne Einbindung in eigenen Code. Verboten: jede Nutzungsbeschränkung (NC, Research-only, „non-commercial“), unklare Lizenzen. *(Abgelöst durch ADR-0012: NC/research-only sind erlaubt und werden im Gate als `NC` markiert; unklare Lizenzen bleiben verboten.)* Für Modelle zählt die Lizenz der Gewichte. Durchsetzung über CI-Lizenz-Gate, Denylist, models.lock, SBOM (SPDX/CycloneDX).

**Umsetzung (KAN-23):**
- **Erlaubt:** `licenses/allowlist.yaml` (SPDX-IDs, `data_licenses` für Modelle/Daten, geprüfte `overrides`).
- **Verboten:** `licenses/denylist.yaml` (seit ADR-0012 werden NC-/Research-Muster und -Pakete nur noch als `NC` markiert; Copyleft-IDs, verbotene Pakete und Modelle, `container_only_modules`).
- **Prüfung:** `make license-gate` als Teil von `make check` und der CI. Sie umfasst installierte Python-Pakete (Laufzeit- und Dev-Scope aus `uv.lock`), `viewer/node_modules`, `models.lock` und die Importe in `src/`.
- **Copyleft in Dev-Werkzeugen:** erzeugt nur eine Warnung, in der Laufzeit ist es ein Fehler.
- **Änderungen:** Änderungen an `licenses/` mergt immer der PO.

## Konsequenzen
+ Rechtssicherheit für spätere Kommerzialisierung *(seit ADR-0012 erst nach Neuprüfung der NC-Komponenten)*
- Einige SOTA-Modelle sind ausgeschlossen *(seit ADR-0012 lizenzrechtlich erlaubt, Stack-Wechsel nur per ADR)*
