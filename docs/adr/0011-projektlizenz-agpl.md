# ADR-0011: Projektlizenz AGPL-3.0-or-later

- **Status:** angenommen (PO, 21.11.2026)
- **Datum:** 2026-11-21
- **Jira:** – (PO-Entscheidung im Chat am 21.11.2026, Anlass: offener Punkt aus KAN-16)

## Kontext
drone-scan hatte bisher keine eigene Lizenz. `pyproject.toml` enthält kein `license`-Feld, und es gibt keine `LICENSE`-Datei. Damit ist der Code rechtlich „all rights reserved“, und Dritte dürfen ihn weder nutzen noch beitragen. ADR-0007 regelt nur, welche **fremden** Lizenzen wir einbinden dürfen, nicht die Lizenz des eigenen Codes.

Der PO möchte eine **strenge Copyleft-Lizenz**. Weiterentwicklungen sollen offen bleiben und nicht in proprietären Produkten verschwinden. Wichtig für die Wahl: drone-scan wird auch **über das Netz genutzt**. Server und PWA-Viewer liefern die Begehung im Browser aus, z. B. für Verkaufsgespräche.

## Entscheidung
Der gesamte eigene Code (`src/`, `tests/`, `tools/`, später `viewer/` und `flight/`) steht unter **GNU Affero General Public License v3.0 or later** (SPDX: `AGPL-3.0-or-later`).

- Der vollständige Lizenztext liegt in `LICENSE` im Repo-Root. `pyproject.toml` (und später `viewer/package.json`) tragen den SPDX-Ausdruck.
- Rechteinhaber ist der PO. Externe Beiträge werden nur mit einer Vereinbarung angenommen, die dem PO die Rechte überträgt bzw. eine Neulizenzierung erlaubt (CLA). So bleibt eine spätere Doppellizenzierung möglich.
- ADR-0007 bleibt unverändert gültig. Fremder GPL/AGPL-Code wird weiterhin nur als separater Prozess oder Container genutzt und nicht in `src/` importiert.

**Verworfene Alternativen**
- **GPL-3.0-or-later:** starkes Copyleft, greift aber nur bei *Weitergabe*. Wer drone-scan verändert und nur als Webdienst betreibt, müsste seine Änderungen nicht offenlegen (sogenannte SaaS-Lücke). Genau das ist bei unserem Server- und PWA-Modell der wahrscheinliche Fall.
- **AGPL-3.0-only:** schließt künftige Lizenzversionen aus. Das bringt kaum Nutzen, kostet aber Flexibilität, falls die FSF eine verbesserte Version veröffentlicht.
- **MPL-2.0 / LGPL-3.0 (schwaches Copyleft):** Diese Lizenzen erlauben, unseren Code in proprietäre Produkte einzubetten. Das widerspricht dem Wunsch nach strengem Copyleft.
- **MIT / Apache-2.0 (permissiv):** kein Copyleft, aus demselben Grund verworfen.

## Konsequenzen
+ Weiterentwicklungen von drone-scan bleiben offen, auch beim Betrieb als Webdienst (AGPL §13).
+ **Kompatibilität mit den Abhängigkeiten:** Alle laut ADR-0007 erlaubten Lizenzen lassen sich mit AGPL-3.0 kombinieren: MIT, BSD-2/3, Apache-2.0, ISC, MPL-2.0, PSF-2.0, LGPL-2.1+/3.0. Der License-Audit zu KAN-16 fand nur solche Lizenzen. Die bisherige Lizenzpolitik muss also nicht geändert werden.
+ Der PO kann als alleiniger Rechteinhaber den Code zusätzlich unter einer kommerziellen Lizenz anbieten (Doppellizenzierung). Damit bleibt die Anforderung „kommerziell verwertbar“ aus ADR-0007 erfüllt.
- **Quellcode-Angebot im Netzbetrieb:** Server und Viewer müssen Nutzern einen Hinweis auf den Quellcode bzw. einen Link dorthin anzeigen (z. B. im Viewer-Menü und in der API-Doku). Das ist als Akzeptanzkriterium in die Stories für Server (EP Server/Viewer, M1) aufzunehmen.
- **Export-Bundle (M4):** Das offline teilbare Viewer-Bundle ist eine Weitergabe. Es muss den Lizenztext und ein Quellcode-Angebot enthalten. Das betrifft die Export-Story.
- **Partner und Kunden:** Wer eine veränderte Version betreibt oder weitergibt, muss deren Quellcode offenlegen. Für Partner, die das nicht möchten, braucht es eine kommerzielle Lizenz vom PO.
- **Beiträge Dritter** brauchen ein CLA. Andernfalls ist eine Doppellizenzierung später nicht mehr möglich.
- **GPL-2.0-only** ist mit AGPL-3.0 nicht kompatibel. Solche Werkzeuge dürfen weiterhin nur als separater Prozess genutzt werden; ADR-0007 deckt das bereits ab.
- Kein Einfluss auf VRAM oder Laufzeit. Für Modellgewichte gilt weiterhin deren eigene Lizenz (`models.lock`).
- Keine Rechtsberatung: Vor den ersten Pilotkunden (M4) sollte die Lizenz rechtlich geprüft werden. Das ist bereits als Budgetposten in `docs/projektmanagement.md` §8 eingeplant.
