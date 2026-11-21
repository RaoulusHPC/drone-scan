# ADR-0011: Projektlizenz AGPL-3.0-or-later

- **Status:** angenommen (PO, 21.11.2026)
- **Datum:** 2026-11-21
- **Jira:** – (PO-Entscheidung im Chat am 21.11.2026, Anlass: offener Punkt aus KAN-16)

## Kontext
drone-scan hatte bis zu diesem ADR keine eigene Lizenz. Es gab weder ein `license`-Feld in `pyproject.toml` noch eine `LICENSE`-Datei. Damit war der Code rechtlich „all rights reserved“, und Dritte durften ihn weder nutzen noch beitragen. ADR-0007 regelt nur, welche **fremden** Lizenzen wir einbinden dürfen, nicht die Lizenz des eigenen Codes.

Der PO möchte eine **strenge Copyleft-Lizenz**. Weiterentwicklungen sollen offen bleiben und nicht in proprietären Produkten verschwinden. Wichtig für die Wahl: drone-scan wird auch **über das Netz genutzt**. Server und PWA-Viewer liefern die Begehung im Browser aus, z. B. bei Verkaufsgesprächen über den lokalen Server. Ein eigener Cloud-Dienst bleibt laut `docs/projektmanagement.md` ein Nicht-Ziel. Dritte könnten drone-scan aber als Webdienst betreiben.

## Entscheidung
Der gesamte eigene Code (`src/`, `tests/`, `tools/`, später `viewer/` und `flight/`) steht unter **GNU Affero General Public License v3.0 or later** (SPDX: `AGPL-3.0-or-later`).

- Der vollständige Lizenztext liegt in `LICENSE` im Repo-Root. `pyproject.toml` (und später `viewer/package.json`) tragen den SPDX-Ausdruck.
- Rechteinhaber ist der PO. Externe Beiträge werden nur mit einer Vereinbarung angenommen, die dem PO die Rechte überträgt bzw. eine Neulizenzierung erlaubt (CLA). So bleibt eine spätere Doppellizenzierung möglich.
- ADR-0007 bleibt unverändert gültig. Fremder GPL/AGPL-Code wird weiterhin nur als separater Prozess oder Container genutzt und nicht in `src/` importiert. Seit diesem ADR wäre ein Import von GPL-3.0- oder AGPL-3.0-Code zwar lizenzrechtlich mit unserem Code vereinbar. Die Trennung bleibt trotzdem bestehen, **weil fremder Copyleft-Code im Kern die Doppellizenzierung durch den PO verhindern würde**: Für fremden Code besitzt er keine Rechte zur Neulizenzierung.

**Verworfene Alternativen**
- **GPL-3.0-or-later:** starkes Copyleft, greift aber nur bei *Weitergabe*. Wer drone-scan verändert und nur als Webdienst betreibt, müsste seine Änderungen nicht offenlegen (sogenannte SaaS-Lücke).
- **AGPL-3.0-only:** schließt künftige Lizenzversionen aus. Das bringt kaum Nutzen, kostet aber Flexibilität, falls die FSF eine verbesserte Version veröffentlicht.
- **MPL-2.0 / LGPL-3.0 (schwaches Copyleft):** Diese Lizenzen erlauben, unseren Code in proprietäre Produkte einzubetten. Das widerspricht dem Wunsch nach strengem Copyleft.
- **MIT / Apache-2.0 (permissiv):** kein Copyleft, aus demselben Grund verworfen.

## Konsequenzen
+ Weiterentwicklungen von drone-scan bleiben offen, auch beim Betrieb als Webdienst (AGPL §13).
+ **Kompatibilität mit den Abhängigkeiten:** Alle laut ADR-0007 erlaubten Lizenzen lassen sich mit AGPL-3.0 kombinieren:
  - MIT, BSD-2/3, ISC, PSF-2.0
  - Apache-2.0 (kompatibel mit GPLv3/AGPLv3, nicht mit GPLv2)
  - LGPL-2.1+/3.0
  - CC-BY-4.0 für Daten (einseitig kompatibel)
  - MPL-2.0 über die „Secondary License“-Klausel (§3.3), sofern die Datei nicht als „Incompatible With Secondary Licenses“ markiert ist

  Der License-Audit zu KAN-16 fand nur solche Lizenzen. Die bisherige Lizenzpolitik muss also nicht geändert werden.
+ Der PO kann als alleiniger Rechteinhaber den Code zusätzlich unter einer kommerziellen Lizenz anbieten (Doppellizenzierung). Damit bleibt die Anforderung „kommerziell verwertbar“ aus ADR-0007 erfüllt.
- **Quellcode-Hinweis im Netzbetrieb:** Server und Viewer zeigen einen Link zum Quellcode an (z. B. im Viewer-Menü und in der API-Doku). AGPL §13 verpflichtet dazu nur Betreiber **veränderter** Versionen. Wir machen es als Projektregel auch für die unveränderte Version, damit die Pflicht bei Änderungen nicht vergessen wird.
- **PWA-Viewer:** Das an den Browser ausgelieferte JavaScript gilt als Weitergabe von Objektcode. Der zugehörige Quellcode muss verfügbar sein, also nicht-minifizierte Quellen bzw. Source Maps und Build-Anleitung. Das betrifft den Vite-Build.
- **Export-Bundle (M4):** Das offline teilbare Viewer-Bundle ist eine Weitergabe. Es muss den Lizenztext und ein Quellcode-Angebot enthalten.
- **Partner und Kunden:** Wer eine veränderte Version betreibt oder weitergibt, muss deren Quellcode offenlegen. Für Partner, die das nicht möchten, braucht es eine kommerzielle Lizenz vom PO.
- **Beiträge Dritter** brauchen ein CLA. Andernfalls ist eine Doppellizenzierung später nicht mehr möglich. CLA-Text und `CONTRIBUTING.md` fehlen noch. Bis dahin werden keine externen Beiträge angenommen.
- **GPL-2.0-only** ist mit AGPL-3.0 nicht kompatibel. Solche Werkzeuge dürfen weiterhin nur als separater Prozess genutzt werden; ADR-0007 deckt das bereits ab.
- Kein Einfluss auf VRAM oder Laufzeit. Für Modellgewichte gilt weiterhin deren eigene Lizenz (`models.lock`).
- **Rechtliche Prüfung:** Dieser ADR ist keine Rechtsberatung. Vor den ersten Pilotkunden (M4) sollen Lizenz, CLA und Quellcode-Angebot rechtlich geprüft werden. Das ist im Budget in `docs/projektmanagement.md` §8 ergänzt.

**Folgearbeiten (noch nicht im Backlog):** Quellcode-Link in Server und Viewer (M1), Lizenz und Quellcode-Angebot im Export-Bundle (M4), CLA und `CONTRIBUTING.md`, optional SPDX-Header in den Quelldateien.
