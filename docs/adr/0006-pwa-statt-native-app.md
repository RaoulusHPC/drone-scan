# ADR-0006: Web-PWA statt nativer Android-App

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Das Android-Smartphone dient als Bedien- und Anzeigegerät. Eine native App erhöht Aufwand und Store-Abhängigkeit.

## Entscheidung
Viewer und Bedienung als PWA (TypeScript, three.js-basiert), ausgeliefert vom lokalen Server; zusätzlich als statisches Bundle exportierbar.

## Konsequenzen
+ Ein Codebasis für Android, Desktop, Betrachtende
- Kein Zugriff auf alle Sensoren; Installation braucht HTTPS im LAN
