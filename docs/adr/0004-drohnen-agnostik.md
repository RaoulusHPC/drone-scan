# ADR-0004: Drohnen-agnostisch: Videodatei als Minimalschnittstelle

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Endnutzer besitzen unterschiedliche Drohnen (≤ 500 €), meist mit geschlossener Firmware. Programmierbare Billigdrohnen sind kaum noch verfügbar (Tello eingestellt).

## Entscheidung
Die einzige Pflichtschnittstelle ist eine Videodatei (+ optional Telemetrie). Hersteller-SDKs sind nicht nötig. Adapter (Tello, ArduPilot) sind optionale Plug-ins.

## Konsequenzen
+ Funktioniert mit Smartphone, DJI, FPV-Cams und eigener Hardware
- Keine Flugsteuerung im MVP; Kamera-Intrinsics müssen geschätzt werden
