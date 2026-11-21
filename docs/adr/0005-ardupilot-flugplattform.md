# ADR-0005: ArduPilot als offene Flugplattform ab M5

- **Status:** vorgeschlagen
- **Datum:** 2026-11-20

## Kontext
Für assistierten bzw. autonomen Indoor-Flug wird eine offene Firmware mit Positionshaltung ohne GPS benötigt. ArduPilot unterstützt viele kleine Flight Controller sowie optische Fluss- und Abstandssensoren (z. B. MatekSys 3901-L0X, MicoAir MTF-01/02) und bringt SITL mit.

## Entscheidung
ArduPilot Copter (GPL-3.0, unverändert genutzt, separater Prozess/Firmware), MAVLink 2, Simulation mit SITL + Gazebo. PX4 (BSD-3) bleibt Alternative, falls ein Pixhawk-Formfaktor gewählt wird.

## Konsequenzen
+ Offen, dokumentiert, Indoor-Optionen, große Community
- Sensorqualität indoor schwankt; Tuning-Aufwand; GPL nur bei Firmware-Änderungen relevant
