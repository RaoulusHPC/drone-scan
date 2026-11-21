# ADR-0002: 3D Gaussian Splatting mit gsplat als primäre Darstellung

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Für fotorealistische Begehungen ist 3DGS Stand der Technik. Die Referenzimplementierung von Inria steht unter einer Lizenz, die kommerzielle Nutzung ausschließt. gsplat (Nerfstudio-Projekt) ist Apache-2.0-lizenziert und speichersparender.

## Entscheidung
Wir nutzen gsplat (direkt oder über Nerfstudio Splatfacto). Inria-Code wird weder verwendet noch kopiert. Ein Mesh ist optionaler Zusatz.

## Konsequenzen
+ Beste visuelle Qualität, Web-Viewer verfügbar, kommerziell nutzbar
- Splats sind schlechter für Messungen → Mesh/Grundriss separat
