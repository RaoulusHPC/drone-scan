# ADR-0003: COLMAP als SfM-Kern

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Benötigt werden robuste Kameraposen aus Videoframes. COLMAP (BSD-3) ist ausgereift, GPU-beschleunigt und das De-facto-Austauschformat für 3DGS-Werkzeuge. Feed-forward-Modelle (z. B. VGGT, MapAnything) sind vielversprechend für texturarme Räume, haben aber teils eingeschränkte Lizenzen und hohen VRAM-Bedarf.

## Entscheidung
COLMAP (inkl. globalem Mapper, sofern in der genutzten Version verfügbar) mit sequenziellem Matching und Loop-Detection. Alternativen werden nur nach dokumentierter Lizenz- und VRAM-Prüfung als Plug-in ergänzt.

## Konsequenzen
+ Bewährt, permissiv, Ökosystem
- Schwächen bei texturarmen Flächen → AprilTags, Capture Guidance, Matcher-Alternativen
