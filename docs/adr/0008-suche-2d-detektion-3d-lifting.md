# ADR-0008: Gegenstandssuche über 2D-Detektion + 3D-Lifting

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Sprachfähige 3D-Felder (LERF, LangSplat u. ä.) brauchen viel VRAM und Trainingszeit. Open-Vocabulary-Detektoren auf Einzelbildern sind ausgereift und mit 6 GB betreibbar.

## Entscheidung
Pro Keyframe: Detektionen + CLIP-Embeddings → FAISS-Index. Anfrage: Text → Kandidatenframes → Detektion → Triangulation über Kameraposen → 3D-Marker mit Belegbildern.

## Konsequenzen
+ Wenig VRAM, erklärbar (Belegbilder), austauschbare Modelle
- Kleine/verdeckte Gegenstände nur bei ausreichender Bildauflösung
