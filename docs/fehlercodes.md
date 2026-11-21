# Fehlercodes

Exitcodes aller Pipeline-Stufen (`drone-scan <stufe> --scan <dir>`), siehe Architektur §5.2. Ursache und Abhilfe stehen zusätzlich in `<scan>/<stufe>/manifest.json` (`error`, `hints`). Der vollständige Katalog mit allen Meldungen folgt in KAN-56 (US-041).

| Code | Status im Manifest | Bedeutung | Typische Abhilfe |
|---|---|---|---|
| 0 | `ok` | Stufe erfolgreich oder unverändert übersprungen (Cache) | – (`--force` rechnet neu) |
| 1 | `error` | Unerwarteter Fehler, fehlende GPU/PyTorch oder Abbruch (Strg+C) | Meldung in `error` lesen. Bei GPU-Stufen: NVIDIA-Treiber prüfen (`nvidia-smi`) |
| 2 | – | Aufruffehler der CLI (z. B. unbekanntes Profil, fehlende Option) | Aufruf mit `--help` prüfen |
| 10 | `input_missing` | Eingabe oder Scan-Verzeichnis fehlt (z. B. Video nicht importiert, vorherige Stufe nicht gelaufen, Tippfehler in `--scan`) | Vorherige Stufe bzw. Import ausführen, Pfad mit `--scan` prüfen |
| 20 | `quality_insufficient` | Aufnahmequalität reicht nicht. `hints` enthält immer konkrete Hinweise | Hinweisen folgen (z. B. langsamer filmen, Licht an, AprilTags) und neu aufnehmen |
| 30 | `oom` | Grafik- oder Arbeitsspeicher reicht nicht | Profil `low_vram_6gb`, andere GPU-Programme schließen, kürzeres Video. Die Job-Queue wiederholt einmal mit kleinerem Profil |
