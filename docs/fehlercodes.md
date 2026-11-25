# Fehlercodes

Exitcodes aller Pipeline-Stufen (`drone-scan <stufe> --scan <dir>`), siehe Architektur §5.2. Ursache und Abhilfe stehen zusätzlich in `<scan>/<stufe>/manifest.json` (`error`, `hints`). Der vollständige Katalog mit allen Meldungen folgt in KAN-56 (US-041).

| Code | Status im Manifest | Bedeutung | Typische Abhilfe |
|---|---|---|---|
| 0 | `ok` | Stufe erfolgreich oder unverändert übersprungen (Cache) | – (`--force` rechnet neu) |
| 1 | `error` | Unerwarteter Fehler, fehlende GPU/PyTorch, fehlendes ffmpeg (`ingest`), Bilder nicht schreibbar oder Abbruch (Strg+C) | Meldung in `error` lesen. Bei GPU-Stufen: NVIDIA-Treiber prüfen (`nvidia-smi`). Bei `ingest`: ffmpeg installieren, freien Speicherplatz prüfen |
| 2 | – | Aufruffehler der CLI (z. B. unbekanntes Profil, fehlende Option) | Aufruf mit `--help` prüfen |
| 10 | `input_missing` | Eingabe oder Scan-Verzeichnis fehlt (z. B. Video nicht importiert, vorherige Stufe nicht gelaufen, Tippfehler in `--scan`) | Vorherige Stufe bzw. Import ausführen, Pfad mit `--scan` prüfen. Gilt auch für `flight add`: Video fehlt, defekt, ohne Videospur oder nicht unterstütztes Format, Scan nicht angelegt (`drone-scan scan create`). `ingest`: Scan ohne Video oder Video nicht dekodierbar (neu importieren) |
| 20 | `quality_insufficient` | Aufnahmequalität reicht nicht. `hints` enthält immer konkrete Hinweise | Hinweisen folgen (bei `ingest`: zu wenige verschiedene scharfe Bilder; z. B. langsamer filmen, Licht an, AprilTags) und neu aufnehmen |
| 30 | `oom` | Grafik- oder Arbeitsspeicher reicht nicht | Profil `low_vram_6gb`, andere GPU-Programme schließen, kürzeres Video. Die Job-Queue wiederholt einmal mit kleinerem Profil |
