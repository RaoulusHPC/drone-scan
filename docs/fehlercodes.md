# Fehlercodes

Exitcodes aller Pipeline-Stufen (`drone-scan <stufe> --scan <dir>`), siehe Architektur §5.2. Ursache und Abhilfe stehen zusätzlich in `<scan>/<stufe>/manifest.json` (`error`, `hints`). Der vollständige Katalog mit allen Meldungen folgt in KAN-56 (US-041).

| Code | Status im Manifest | Bedeutung | Typische Abhilfe |
|---|---|---|---|
| 0 | `ok` | Stufe erfolgreich oder unverändert übersprungen (Cache) | – (`--force` rechnet neu) |
| 1 | `error` | Unerwarteter Fehler, fehlende GPU/PyTorch, GPU durch andere Stufe belegt (Lock-Timeout) oder GPU-Sperrdatei nicht nutzbar (Verzeichnis prüfen, `DRONESCAN_GPU_LOCK_DIR`), fehlendes ffmpeg (`ingest`), Docker oder COLMAP-Image fehlt bzw. COLMAP-Fehler (`sfm`), Bilder nicht schreibbar oder Abbruch (Strg+C) | Meldung in `error` lesen. Bei GPU-Stufen: NVIDIA-Treiber prüfen (`nvidia-smi`). GPU belegt: eine andere GPU-Stufe hält den GPU-Lock länger als `gpu.lock_timeout_s` (Profil, 1800 s); in diesem Fall wird ein vorhandenes Manifest nicht verändert; warten, bis sie fertig ist (`docker ps`, `nvidia-smi`), und neu starten oder `DRONESCAN_GPU__LOCK_TIMEOUT_S` erhöhen. Nach einem gekillten Lauf ggf. übrig gebliebene Docker-Prozesse beenden (COLMAP-Container von `sfm` enden selbst), sie halten den Lock nicht mehr. Bei `ingest`: ffmpeg installieren, freien Speicherplatz prüfen. Bei `sfm`: Docker starten, `docker compose build colmap`, `make docker-check`; COLMAP-Ausgaben stehen in `sfm/colmap/logs/`. Bei `recon`: Docker starten, `docker compose build worker colmap`; Trainer- und COLMAP-Ausgaben stehen in `recon/work/logs/`, die Ursache des Trainers steht in `error`. Training nach `splat.timeout_s` abgebrochen: andere GPU-Programme schließen, `DRONESCAN_SPLAT__TIMEOUT_S` erhöhen oder `DRONESCAN_SPLAT__MAX_ITERATIONS` senken |
| 2 | – | Aufruffehler der CLI (z. B. unbekanntes Profil, fehlende Option, unbekanntes oder ungültiges Kameraprofil bei `flight add --camera`) | Aufruf mit `--help` prüfen |
| 10 | `input_missing` | Eingabe oder Scan-Verzeichnis fehlt (z. B. Video nicht importiert, vorherige Stufe nicht gelaufen, Tippfehler in `--scan`) | Vorherige Stufe bzw. Import ausführen, Pfad mit `--scan` prüfen. Gilt auch für `flight add`: Video fehlt, defekt, ohne Videospur oder nicht unterstütztes Format, Scan nicht angelegt (`drone-scan scan create`). `ingest`: Scan ohne Video oder Video nicht dekodierbar (neu importieren). `sfm`: `ingest` nicht erfolgreich gelaufen oder Einzelbilder eines Flugs fehlen. `recon`: `sfm` nicht erfolgreich gelaufen |
| 20 | `quality_insufficient` | Aufnahmequalität reicht nicht. `hints` enthält immer konkrete Hinweise | Hinweisen folgen (bei `ingest`: zu wenige verschiedene scharfe Bilder, oder die Gesamtampel aus Schärfe/Belichtung/Bewegung/Abdeckung steht auf Rot, Einzelbewertung je Flug in `ingest/<flight_id>/quality.json`, Zusammenfassung in `ingest/manifest.json` unter `metrics.quality`; z. B. langsamer filmen, Licht an, AprilTags, KAN-34; bei `sfm`: weniger als 80 % der Einzelbilder im größten Teilmodell, die Messwerte stehen unter `metrics`; bei `recon`: weniger als zwei verortete Einzelbilder) und neu aufnehmen |
| 30 | `oom` | Grafik- oder Arbeitsspeicher reicht nicht | Profil `low_vram_6gb`, andere GPU-Programme schließen, kürzeres Video. `sfm` und `recon` wiederholen selbst einmal mit kleineren Werten (`oom_retry_*`; `recon` zusätzlich mit gsplat `packed`, Warnung im Manifest). Erst ein zweiter Speichermangel endet mit 30. Beendet der Kernel den Trainer-Container wegen Arbeitsspeichermangel (Docker-Code 137), endet `recon` ohne Wiederholung mit 30 (andere Programme schließen). Die Job-Queue wiederholt einmal mit kleinerem Profil |

### `drone-scan camera calibrate` (KAN-29)

Der Kalibrierbefehl ist keine Pipeline-Stufe (kein Manifest), nutzt aber dieselben Codes:

| Code | Bedeutung | Abhilfe |
|---|---|---|
| 0 | Profil geschrieben (`--output`, Standard `<name>.yaml`) | Mit `drone-scan flight add <scan> <video> --camera <datei>` verwenden |
| 1 | OpenCV fehlt, ChArUco ohne gültige `--marker-mm`, unbekanntes `--dictionary` | `uv sync --extra calibration`; Markergröße messen und angeben |
| 2 | Aufruffehler (ungültiger `--name`, `--pattern`, `--model`, Zieldatei existiert) | Aufruf prüfen, `--force` zum Ersetzen |
| 10 | Video fehlt oder ist nicht lesbar | Pfad prüfen, Video neu exportieren |
| 20 | Muster in zu wenigen Bildern gefunden (`calibration.min_views`) oder Restfehler zu groß (`calibration.max_rms_error_px`) | Muster scharf, flach und aus vielen Winkeln bis in die Bildecken filmen; `--cols`/`--rows` zählen Felder |
