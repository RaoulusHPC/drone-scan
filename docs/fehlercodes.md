# Fehlercodes

Exitcodes aller Pipeline-Stufen (`drone-scan <stufe> --scan <dir>`), siehe Architektur §5.2. Ursache und Abhilfe stehen zusätzlich in `<scan>/<stufe>/manifest.json` (`error`, `hints`). Jede Fehlerursache hat zusätzlich einen stabilen Fehlercode (`<bereich>.<name>`, z. B. `sfm.no_model`), siehe [Katalog](#katalog-der-fehlercodes). Er steht im Manifest (`error_code`) und wird bei Fehlern in der CLI ausgegeben (`Fehlercode: ...`).

| Code | Status im Manifest | Bedeutung | Typische Abhilfe |
|---|---|---|---|
| 0 | `ok` | Stufe erfolgreich oder unverändert übersprungen (Cache) | – (`--force` rechnet neu) |
| 1 | `error` | Unerwarteter Fehler, fehlende GPU/PyTorch, GPU durch andere Stufe belegt (Lock-Timeout) oder GPU-Sperrdatei nicht nutzbar (Verzeichnis prüfen, `DRONESCAN_GPU_LOCK_DIR`), fehlendes ffmpeg (`ingest`), Docker oder COLMAP-Image fehlt bzw. COLMAP-Fehler (`sfm`), Bilder nicht schreibbar oder Abbruch (Strg+C) | Meldung in `error` lesen. Bei GPU-Stufen: NVIDIA-Treiber prüfen (`nvidia-smi`). GPU belegt: eine andere GPU-Stufe hält den GPU-Lock länger als `gpu.lock_timeout_s` (Profil, 1800 s); in diesem Fall wird ein vorhandenes Manifest nicht verändert; warten, bis sie fertig ist (`docker ps`, `nvidia-smi`), und neu starten oder `DRONESCAN_GPU__LOCK_TIMEOUT_S` erhöhen. Nach einem gekillten Lauf ggf. übrig gebliebene Docker-Prozesse beenden (COLMAP-Container von `sfm` enden selbst), sie halten den Lock nicht mehr. Bei `ingest`: ffmpeg installieren, freien Speicherplatz prüfen. Bei `sfm`: Docker starten, `docker compose build colmap`, `make docker-check`; COLMAP-Ausgaben stehen in `sfm/colmap/logs/`. Bei `recon`: Docker starten, `docker compose build worker colmap`; Trainer- und COLMAP-Ausgaben stehen in `recon/work/logs/`, die Ursache des Trainers steht in `error`. Training nach `splat.timeout_s` abgebrochen: andere GPU-Programme schließen, `DRONESCAN_SPLAT__TIMEOUT_S` erhöhen oder `DRONESCAN_SPLAT__MAX_ITERATIONS` senken |
| 2 | – | Aufruffehler der CLI (z. B. unbekanntes Profil, fehlende Option, unbekanntes oder ungültiges Kameraprofil bei `flight add --camera`) | Aufruf mit `--help` prüfen |
| 10 | `input_missing` | Eingabe oder Scan-Verzeichnis fehlt (z. B. Video nicht importiert, vorherige Stufe nicht gelaufen, Tippfehler in `--scan`) | Vorherige Stufe bzw. Import ausführen, Pfad mit `--scan` prüfen. Gilt auch für `flight add`: Video fehlt, defekt, ohne Videospur oder nicht unterstütztes Format, Scan nicht angelegt (`drone-scan scan create`). `ingest`: Scan ohne Video oder Video nicht dekodierbar (neu importieren). `sfm`: `ingest` nicht erfolgreich gelaufen oder Einzelbilder eines Flugs fehlen. `recon`: `sfm` nicht erfolgreich gelaufen |
| 20 | `quality_insufficient` | Aufnahmequalität reicht nicht. `hints` enthält immer konkrete Hinweise | Hinweisen folgen (bei `ingest`: zu wenige verschiedene scharfe Bilder, oder die Gesamtampel aus Schärfe/Belichtung/Bewegung/Abdeckung steht auf Rot, Einzelbewertung je Flug in `ingest/<flight_id>/quality.json`, Zusammenfassung in `ingest/manifest.json` unter `metrics.quality`; z. B. langsamer filmen, Licht an, AprilTags, KAN-34; bei `sfm`: weniger als 80 % der Einzelbilder im größten Teilmodell, die Messwerte stehen unter `metrics`; bei `recon`: weniger als zwei verortete Einzelbilder) und neu aufnehmen |
| 30 | `oom` | Grafik- oder Arbeitsspeicher reicht nicht | Profil `low_vram_6gb`, andere GPU-Programme schließen, kürzeres Video. `sfm` und `recon` wiederholen selbst einmal mit kleineren Werten (`oom_retry_*`; `recon` zusätzlich mit gsplat `packed`, Warnung im Manifest). Erst ein zweiter Speichermangel endet mit 30. Beendet der Kernel den Trainer-Container wegen Arbeitsspeichermangel (Docker-Code 137), endet `recon` ohne Wiederholung mit 30 (andere Programme schließen). Die Job-Queue (`drone-scan serve`) wiederholt die Stufe einmal mit dem Profilwert server.oom_retry_profile des Job-Profils (`default_8gb` → `low_vram_6gb`) |

### `drone-scan camera calibrate` (KAN-29)

Der Kalibrierbefehl ist keine Pipeline-Stufe (kein Manifest), nutzt aber dieselben Codes:

| Code | Bedeutung | Abhilfe |
|---|---|---|
| 0 | Profil geschrieben (`--output`, Standard `<name>.yaml`) | Mit `drone-scan flight add <scan> <video> --camera <datei>` verwenden |
| 1 | OpenCV fehlt, ChArUco ohne gültige `--marker-mm`, unbekanntes `--dictionary` | `uv sync --extra calibration`; Markergröße messen und angeben |
| 2 | Aufruffehler (ungültiger `--name`, `--pattern`, `--model`, Zieldatei existiert) | Aufruf prüfen, `--force` zum Ersetzen |
| 10 | Video fehlt oder ist nicht lesbar | Pfad prüfen, Video neu exportieren |
| 20 | Muster in zu wenigen Bildern gefunden (`calibration.min_views`) oder Restfehler zu groß (`calibration.max_rms_error_px`) | Muster scharf, flach und aus vielen Winkeln bis in die Bildecken filmen; `--cols`/`--rows` zählen Felder |

## Katalog der Fehlercodes

Quelle ist `src/dronescan/common/error_catalog.yaml`; die Tabelle wird daraus erzeugt (`uv run python -m dronescan.common.error_catalog --write`) und von einem Test auf Abweichungen geprüft. Neue Fehler brauchen zuerst einen Eintrag im Katalog. Die Spalte „Abhilfe“ zeigt die festen Hinweise; Meldungen nennen zusätzlich Details des konkreten Falls (z. B. Flug-ID, Zahlen).

<!-- BEGIN error-catalog (generated: python -m dronescan.common.error_catalog) -->
| Fehlercode | Exitcode | Ursache | Abhilfe |
|---|---|---|---|
| `capture.calibration_board_invalid` | 1 | Ungültige Mustereinstellung (ChArUco ohne gültige --marker-mm, unbekanntes --dictionary). | Abhilfe: Markergröße messen und angeben bzw. ein Wörterbuch wie DICT_4X4_50 oder DICT_5X5_100 verwenden. |
| `capture.calibration_ffmpeg_missing` | 1 | ffmpeg/ffprobe ist nicht installiert. | Abhilfe: ffmpeg installieren (z. B. `sudo apt install ffmpeg`). |
| `capture.calibration_opencv_missing` | 1 | Für `camera calibrate` fehlt OpenCV. | Abhilfe: das optionale Paket installieren mit `uv sync --extra calibration` (bzw. `pip install 'drone-scan[calibration]'`). |
| `capture.calibration_write_failed` | 1 | Das Kameraprofil konnte nicht geschrieben werden. | Abhilfe: Zielordner und Schreibrechte prüfen oder mit --output einen anderen Pfad angeben. |
| `cli.config_invalid` | 1 | Die Konfiguration (Profil plus Umgebungsvariablen DRONESCAN_*) enthält einen ungültigen Wert. | Abhilfe: den in der Meldung genannten Wert bzw. die Umgebungsvariable korrigieren. |
| `common.gpu_busy` | 1 | Eine andere GPU-Stufe hält die GPU-Sperre länger als gpu.lock_timeout_s (Profil, 1800 s). Ein vorhandenes Manifest wird dabei nicht verändert. | Abhilfe: warten, bis die andere Stufe fertig ist (`docker ps`, `nvidia-smi`), und neu starten oder DRONESCAN_GPU__LOCK_TIMEOUT_S erhöhen. |
| `common.gpu_lock_unusable` | 1 | Die GPU-Sperrdatei ist nicht nutzbar (Verzeichnis nicht beschreibbar). | Abhilfe: Verzeichnis prüfen oder DRONESCAN_GPU_LOCK_DIR setzen; die Meldung nennt Details. |
| `common.gpu_unavailable` | 1 | Kein nutzbares PyTorch/CUDA (fehlende GPU-Abhängigkeiten, Treiber oder GPU). | Abhilfe: NVIDIA-Treiber prüfen (`nvidia-smi`); die Meldung nennt die konkrete Abhilfe. |
| `common.interrupted` | 1 | Abgebrochen (Strg+C bzw. Prozess beendet). | Abhilfe: Stufe erneut starten; bereits erzeugte Dateien werden neu berechnet. |
| `common.outside_scan` | 1 | Eine Eingabe liegt außerhalb des Scan-Verzeichnisses. | Abhilfe: Dateien in das Scan-Verzeichnis importieren (Stufe 'import'). |
| `common.unexpected` | 1 | Unerwarteter Fehler in einer Stufe (nicht vorhergesehener Programmfehler). | Unerwarteter Fehler. Abhilfe: mit --force erneut ausführen; tritt er wieder auf, Log-Ausgabe und manifest.json an das Entwicklungsteam geben. |
| `common.write_failed` | 1 | Ergebnisse oder Einzelbilder konnten nicht geschrieben werden. | Abhilfe: freien Speicherplatz und Schreibrechte im Workspace prüfen. |
| `ingest.ffmpeg_missing` | 1 | ffmpeg ist nicht installiert. | Abhilfe: ffmpeg installieren (z. B. `sudo apt install ffmpeg`). |
| `models.download_failed` | 1 | Der Download ist fehlgeschlagen (HTTP-Fehler, keine Verbindung, größer als size_bytes oder Weiterleitung auf eine unsichere Adresse). | Abhilfe: Internetverbindung und Proxy prüfen, URL in models.lock prüfen, dann erneut `drone-scan models pull` ausführen. |
| `models.hash_mismatch` | 1 | Die SHA-256 der heruntergeladenen Datei weicht von models.lock ab. Der Download wird verworfen und der Lauf bricht vor dem nächsten Modell ab. | Abhilfe: später erneut versuchen; tritt es wieder auf, hat sich die Datei an der Quelle geändert und der Eintrag muss neu geprüft werden (PO, license-auditor). |
| `models.lock_invalid` | 1 | models.lock ist nicht lesbar oder ungültig (jedes Modell braucht name, version, source, url, sha256, weights_license und code_license). | Abhilfe: die in der Meldung genannte Stelle in models.lock korrigieren. |
| `models.write_failed` | 1 | Eine Datei im Modellverzeichnis ist nicht les- oder schreibbar. | Abhilfe: freien Speicherplatz und Schreibrechte prüfen (--models-dir bzw. DRONESCAN_MODELS_DIR). |
| `recon.image_unavailable` | 1 | Docker oder das Worker- bzw. COLMAP-Image fehlt, oder die GPU ist im Container nicht erreichbar. | Abhilfe: Docker starten und die Images bauen mit `docker compose build worker colmap`.<br>GPU im Container prüfen mit `make docker-check` (NVIDIA Container Toolkit, siehe README). |
| `recon.prepare_failed` | 1 | Entzerren, Bilder lesen oder Szene laden ist fehlgeschlagen, oder das Training schrieb keine Messwerte bzw. endete mit einem Fehler. | Abhilfe: Logdateien in recon/work/logs/ prüfen und mit --force erneut ausführen. |
| `recon.train_timeout` | 1 | Das Training wurde nach splat.timeout_s abgebrochen oder der Worker antwortet nicht. | Abhilfe: andere GPU-Programme schließen; auf langsamen Rechnern DRONESCAN_SPLAT__TIMEOUT_S erhöhen oder DRONESCAN_SPLAT__MAX_ITERATIONS senken. |
| `server.extra_missing` | 1 | Für `drone-scan serve` fehlen FastAPI bzw. uvicorn. | Abhilfe: das optionale Paket installieren mit `uv sync --extra server` (bzw. `pip install 'drone-scan[server]'`). |
| `server.host_not_allowed` | 1 | Anfrage abgelehnt: Sie kam nicht über localhost oder von einer fremden Webseite (Schutz vor DNS-Rebinding und Cross-Site-Anfragen). | Abhilfe: die Oberfläche über http://127.0.0.1:<port> bzw. http://localhost:<port> öffnen. |
| `server.job_conflict` | 1 | Für diesen Scan wartet oder läuft bereits ein Job. | Abhilfe: den laufenden Job abwarten oder abbrechen (POST /api/jobs/{id}/cancel). |
| `server.job_interrupted` | 1 | Der Server wurde beendet, während der Job lief. | Abhilfe: Job fortsetzen (POST /api/jobs/{id}/resume); fertige Stufen werden übersprungen. |
| `server.job_not_cancellable` | 1 | Der Job ist bereits beendet und kann nicht mehr abgebrochen werden. | Abhilfe: nichts zu tun; zum Fortsetzen POST /api/jobs/{id}/resume verwenden. |
| `server.job_not_resumable` | 1 | Nur fehlgeschlagene, abgebrochene oder unterbrochene Jobs lassen sich fortsetzen. | Abhilfe: den Job abwarten bzw. für einen erfolgreichen Scan einen neuen Job starten. |
| `server.stage_crashed` | 1 | Der Prozess einer Stufe endete, ohne ein Manifest zu schreiben (z. B. vom System beendet). | Abhilfe: Job fortsetzen (POST /api/jobs/{id}/resume); tritt es wieder auf, das Job-Log `<scan>/jobs/<job-id>.log` an das Entwicklungsteam geben. |
| `server.upload_failed` | 1 | Das Video konnte nicht gespeichert werden (Verbindung abgebrochen oder Speicher voll). | Abhilfe: freien Speicherplatz prüfen und den Upload wiederholen. |
| `server.upload_too_large` | 1 | Das hochgeladene Video ist größer als server.max_upload_gb (Profil). | Abhilfe: das Video kürzen bzw. in mehrere Flüge teilen oder DRONESCAN_SERVER__MAX_UPLOAD_GB erhöhen. |
| `sfm.colmap_failed` | 1 | COLMAP ist fehlgeschlagen oder antwortet nicht. | Abhilfe: Logdateien in sfm/colmap/logs/ prüfen und mit --force erneut ausführen. |
| `sfm.image_unavailable` | 1 | Docker oder das COLMAP-Image fehlt, oder die GPU ist im Container nicht erreichbar. | Abhilfe: Docker starten und das Image bauen mit `docker compose build colmap`.<br>GPU im Container prüfen mit `make docker-check` (NVIDIA Container Toolkit, siehe README). |
| `capture.camera_profile_invalid` | 2 | Das Kameraprofil bei `flight add --camera` ist unbekannt, nicht lesbar oder ungültig. | Abhilfe: Name oder Pfad prüfen (`drone-scan camera list`) oder neu kalibrieren. |
| `cli.usage` | 2 | Aufruffehler der CLI (unbekanntes Profil, fehlende oder ungültige Option, Zieldatei existiert). | Abhilfe: Aufruf mit `--help` prüfen. |
| `models.unknown_name` | 2 | Der Modellname steht nicht in models.lock (es wird nichts geladen). | Abhilfe: nur gelistete Modelle laden; neue Einträge prüft der PO (Lizenz der Gewichte). |
| `server.invalid_request` | 2 | Die Anfrage an die API ist ungültig (fehlendes oder falsches Feld, unbekannte Stufe). | Abhilfe: Anfrage mit der API-Beschreibung abgleichen (/api/openapi.json). |
| `server.not_loopback` | 2 | `serve` darf nur an localhost gebunden werden (Zugriff aus dem LAN folgt mit HTTPS, KAN-54). | Abhilfe: --host weglassen oder 127.0.0.1, ::1 bzw. localhost angeben. |
| `server.profile_unknown` | 2 | Das für den Job angegebene Profil gibt es nicht. | Abhilfe: ein vorhandenes Profil wählen, z. B. low_vram_6gb (`drone-scan config`). |
| `capture.calibration_video_unreadable` | 10 | Das Kalibriervideo fehlt oder ist nicht lesbar. | Abhilfe: Video prüfen oder neu exportieren (MP4/MOV mit H.264/H.265). |
| `capture.scan_invalid` | 10 | `flight add`: Scan nicht angelegt, Video fehlt, ist defekt, hat keine Videospur oder ein nicht unterstütztes Format. | Abhilfe: Scan anlegen (`drone-scan scan create`), Pfad prüfen oder das Video neu exportieren. |
| `common.input_missing` | 10 | Eine Eingabe der Stufe fehlt oder der Scan ist nicht lesbar (vorherige Stufe nicht gelaufen). | Abhilfe: vorherige Stufe bzw. den Import für diesen Scan ausführen und den Pfad mit --scan prüfen. |
| `common.no_flight` | 10 | Der Scan enthält noch kein Video. | Abhilfe: Video importieren mit `drone-scan flight add <scan> <video>`. |
| `common.scan_dir_missing` | 10 | Das mit --scan angegebene Scan-Verzeichnis existiert nicht. | Abhilfe: Pfad mit --scan prüfen bzw. den Scan zuerst anlegen. |
| `ingest.video_broken` | 10 | Das Video kann nicht (vollständig) dekodiert werden. | Abhilfe: Video am Rechner abspielen und neu importieren, ggf. vorher neu exportieren (MP4/MOV mit H.264/H.265). |
| `models.lock_missing` | 10 | models.lock wurde nicht gefunden. | Abhilfe: im Repository-Verzeichnis ausführen oder den Pfad mit --lock bzw. DRONESCAN_MODELS_LOCK angeben. |
| `models.not_available` | 10 | Ein von einer Stufe benötigtes Modell steht nicht in models.lock, ist nicht heruntergeladen, nicht lesbar oder beschädigt. | Abhilfe: einmalig `drone-scan models pull <name>` ausführen (einziger Netzwerkzugriff). |
| `recon.sfm_missing` | 10 | Die Kameraposen fehlen, weil 'sfm' nicht erfolgreich gelaufen ist. | Abhilfe: vorherige Stufe bzw. den Import für diesen Scan ausführen und den Pfad mit --scan prüfen. |
| `server.job_not_found` | 10 | Den angefragten Job gibt es nicht. | Abhilfe: Job-ID prüfen (GET /api/jobs). |
| `server.result_not_found` | 10 | Für diese Stufe liegt (noch) kein Ergebnis vor. | Abhilfe: zuerst einen Job mit dieser Stufe starten und abwarten, bis er fertig ist. |
| `server.scan_not_found` | 10 | Den angefragten Scan gibt es im Workspace nicht. | Abhilfe: Scan-ID prüfen (GET /api/scans) oder den Scan zuerst anlegen. |
| `sfm.frames_missing` | 10 | Die Einzelbilder fehlen, weil 'ingest' nicht erfolgreich gelaufen ist oder Dateien eines Flugs fehlen. | Abhilfe: vorherige Stufe bzw. den Import für diesen Scan ausführen und den Pfad mit --scan prüfen. |
| `capture.calibration_quality` | 20 | Das Muster wurde in zu wenigen Bildern gefunden (calibration.min_views), die Kalibrierung ist nicht konvergiert oder der Restfehler ist zu groß (calibration.max_rms_error_px). | Muster vollständig, scharf und gut beleuchtet filmen; die Kamera langsam bewegen.<br>Aus verschiedenen Winkeln und Abständen filmen, das Muster auch in die Bildecken und an die Bildränder bringen.<br>Muster flach auf eine feste Unterlage kleben (nicht gewellt).<br>Mustergröße prüfen: --cols und --rows zählen die Felder (Quadrate), nicht die inneren Ecken; bei ChArUco auch --dictionary prüfen. |
| `ingest.quality_red` | 20 | Die Gesamtampel aus Schärfe, Belichtung, Bewegung und Abdeckung steht auf Rot (Einzelbewertung je Flug in ingest/<flight_id>/quality.json). | Die Hinweise der Meldung zu den roten Kriterien befolgen und neu aufnehmen. |
| `ingest.too_few_frames` | 20 | Zu wenige verschiedene, scharfe Einzelbilder (Mindestzahl im Profil, ingest.min_keyframes). | Kamera während der Aufnahme langsam und stetig durch den Raum bewegen (≤ 0,5 m/s).<br>Länger aufnehmen: mindestens 30 Sekunden pro Raum.<br>Licht einschalten, damit die Bilder nicht verwackeln. |
| `recon.few_images` | 20 | Weniger als zwei verortete Einzelbilder, zu wenig für das Training. | Mehr Einzelbilder aufnehmen: langsamer und länger filmen.<br>Die Stufe 'sfm' muss mindestens zwei Einzelbilder verorten. |
| `sfm.no_model` | 20 | Aus den Einzelbildern entstand kein 3D-Modell, oder weniger als 80 % der Einzelbilder liegen im größten Teilmodell (Messwerte unter metrics). | Licht einschalten und möglichst bei Tageslicht aufnehmen.<br>Langsamer bewegen (≤ 0,5 m/s) und 1 bis 1,5 m Abstand zu den Wänden halten.<br>Türdurchgänge doppelt aufnehmen (hin und zurück).<br>Bei weißen Wänden, Glas oder Spiegeln 2 bis 4 AprilTags aufhängen. |
| `common.gpu_oom` | 30 | Der Grafikspeicher (VRAM) hat nicht gereicht, auch nicht nach der automatischen Wiederholung mit kleineren Werten. | Der Grafikspeicher hat nicht gereicht. Abhilfe: Profil 'low_vram_6gb' verwenden, andere GPU-Programme schließen oder ein kürzeres Video bzw. weniger Räume pro Scan verarbeiten. |
| `common.ram_oom` | 30 | Der Arbeitsspeicher (RAM) hat nicht gereicht, oder der Kernel hat den Container beendet (Docker-Code 137). | Der Arbeitsspeicher hat nicht gereicht. Abhilfe: andere Programme schließen oder ein kürzeres Video bzw. weniger Räume pro Scan verarbeiten. |
<!-- END error-catalog -->

### `drone-scan models pull` / `models list` (KAN-24)

`models pull [NAME…]` ist der **einzige Netzwerkzugriff** von drone-scan (CLAUDE.md Regel 3). Er lädt nur Modelle aus `models.lock` nach `models/<name>/<version>/` (`--models-dir` bzw. `DRONESCAN_MODELS_DIR`) und prüft jede Datei per SHA-256. Erst eine geprüfte Datei wird an ihren Platz verschoben. `models list` zeigt Status (`ok`, `fehlt`, `beschädigt`) und Lizenzen und läuft offline. Beide sind keine Pipeline-Stufen (kein Manifest), nutzen aber dieselben Codes:

Exitcodes: 0 alle angeforderten Modelle geprüft vorhanden (`--force` lädt neu), 1 Download- oder Prüffehler bzw. ungültige `models.lock`, 2 Modellname nicht in `models.lock` (es wird nichts geladen), 10 `models.lock` nicht gefunden. Ursache und Abhilfe stehen im Katalog oben unter `models.*`; die CLI gibt den Fehlercode aus.

Stufen lesen Gewichte nur über `resolve_model` und laden nie selbst nach. Fehlt ein Modell oder ist es beschädigt, meldet `resolve_model` einen Fehler mit dem Hinweis `drone-scan models pull <name>`. Die erste Stufe mit Modellen (KAN-46) bildet das auf Exit 10 ab. `models.lock` erlaubt `http://` nur für 127.0.0.1/localhost (lokaler Test-Server oder Mirror), das Lizenz-Gate warnt dann. Weiterleitungen werden nur auf zulässige Adressen verfolgt.
