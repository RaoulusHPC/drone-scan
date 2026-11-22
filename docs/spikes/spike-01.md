# Spike 01: COLMAP + gsplat auf der GTX 1060 (KAN-19, US-004)

- **Datum:** 21./22.11.2026
- **Hardware:** GTX 1060 6 GiB (Pascal, `sm_61`, steuert den Bildschirm: ca. 1,0 GiB belegt), `CUDA_VISIBLE_DEVICES=0`, fp32. Die RTX 2070 SUPER wurde nicht benutzt.
- **Software:** Images aus KAN-18 (CUDA 12.6.3, PyTorch 2.14.1+cu126, gsplat v1.5.3 mit Pascal-Patch, COLMAP 4.2.1). PyTorch-Allocator-Grenze `DRONESCAN_VRAM_LIMIT_GB=5.0`.
- **Daten:** zwei private Smartphone-Clips (je ca. 86 s, H.264, 30 fps variabel), **nur lokal verarbeitet**. Dieser Bericht enthält nur Zahlen und Text: keine Frames, Splats, Screenshots oder Renderings. Die Sichtprüfung der Ergebnisse macht der PO lokal (siehe Abschnitt 8).
- **Werkzeuge:** `tools/spike01/` (nur Code; Aufruf siehe Abschnitt 9).

## 1. Ergebnis in Kürze

| Frage | Ergebnis |
|---|---|
| **AK2:** gsplat-Training auf Pascal (`sm_61`, fp32) | **Nachgewiesen.** 7000 Iterationen laufen durch, mit Default- und MCMC-Strategie, ohne Fallback. Das schließt den offenen Punkt aus R-13/ADR-0010. |
| **R-03:** reichen 6 GiB? | **Ja, mit großem Abstand.** Training bei 1,0 Mio. Gaussians: 0,8 bis 1,5 GiB Allocator-Peak, **höchstens 2,5 GiB** GPU-Gesamtverbrauch inkl. CUDA-Kontext (Q-2: ≤ 5,5 GiB). Obergrenze auf der 1060: ca. 3 Mio. Gaussians bei 1200 px. |
| **Q-3 / R-01:** ≥ 90 % registrierte Keyframes | **Clip 1: 100 %** (164/164). **Clip 2: nein**, beste Variante 58,5 % (138/236), Modell zerfällt in 4 bis 5 Teile. R-01 tritt ein. Keine der getesteten Gegenmaßnahmen reicht. |
| **Q-1:** ≤ 60 min Gesamtlaufzeit | Auf der 1060 **nicht belegbar**: Training allein braucht für 30 000 Iterationen bei 1200 px 67 min. Es fehlt eine Messung auf der RTX 2060 (siehe Abschnitt 8). |
| **AK1:** 3 Referenzräume | **Nur 2 Clips** (Keller und Treppenhaus, keiner passt genau auf die drei Kategorien). Der PO hat entschieden, den Spike mit diesen zwei Räumen abzuschließen. |
| **Qualität (AK3)** | Objektiv gemessen (PSNR/SSIM auf zurückgehaltenen Frames). **Subjektive Qualität (Screenshots) steht aus** und wird vom PO lokal beurteilt, weil Bildinhalte die Maschine nicht verlassen sollen. |

**Empfehlung für `low_vram_6gb`:** MCMC-Strategie, Trainingsbreite 1200 px, Obergrenze 1,0 Mio. Gaussians, erschöpfendes Matching bei kurzen Clips (Abschnitt 7).

## 2. Methode

Je Clip eine Kette aus Ingest, SfM und Training, nacheinander (nie zwei GPU-Jobs gleichzeitig, Dateisperre im Runner):

1. **Ingest** (ffmpeg auf dem Host): Frames mit angewandter Rotation dekodieren. Zuerst 6 Analysebilder pro Sekunde (klein, grau), Schärfe = Varianz des Laplace-Operators. Je Fenster (Standard: 3 Analysebilder = 2 Keyframes/s) wird das schärfste Bild gewählt. Fenster, deren bestes Bild unter 40 % der Median-Schärfe liegt, entfallen (Bewegungsunschärfe). Ausgabe als JPEG, lange Kante 1600 px.
2. **SfM** (Container `colmap`): SIFT auf der GPU (8192 Features), eine gemeinsame Kamera (SIMPLE_RADIAL), Matching, Mapper (CPU). Registrierungsquote = Frames im größten Modell / extrahierte Keyframes.
3. **Training** (Container `worker`): COLMAP-Undistortion (PINHOLE), eigener schlanker Trainer mit `gsplat.rasterization` und `DefaultStrategy` bzw. `MCMCStrategy`, SH-Grad 3, L1 + 0,2·(1−SSIM). Jedes 8. Bild wird zurückgehalten. Metriken: PSNR/SSIM (eigene Implementierung) auf den zurückgehaltenen Bildern.
4. **Messung:** Laufzeit je Stufe (Wandzeit, inkl. Container-Start von ca. 1 bis 2 s). GPU-Speicher: `nvidia-smi`-Gesamtverbrauch der GPU 0 während der Stufe (Abtastung alle 0,4 s) **abzüglich Bildschirm-Baseline (ca. 1,0 GiB)**. Das enthält den CUDA-Kontext. Dazu beim Training `torch.cuda.max_memory_allocated/reserved`.

## 3. Eingangsdaten

| | Clip 1 | Clip 2 |
|---|---|---|
| Kodierte Größe / Rotations-Flag | 1080×1920 / 90° | 1080×1920 / 90° |
| **Größe nach Rotation (ffmpeg-Standard)** | **1920×1080 (Querformat)** | **1920×1080 (Querformat)** |
| Dauer | 85,9 s | 86,5 s |
| Schärfe (Laplace-Varianz, Median / Min / Max) | 582 / 51 / 1914 | 335 / 20 / 1209 |

**Befund zur Ausrichtung:** Die Angabe „Hochformat 1080×1920“ beschreibt die *kodierte* Größe. Das Rotations-Flag dreht die Anzeige auf Querformat 1920×1080. ffmpeg wendet die Rotation standardmäßig an; die Pipeline (Story zu `ingest`) muss das Flag berücksichtigen.

Clip 2 ist deutlich weniger scharf (Median-Schärfe 335 gegenüber 582) und verliert 26 von 173 Fenstern durch den Schärfefilter, Clip 1 nur 8 von 172. Welcher Clip Keller und welcher Treppenhaus ist, lässt sich aus den Messwerten nicht ableiten; sie heißen hier neutral Clip 1 und Clip 2.

## 4. SfM und Laufzeiten (Standardparameter)

| | Clip 1 | Clip 2 |
|---|---|---|
| Keyframes (Fenster gesamt) | 164 (172) | 147 (173) |
| **Registriert** | **164 (100 %)**, 1 Modell | **41 (27,9 %)**, 5 Modelle (41/12/22/26/22) |
| Ingest (ffmpeg, Host) | 45 s | 53 s |
| SIFT-Extraktion (GPU) | 18,5 s, +1,19 GiB GPU | 16,7 s, +1,19 GiB GPU |
| Matching (sequentiell, Overlap 10, GPU) | 150 s, +1,15 GiB GPU | 139 s, +0,91 GiB GPU |
| Mapper (CPU, Bündelausgleich ohne GPU) | 593 s | 83 s |
| Undistortion + Rohbildumwandlung (Host/CPU) | 26 s + 6 s | 18 s + 4 s (bestes Modell, Abschnitt 5) |

Die Mapper-Zeit ist bei Clip 2 nur deshalb kurz, weil das Modell in kleine Teile zerfällt.

## 5. Clip 2: Gegenmaßnahmen (Timebox: 3 bis 4)

Alle Varianten laufen auf Clip 2. Die Zeilen 4 bis 6 ändern je eine Sache gegenüber der Basis „erschöpfendes Matching“ (Zeile 3), die letzte zwei (Keyframe-Rate und Schärfefilter).

| Variante | Keyframes | Registriert | % | Modelle | Matching | Mapper |
|---|---|---|---|---|---|---|
| Standard (2/s, sequentiell, Overlap 10) | 147 | 41 | 27,9 | 41/12/22/26/22 | 139 s | 83 s |
| 3/s, sequentiell, Overlap 20 | 213 | 84 | 39,4 | 84/14/40/35/31 | 225 s | 180 s |
| **erschöpfendes Matching** | 147 | 79 | 53,7 | 2/79/11/22/26 | 452 s | 85 s |
| erschöpfend + 16 384 Features + `peak_threshold` 0,003 | 147 | 83 | 56,5 | 83/10/26/22 | 710 s | 150 s |
| erschöpfend + 1920 px + 16 384 Features | 147 | 78 | 53,1 | 78/22/26 | 377 s | 154 s |
| erschöpfend + `init_min_tri_angle` 6, `init_min_num_inliers` 50 | 147 | 82 | 55,8 | 82/22/11/26 | 532 s | 102 s |
| erschöpfend + 3/s + Schärfefilter 0,2 | 236 | **138** | **58,5** | 138/15/40/34 | 1073 s | 276 s |

**Lesart**
- Den größten Sprung bringt das **erschöpfende Matching** (27,9 % → 53,7 %). Ein größeres sequentielles Fenster oder dichtere Frames allein helfen weniger (39,4 %).
- Alles Weitere (mehr Features, niedrigere Peak-Schwelle, native Auflösung, lockerere Mapper-Parameter) bringt nur **0 bis 3 Prozentpunkte** und kostet Laufzeit. Die native Auflösung bringt nichts (53,1 %) und erhöht den GPU-Speicher der Extraktion von 1,19 auf 1,65 GiB.
- Dichtere Frames mit weniger strengem Schärfefilter geben die beste Quote (58,5 %), aber bei 3,2-facher Matching-Zeit.
- **Kein Ansatz erreicht 80 % (Q-3: 90 %).** Die Quote scheint bei etwa 55 bis 60 % zu sättigen. Das Modell bleibt in 3 bis 5 Teile zerstückelt. Die Ursache liegt vermutlich in der Szene (wenig Licht, wenig Textur, wiederkehrende Strukturen, geringere Schärfe) und nicht in den COLMAP-Parametern. Das ist aus den Messwerten **nicht bewiesen**: Ob es an der Aufnahme oder an SIFT liegt, zeigt erst ein Vergleich mit anderen Merkmalen (siehe Abschnitt 8).
- Nicht getestet (außerhalb der Timebox): Zusammenführen der Teilmodelle (`model_merger`), ALIKED + LightGlue, AprilTags, andere Kameramodelle.

## 6. Training auf der GTX 1060 (fp32, 7000 Iterationen, SH-Grad 3)

**Clip 1** (143 Trainings-, 21 zurückgehaltene Bilder, 44 393 Start-Punkte aus SfM):

| Variante | Gaussians | PSNR dB | SSIM | s/Iteration | 30 000 Iter. | Allocator-Peak | reserviert | GPU gesamt |
|---|---|---|---|---|---|---|---|---|
| Default, 1200 px | 1 005 661 | 25,53 | 0,880 | 0,158 | 79 min | 1,52 GiB | 2,27 GiB | 1,81 GiB |
| Default, 1600 px | 954 930 | 25,37 | 0,887 | 0,249 | 124 min | 1,46 GiB | 2,52 GiB | 2,45 GiB |
| MCMC (Cap 1 Mio.), 1200 px | 508 981 | 26,21 | 0,894 | 0,138 | 69 min | 0,78 GiB | 0,95 GiB | 1,07 GiB |
| MCMC (Cap 1,5 Mio.), 1200 px | 508 981 | 26,38 | 0,894 | 0,139 | 69 min | 0,78 GiB | 0,93 GiB | 1,05 GiB |
| MCMC (Cap 0,5 Mio.), 960 px | 500 000 | 26,44 | 0,893 | 0,097 | 48 min | 0,70 GiB | 1,02 GiB | 1,14 GiB |
| **MCMC (Cap 1 Mio.), 1200 px, Wiederholung** | 508 981 | **26,54** | **0,894** | 0,134 | 67 min | 0,78 GiB | 0,93 GiB | 1,11 GiB |

**Clip 2**, bestes Teilmodell (138 Bilder: 120 Training, 18 zurückgehalten; 20 144 Start-Punkte), MCMC, 1200 px: 230 908 Gaussians, **PSNR 23,78 dB, SSIM 0,859**, 0,119 s/Iteration (59 min für 30 000), Allocator-Peak 0,45 GiB, GPU gesamt 0,92 GiB.

**Lesart**
- **gsplat trainiert auf Pascal ohne Fallback.** Der Trainer prüft Posen, Kameramatrix und Aufteilung zusätzlich in einem Test auf synthetischen Daten (28 dB nach 300 Iterationen).
- **MCMC schlägt Default** in Qualität (≈ +0,7 bis +1,0 dB, SSIM +0,014) und Speicher (halbe Gaussian-Zahl, halber Peak) bei geringerer Zeit pro Iteration. Unterschiede unter etwa 0,2 dB sind nicht aussagekräftig: Die drei MCMC-Läufe mit 1200 px haben dieselben 508 981 Gaussians, streuen aber um 0,3 dB (Seed fest, `atomicAdd`-Reihenfolge ist nicht deterministisch). Jede Variante ist ein einzelner Lauf.
- **Die Caps 1 Mio. und 1,5 Mio. wurden nie erreicht:** MCMC wächst in 7000 Iterationen nur auf 509 000 Gaussians (vermutlich begrenzt die Wachstumsrate je Aufräumschritt, das habe ich nicht geprüft). Nur der 0,5-Mio.-Lauf stieß an sein Cap. Ob mehr Gaussians bessere Qualität bringen, ist damit **nicht gemessen**.
- **1600 px statt 1200 px** kostet 58 % mehr Zeit und bringt keine bessere Qualität (Default: 25,37 gegenüber 25,53 dB). 960 px ist 28 % schneller als 1200 px bei gleicher Qualität (MCMC: 26,44 gegenüber 26,2 bis 26,5 dB).
- Zurückgehaltene Frames liegen 1,6 bis 3,7 dB unter den Trainingsansichten (Clip 1: MCMC 26,5 gegenüber 28,1 dB, Default 25,5 gegenüber 29,2 dB). Der Abstand ist bei Default größer, das Modell passt sich stärker an die Trainingsbilder an.
- **Clip 2** erreicht auf seinem Teilmodell 2,8 dB weniger als Clip 1. Die Rekonstruktion deckt nur etwa 58 % der Aufnahme ab, die Szene ist schwieriger.

### Speicher-Stresstest (Clip 1, Start mit N zufälligen Gaussians, 30 Iterationen, Default-Strategie)

| Breite | Start-Gaussians | Ergebnis | Allocator-Peak | reserviert | GPU gesamt | s/Iteration |
|---|---|---|---|---|---|---|
| 1200 | 1 Mio. | ok | 1,56 GiB | 1,68 GiB | 1,81 GiB | 0,23 |
| 1200 | 2 Mio. | ok | 2,74 GiB | 3,42 GiB | 3,55 GiB | 0,28 |
| 1200 | 3 Mio. | ok | 3,90 GiB | 4,47 GiB | 4,59 GiB | 0,33 |
| 1200 | 4 Mio. | **OOM** | 3,72 GiB | 4,49 GiB | 4,61 GiB | – |
| 1200 | 6 Mio. | **OOM** | 4,36 GiB | 4,61 GiB | 4,73 GiB | – |
| 1600 | 2 Mio. | ok | 3,16 GiB | 3,34 GiB | 3,46 GiB | 0,40 |
| 1600 | 4 Mio. | **OOM** | 3,84 GiB | 4,06 GiB | 4,14 GiB | – |

Der Speicher wächst etwa linear (ca. 1,2 GiB je Million Gaussians bei 1200 px plus ca. 0,4 GiB Grundlast). Der OOM tritt auf, bevor die 5,0-GiB-Allocator-Grenze greift: Auf der 1060 sind wegen Bildschirm (ca. 1,0 GiB) und CUDA-Kontext nur **ca. 4,6 bis 4,7 GiB** nutzbar, das ist die harte Grenze dieses Entwicklungsrechners. Auf der Zielkarte RTX 2060 (Q-2: ≤ 5,5 GiB gesamt) ist mehr Platz, wenn dort kein Bildschirm hängt; das ist nicht gemessen.

## 7. Empfehlung für die Profilparameter `low_vram_6gb`

| Parameter | Aktuell (`low_vram_6gb.yaml`) | Empfehlung | Begründung |
|---|---|---|---|
| `splat.strategy` (neu) | – | `mcmc` | bessere Qualität bei halbem Speicher und kürzerer Laufzeit (Abschnitt 6) |
| `splat.max_image_edge_px` (neu, Trainingsbreite) | – | **1200** (960 als Schnellmodus) | 1600 px kostet 58 % mehr Zeit ohne Qualitätsgewinn |
| `splat.max_gaussians` | 1 000 000 | **1 000 000 beibehalten**; 1 500 000 ist zulässig | Bei 1,5 Mio. liegt der GPU-Gesamtverbrauch laut Interpolation aus dem Stresstest bei ca. 2,7 GiB (nicht direkt gemessen), die Bedingung aus dem Profilkommentar (≤ 5,5 GiB) ist erfüllt. Ein Qualitätsgewinn über 1 Mio. ist nicht gemessen, kein Grund zur Erhöhung. Harte Grenze auf der 1060: ca. 3 Mio. |
| `splat.max_iterations` | 30 000 | **15 000 als Standard** (30 000 als Obergrenze) | 30 000 Iterationen kosten bei 1200 px 67 min auf der 1060; mit den Stufen davor reißt das Q-1 (≤ 60 min). Die Qualitätskurve über die Iterationen ist nicht gemessen und gehört in die `recon`-Story |
| `gpu.allow_fp16` | `true` (nur sm_70+) | unverändert | auf Pascal fp32 (gemessen), sm_75 nicht gemessen |
| `sfm.max_image_edge_px` | 1600 | **1600 beibehalten** | größere Bilder erhöhen den GPU-Speicher (1920 px: +0,46 GiB) ohne bessere Quote |
| SfM-Matcher (neu) | – | **erschöpfend bis ca. 250 Keyframes**, darüber sequentiell | größter Einzelgewinn bei Clip 2; kostet bei 147 Keyframes 452 s statt 139 s und wächst quadratisch (236 Keyframes: 1073 s) |
| Keyframe-Rate | – | 2/s, Schärfefilter 0,4 | höhere Raten und schwächerer Filter bringen bei Clip 2 nur wenige Punkte |

Die YAML-Datei wird in diesem Spike **nicht** geändert. Die Werte wandern in die Stufen-Stories (`sfm`, `recon.splat`), sobald diese das Schema um `strategy` und die Trainingsbreite erweitern.

## 8. Abweichungen, Grenzen und offene Punkte

- **AK1 nur zu zwei Dritteln erfüllt:** 2 statt 3 Räume, und keiner entspricht genau den vorgesehenen Kategorien (hell/texturiert, weiße Wände, Küche/Bad mit Spiegel). Der PO hat das akzeptiert; weitere Clips folgen später. Die Raumtypen gehen als Anforderung in die Benchmark-Story KAN-27.
- **Subjektive Qualität (AK3, „Screenshots“) offen:** Aus Datenschutzgründen habe ich keine Bilder geöffnet. Vergleichsbilder (zurückgehaltener Frame links, Rendering rechts) liegen lokal in `workspace/spike-kan19/review/` (gitignored) zur Sichtprüfung durch den PO. Hier stehen nur PSNR/SSIM.
- **Die Laufzeiten gelten für die GTX 1060**, nicht für die Zielkarte RTX 2060 (ADR-0010). Q-1 ist damit **nicht verifiziert**. Es fehlt eine Messung auf Turing (fp16 und der Originalpfad des gsplat-Kernels). Der Wert 67 min je 30 000 Iterationen ist eine Hochrechnung aus 7000 Iterationen.
- **Einzelläufe, ein Seed:** PSNR-Unterschiede unter etwa 0,2 dB sind nicht belastbar (Abschnitt 6).
- **GPU-Speicher gemessen mit `nvidia-smi`** (Abtastung alle 0,4 s): Kurze Spitzen können fehlen. Das Training ist durch die Allocator-Zahlen abgesichert.
- **Nicht gemessen:** Qualität über die Iterationen, SH-Grad < 3, `packed`-Rendering, Caps oberhalb der in 7000 Iterationen erreichten Gaussian-Zahl, Dichte-Steuerung für größere Szenen, mehrere Clips pro Raum (Merge), die 2DGS-Abweichung auf sm_75 (aus KAN-18, braucht die reservierte RTX 2070 SUPER).
- **R-01 bleibt offen:** Clip 2 verfehlt Q-3. Der Spike beantwortet nicht, ob die Ursache Szene oder Merkmale sind.

## 9. Empfehlungen an Folgestories

- **`sfm`-Stufe (KAN-36):** erschöpfendes Matching als Standard für kurze Clips (≤ 250 Keyframes), sequentiell darüber. Zerfällt das Ergebnis (größtes Modell < 90 %), Exitcode 20 mit Hinweisen (Licht, langsamer gehen, Türdurchgänge doppelt, AprilTags) statt still ein Teilmodell zu liefern. COLMAP-Startwerte siehe [PM]-Kommentar an KAN-36.
- **Alternative Merkmale (KAN-39):** ALIKED + LightGlue für texturarme/dunkle Szenen prüfen (ONNX-Gewichte über `models.lock`). Das ist der naheliegende nächste Test an Clip 2.
- **Aufnahme-Leitfaden / Erfassungs-Stories:** Clip 2 stützt die Checkliste (Licht, ≤ 0,5 m/s, 1 bis 1,5 m Wandabstand, Türdurchgänge doppelt, AprilTags). Bewegungsunschärfe zeigt sich im Schärfemaß (Median 335 gegenüber 582).
- **`recon.splat`:** MCMC-Strategie mit Trainingsbreite 1200 px; Qualität über Iterationen messen, bevor `max_iterations` festgelegt wird.
- **`ingest`:** Rotations-Flag auswerten (Abschnitt 3).
- **GPU-Lock (KAN-80):** Der Spike-Runner nutzte eine Datei-Sperre (`flock`) für GPU-Stufen. Das Muster lässt sich übernehmen.

## 10. Reproduktion

```bash
# Messungen (Host; Zwischenergebnisse in --work, außerhalb des Repos, nie committen)
uv run python tools/spike01/run_room.py ingest sfm --video data/samples/<clip>.mp4 --room <name> --work <dir>
uv run python tools/spike01/run_room.py prep train --video data/samples/<clip>.mp4 --room <name> --work <dir> \
    --train-width 1200 --iterations 7000 --strategy mcmc --cap 1000000 --tag rec_w1200
# Gegenmaßnahmen für SfM, z. B.:  --matcher exhaustive --fps 3 --min-sharpness 0.2
# Speicher-Stresstest:            --init-random 3000000 --iterations 30 --strategy default
# Vergleichsbilder (nur lokal):   beim Training `--save-renders 4`, danach die Stufe `compare`
make test-gpu   # u. a. Tests der Spike-Werkzeuge auf synthetischen Daten
```
