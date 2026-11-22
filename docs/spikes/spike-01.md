# Spike 01: COLMAP + gsplat auf der GTX 1060 (KAN-19, US-004)

- **Datum:** 21. bis 22.11.2026
- **Hardware:** GTX 1060 6 GiB (Pascal, `sm_61`, steuert den Bildschirm), `CUDA_VISIBLE_DEVICES=0`, fp32. Die RTX 2070 SUPER wurde nicht benutzt.
- **Software:** Images aus KAN-18 (CUDA 12.6.3, PyTorch 2.14.1+cu126, gsplat v1.5.3 mit Pascal-Patch, COLMAP 4.2.1). PyTorch-Allocator-Grenze `DRONESCAN_VRAM_LIMIT_GB=5.0` (Anteil 0,8443 der Karte, in jeder `metrics.json` festgehalten).
- **Daten:** zwei private Smartphone-Clips (je ca. 86 s, H.264, 30 fps variabel), **nur lokal verarbeitet**. Dieser Bericht enthält nur Zahlen und Text: keine Frames, Splats, Screenshots oder Renderings. Die Sichtprüfung der Ergebnisse macht der PO lokal (Abschnitt 8).
- **Werkzeuge:** `tools/spike01/` (nur Code; Aufruf in Abschnitt 10).

## 1. Ergebnis in Kürze

| Frage | Ergebnis |
|---|---|
| **AK2:** gsplat-Training auf Pascal (`sm_61`, fp32) | **Nachgewiesen.** Training bis 15 000 Iterationen und 1,5 Mio. Gaussians läuft durch, mit Default- und MCMC-Strategie, ohne Fallback. Das schließt den offenen Punkt aus R-03/R-13/ADR-0010. |
| **R-03 / Q-2:** Speicher ≤ 5,5 GiB je Prozess inkl. CUDA-Kontext | **Eingehalten, mit großem Abstand.** Realer Trainingslauf mit Wachstum bis zum Cap von 1,5 Mio. Gaussians (inkl. Verschiebe- und Auswahlschritten): **Prozess-Peak 2,6 bis 2,7 GiB** (Allocator-Peak 2,0 GiB, reserviert 2,5 bis 2,6 GiB, Kontext 0,12 GiB). |
| **Q-3 / R-01:** ≥ 90 % registrierte Keyframes | **Clip 1: 100 %** (164/164). **Clip 2: nein**, beste Variante 58,5 % (138/236), das Modell zerfällt in 4 bis 5 Teile. R-01 tritt ein. Keine der getesteten Gegenmaßnahmen reicht. |
| **Q-1:** ≤ 60 min Gesamtlaufzeit | **Nicht verifizierbar.** Für den 86-s-Clip 1 braucht die Kette auf der GTX 1060 **63 min** (Training mit 15 000 Iterationen: 49 min). Q-1 gilt für ca. 4 min Video auf der RTX 2060, die nicht gemessen wurde (Abschnitt 8). |
| **AK1:** 3 Referenzräume | **Nur 2 Clips** (Keller und Treppenhaus, keiner passt genau auf die drei Kategorien). Der PO hat entschieden, den Spike mit diesen zwei Räumen abzuschließen. |
| **Qualität (AK3)** | Objektiv gemessen (PSNR/SSIM auf zurückgehaltenen Frames). **Subjektive Qualität (Screenshots) steht aus** und wird vom PO lokal beurteilt, weil Bildinhalte die Maschine nicht verlassen sollen. |

**Empfehlung für `low_vram_6gb`** (Abschnitt 7): MCMC-Strategie, Trainingsbreite 1200 px, bis 1,5 Mio. Gaussians, 15 000 Iterationen (vorläufig), erschöpfendes Matching bei kurzen Clips.

## 2. Methode

Je Clip eine Kette aus Ingest, SfM und Training, nacheinander (nie zwei GPU-Jobs gleichzeitig: Dateisperre `/tmp/dronescan-gpu.lock` im Spike-Runner):

1. **Ingest** (ffmpeg auf dem Host): Frames mit angewandter Rotation dekodieren. Zuerst 6 Analysebilder pro Sekunde (klein, grau), Schärfe = Varianz des Laplace-Operators. Je Fenster (Standard: 3 Analysebilder = 2 Keyframes/s) wird das schärfste Bild gewählt. Fenster, deren bestes Bild unter 40 % der Median-Schärfe liegt, entfallen (Bewegungsunschärfe). Ausgabe als JPEG, lange Kante 1600 px.
2. **SfM** (Container `colmap`): SIFT auf der GPU (8192 Features), eine gemeinsame Kamera (SIMPLE_RADIAL), Matching, Mapper (CPU). Registrierungsquote = Frames im größten Modell / extrahierte Keyframes.
3. **Training** (Container `worker`): COLMAP-Undistortion (PINHOLE), eigener schlanker Trainer mit `gsplat.rasterization` und `DefaultStrategy` bzw. `MCMCStrategy`, SH-Grad 3, L1 + 0,2·(1−SSIM). Jedes 8. Bild (in Namensreihenfolge) wird zurückgehalten. Metriken PSNR/SSIM auf den zurückgehaltenen Bildern.
4. **Messung:**
   - *Laufzeit:* Wandzeit je Stufe inkl. Container-Start (ca. 1 bis 2 s).
   - *GPU-Speicher, Training:* `torch.cuda.max_memory_allocated/reserved` und der **Prozess-Peak** aus `nvidia-smi --query-compute-apps` (größter Rechenprozess auf GPU 0, Abtastung alle 0,25 s). Er enthält den CUDA-Kontext und ist die Größe aus Q-2. Der Bildschirm taucht dort nicht auf.
   - *Einschränkung früherer Messungen:* Zuerst habe ich den Gesamtverbrauch der GPU abzüglich einer Bildschirm-Baseline gemessen. Das war unzuverlässig (Baseline schwankte 1025 bis 1273 MiB, ein Wert lag unter dem reservierten PyTorch-Speicher). **Diese Werte sind im Bericht verworfen**, die Trainingsläufe wurden mit der Prozessmessung wiederholt. Nur die SfM-Werte in Abschnitt 4 stammen noch aus der alten Messung und gelten als Größenordnung (±0,2 GiB).

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
| SIFT-Extraktion (GPU) | 18,5 s, ca. +1,2 GiB | 16,7 s, ca. +1,2 GiB |
| Matching (sequentiell, Overlap 10, GPU) | 150 s, ca. +1,1 GiB | 139 s, ca. +0,9 GiB |
| Mapper (CPU, Bündelausgleich ohne GPU) | 593 s | 83 s |
| Undistortion + Rohbildumwandlung (CPU) | 26 s + 6 s | 18 s + 4 s (bestes Modell, Abschnitt 5) |

Die GPU-Speicherwerte der SfM-Stufen stammen aus der verworfenen Gesamt-GPU-Messung (Größenordnung, ±0,2 GiB). Die Mapper-Zeit ist bei Clip 2 nur deshalb kurz, weil das Modell in kleine Teile zerfällt. Bei Clip 1 ist der Mapper mit 593 s der größte SfM-Posten und für Q-1 relevant.

## 5. Clip 2: Gegenmaßnahmen (Timebox: 3 bis 4)

Alle Varianten laufen auf Clip 2. Die Zeilen 4 und 6 ändern je einen Parameter gegenüber der Basis „erschöpfendes Matching“ (Zeile 3), Zeile 5 und 7 je zwei.

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
- Alles Weitere (mehr Features, niedrigere Peak-Schwelle, native Auflösung, lockerere Mapper-Parameter) bringt nur **0 bis 3 Prozentpunkte** und kostet Laufzeit. Die native Auflösung bringt nichts (53,1 %).
- Dichtere Frames mit weniger strengem Schärfefilter geben die beste Quote (58,5 %), aber bei 2,4-facher Matching-Zeit gegenüber dem erschöpfenden Matching mit 147 Frames (1073 s gegenüber 452 s; gegenüber dem sequentiellen Standard 7,7-fach).
- **Kein Ansatz erreicht 80 % (Q-3: 90 %).** Die Quote scheint bei etwa 55 bis 60 % zu sättigen, das Modell bleibt in 3 bis 5 Teile zerstückelt. Die Ursache liegt vermutlich in der Szene (wenig Licht, wenig Textur, wiederkehrende Strukturen, geringere Schärfe) und nicht in den COLMAP-Parametern. Das ist aus den Messwerten **nicht bewiesen**: Ob es an der Aufnahme oder an SIFT liegt, zeigt erst ein Vergleich mit anderen Merkmalen (Abschnitt 9).
- Nicht getestet (außerhalb der Timebox): Zusammenführen der Teilmodelle (`model_merger`), ALIKED + LightGlue, AprilTags, andere Kameramodelle.

## 6. Training auf der GTX 1060 (fp32, SH-Grad 3)

### 6.1 Vergleich bei 7000 Iterationen (Clip 1, 143 Trainings- und 21 zurückgehaltene Bilder, 44 393 Start-Punkte)

| Variante | End-Gaussians | PSNR dB | SSIM | s/Iteration (Mittel) | Allocator-Peak | reserviert |
|---|---|---|---|---|---|---|
| Default, 1200 px | 1 005 661 | 25,53 | 0,880 | 0,158 | 1,52 GiB | 2,27 GiB |
| Default, 1600 px | 954 930 | 25,37 | 0,887 | 0,249 | 1,46 GiB | 2,52 GiB |
| MCMC (Cap 1 Mio.), 1200 px | 508 981 | 26,21 | 0,894 | 0,138 | 0,78 GiB | 0,95 GiB |
| MCMC (Cap 1,5 Mio.), 1200 px | 508 981 | 26,38 | 0,894 | 0,139 | 0,78 GiB | 0,93 GiB |
| MCMC (Cap 0,5 Mio.), 960 px | 500 000 | 26,44 | 0,893 | 0,097 | 0,70 GiB | 1,02 GiB |
| MCMC (Cap 1 Mio.), 1200 px, Wiederholung | 508 981 | 26,54 | 0,894 | 0,134 | 0,78 GiB | 0,93 GiB |

**Lesart und Vorbehalte**
- **MCMC schlägt Default** in diesem Vergleich um ca. 0,7 bis 1,0 dB PSNR und 0,014 SSIM, bei halber Gaussian-Zahl und halbem Speicher. **Einschränkung:** Den Zeitplan der Default-Strategie habe ich für 7000 Iterationen selbst skaliert (`refine_stop` bei 3500, `reset_every` 700; die Referenz nutzt 15 000 bzw. 3000 bei 30 000 Iterationen). Das kann Default benachteiligen. Der Vorsprung ist also nicht mit dem Referenzzeitplan abgesichert.
- **Unterschiede unter etwa 0,2 dB sind nicht aussagekräftig:** Die drei MCMC-Läufe mit 1200 px haben dieselben 508 981 Gaussians, streuen aber um 0,3 dB (Seed fest, `atomicAdd`-Reihenfolge nicht deterministisch). Jede Variante ist ein einzelner Lauf.
- **Die Caps 1 Mio. und 1,5 Mio. wurden bei 7000 Iterationen nie erreicht.** MCMC wächst von 44 393 Start-Punkten um 5 % je Aufräumschritt ab Iteration 500 und endet mit `refine_stop` bei 5600 bei 509 000 Gaussians. Das erklärt die identischen Zahlen (die Rechnung 44 393 · 1,05^50 ≈ 509 000 passt). Die Mittelwerte der Zeit je Iteration sind deshalb bei 7000 Iterationen **nicht auf längere Läufe übertragbar** (Abschnitt 6.2).
- **1600 px statt 1200 px** kostet 58 % mehr Zeit und bringt keine bessere Qualität (Default: 25,37 gegenüber 25,53 dB). 960 px ist 28 % schneller als 1200 px bei vergleichbarer Qualität (26,44 gegenüber 26,2 bis 26,5 dB, MCMC).
- Der Abstand zwischen zurückgehaltenen Frames und Trainingsansichten beträgt bei Clip 1 1,5 bis 3,7 dB (MCMC 1,5 bis 1,8 dB, Default 3,7 dB).
- **SSIM** ist eine eigene Implementierung mit Null-Auffüllung am Rand (Gauß-Fenster 11, σ 1,5). Die Absolutwerte sind mit Literaturwerten nicht direkt vergleichbar, nur untereinander.

### 6.2 Langer Lauf bis zum Cap (15 000 Iterationen, MCMC, Cap 1,5 Mio., 1200 px)

| Clip / Lauf | Gaussians | PSNR dB | SSIM | Trainingsansichten dB | Zeit | Allocator-Peak | reserviert | **Prozess-Peak** |
|---|---|---|---|---|---|---|---|---|
| Clip 1 | 1 500 000 | **27,60** | 0,909 | 30,80 | 2943 s (49 min) | 2,00 GiB | 2,56 GiB | **2,68 GiB** |
| Clip 1, Wiederholung | 1 500 000 | **27,66** | 0,910 | 30,84 | 2946 s | 2,01 GiB | 2,51 GiB | **2,63 GiB** (*) |
| Clip 2, bestes Teilmodell (138 Bilder, 120/18) | 1 500 000 | 25,15 | 0,882 | 31,01 | 2567 s (43 min) | 1,92 GiB | 2,49 GiB | **2,61 GiB** (*) |

(*) Der abgetastete Prozess-Peak lag bei diesen beiden Läufen unter dem von PyTorch reservierten Speicher (2598 MiB gegenüber 2572 MiB reserviert; 2496 MiB gegenüber 2552 MiB), das ist nicht möglich. Vermutlich ist eine `nvidia-smi`-Abfrage unter Last fehlgeschlagen und hat die Abtastung beendet (im Runner behoben, die Läufe sind nicht wiederholt). Angegeben ist deshalb **reserviert + 124 MiB Kontext**. Der Kontext war in allen Stresstests konstant 124 MiB (Prozess-Peak minus reserviert). Beim ersten Lauf stimmt die Abtastung mit dieser Rechnung überein (2746 MiB).

**Wachstum und Zeit je Iteration (Clip 1, aus dem Trainingslog):**

| Iteration | Gaussians | s/Iteration | Allocator-Peak | reserviert |
|---|---|---|---|---|
| 500 | 44 393 | 0,10 | 0,69 GiB | 0,72 GiB |
| 3000 | 143 153 | 0,12 | 0,69 GiB | 0,72 GiB |
| 5500 | 484 744 | 0,16 | 0,76 GiB | 0,94 GiB |
| 7000 | 1 007 740 | 0,21 | 1,37 GiB | 1,76 GiB |
| 8000 | **1 500 000 (Cap erreicht)** | 0,26 | 2,00 GiB | 2,56 GiB |
| 8000 bis 15 000 | 1 500 000 | 0,23 bis 0,26 | 2,00 GiB | 2,56 GiB |

**Lesart**
- Der Lauf reicht **durch die Wachstumsphase bis ins Plateau am Cap** und enthält alle Auswahl- und Verschiebeschritte der MCMC-Strategie bei voller Größe (7000 Iterationen im Plateau). Der Peak ändert sich ab Iteration 8000 nicht mehr. Damit ist das Budget für **Training bis 1,5 Mio. Gaussians bei 1200 px gemessen**: Prozess-Peak ≤ 2,7 GiB inkl. CUDA-Kontext, weit unter 5,5 GiB.
- Qualität: 27,6 bis 27,7 dB gegenüber 26,2 bis 26,5 dB bei 7000 Iterationen (**+1,2 dB**). Der Gewinn stammt aus beidem (mehr Iterationen und mehr Gaussians), das ist nicht getrennt.
- **Zeit:** Am Anfang kostet eine Iteration 0,10 s, im Plateau 0,25 s. Ein konstanter Mittelwert führt in die Irre. Aus dem Log: 15 000 Iterationen = 49 min. 30 000 Iterationen wären **ca. 112 min** (8000 Iterationen bis zum Cap in 1210 s, danach 22 000 · 0,25 s; Rechnung aus dem Plateauwert, nicht gemessen).
- **Clip 2** erreicht auf seinem Teilmodell 25,15 dB auf zurückgehaltenen Frames, aber 31,0 dB auf den Trainingsansichten: **5,9 dB Abstand** gegenüber 3,2 dB bei Clip 1. Mit 120 Trainingsbildern aus einem zerstückelten Modell passt sich das Modell stark an die Trainingsansichten an. Das ist ein Hinweis auf geringe Abdeckung, nicht auf einen Fehler im Trainer. Das Modell deckt nur 58 % der Aufnahme ab.

### 6.3 Speicher-Stresstest (Clip 1, Start mit N zufälligen Gaussians, 30 Iterationen, Default-Strategie, 1200 px)

| Start-Gaussians | Ergebnis | Allocator-Peak | reserviert | **Prozess-Peak** | s/Iteration |
|---|---|---|---|---|---|
| 1,0 Mio. | ok | 1,56 GiB | 1,68 GiB | 1,80 GiB | 0,22 |
| 1,5 Mio. | ok | 2,16 GiB | 2,40 GiB | 2,52 GiB | 0,25 |
| 2,0 Mio. | ok | 2,74 GiB | 3,42 GiB | 3,54 GiB | 0,28 |
| 3,0 Mio. | ok | 3,90 GiB | 4,47 GiB | 4,59 GiB | 0,33 |
| 4,0 Mio. | **OOM** | 3,72 GiB | 4,49 GiB | (nicht gemessen) | – |
| 6,0 Mio. | **OOM** | 4,36 GiB | 4,61 GiB | (nicht gemessen) | – |

Bei 1600 px: 2 Mio. ok (3,16 GiB Allocator-Peak), 4 Mio. OOM.

- Der Speicher wächst etwa linear (ca. 1,2 GiB je Million Gaussians bei 1200 px plus ca. 0,4 GiB Grundlast). Der Stresstest bei 1,5 Mio. (2,52 GiB) stimmt mit dem realen Lauf am Cap (2,61 bis 2,68 GiB) gut überein.
- Der OOM tritt auf, bevor die 5,0-GiB-Allocator-Grenze greift: Auf der 1060 sind wegen des Bildschirms (ca. 1,0 GiB) und des CUDA-Kontexts nur **ca. 4,6 bis 4,7 GiB** nutzbar. Das ist die harte Grenze dieses Entwicklungsrechners. Die Allocator-Grenze selbst wurde nie ausgelöst (Durchsetzung nicht getestet).
- **Der Stresstest läuft ohne Verdichtung** (30 Iterationen, keine Auswahl- oder Verschiebeschritte). Die „ca. 3 Mio.“ sind daher eine **obere Schranke für den statischen Speicher**, die reale Grenze mit Wachstum liegt darunter. Gemessen mit Wachstum ist nur 1,5 Mio. (Abschnitt 6.2).

## 7. Empfehlung für die Profilparameter `low_vram_6gb`

| Parameter | Aktuell (`low_vram_6gb.yaml`) | Empfehlung | Begründung |
|---|---|---|---|
| `splat.strategy` (neu) | – | `mcmc` | bessere Qualität und kleinerer Speicher als Default (6.1, mit dem Vorbehalt zum Default-Zeitplan) |
| `splat.max_image_edge_px` (neu, Trainingsbreite) | – | **1200** (960 als Schnellmodus) | 1600 px kostet 58 % mehr Zeit ohne Qualitätsgewinn |
| `splat.max_gaussians` | 1 000 000 | **1 500 000** (gemessen) | Die Bedingung aus dem Profilkommentar („1,5 M nur, wenn KAN-19 ≤ 5,5 GiB gesamt misst“) ist erfüllt: Prozess-Peak 2,6 bis 2,7 GiB inkl. Kontext (6.2). Mehr als 1,5 Mio. ist mit Wachstum nicht gemessen |
| `splat.max_iterations` | 30 000 | **15 000 (vorläufig)** | 15 000 Iterationen = 49 min auf der 1060, 30 000 wären ca. 112 min und reißen Q-1. Die Qualitätskurve über die Iterationen ist nicht gemessen (nur 7000 und 15 000), die Zahl gehört in die `recon`-Story |
| `gpu.allow_fp16` | `true` (nur sm_70+) | unverändert | auf Pascal fp32 (gemessen), sm_75 nicht gemessen |
| `sfm.max_image_edge_px` | 1600 | **1600 beibehalten** | größere Bilder bringen keine bessere Quote (Abschnitt 5) |
| SfM-Matcher (neu) | – | **erschöpfend bis ca. 250 Keyframes**, darüber sequentiell | größter Einzelgewinn bei Clip 2 (+25,8 Punkte). Nur an diesem einen Clip getestet: Clip 1 erreichte mit sequentiellem Matching schon 100 %. Kosten: bei 147 Keyframes 452 s statt 139 s, bei 236 Keyframes 1073 s (quadratisches Wachstum) |
| Keyframe-Rate | – | 2/s, Schärfefilter 0,4 | höhere Raten und schwächerer Filter bringen bei Clip 2 nur wenige Punkte |

Die YAML-Datei wird in diesem Spike **nicht** geändert. Die Werte wandern in die Stufen-Stories (`sfm`, `recon.splat`), sobald diese das Schema um `strategy` und die Trainingsbreite erweitern.

**Gesamtlaufzeit Clip 1 auf der GTX 1060 (Q-1):** Ingest 45 s + SIFT 19 s + Matching 150 s + Mapper 593 s + Undistortion/Rohbilder 32 s + Training (15 000 Iterationen) 2943 s = **3782 s (63 min)** für 86 s Video und 164 Keyframes. Q-1 meint ca. 4 min Video auf der RTX 2060: Bildzahl, Matching und Mapper wachsen mit der Länge, die Kartendifferenz ist nicht gemessen. Für Q-1 bleiben als Hebel weniger Iterationen, 960 px (28 % schneller), ein kleineres Cap (0,21 s je Iteration bei 1 Mio. Gaussians gegenüber 0,25 s bei 1,5 Mio.) oder eine schnellere Karte.

## 8. Abweichungen, Grenzen und offene Punkte

- **AK1 nur zu zwei Dritteln erfüllt:** 2 statt 3 Räume, und keiner entspricht genau den vorgesehenen Kategorien (hell/texturiert, weiße Wände, Küche/Bad mit Spiegel). Der PO hat das akzeptiert; weitere Clips folgen später. Die Raumtypen gehen als Anforderung in die Benchmark-Story KAN-27.
- **Subjektive Qualität (AK3, „Screenshots“) offen:** Aus Datenschutzgründen habe ich keine Bilder geöffnet. Vergleichsbilder (zurückgehaltener Frame links, Rendering rechts) liegen lokal in `workspace/spike-kan19/review/` (gitignored) zur Sichtprüfung durch den PO: zwei Läufe mit 15 000 Iterationen (Clip 1 und Clip 2, bestes Modell) und drei ältere mit 7000 Iterationen. Hier stehen nur PSNR/SSIM.
- **Die Laufzeiten gelten für die GTX 1060**, nicht für die Zielkarte RTX 2060 (ADR-0010). Q-1 ist **nicht verifiziert**, ebenso fp16 und der Originalpfad des gsplat-Kernels auf sm_75. Die Messung der 2DGS-Abweichung auf sm_75 aus KAN-18 steht ebenfalls aus (braucht die reservierte RTX 2070 SUPER, nur mit Zustimmung des POs).
- **Einzelläufe, ein Seed:** PSNR-Unterschiede unter etwa 0,2 dB sind nicht belastbar. Die einzige Wiederholung (Clip 1, 15 000 Iterationen) lag 0,06 dB auseinander.
- **GPU-Sperre:** Der Spike-Runner serialisiert GPU-Stufen über `/tmp/dronescan-gpu.lock` (alle Jobs dieses Benutzers). Die GPU-Tests von `pytest` und fremde Prozesse nehmen die Sperre nicht. Eine echte, überall genutzte Sperre ist Teil von KAN-80.
- **Nicht gemessen:** Qualität über die Iterationen, SH-Grad < 3, `packed`-Rendering, Caps oberhalb 1,5 Mio. mit Wachstum, mehrere Clips pro Raum (Merge), Verhalten bei größeren Räumen (mehr Gaussians für dieselbe Qualität), Allocator-Grenze als Auslöser (nie erreicht).
- **R-01 bleibt offen:** Clip 2 verfehlt Q-3. Der Spike beantwortet nicht, ob die Ursache Szene oder Merkmale sind.

## 9. Empfehlungen an Folgestories

- **`sfm`-Stufe (KAN-36):** erschöpfendes Matching als Standard für kurze Clips (≤ 250 Keyframes), sequentiell darüber. Zerfällt das Ergebnis (größtes Modell < 90 %), Exitcode 20 mit Hinweisen (Licht, langsamer gehen, Türdurchgänge doppelt, AprilTags) statt still ein Teilmodell zu liefern. COLMAP-Startwerte siehe [PM]-Kommentar an KAN-36. Der Mapper (593 s bei Clip 1) ist der größte CPU-Posten und gehört in die Q-1-Betrachtung.
- **Alternative Merkmale (KAN-39):** ALIKED + LightGlue für texturarme/dunkle Szenen prüfen (ONNX-Gewichte über `models.lock`). Das ist der naheliegende nächste Test an Clip 2.
- **Aufnahme-Leitfaden / Erfassungs-Stories:** Clip 2 stützt die Checkliste (Licht, ≤ 0,5 m/s, 1 bis 1,5 m Wandabstand, Türdurchgänge doppelt, AprilTags). Bewegungsunschärfe zeigt sich im Schärfemaß (Median 335 gegenüber 582).
- **`recon.splat`:** MCMC-Strategie mit Trainingsbreite 1200 px und bis 1,5 Mio. Gaussians. Qualität über Iterationen messen, bevor `max_iterations` festgelegt wird. Bei Teilmodellen mit großem Abstand zwischen Trainings- und zurückgehaltenen Ansichten (Clip 2: 5,9 dB) Warnung im Qualitätsreport.
- **`ingest`:** Rotations-Flag auswerten (Abschnitt 3).
- **OOM-Verhalten (Exitcode 30):** Der Trainer meldet Out-of-Memory mit Exitcode 30 und Messwerten. Ein automatischer Wiederholungsversuch mit kleinerem Cap, 960 px und `--packed` ist nicht umgesetzt (gehört in die Job-Queue). Auf einer Karte mit Bildschirm sind nur ca. 4,6 GiB nutzbar.
- **GPU-Sperre (KAN-80):** Das Muster `flock` auf einer festen Datei lässt sich übernehmen.

## 10. Reproduktion

```bash
# Messungen (Host). --work muss außerhalb des Repos oder git-ignoriert sein, z. B. unter workspace/.
uv run python tools/spike01/run_room.py ingest sfm --video data/samples/<clip>.mp4 --room <name> --work <dir>
uv run python tools/spike01/run_room.py prep train --video data/samples/<clip>.mp4 --room <name> --work <dir> \
    --train-width 1200 --iterations 15000 --strategy mcmc --cap 1500000 --tag <tag> --save-renders 4
# Gegenmaßnahmen für SfM, z. B.:  --matcher exhaustive --fps 3 --min-sharpness 0.2
# Speicher-Stresstest:            --init-random 3000000 --iterations 30 --strategy default
# Vergleichsbilder (nur lokal):   Stufe `compare` mit --tag <tag>; erzeugt <work>/<room>/compare/.
#                                 Von dort in den privaten Ordner kopieren (workspace/spike-kan19/review/).
make test-gpu   # u. a. Tests der Spike-Werkzeuge auf synthetischen Daten
```
