# ADR-0012: Non-commercial-Lizenzen erlaubt, Images nur lokal

- **Status:** angenommen (PO im Chat, 21.11.2026); löst ADR-0007 teilweise ab
- **Datum:** 2026-11-21
- **Jira:** – (Anlass: license-auditor-Befunde zu KAN-18, PO-Entscheidungen F1–F6 vom 21.11.2026)

## Kontext
ADR-0007 verbietet alle Komponenten mit Nutzungsbeschränkung („non-commercial“, „research-only“), damit drone-scan später kommerziell verwertbar bleibt. Die Lizenzprüfung der Docker-Images in KAN-18 hat gezeigt, wie weit dieses Verbot reicht:

- **COLMAP mit CUDA** kompiliert immer **SiftGPU** ein (UNC-Lizenz: „educational, research and non-profit purposes“). Ohne SiftGPU gibt es nur zwei Wege: COLMAP ganz ohne CUDA, also langsamere SIFT auf der CPU, oder ein dauerhaft gepflegter Patch.
- **PyTorch auf der GPU** braucht die proprietären NVIDIA-Laufzeitbibliotheken (CUDA-Runtime, cuBLAS, cuDNN, cuSPARSELt, NCCL, NVSHMEM, cuda-bindings) und die `nvidia/cuda`-Basis-Images. Das ist kein Open Source im Wortsinn.
- Weitere SOTA-Komponenten (Inria-3DGS, DUSt3R/MASt3R, SuperPoint/SuperGlue) sind NC-lizenziert.

Der PO hat festgelegt: drone-scan ist ein **Non-Profit-Projekt**, die Images werden **nur lokal** gebaut und genutzt. Eine kommerzielle Nutzung ist derzeit nicht geplant. Das Repository bleibt **öffentlich** (GitHub). Deshalb darf NC-Code bzw. NC-Gewicht selbst nicht darin landen.

## Entscheidung
1. **NC und research-only sind erlaubt.** Komponenten mit Nutzungsbeschränkung (Code und Modellgewichte) dürfen genutzt werden. Das gilt auch für SiftGPU in COLMAP: Das COLMAP-Image wird mit CUDA `61;75` gebaut.
2. **Copyleft-Regel bleibt unverändert** (ADR-0007, ADR-0011). Fremder GPL- bzw. AGPL-Code läuft nur als separater Prozess oder Container und wird nie aus `src/` importiert. Das gilt auch für NC-Lizenzen mit Share-alike-Klausel (z. B. CC-BY-NC-SA): solchen Code ebenfalls nicht aus `src/` importieren, bei Bedarf in `container_only_modules` eintragen. Kombiniert eine Lizenz NC **und** Copyleft (`AGPL-3.0-only AND CC-BY-NC-4.0`), prüft das Gate den Copyleft-Teil wie bisher.
3. **Ausnahme NVIDIA-Laufzeit.** Die proprietären NVIDIA-GPU-Laufzeitbibliotheken (siehe Kontext) und die `nvidia/cuda`-Basis-Images sind erlaubt, so wie der Treiber. Die Build-Stufen (`-devel`-Images, nvcc) werden nie ausgeliefert.
4. **Images nur lokal.** Docker-Images werden lokal gebaut (`docker compose build`) und nie in eine Registry gepusht, auch nicht in eine private.
5. **Kein NC-Code und keine NC-Gewichte im öffentlichen Repository.** NC-Komponenten werden weder vendored noch kopiert noch als Git-Submodule eingebunden, NC-Gewichte nie eingecheckt. Erlaubt ist nur:
   - Ein Dockerfile baut die Komponente lokal aus der Upstream-Quelle (z. B. SiftGPU im COLMAP-Build).
   - `drone-scan models pull` lädt Gewichte lokal herunter.

   Das Lizenz-Gate prüft das: Getrackte Pfade und Submodule-URLs dürfen nicht nach einem bekannten NC-Projekt (`nc_packages`, `nc_image_components`) benannt sein, und Modellgewichte (`.pt`, `.pth`, `.ckpt`, `.onnx`, `.safetensors`, `.tflite`, `.engine`) dürfen nicht in Git liegen. Kopierter NC-Code unter anderem Namen lässt sich so nicht erkennen. Hier bleiben Review und `license-auditor` zuständig.
6. **Pflicht vor jeder Weitergabe oder kommerziellen Nutzung:** Bevor Images oder Binärpakete an Dritte gehen oder drone-scan kommerziell genutzt wird, werden **alle NC-Komponenten neu geprüft** und ersetzt oder lizenziert. Dazu gehören auch die zurückgestellten Punkte aus KAN-18:
   - Quellangebot bzw. Quellcode für die GPL- und LGPL-Pakete in beiden Images (u. a. COLMAP-Gesamtwerk mit CHOLMOD/SPQR, ffmpeg)
   - NOTICE-Datei bzw. SBOM pro Image
   - Ergänzung der Projektlizenz nach AGPL §7 („Additional Permission“) für das Linken mit NVIDIA-CUDA-, cuDNN- und NCCL-Bibliotheken, weil die NVIDIA-Lizenzen eine Unterwerfung unter Copyleft-Quellpflichten ausschließen
   - optional ein schlankerer COLMAP-Build (OpenImageIO ohne ffmpeg, OpenCV und GDAL; Ceres ohne SuiteSparse)
7. **Kein Freibrief für Stack-Wechsel.** Die Lockerung betrifft nur die Lizenzfrage. Ein Wechsel etablierter Komponenten, z. B. gsplat → Inria-3DGS (ADR-0002) oder ALIKED + LightGlue → SuperPoint/SuperGlue, braucht weiterhin ein eigenes ADR und die Freigabe des POs.
8. **Durchsetzung:**
   - Das Lizenz-Gate (`make license-gate`) lässt den Build bei NC **nicht** mehr scheitern. Es erkennt NC-Lizenzen sowie bekannte NC-Pakete und -Modelle weiterhin und führt sie im Report als `NC` auf. So sind sie vor einer Weitergabe vollständig auffindbar.
   - Die bisherigen NC-Einträge stehen in `licenses/denylist.yaml` unter `nc_license_patterns`, `nc_packages` und `nc_models`.
   - Das Gate sieht nur Python-Pakete, npm-Pakete, `models.lock` und Importe in `src/`. NC-Komponenten, die in die Docker-Images einkompiliert sind (z. B. SiftGPU in COLMAP), werden deshalb von Hand in `nc_image_components` gepflegt und bei jedem Lauf als `NC` gemeldet. Wer ein Image um eine NC-Komponente erweitert, trägt sie dort ein.
   - Ein bekanntes NC-Paket ohne Lizenzangabe in den Metadaten bleibt ein Fehler („Lizenz unklar“). Abhilfe: Override in `allowlist.yaml` mit der NC-Kennung (z. B. `LicenseRef-NC`).
   - Harte Verbote bleiben in `packages` und `models` (derzeit leer) sowie in `copyleft`.
   - Unklare Lizenzen bleiben ein Fehler.
9. **Allowlist erweitert** um BSL-1.0, Apache-2.0 WITH LLVM-exception, Zlib, 0BSD, CC0-1.0 und curl (PO, 21.11.2026).

**Verworfene Alternativen**
- **ADR-0007 unverändert lassen, COLMAP ohne CUDA bauen:** hält die Kommerzialisierbarkeit offen, macht die Merkmalsextraktion auf dem Zielrechner aber deutlich langsamer und erfüllt das Akzeptanzkriterium von KAN-18 („colmap mit CUDA 61;75“) nicht.
- **SiftGPU per Patch aus COLMAP entfernen:** erfüllt das Kriterium, erzeugt aber Pflegeaufwand bei jedem COLMAP-Update. Bei einem Non-Profit-Projekt mit lokaler Nutzung steht dem kein Nutzen gegenüber.
- **NC nur als Einzelfall-Ausnahme (nur SiftGPU):** wäre die engere Lösung. Der PO hat sich bewusst für eine generelle Lockerung entschieden, damit Spikes nicht an jeder NC-Komponente auf eine Einzelfreigabe warten.

## Konsequenzen
+ COLMAP läuft mit GPU-SIFT, KAN-18 wird ohne Patch umsetzbar. Spikes dürfen NC-Modelle evaluieren (nach ADR und PO-Freigabe für den Stack-Wechsel).
+ Das Gate bleibt aussagekräftig: Jede NC-Komponente steht im Report.
- **Kommerzielle Verwertung ist bis zu einer vollständigen Neuprüfung ausgeschlossen.** Ziel 2 in `docs/architektur.md` (Kommerzialisierbarkeit) wird dafür zurückgestellt. Eine spätere Ablösung von NC-Komponenten kann Aufwand kosten.
- Die zurückgestellten Pflichten (Quellangebot, SBOM, AGPL-§7-Zusatz) werden bei einer Weitergabe fällig.
- Die in ADR-0011 genannte Doppellizenzierung (kommerzielle Lizenz durch den PO) und jede Weitergabe des AGPL-Codes zusammen mit NC-Abhängigkeiten sind erst nach der Neuprüfung möglich. README und ADR-0011 weisen darauf hin.
- Image-Weitergabe ist nur nach Punkt 6 möglich. Nutzer bauen die Images selbst.
- Lizenz: keine neuen Abhängigkeiten durch dieses ADR. VRAM: keine Auswirkung.
