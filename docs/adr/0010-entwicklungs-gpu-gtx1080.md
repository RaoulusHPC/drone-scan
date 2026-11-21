# ADR-0010: Entwicklungs-GPU GTX 1060 6GB neben Ziel-GPU RTX 2060

- **Status:** angenommen (korrigiert am 2026-11-21: GTX 1060 statt GTX 1080, VRAM-Werte in GiB)
- **Datum:** 2026-11-21

## Kontext
Claude Code läuft lokal auf dem Entwicklungsrechner mit einer **GTX 1060 6GB (6 GiB, Pascal, sm_61)**. Sie steuert auch den Bildschirm, dadurch ist etwa 1 GiB belegt. Zielhardware der Nutzer bleibt die **RTX 2060 (6 GiB, Turing, sm_75)**.
Im selben Rechner steckt eine **RTX 2070 SUPER (8 GiB, sm_75)**. Sie ist für ein anderes Projekt reserviert und darf nur nach vorheriger Zustimmung des POs genutzt werden, wenn sie frei ist.

*Korrektur:* Die ursprüngliche Fassung nannte eine GTX 1080 (8 GB). Gemeint war die GTX 1060 6GB (PO, 21.11.2026). Die Architektur (sm_61) und damit die Toolchain-Entscheidung bleiben gleich.
PyTorch hat Pascal aus den CUDA-12.8/12.9- und CUDA-13-Builds entfernt. Nur die CUDA-12.6-Builds unterstützen Pascal noch, und laut PyTorch-Ankündigung ist **2.14 die letzte Version mit CUDA-12.x-Wheels**; ab 2.15 ist Pascal nur noch per Eigenbau nutzbar. Pascal hat zudem keine Tensor Cores (fp16 bringt kaum Vorteile).

## Entscheidung
- Basis-Image und Wheels: **CUDA 12.6**, **PyTorch < 2.15** (cu126-Index), in `uv.lock` gepinnt.
- Eigene CUDA-Erweiterungen (gsplat) und COLMAP werden für **`TORCH_CUDA_ARCH_LIST="6.1;7.5"`** bzw. `CMAKE_CUDA_ARCHITECTURES="61;75"` gebaut.
- Präzision: Profilparameter `precision` = `fp32` auf Pascal, `fp16`/`mixed` erlaubt ab Turing (automatische Erkennung über `torch.cuda.get_device_capability()`).
- VRAM-Werte gelten in **GiB**. Budget pro Prozess ≤ 5,5 GiB gesamt, PyTorch-Allocator-Grenze 5,0 GiB (`low_vram_6gb`, `DRONESCAN_VRAM_LIMIT_GB`).
- VRAM-Tests laufen auf der GTX 1060 (`make test-gpu` legt `CUDA_VISIBLE_DEVICES` fest). Die RTX 2070 SUPER wird nur mit Zustimmung genutzt.
- Laufzeitmessungen auf der GTX 1060 sind **nicht** 1:1 auf die RTX 2060 übertragbar; Q-1 wird vor M1 einmal auf Turing-Hardware verifiziert (eigener Laptop oder geliehenes Gerät).

## Konsequenzen
+ Entwicklung ohne neue Hardware möglich. Die GTX 1060 hat genau die Speichergröße der Zielkarte und bildet sie realistisch ab.
- Weil die GTX 1060 auch den Bildschirm steuert, sind nur etwa 4,8 GiB frei, also weniger als auf der Zielkarte. Volle Läufe im Profil `low_vram_6gb` (5,0 GiB + Kontext) passen dort nicht. Abhilfe im Spike US-004: Bildschirm an eine andere Grafikausgabe hängen, kleineres Testprofil verwenden oder nach Zustimmung die RTX 2070 SUPER nutzen.
- Toolchain eingefroren auf CUDA 12.6 / PyTorch ≤ 2.14 → Sicherheits- und Feature-Updates enden; spätestens bei M5 GPU-Wechsel prüfen (Risiko R-13)
- gsplat-Build für sm_61 muss im Spike US-004 nachgewiesen werden; Fallback: Training im Container auf Turing-Hardware

## Nachtrag 2026-11-21: gsplat-Patch für Pascal (KAN-18, PO-Entscheidung)
**Befund beim Image-Build:** gsplat lässt sich unverändert nicht für sm_61 bauen.
- Seit v1.0 nutzen die Backward-Kernels der Projektion `cooperative_groups::labeled_partition`. Betroffen sind 3DGS und 2DGS, jeweils fused und packed. CUDA stellt diese Funktion erst ab sm_70 bereit (Match-Collectives). Der Guard `#if __CUDA_ARCH__ >= 700` ist im gsplat-Code auskommentiert. Geprüft: v1.0.0 bis v1.5.3 (`git grep -n "__CUDA_ARCH__ >= 700\|labeled_partition" <tag> -- gsplat/cuda/csrc` im gsplat-Repo).
- Zusätzlich brauchen die Header von PyTorch 2.14 C++20. gsplat erzwingt für nvcc aber `-std=c++17`.

**Entscheidung (PO, Option A):** gsplat v1.5.3 wird mit einem gepinnten Patch gebaut: `docker/patches/gsplat-v1.5.3-pascal.patch`.
- `setup.py`: `-std=c++17` → `-std=c++20`.
- `Utils.cuh`: neue Hilfsfunktion `labeled_partition_compat`. In den 4 betroffenen Kernel-Dateien ersetzt sie an 8 Stellen `cg::labeled_partition`.
  - **Ab sm_70** (also sm_75, RTX 2060) ruft sie unverändert `cg::labeled_partition` auf. Der Codepfad der Zielhardware bleibt gleich.
  - **Unter sm_70** (sm_61, GTX 1060) bildet jeder Thread eine eigene Gruppe der Größe 1 (`cg::tiled_partition<1>`). Die Warp-Summe entfällt, und jeder Thread addiert seine Gradienten direkt per `atomicAdd`. Die Summe ist dieselbe, nur die Reihenfolge der Gleitkomma-Additionen ändert sich. Auf Pascal ist das etwas langsamer, weil mehr Atomics anfallen.
- Das Dockerfile prüft den Patch mit `git apply --check`. Passt er nicht mehr sauber (z. B. nach einem gsplat-Update), schlägt der Build fehl.

**Nachweis:**
- Der Patch ist öffentlich dokumentiert, trägt einen Apache-2.0-Hinweis und liegt im Image unter `/usr/share/doc/gsplat/MODIFICATIONS.patch`.
- CPU-Test `test_gsplat_patch_keeps_the_sm70_path_unchanged`: Der Patch ändert nur die 8 Aufrufe und die C++-Version, und der Zweig ab sm_70 ruft `cg::labeled_partition` auf.
- Der Image-Build prüft mit `cuobjdump --list-elf`, dass gsplat und COLMAP Gerätecode für `sm_61` **und** `sm_75` enthalten, sonst schlägt er fehl.
- GPU-Test `test_patched_gsplat_gradients_match_reference_on_gtx_1060` (`docker/checks/gsplat_grads.py`, synthetische Daten, 5,0-GiB-Allocator-Grenze): Die Gradienten des Backward-Passes (3DGS und 2DGS, fused und packed) werden mit einer **float64-Referenz** (gsplats PyTorch-Implementierung in double) als relative L2-Abweichung verglichen. Gemessen auf der GTX 1060:

  | Kernel | Gradient | Abweichung | Grenze |
  |---|---|---|---|
  | 3DGS fused | alle | ≤ 1,2e-6 | 1e-4 |
  | 3DGS packed vs. fused | alle | ≤ 1,2e-6 | 1e-4 |
  | 2DGS fused | `means` | 8e-8 | 1e-4 |
  | 2DGS fused | `viewmats` | 2,1e-3 | 1e-2 |
  | 2DGS fused | `scales` | 3,7e-3 | 1e-2 |
  | 2DGS fused | `quats` | 1,9e-2 | 5e-2 |
  | 2DGS packed vs. fused | alle | ≤ 1e-6 | 1e-4 |

  Die 3DGS-Kernel stimmen also bis auf Float32-Rauschen überein. Der 2DGS-Kernel weicht bei `quats`, `scales` und `viewmats` systematisch um 0,2 bis 1,9 % von der Referenz ab (über Szenengrößen und Seeds gleich). Das liegt nicht am Patch: `means` läuft durch dieselbe Reduktion und ist exakt, und packed entspricht fused. Die PyTorch-Referenz in float32 liegt dagegen bei etwa 2e-7. gsplat lässt für `quats` in den eigenen Tests 20 % Toleranz zu. Eine kaputte Reduktion (fehlendes oder doppeltes `atomicAdd`) läge um Größenordnungen darüber. **Nicht gemessen:** ob dieselbe 2DGS-Abweichung auf sm_75 mit dem unveränderten Pfad auftritt. Das ginge nur mit der reservierten RTX 2070 SUPER (Zustimmung des POs).
- `peak_vram_mb` im Test (21,5 MB) misst nur den PyTorch-Allocator, nicht den CUDA-Kontext. Die Grenze erzwingt der Allocator.

**Pflege:** Bei jedem gsplat-Update muss der Patch neu erstellt und beide Tests müssen wiederholt werden. Unterstützt gsplat Pascal irgendwann selbst, entfällt der Patch. Ein Upstream-Issue oder -PR wird nicht gestellt; eine öffentliche Aktion ist nicht freigegeben.

Damit ist der offene Punkt „gsplat-Build für sm_61 muss im Spike US-004 nachgewiesen werden“ für den **Build** erledigt. Laufzeit und VRAM des Trainings auf der 1060 weist weiterhin der Spike nach.
