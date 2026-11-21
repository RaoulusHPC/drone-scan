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
