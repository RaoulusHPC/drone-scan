# ADR-0010: Entwicklungs-GPU GTX 1080 neben Ziel-GPU RTX 2060

- **Status:** angenommen
- **Datum:** 2026-11-21

## Kontext
Claude Code läuft lokal auf dem Entwicklungsrechner mit einer **GTX 1080 (8 GB, Pascal, sm_61)**. Zielhardware der Nutzer bleibt die **RTX 2060 (6 GB, Turing, sm_75)**.
PyTorch hat Pascal aus den CUDA-12.8/12.9- und CUDA-13-Builds entfernt. Nur die CUDA-12.6-Builds unterstützen Pascal noch, und laut PyTorch-Ankündigung ist **2.14 die letzte Version mit CUDA-12.x-Wheels**; ab 2.15 ist Pascal nur noch per Eigenbau nutzbar. Pascal hat zudem keine Tensor Cores (fp16 bringt kaum Vorteile).

## Entscheidung
- Basis-Image und Wheels: **CUDA 12.6**, **PyTorch < 2.15** (cu126-Index), in `uv.lock` gepinnt.
- Eigene CUDA-Erweiterungen (gsplat) und COLMAP werden für **`TORCH_CUDA_ARCH_LIST="6.1;7.5"`** bzw. `CMAKE_CUDA_ARCHITECTURES="61;75"` gebaut.
- Präzision: Profilparameter `precision` = `fp32` auf Pascal, `fp16`/`mixed` erlaubt ab Turing (automatische Erkennung über `torch.cuda.get_device_capability()`).
- VRAM-Tests laufen auf der GTX 1080 mit 5,5-GB-Grenze (Zielrechner-Simulation).
- Laufzeitmessungen auf der GTX 1080 sind **nicht** 1:1 auf die RTX 2060 übertragbar; Q-1 wird vor M1 einmal auf Turing-Hardware verifiziert (eigener Laptop oder geliehenes Gerät).

## Konsequenzen
+ Entwicklung ohne neue Hardware möglich, 8 GB geben Luft für Profilvergleiche
- Toolchain eingefroren auf CUDA 12.6 / PyTorch ≤ 2.14 → Sicherheits- und Feature-Updates enden; spätestens bei M5 GPU-Wechsel prüfen (Risiko R-13)
- gsplat-Build für sm_61 muss im Spike US-004 nachgewiesen werden; Fallback: Training im Container auf Turing-Hardware
