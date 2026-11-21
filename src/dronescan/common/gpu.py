"""GPU helpers shared by all GPU stages: VRAM limit and precision choice.

`apply_vram_limit` must be called once at the start of every GPU stage
(wired up by the stage framework, KAN-21). PyTorch is imported lazily so
CPU-only code and tests never need it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from dronescan.common.config import Settings

GIB = 1024**3
MIN_FP16_CAPABILITY = (7, 0)  # Volta+; Pascal (sm_61) stays fp32, ADR-0010


class GpuUnavailableError(RuntimeError):
    """No usable PyTorch/CUDA setup for a GPU stage."""


def _import_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise GpuUnavailableError(
            "PyTorch ist nicht installiert. Abhilfe: GPU-Abhängigkeiten installieren "
            "(PyTorch < 2.15 mit CUDA 12.6, siehe ADR-0010)."
        ) from exc


def apply_vram_limit(settings: Settings, device: int = 0) -> float:
    """Cap PyTorch's allocator at ``settings.vram_limit_gb`` GiB on ``device``.

    Returns the fraction passed to ``torch.cuda.set_per_process_memory_fraction``.
    The CUDA context itself (a few hundred MB) is not covered by this limit, so
    profiles keep headroom below the physical 6 GB of the target card.
    """
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise GpuUnavailableError(
            "Keine CUDA-GPU verfügbar. Abhilfe: NVIDIA-Treiber prüfen (nvidia-smi) bzw. "
            "den Container mit '--gpus all' starten."
        )
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = min(1.0, settings.vram_limit_gb * GIB / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return fraction


def use_fp16(settings: Settings, compute_capability: tuple[int, int]) -> bool:
    """Whether a stage may use fp16/AMP on a GPU with ``compute_capability``."""
    return settings.gpu.allow_fp16 and compute_capability >= MIN_FP16_CAPABILITY
