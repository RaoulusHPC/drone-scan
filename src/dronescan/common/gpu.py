"""GPU helpers shared by all GPU stages: VRAM limit and precision choice.

`apply_vram_limit` must be called once at the start of every GPU stage, before
the first allocation (wired up by the stage framework, KAN-21). PyTorch is
imported lazily so CPU-only code and tests never need it.

The limit only caps PyTorch's caching allocator in this process. It does not
cover the CUDA context (~0.3-0.5 GiB) or separate processes such as COLMAP;
their VRAM must be bounded by profile parameters (e.g. ``sfm.max_image_edge_px``).
"""

from __future__ import annotations

import importlib
from types import ModuleType

from dronescan.common.config import Settings

GIB = 1024**3
MIN_FP16_CAPABILITY = (7, 0)  # sm_70 (Volta) and newer; Pascal (sm_61) stays fp32, ADR-0010


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


def apply_vram_limit(settings: Settings, device: int | None = None) -> float:
    """Cap PyTorch's allocator at ``settings.vram_limit_gb`` GiB.

    ``device`` defaults to the current CUDA device. Returns the fraction passed
    to ``torch.cuda.set_per_process_memory_fraction`` (clamped to 1.0 if the
    limit exceeds the card). Calling it again simply overwrites the fraction.
    """
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise GpuUnavailableError(
            "Keine CUDA-GPU verfügbar. Abhilfe: NVIDIA-Treiber prüfen (nvidia-smi) bzw. "
            "den Container mit '--gpus all' starten."
        )
    index = torch.cuda.current_device() if device is None else device
    total_bytes = int(torch.cuda.get_device_properties(index).total_memory)
    fraction = min(1.0, settings.vram_limit_gb * GIB / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction


def use_fp16(settings: Settings, compute_capability: tuple[int, int]) -> bool:
    """Whether a stage may use fp16/AMP on a GPU with ``compute_capability``.

    Pass the capability of the same device that `apply_vram_limit` used.
    """
    return settings.gpu.allow_fp16 and compute_capability >= MIN_FP16_CAPABILITY
