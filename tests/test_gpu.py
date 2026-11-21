"""VRAM limit and precision helpers for GPU stages (KAN-20, AC 2).

CPU tests use a fake `torch` module; the `gpu` test uses the real one.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field

import pytest

from dronescan.common.config import load_settings
from dronescan.common.gpu import GpuUnavailableError, apply_vram_limit, use_fp16

GIB = 1024**3


@dataclass
class FakeCuda:
    available: bool = True
    total_memory: int = 8 * GIB
    calls: list[tuple[float, int]] = field(default_factory=list)

    def is_available(self) -> bool:
        return self.available

    def get_device_properties(self, device: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(total_memory=self.total_memory)

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None:
        self.calls.append((fraction, device))


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
    cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.delenv("DRONESCAN_VRAM_LIMIT_GB", raising=False)
    monkeypatch.delenv("DRONESCAN_PROFILE", raising=False)
    return cuda


def test_env_limit_sets_memory_fraction(
    fake_cuda: FakeCuda, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given DRONESCAN_VRAM_LIMIT_GB=4 and an 8 GiB GPU,
    When a GPU stage applies the VRAM limit,
    Then torch.cuda.set_per_process_memory_fraction(0.5, device) is called.
    """
    monkeypatch.setenv("DRONESCAN_VRAM_LIMIT_GB", "4")

    fraction = apply_vram_limit(load_settings(), device=1)

    assert fraction == pytest.approx(0.5)
    assert len(fake_cuda.calls) == 1
    called_fraction, called_device = fake_cuda.calls[0]
    assert called_fraction == pytest.approx(0.5)
    assert called_device == 1


def test_profile_limit_on_8gb_card(fake_cuda: FakeCuda) -> None:
    """Given the low_vram_6gb profile (5.5 GB) on an 8 GiB development GPU,
    When the limit is applied,
    Then the fraction simulates the 6 GB target card.
    """
    fraction = apply_vram_limit(load_settings("low_vram_6gb"))

    assert fraction == pytest.approx(5.5 / 8)


def test_limit_above_card_size_is_clamped(fake_cuda: FakeCuda) -> None:
    """Given a limit larger than the physical GPU memory,
    When the limit is applied,
    Then the fraction is clamped to 1.0.
    """
    fake_cuda.total_memory = 4 * GIB

    assert apply_vram_limit(load_settings("default_8gb")) == pytest.approx(1.0)


def test_missing_cuda_raises_helpful_error(fake_cuda: FakeCuda) -> None:
    """Given a machine without a usable CUDA device,
    When a GPU stage applies the VRAM limit,
    Then a GpuUnavailableError explains the cause and remedy.
    """
    fake_cuda.available = False

    with pytest.raises(GpuUnavailableError, match="CUDA"):
        apply_vram_limit(load_settings())


def test_missing_torch_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given an environment without PyTorch,
    When a GPU stage applies the VRAM limit,
    Then a GpuUnavailableError explains how to install it.
    """
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(GpuUnavailableError, match="PyTorch"):
        apply_vram_limit(load_settings())


@pytest.mark.parametrize(
    ("allow", "capability", "expected"),
    [
        (True, (7, 5), True),  # RTX 2060 (target)
        (True, (6, 1), False),  # GTX 1080 (Pascal): always fp32, ADR-0010
        (False, (7, 5), False),  # profile forbids fp16
    ],
)
def test_fp16_only_on_turing_or_newer(
    allow: bool, capability: tuple[int, int], expected: bool
) -> None:
    """Given a profile's allow_fp16 flag and the GPU compute capability,
    When the precision is chosen,
    Then fp16 is used only if allowed and the GPU is sm_70 or newer.
    """
    settings = load_settings()
    settings = settings.model_copy(
        update={"gpu": settings.gpu.model_copy(update={"allow_fp16": allow})}
    )

    assert use_fp16(settings, capability) is expected


@pytest.mark.gpu
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_real_gpu_respects_limit() -> None:
    """Given a real CUDA GPU and DRONESCAN_VRAM_LIMIT_GB (set by `make test-gpu`),
    When the limit is applied and more memory is requested than allowed,
    Then PyTorch raises an out-of-memory error.
    """
    import importlib

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    limit_gb = float(os.environ.get("DRONESCAN_VRAM_LIMIT_GB", "5.5"))

    apply_vram_limit(load_settings())

    with pytest.raises(torch.cuda.OutOfMemoryError):
        torch.empty(int((limit_gb + 0.5) * GIB), dtype=torch.uint8, device="cuda")
