"""Image quality metrics for the spike (PyTorch): PSNR and SSIM on [H, W, 3] images in 0..1.

Own implementation (gaussian window 11, sigma 1.5, as in the SSIM paper), so the worker image
needs no extra dependency (torchmetrics, scikit-image).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

_C1 = 0.01**2
_C2 = 0.03**2


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(torch.mean((prediction - target) ** 2))
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def _window(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    gauss = torch.exp(-(axis**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :])[None, None]
    return kernel.expand(channels, 1, size, size).contiguous()


def ssim_tensor(
    prediction: torch.Tensor, target: torch.Tensor, size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """Mean SSIM of two [N, C, H, W] batches (differentiable, used as training loss too)."""
    channels = prediction.shape[1]
    window = _window(size, sigma, channels, prediction.device)

    def blur(image: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(image, window, padding=size // 2, groups=channels)

    mu_x, mu_y = blur(prediction), blur(target)
    var_x = blur(prediction * prediction) - mu_x**2
    var_y = blur(target * target) - mu_y**2
    covariance = blur(prediction * target) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + _C1) * (2 * covariance + _C2)
    denominator = (mu_x**2 + mu_y**2 + _C1) * (var_x + var_y + _C2)
    return (numerator / denominator).mean()


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM of two [H, W, 3] images in 0..1."""
    to_nchw = lambda image: image.permute(2, 0, 1)[None]  # noqa: E731
    return float(ssim_tensor(to_nchw(prediction), to_nchw(target)))
