# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pruning and SH reduction of splats for playback on phones (KAN-42)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dronescan.recon.scene import Splats


def _opacity(splats: Splats) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-splats.opacity_logits.astype(np.float64)))


def importance(splats: Splats) -> NDArray[np.float64]:
    """Opacity × area of the two largest axes: how much a Gaussian covers on screen."""
    log_scales = splats.log_scales.astype(np.float64)
    return _opacity(splats) * np.exp(log_scales.sum(axis=1) - log_scales.min(axis=1))


def drop_transparent(splats: Splats, min_opacity: float) -> Splats:
    """Drop Gaussians whose opacity is below ``min_opacity`` (order is kept)."""
    return splats.subset(np.flatnonzero(_opacity(splats) >= min_opacity))


def cap(splats: Splats, max_gaussians: int) -> Splats:
    """Keep the ``max_gaussians`` most important Gaussians in their original order."""
    if len(splats) <= max_gaussians:
        return splats
    top = np.argpartition(-importance(splats), max_gaussians - 1)[:max_gaussians]
    return splats.subset(np.sort(top))


def reduce_sh(splats: Splats, degree: int) -> Splats:
    """Drop SH coefficients above ``degree`` (no-op if the splats have fewer)."""
    if degree >= splats.sh_degree:
        return splats
    rest = (degree + 1) ** 2 - 1
    return Splats(
        means=splats.means,
        sh0=splats.sh0,
        sh_rest=np.ascontiguousarray(splats.sh_rest[:, :rest]),
        opacity_logits=splats.opacity_logits,
        log_scales=splats.log_scales,
        quats_wxyz=splats.quats_wxyz,
    )
