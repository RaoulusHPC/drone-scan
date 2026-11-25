# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image measures on small grey frames: sharpness, exposure and similarity (numpy, CPU)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Gray = NDArray[np.uint8]
Thumbnail = NDArray[np.float32]

_FLAT_STD = 0.5  # grey levels; below this a thumbnail carries no usable structure


def laplacian_variance(frame: Gray) -> float:
    """Variance of the 4-neighbour Laplacian; high for sharp, low for blurred frames."""
    data = frame.astype(np.float32)
    laplacian = (
        4 * data[1:-1, 1:-1] - data[:-2, 1:-1] - data[2:, 1:-1] - data[1:-1, :-2] - data[1:-1, 2:]
    )
    return float(laplacian.var())


def brightness(frame: Gray) -> float:
    """Mean grey level (0..255); low = underexposed, high = overexposed (KAN-34)."""
    return float(frame.astype(np.float32).mean())


def thumbnail(frame: Gray, long_edge: int) -> Thumbnail:
    """Block average; suppresses sensor noise so that similarity reflects the view."""
    block = max(1, max(frame.shape) // long_edge)
    rows, cols = max(1, frame.shape[0] // block), max(1, frame.shape[1] // block)
    block_rows, block_cols = frame.shape[0] // rows, frame.shape[1] // cols
    cropped = frame[: rows * block_rows, : cols * block_cols].astype(np.float32)
    blocks = cropped.reshape(rows, block_rows, cols, block_cols)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def similarity(first: Thumbnail, second: Thumbnail) -> float:
    """Normalised cross-correlation in [-1, 1]; 1 = same view (exposure does not matter).

    Two structureless views (e.g. a plain wall) are the same; a structureless and a
    structured one are not.
    """
    flat = (float(first.std()) < _FLAT_STD, float(second.std()) < _FLAT_STD)
    if any(flat):
        return 1.0 if all(flat) else 0.0
    a, b = first - first.mean(), second - second.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
