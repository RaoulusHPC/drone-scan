# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sharpness (variance of the Laplacian) per frame from raw 8-bit grey frames (numpy).

Usage (inside the worker container): python -m tools.spike01.sharpness RAW WIDTH HEIGHT
Prints a JSON list with one score per frame.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def laplacian_variance(frames: np.ndarray) -> list[float]:
    """Variance of the 4-neighbour Laplacian of each [H, W] frame in a [N, H, W] array."""
    data = frames.astype(np.float32)
    centre = data[:, 1:-1, 1:-1]
    laplacian = (
        4 * centre - data[:, :-2, 1:-1] - data[:, 2:, 1:-1] - data[:, 1:-1, :-2] - data[:, 1:-1, 2:]
    )
    return [round(float(value), 3) for value in laplacian.reshape(len(frames), -1).var(axis=1)]


def main() -> int:
    path, width, height = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    raw = np.fromfile(path, dtype=np.uint8)
    frames = raw.reshape(-1, height, width)
    print(json.dumps(laplacian_variance(frames)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
