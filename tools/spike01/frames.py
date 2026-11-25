# SPDX-License-Identifier: AGPL-3.0-or-later
"""Frame selection and geometry helpers for the spike (standard library only)."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from tools.spike01.colmap_text import Camera


def display_size(width: int, height: int, rotation: int) -> tuple[int, int]:
    """Frame size after decoding with autorotate: 90/270 degrees swap width and height."""
    return (height, width) if rotation % 180 == 90 else (width, height)


def select_keyframes(
    scores: Sequence[float], *, window: int, min_rel_sharpness: float
) -> list[int]:
    """Indices of the sharpest frame in each non-overlapping window of ``window`` frames.

    Windows whose best score is below ``min_rel_sharpness`` times the median of all scores
    are dropped (motion blur). The last window may be shorter.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if not scores:
        return []
    threshold = min_rel_sharpness * statistics.median(scores)
    selected = []
    for start in range(0, len(scores), window):
        chunk = range(start, min(start + window, len(scores)))
        best = max(chunk, key=lambda index: scores[index])
        if scores[best] >= threshold:
            selected.append(best)
    return selected


def scale_pinhole(camera: Camera, *, new_width: int) -> Camera:
    """Scale an undistorted PINHOLE camera (fx, fy, cx, cy) to a new image width."""
    if camera.model != "PINHOLE":
        raise ValueError(f"expected a PINHOLE camera, got {camera.model}")
    factor = new_width / camera.width
    return Camera(
        camera_id=camera.camera_id,
        model=camera.model,
        width=new_width,
        height=round(camera.height * factor),
        params=tuple(value * factor for value in camera.params),
    )


def ffmpeg_select_expression(indices: Sequence[int]) -> str:
    """ffmpeg ``select`` expression for the given frame numbers (commas escaped for -vf)."""
    if not indices:
        raise ValueError("no frames to select")
    return "+".join(f"eq(n\\,{index})" for index in indices)


def analysis_size(width: int, height: int, *, long_edge: int) -> tuple[int, int]:
    """Size of the small frames used for the sharpness analysis (orientation kept, even)."""
    scale = long_edge / max(width, height)
    small_width = max(2, round(width * scale / 2) * 2)
    small_height = max(2, round(height * scale / 2) * 2)
    return small_width, small_height


def colmap_option_args(group: str, options: Sequence[str]) -> list[str]:
    """``key=value`` pairs as COLMAP flags: ``--<group>.<key> <value>``."""
    args: list[str] = []
    for option in options:
        key, separator, value = option.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"option must look like key=value: {option}")
        args += [f"--{group}.{key}", value]
    return args
