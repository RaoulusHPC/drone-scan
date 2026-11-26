# SPDX-License-Identifier: AGPL-3.0-or-later
"""COLMAP camera models drone-scan supports (KAN-29); shared by config and camera profiles."""

from __future__ import annotations

from typing import Literal

CameraModel = Literal[
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"
]
CAMERA_MODELS: tuple[CameraModel, ...] = (
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
)
