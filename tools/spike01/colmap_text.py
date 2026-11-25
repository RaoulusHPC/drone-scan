# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure-Python reader for COLMAP text models (cameras.txt, images.txt, points3D.txt).

Standard library only, so it runs on the host (tests) and inside the worker container.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class Image:
    image_id: int
    quat_wxyz: tuple[float, float, float, float]
    translation: tuple[float, float, float]
    camera_id: int
    name: str


@dataclass(frozen=True)
class Point3D:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float


@dataclass(frozen=True)
class RegistrationStats:
    registered: int
    total: int
    percent: float


def _data_lines(text: str) -> list[str]:
    """All lines that are not comments; empty lines are kept (images.txt uses them)."""
    return [line.strip() for line in text.splitlines() if not line.lstrip().startswith("#")]


def parse_cameras_txt(text: str) -> dict[int, Camera]:
    cameras = {}
    for line in _data_lines(text):
        if not line:
            continue
        fields = line.split()
        camera = Camera(
            camera_id=int(fields[0]),
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            params=tuple(float(value) for value in fields[4:]),
        )
        cameras[camera.camera_id] = camera
    return cameras


def parse_images_txt(text: str) -> list[Image]:
    """Two lines per image (pose, then 2D points; the second one may be empty)."""
    images = []
    for pose in _data_lines(text)[0::2]:  # every second line is a pose; the others are 2D points
        if not pose:
            continue
        fields = pose.split()
        images.append(
            Image(
                image_id=int(fields[0]),
                quat_wxyz=(
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                ),
                translation=(float(fields[5]), float(fields[6]), float(fields[7])),
                camera_id=int(fields[8]),
                name=" ".join(fields[9:]),
            )
        )
    return images


def parse_points3d_txt(text: str) -> list[Point3D]:
    points = []
    for line in _data_lines(text):
        if not line:
            continue
        fields = line.split()
        points.append(
            Point3D(
                point_id=int(fields[0]),
                xyz=(float(fields[1]), float(fields[2]), float(fields[3])),
                rgb=(int(fields[4]), int(fields[5]), int(fields[6])),
                error=float(fields[7]),
            )
        )
    return points


def registration_stats(images: list[Image], total_frames: int) -> RegistrationStats:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    return RegistrationStats(
        registered=len(images),
        total=total_frames,
        percent=round(100.0 * len(images) / total_frames, 2),
    )
