# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic textured room for SfM tests (numpy only; no real recordings, CLAUDE.md rule 3).

A camera walks a closed circle inside a box room and looks across it, so consecutive views
overlap and the last views close the loop to the first ones. Poses are known exactly.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

ROOM = np.array([5.0, 4.0, 2.6])  # metres
TEXELS_PER_M = 220
CIRCLE_RADIUS_M = 0.9
PIXEL_NOISE = 2.0  # grey levels


@dataclass(frozen=True)
class View:
    rotation: Array  # world to camera (OpenCV axes: x right, y down, z forward)
    centre: Array


def _texture(rng: np.random.Generator, width_m: float, height_m: float) -> NDArray[np.float32]:
    """Random blocks at several scales: many corners, unique everywhere."""
    rows, cols = round(height_m * TEXELS_PER_M), round(width_m * TEXELS_PER_M)
    image = np.zeros((rows, cols), dtype=np.float32)
    for block, weight in ((64, 0.35), (24, 0.3), (9, 0.2), (4, 0.15)):
        noise = rng.random((rows // block + 1, cols // block + 1), dtype=np.float32)
        tiled = np.repeat(np.repeat(noise, block, axis=0), block, axis=1)
        image += weight * tiled[:rows, :cols]
    return 30 + 200 * image


def views(count: int, turns: float = 1.0) -> list[View]:
    result = []
    for index in range(count):
        angle = 2 * np.pi * turns * index / count
        outward = np.array([np.cos(angle), np.sin(angle), 0.0])
        centre = ROOM / 2 + CIRCLE_RADIUS_M * outward
        centre[2] += 0.2 * np.sin(3 * angle)
        # Looking across the room: sideways motion gives parallax (outward views mostly rotate).
        forward = -outward + np.array([0, 0, 0.12 * np.sin(5 * angle)])
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        result.append(View(rotation=np.stack([right, down, forward]), centre=centre))
    return result


def render(
    view: View, textures: dict[tuple[int, int], NDArray[np.float32]], size: tuple[int, int]
) -> NDArray[np.float32]:
    width, height = size
    focal = 0.8 * width
    u, v = np.meshgrid(np.arange(width) + 0.5 - width / 2, np.arange(height) + 0.5 - height / 2)
    rays = np.stack([u / focal, v / focal, np.ones_like(u)], axis=-1) @ view.rotation
    # Distance to the six walls of the box; the camera is inside, so the nearest hit counts.
    target = np.where(rays > 0, ROOM, 0.0)
    with np.errstate(divide="ignore"):
        distance = np.where(rays != 0, (target - view.centre) / rays, np.inf)
    axis = distance.argmin(axis=-1)
    hit = view.centre + rays * distance.min(axis=-1, keepdims=True)
    image = np.zeros((height, width), dtype=np.float32)
    for (wall_axis, side), texture in textures.items():
        positive = np.take_along_axis(rays, axis[..., None], axis=-1)[..., 0] > 0
        mask = (axis == wall_axis) & (positive == bool(side))
        a, b = (i for i in range(3) if i != wall_axis)
        x = np.clip(hit[mask][:, a] * TEXELS_PER_M - 0.5, 0, texture.shape[1] - 1.001)
        y = np.clip(hit[mask][:, b] * TEXELS_PER_M - 0.5, 0, texture.shape[0] - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        top = texture[y0, x0] * (1 - fx) + texture[y0, x0 + 1] * fx
        bottom = texture[y0 + 1, x0] * (1 - fx) + texture[y0 + 1, x0 + 1] * fx
        image[mask] = top * (1 - fy) + bottom * fy
    return image


def write_jpegs(out_dir: Path, poses: list[View], size: tuple[int, int], seed: int = 0) -> None:
    """Render ``poses`` to ``000001.jpg``, ... (grey noise added, encoded with ffmpeg)."""
    rng = np.random.default_rng(seed)
    textures = {}
    for axis in range(3):
        a, b = (i for i in range(3) if i != axis)
        for side in (0, 1):
            textures[axis, side] = _texture(rng, ROOM[a], ROOM[b])
    out_dir.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray"]
    command += ["-s", f"{size[0]}x{size[1]}", "-i", "-", "-pix_fmt", "yuvj420p", "-qscale:v", "2"]
    with subprocess.Popen([*command, str(out_dir / "%06d.jpg")], stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for pose in poses:
            noise = rng.normal(0, PIXEL_NOISE, (size[1], size[0]))
            image = np.clip(render(pose, textures, size) + noise, 0, 255)
            proc.stdin.write(image.astype(np.uint8).tobytes())
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg could not write the synthetic views")


def aligned_position_errors(estimated: Array, truth: Array) -> Array:
    """Camera position errors (m) after the best similarity transform (Umeyama 1991)."""
    mean_e, mean_t = estimated.mean(axis=0), truth.mean(axis=0)
    e, t = estimated - mean_e, truth - mean_t
    u, s, vt = np.linalg.svd(t.T @ e / len(e))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(s) @ d) / (e**2).sum() * len(e))
    errors: Array = np.linalg.norm(scale * e @ rotation.T - t, axis=1)
    return errors
