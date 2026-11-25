# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic scene for tests of the spike trainer (runs in the worker container).

Renders a few hundred random coloured Gaussians from cameras on a circle with gsplat and
writes them as a COLMAP text model (PINHOLE) plus raw RGB frames, the same input layout the
trainer reads from real recordings. No real data involved.

Usage: python -m tools.spike01.synth_scene OUT_DIR
Writes OUT_DIR/{cameras.txt,images.txt,points3D.txt} and OUT_DIR/raw/<name>.rgb.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from gsplat import rasterization

WIDTH, HEIGHT, FOCAL = 320, 240, 300.0
VIEWS = 12


def rotation_to_quat(matrix: torch.Tensor) -> tuple[float, float, float, float]:
    """Rotation matrix to unit quaternion (w, x, y, z), w >= 0."""
    m = matrix.tolist()
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w, x, y, z = (
            s / 4,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w, x, y, z = (
            (m[2][1] - m[1][2]) / s,
            s / 4,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w, x, y, z = (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            s / 4,
            (m[1][2] + m[2][1]) / s,
        )
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w, x, y, z = (
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            s / 4,
        )
    return (w, x, y, z) if w >= 0 else (-w, -x, -y, -z)


def look_at(centre: torch.Tensor) -> torch.Tensor:
    """World-to-camera 4x4 for a camera at ``centre`` looking at the origin (OpenCV axes)."""
    forward = -centre / centre.norm()
    right = torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0]))
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward])  # rows: world-to-camera
    viewmat = torch.eye(4)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ centre
    return viewmat


def write_scene(out: Path) -> None:
    torch.manual_seed(1)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    count = 400
    means = torch.rand(count, 3) - 0.5
    colors = torch.rand(count, 3)
    scales = torch.full((count, 3), 0.06)
    quats = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)
    opacities = torch.full((count,), 0.9)
    intrinsics = torch.tensor([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1.0]])

    poses = []
    for view in range(VIEWS):
        angle = 2 * math.pi * view / VIEWS
        centre = torch.tensor([3.0 * math.cos(angle), 3.0 * math.sin(angle), 0.8])
        poses.append(look_at(centre))
    with torch.no_grad():
        images, _alphas, _info = rasterization(
            means.cuda(),
            quats.cuda(),
            scales.cuda(),
            opacities.cuda(),
            colors.cuda(),
            torch.stack(poses).cuda(),
            intrinsics[None].repeat(VIEWS, 1, 1).cuda(),
            WIDTH,
            HEIGHT,
        )

    (out / "cameras.txt").write_text(
        f"1 PINHOLE {WIDTH} {HEIGHT} {FOCAL} {FOCAL} {WIDTH / 2} {HEIGHT / 2}\n"
    )
    lines = []
    for view, pose in enumerate(poses):
        name = f"v_{view:02d}.jpg"
        w, x, y, z = rotation_to_quat(pose[:3, :3])
        tx, ty, tz = pose[:3, 3].tolist()
        lines += [f"{view + 1} {w} {x} {y} {z} {tx} {ty} {tz} 1 {name}", ""]
        frame = (images[view].clamp(0, 1) * 255).round().byte().cpu()
        (out / "raw" / f"{name}.rgb").write_bytes(frame.numpy().tobytes())
    (out / "images.txt").write_text("\n".join(lines) + "\n")

    noisy = means + 0.02 * torch.randn_like(means)  # like a sparse SfM cloud: close, not exact
    rgb = (colors * 255).round().int().tolist()
    (out / "points3D.txt").write_text(
        "".join(
            f"{i + 1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0.5 1 0\n"
            for i, (p, c) in enumerate(zip(noisy.tolist(), rgb, strict=True))
        )
    )


if __name__ == "__main__":
    write_scene(Path(sys.argv[1]))
