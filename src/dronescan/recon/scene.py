# SPDX-License-Identifier: AGPL-3.0-or-later
"""Undistorted COLMAP scene and splat files for stage ``recon`` (numpy only, host and worker).

The scene is the PINHOLE model (binary, or text in tests) written by COLMAP's
``image_undistorter`` plus one raw
RGB file (``<image name>.rgb``, 8 bit, row-major) per image. The splat is written as the usual
3DGS PLY (positions, SH coefficients, logit opacity, log scales, rotation quaternions).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dronescan.sfm.model import read_cameras

RAW_SUFFIX = ".rgb"
_POINT_FIELDS = 8  # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]


class SceneError(Exception):
    """The undistorted scene is incomplete or not PINHOLE (message is user-facing, German)."""


@dataclass(frozen=True)
class View:
    """One registered image: world-to-camera pose and PINHOLE intrinsics at its file size."""

    name: str
    quat_wxyz: tuple[float, float, float, float]
    translation: tuple[float, float, float]
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def raw_name(self) -> str:
        return self.name + RAW_SUFFIX

    def scaled(self, max_edge: int) -> View:
        """The same view downscaled so that its longest edge is at most ``max_edge``."""
        factor = min(1.0, max_edge / max(self.width, self.height))
        if factor == 1.0:
            return self
        width, height = max(1, round(self.width * factor)), max(1, round(self.height * factor))
        sx, sy = width / self.width, height / self.height
        return View(
            name=self.name,
            quat_wxyz=self.quat_wxyz,
            translation=self.translation,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
        )


@dataclass(frozen=True)
class Scene:
    views: list[View]  # sorted by name
    points: NDArray[np.float32]  # [P, 3]
    colors: NDArray[np.float32]  # [P, 3] in 0..1


_Pose = tuple[str, tuple[float, float, float, float], tuple[float, float, float], int]
_CAMERA_MODELS = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}
_CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}
_BIN_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_TXT_FILES = ("cameras.txt", "images.txt", "points3D.txt")


def read_scene(model_dir: Path) -> Scene:
    """Read an undistorted COLMAP model, binary (as the undistorter writes it) or text."""
    binary = all((model_dir / name).is_file() for name in _BIN_FILES)
    if not binary and not all((model_dir / name).is_file() for name in _TXT_FILES):
        raise SceneError("Das unverzerrte Modell ist unvollständig (cameras, images, points3D).")
    if binary:
        cameras, poses, points, colors = _read_binary(model_dir)
    else:
        cameras, poses, points, colors = _read_text(model_dir)
    views: list[View] = []
    for name, quat, translation, camera_id in poses:
        model, width, height, params = cameras[camera_id]
        if model != "PINHOLE":
            raise SceneError(f"Kamera ist nicht entzerrt (Modell {model}).")
        fx, fy, cx, cy = params
        views.append(View(name, quat, translation, fx, fy, cx, cy, width, height))
    if not views or not len(points):
        raise SceneError("Das unverzerrte Modell enthält keine Bilder oder keine Punkte.")
    return Scene(
        views=sorted(views, key=lambda view: view.name),
        points=np.asarray(points, dtype=np.float32).reshape(-1, 3),
        colors=np.asarray(colors, dtype=np.float32).reshape(-1, 3) / 255.0,
    )


_Cameras = dict[int, tuple[str, int, int, list[float]]]
_Model = tuple[_Cameras, list[_Pose], list[list[float]], list[list[int]]]


def _read_text(text_dir: Path) -> _Model:
    cameras = {
        key: (camera.model, camera.width, camera.height, camera.params)
        for key, camera in read_cameras(text_dir).items()
    }
    poses: list[_Pose] = []
    with (text_dir / "images.txt").open(encoding="utf-8") as handle:
        rows = (line for line in handle if not line.startswith("#"))
        for row in rows:  # two lines per image: pose, then its 2D points
            fields = row.split()
            next(rows, None)
            if fields:
                qw, qx, qy, qz, tx, ty, tz = (float(value) for value in fields[1:8])
                poses.append((fields[9], (qw, qx, qy, qz), (tx, ty, tz), int(fields[8])))
    points, colors = [], []
    with (text_dir / "points3D.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                fields = line.split(maxsplit=_POINT_FIELDS)
                points.append([float(value) for value in fields[1:4]])
                colors.append([int(value) for value in fields[4:7]])
    return cameras, poses, points, colors


def _read_binary(model_dir: Path) -> _Model:
    """COLMAP binary model (layout unchanged in COLMAP 4: cameras, images, points3D)."""
    data = (model_dir / "cameras.bin").read_bytes()
    (count,), offset = struct.unpack_from("<Q", data), 8
    cameras: _Cameras = {}
    for _ in range(count):
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
        offset += 24
        size = _CAMERA_PARAMS.get(model_id)
        if size is None:
            raise SceneError(f"Kamera ist nicht entzerrt (Modell-Nr. {model_id}).")
        params = list(struct.unpack_from(f"<{size}d", data, offset))
        offset += 8 * size
        cameras[camera_id] = (_CAMERA_MODELS[model_id], width, height, params)

    data = (model_dir / "images.bin").read_bytes()
    (count,), offset = struct.unpack_from("<Q", data), 8
    poses: list[_Pose] = []
    for _ in range(count):
        _image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack_from("<I7dI", data, offset)
        offset += 64
        end = data.index(b"\0", offset)
        name = data[offset:end].decode("utf-8")
        (points2d,) = struct.unpack_from("<Q", data, end + 1)
        offset = end + 9 + points2d * 24  # x, y (double), point3D_id (int64)
        poses.append((name, (qw, qx, qy, qz), (tx, ty, tz), camera_id))

    data = (model_dir / "points3D.bin").read_bytes()
    (count,), offset = struct.unpack_from("<Q", data), 8
    points, colors = [], []
    for _ in range(count):
        _point_id, x, y, z, r, g, b, _error, track = struct.unpack_from("<Q3d3BdQ", data, offset)
        offset += 51 + track * 8  # track: image_id, point2D_idx (int32 each)
        points.append([x, y, z])
        colors.append([r, g, b])
    return cameras, poses, points, colors


def split_holdout(count: int, every: int) -> tuple[list[int], list[int]]:
    """Indices (train, held out): every ``every``-th image in name order is held out."""
    held_out = [index for index in range(count) if index % every == 0]
    train = [index for index in range(count) if index % every != 0]
    return train, held_out


def read_raw(path: Path, width: int, height: int) -> NDArray[np.uint8]:
    """A raw RGB frame as [height, width, 3] uint8."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != width * height * 3:
        raise SceneError(f"Rohbild {path.name} hat nicht die Größe {width}×{height}.")
    return data.reshape(height, width, 3)


def write_ply(
    path: Path,
    *,
    means: NDArray[np.float32],
    sh0: NDArray[np.float32],
    sh_rest: NDArray[np.float32],
    opacity_logits: NDArray[np.float32],
    log_scales: NDArray[np.float32],
    quats_wxyz: NDArray[np.float32],
) -> None:
    """Write Gaussians as binary little-endian 3DGS PLY.

    ``sh0`` is [N, 1, 3], ``sh_rest`` [N, K-1, 3] (K = (degree+1)²); the PLY stores the rest
    coefficients channel-major (all R, then G, then B) like the reference implementation.
    Quaternions are normalised.
    """
    count = len(means)
    rest = np.ascontiguousarray(sh_rest.transpose(0, 2, 1)).reshape(count, -1)
    norms = np.linalg.norm(quats_wxyz, axis=1, keepdims=True)
    unit_quats = (quats_wxyz / np.maximum(norms, 1e-12)).astype(np.float32)
    columns: list[tuple[str, NDArray[np.float32]]] = [
        *((name, means[:, i]) for i, name in enumerate("xyz")),
        *((f"n{axis}", np.zeros(count, np.float32)) for axis in "xyz"),
        *((f"f_dc_{i}", sh0[:, 0, i]) for i in range(3)),
        *((f"f_rest_{i}", rest[:, i]) for i in range(rest.shape[1])),
        ("opacity", opacity_logits.reshape(count)),
        *((f"scale_{i}", log_scales[:, i]) for i in range(3)),
        *((f"rot_{i}", unit_quats[:, i]) for i in range(4)),
    ]
    data = np.empty(count, dtype=[(name, "<f4") for name, _ in columns])
    for name, values in columns:
        data[name] = values
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name, _ in columns]
    header.append("end_header")
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(data.tobytes())
    tmp.replace(path)


def read_ply_header(path: Path) -> tuple[int, list[str]]:
    """Vertex count and property names of a PLY written by `write_ply`."""
    names: list[str] = []
    count = 0
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
    return count, names
