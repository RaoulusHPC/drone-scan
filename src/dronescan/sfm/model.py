# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read COLMAP sparse models: image/point counts (binary) and poses statistics (text)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_COUNT = struct.Struct("<Q")  # first field of images.bin and points3D.bin
_NAME_FIELD = 9  # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
_ERROR_FIELD = 7  # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]


@dataclass(frozen=True)
class SparseModel:
    """One (partial) reconstruction written by the mapper."""

    path: Path
    images: int


@dataclass(frozen=True)
class ModelReport:
    image_names: list[str]
    points: int
    mean_reprojection_error_px: float


def _count(path: Path) -> int:
    with path.open("rb") as handle:
        (value,) = _COUNT.unpack(handle.read(_COUNT.size))
    return int(value)


def list_models(models_dir: Path) -> list[SparseModel]:
    """Models with at least one point, largest first (the mapper's numbering as tie-break)."""
    models = []
    for path in models_dir.iterdir():
        files = (path / "images.bin", path / "points3D.bin")
        # The mapper may leave directories without points (failed initialisation).
        if all(file.is_file() for file in files) and _count(files[1]) > 0:
            models.append(SparseModel(path=path, images=_count(files[0])))
    return sorted(models, key=lambda model: (-model.images, int(model.path.name)))


def read_report(text_dir: Path) -> ModelReport:
    """Statistics of a model in COLMAP text format."""
    names: list[str] = []
    with (text_dir / "images.txt").open(encoding="utf-8") as handle:
        rows = (line for line in handle if not line.startswith("#"))
        for row in rows:  # two lines per image: pose, then its 2D points (may be empty)
            if row.strip():
                names.append(row.split()[_NAME_FIELD])
            next(rows, None)
    points, error_sum = 0, 0.0
    with (text_dir / "points3D.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                points += 1
                error_sum += float(line.split(maxsplit=_ERROR_FIELD + 1)[_ERROR_FIELD])
    return ModelReport(
        image_names=names,
        points=points,
        mean_reprojection_error_px=error_sum / points if points else 0.0,
    )
