# SPDX-License-Identifier: AGPL-3.0-or-later
"""Camera calibration from a chessboard or ChArUco video (KAN-29, optional).

Frames come from the ffmpeg adapter (display orientation), pattern detection and calibration
from OpenCV (extra ``calibration``, imported lazily). The result is a camera profile in COLMAP
convention: OpenCV puts the centre of the first pixel at (0, 0), COLMAP at (0.5, 0.5).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from dronescan.capture.camera import CameraProfile, Intrinsics, principal_point_index
from dronescan.capture.probe import ProbeError, probe_video
from dronescan.common import error_catalog as catalog
from dronescan.common.camera_models import CameraModel
from dronescan.common.config import CalibrationSettings
from dronescan.common.stage import ExitCode
from dronescan.ingest.ffmpeg import FfmpegError, decode_gray, fit_size

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

Pattern = Literal["chessboard", "charuco"]
_COLMAP_PIXEL_OFFSET = 0.5

log = structlog.get_logger(__name__)


class CalibrationError(Exception):
    """User-facing (German) failure with hints and the exit code of the command."""

    def __init__(self, message: str, code: str, hints: list[str] | None = None) -> None:
        super().__init__(message)
        entry = catalog.get(code)
        self.code = code
        self.hints = list(entry.hints if hints is None else hints)
        self.exit_code = ExitCode(entry.exit_code)


@dataclass(frozen=True)
class Board:
    """Calibration pattern; ``cols`` x ``rows`` count squares, lengths in millimetres."""

    pattern: Pattern
    cols: int
    rows: int
    square_mm: float
    marker_mm: float | None = None
    dictionary: str = "DICT_5X5_100"


@dataclass(frozen=True)
class CalibrationResult:
    profile: CameraProfile
    views: int
    frames: int


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise CalibrationError(
            "Für die Kalibrierung fehlt OpenCV.", "capture.calibration_opencv_missing"
        ) from exc
    return cv2


class _Detector:
    """Finds the pattern in one grey frame: object points (mm) and image points (px)."""

    def __init__(self, cv2: Any, board: Board) -> None:
        import numpy as np

        self._cv2 = cv2
        self._board = board
        if board.pattern == "chessboard":
            inner = (board.cols - 1, board.rows - 1)
            grid = np.mgrid[0 : inner[0], 0 : inner[1]].T.reshape(-1, 2)
            self._chessboard = np.zeros((len(grid), 3), np.float32)
            self._chessboard[:, :2] = grid * board.square_mm
            return
        if board.marker_mm is None or board.marker_mm >= board.square_mm:
            raise CalibrationError(
                "Für ChArUco fehlt die Markergröße oder sie ist nicht kleiner als ein Feld.",
                "capture.calibration_board_invalid",
                ["Abhilfe: --marker-mm mit der gemessenen Kantenlänge eines Markers angeben."],
            )
        dictionary_id = getattr(cv2.aruco, board.dictionary, None)
        if not board.dictionary.startswith("DICT_") or dictionary_id is None:
            raise CalibrationError(
                f"Unbekanntes ArUco-Wörterbuch '{board.dictionary}'.",
                "capture.calibration_board_invalid",
                ["Abhilfe: ein Wörterbuch wie DICT_4X4_50 oder DICT_5X5_100 angeben."],
            )
        charuco = cv2.aruco.CharucoBoard(
            (board.cols, board.rows),
            board.square_mm,
            board.marker_mm,
            cv2.aruco.getPredefinedDictionary(dictionary_id),
        )
        self._charuco_corners = charuco.getChessboardCorners()
        self._charuco = cv2.aruco.CharucoDetector(charuco)

    def detect(
        self, gray: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
        import numpy as np

        cv2 = self._cv2
        if self._board.pattern == "chessboard":
            inner = (self._board.cols - 1, self._board.rows - 1)
            found, corners = cv2.findChessboardCornersSB(gray, inner, flags=cv2.CALIB_CB_ACCURACY)
            if not found:
                return None
            return self._chessboard, np.asarray(corners, np.float32).reshape(-1, 2)
        corners, ids, _, _ = self._charuco.detectBoard(gray)
        if ids is None:
            return None
        objects = np.asarray(self._charuco_corners, np.float32)[ids.flatten()]
        return objects, np.asarray(corners, np.float32).reshape(-1, 2)


def _spread[T](items: list[T], count: int) -> list[T]:
    """At most ``count`` items, evenly spread (first and last included)."""
    if len(items) <= count:
        return items
    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


_FLAGS: dict[str, tuple[str, ...]] = {
    "SIMPLE_PINHOLE": (
        "CALIB_FIX_ASPECT_RATIO",
        "CALIB_ZERO_TANGENT_DIST",
        "CALIB_FIX_K1",
        "CALIB_FIX_K2",
        "CALIB_FIX_K3",
    ),
    "PINHOLE": ("CALIB_ZERO_TANGENT_DIST", "CALIB_FIX_K1", "CALIB_FIX_K2", "CALIB_FIX_K3"),
    "SIMPLE_RADIAL": (
        "CALIB_FIX_ASPECT_RATIO",
        "CALIB_ZERO_TANGENT_DIST",
        "CALIB_FIX_K2",
        "CALIB_FIX_K3",
    ),
    "RADIAL": ("CALIB_FIX_ASPECT_RATIO", "CALIB_ZERO_TANGENT_DIST", "CALIB_FIX_K3"),
    "OPENCV": ("CALIB_FIX_K3",),
}


def _solve(
    cv2: Any,
    model: CameraModel,
    objects: list[NDArray[np.float32]],
    images: list[NDArray[np.float32]],
    size: tuple[int, int],
) -> tuple[float, list[float]]:
    """Calibrate; returns the RMS error (px) and the COLMAP parameters (OpenCV convention)."""
    import numpy as np

    if model == "OPENCV_FISHEYE":
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, matrix, dist, _, _ = cv2.fisheye.calibrate(
            [o.reshape(-1, 1, 3).astype(np.float64) for o in objects],
            [i.reshape(-1, 1, 2).astype(np.float64) for i in images],
            size,
            np.eye(3),
            np.zeros(4),
            flags=flags,
        )
        k = dist.flatten()
        return float(rms), [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2], *k[:4]]
    flags = 0
    for name in _FLAGS[model]:
        flags |= getattr(cv2, name)
    rms, matrix, dist, _, _ = cv2.calibrateCamera(
        objects, images, size, np.eye(3), np.zeros(5), flags=flags
    )
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    k1, k2, p1, p2 = dist.flatten()[:4]
    params = {
        "SIMPLE_PINHOLE": [fx, cx, cy],
        "PINHOLE": [fx, fy, cx, cy],
        "SIMPLE_RADIAL": [fx, cx, cy, k1],
        "RADIAL": [fx, cx, cy, k1, k2],
        "OPENCV": [fx, fy, cx, cy, k1, k2, p1, p2],
    }[model]
    return float(rms), params


def _to_colmap(model: CameraModel, params: list[float]) -> list[float]:
    """Shift the principal point from OpenCV to COLMAP pixel convention."""
    shifted = [float(value) for value in params]
    cx = principal_point_index(model)
    for index in (cx, cx + 1):
        shifted[index] += _COLMAP_PIXEL_OFFSET
    return shifted


def calibrate(
    video: Path,
    board: Board,
    model: CameraModel,
    name: str,
    cfg: CalibrationSettings,
) -> CalibrationResult:
    """Detect ``board`` in ``video`` and calibrate ``model``; raises CalibrationError."""
    cv2 = _cv2()
    detector = _Detector(cv2, board)
    if not video.is_file():
        raise CalibrationError(
            f"Video '{video.name}' nicht gefunden.",
            "capture.calibration_video_unreadable",
            ["Abhilfe: Pfad prüfen."],
        )
    if shutil.which("ffprobe") is None:
        raise CalibrationError(
            "ffmpeg/ffprobe ist nicht installiert.",
            "capture.calibration_ffmpeg_missing",
        )
    try:
        meta = probe_video(video)
    except ProbeError as exc:
        raise CalibrationError(str(exc), "capture.calibration_video_unreadable") from exc
    size = fit_size(meta.display_width, meta.display_height, cfg.max_image_edge_px)
    views: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
    frames = 0

    def on_frame(gray: NDArray[np.uint8]) -> None:
        nonlocal frames
        frames += 1
        found = detector.detect(gray)
        if found is not None and len(found[0]) >= cfg.min_corners_per_view:
            views.append(found)

    try:
        decode_gray(video, size, cfg.sample_fps, on_frame)
    except FfmpegError as exc:
        raise CalibrationError(str(exc), "capture.calibration_video_unreadable") from exc
    log.info("calibration_views", frames=frames, views=len(views))
    if len(views) < cfg.min_views:
        raise CalibrationError(
            f"Das Muster wurde nur in {len(views)} von {frames} untersuchten Bildern gefunden "
            f"(mindestens {cfg.min_views} nötig).",
            "capture.calibration_quality",
        )
    used = _spread(views, cfg.max_views)
    try:
        rms, params = _solve(cv2, model, [v[0] for v in used], [v[1] for v in used], size)
    except cv2.error as exc:
        raise CalibrationError(
            "Die Kalibrierung ist nicht konvergiert.", "capture.calibration_quality"
        ) from exc
    log.info("calibration_finished", views=len(used), rms_px=round(rms, 3), model=model)
    if rms > cfg.max_rms_error_px:
        raise CalibrationError(
            f"Der Restfehler der Kalibrierung ist mit {rms:.2f} px zu groß "
            f"(höchstens {cfg.max_rms_error_px:g} px).",
            "capture.calibration_quality",
        )
    profile = CameraProfile(
        name=name,
        description=(
            f"Kalibriert mit {len(used)} Ansichten ({board.pattern}, {board.cols}x{board.rows} "
            f"Felder à {board.square_mm:g} mm). Gilt nur für denselben Videomodus."
        ),
        model=model,
        intrinsics=Intrinsics(width=size[0], height=size[1], params=_to_colmap(model, params)),
        source="calibration",
        rms_error_px=round(rms, 4),
    )
    return CalibrationResult(profile=profile, views=len(used), frames=frames)
