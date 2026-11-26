# SPDX-License-Identifier: AGPL-3.0-or-later
"""`drone-scan camera calibrate` (KAN-29, AC 3): chessboard/ChArUco video -> camera profile.

The videos are rendered from a known pinhole camera, so the calibration can be checked
against the true intrinsics.
"""

from __future__ import annotations

import builtins
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from dronescan.capture import calibration
from dronescan.capture.calibration import _to_colmap
from dronescan.capture.camera import load_camera
from dronescan.cli import app
from tests.test_scan_import import require_ffmpeg  # noqa: F401  (fixture)

runner = CliRunner()

WIDTH, HEIGHT = 960, 540
FOCAL = 700.0
CX, CY = 480.0, 270.0  # OpenCV convention (centre of the first pixel is 0, 0)
COLS, ROWS, SQUARE_MM, MARKER_MM = 9, 6, 30.0, 22.0
PX_PER_MM = 4
MARGIN_MM = 30


def _rotation(rx: float, ry: float) -> NDArray[np.float64]:
    a, b = math.radians(rx), math.radians(ry)
    rot_x = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    rot_y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rotation: NDArray[np.float64] = rot_y @ rot_x
    return rotation


def chessboard_texture() -> NDArray[np.uint8]:
    square = int(SQUARE_MM * PX_PER_MM)
    margin = MARGIN_MM * PX_PER_MM
    board = np.full((ROWS * square + 2 * margin, COLS * square + 2 * margin), 255, np.uint8)
    for row in range(ROWS):
        for col in range(COLS):
            if (row + col) % 2 == 0:
                y, x = margin + row * square, margin + col * square
                board[y : y + square, x : x + square] = 0
    return board


def charuco_texture() -> NDArray[np.uint8]:
    board = cv2.aruco.CharucoBoard(
        (COLS, ROWS),
        SQUARE_MM,
        MARKER_MM,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    )
    margin = MARGIN_MM * PX_PER_MM
    size = (int(COLS * SQUARE_MM * PX_PER_MM) + 2 * margin, int(ROWS * SQUARE_MM * PX_PER_MM))
    size = (size[0], size[1] + 2 * margin)
    return np.asarray(board.generateImage(size, marginSize=margin), np.uint8)


def render_video(path: Path, texture: NDArray[np.uint8], views: int = 16) -> Path:
    """Board seen from ``views`` poses by the true camera (one pose per frame, 2 fps)."""
    camera = np.array([[FOCAL, 0, CX], [0, FOCAL, CY], [0, 0, 1]])
    board_w, board_h = COLS * SQUARE_MM, ROWS * SQUARE_MM
    # texture pixel -> board millimetres, board centred on the optical axis
    to_board = np.array(
        [
            [1 / PX_PER_MM, 0, -MARGIN_MM - board_w / 2],
            [0, 1 / PX_PER_MM, -MARGIN_MM - board_h / 2],
            [0, 0, 1],
        ]
    )
    frames = []
    for index in range(views):
        angle = 2 * math.pi * index / views
        rotation = _rotation(25 * math.sin(angle), 25 * math.cos(angle))
        offset = np.array([60 * math.cos(angle), 40 * math.sin(angle), 420 + 12 * index])
        homography = camera @ np.column_stack([rotation[:, 0], rotation[:, 1], offset]) @ to_board
        frame = cv2.warpPerspective(
            texture,
            homography,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=128,
        )
        frames.append(frame)
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray"]
    command += ["-s", f"{WIDTH}x{HEIGHT}", "-framerate", "2", "-i", "pipe:0"]
    command += ["-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, input=b"".join(f.tobytes() for f in frames), check=True)
    return path


@pytest.fixture(scope="module")
def chessboard_video(tmp_path_factory: pytest.TempPathFactory, require_ffmpeg: None) -> Path:  # noqa: F811
    return render_video(tmp_path_factory.mktemp("calib") / "chessboard.mp4", chessboard_texture())


@pytest.fixture(scope="module")
def charuco_video(tmp_path_factory: pytest.TempPathFactory, require_ffmpeg: None) -> Path:  # noqa: F811
    return render_video(tmp_path_factory.mktemp("calib") / "charuco.mp4", charuco_texture())


def calibrate(video: Path, output: Path, *extra: str) -> Any:
    arguments = ["camera", "calibrate", str(video), "--name", "testcam", "--output", str(output)]
    arguments += ["--cols", str(COLS), "--rows", str(ROWS), "--square-mm", str(SQUARE_MM)]
    return runner.invoke(app, [*arguments, *extra])


def test_chessboard_calibration_recovers_the_intrinsics(
    chessboard_video: Path, tmp_path: Path
) -> None:
    """Given a chessboard video of a camera with known intrinsics,
    When `camera calibrate` runs with the OPENCV model,
    Then the profile holds focal length and principal point within 2 % (COLMAP convention),
    a small reprojection error, and can be loaded as camera profile.
    """
    output = tmp_path / "testcam.yaml"

    result = calibrate(chessboard_video, output)

    assert result.exit_code == 0, result.output
    assert "Kameraprofil gespeichert: testcam.yaml" in result.output
    profile = load_camera(str(output))
    assert profile.source == "calibration"
    assert profile.model == "OPENCV"
    assert profile.intrinsics is not None
    assert (profile.intrinsics.width, profile.intrinsics.height) == (WIDTH, HEIGHT)
    fx, fy, cx, cy, *distortion = profile.intrinsics.params
    assert fx == pytest.approx(FOCAL, rel=0.02)
    assert fy == pytest.approx(FOCAL, rel=0.02)
    assert cx == pytest.approx(CX + 0.5, abs=0.02 * WIDTH)
    assert cy == pytest.approx(CY + 0.5, abs=0.02 * HEIGHT)
    assert np.abs(distortion).max() < 0.05
    assert profile.rms_error_px is not None
    assert profile.rms_error_px < 1.0


def test_charuco_calibration_recovers_the_intrinsics(charuco_video: Path, tmp_path: Path) -> None:
    """Given a ChArUco video,
    When `camera calibrate --pattern charuco` runs with SIMPLE_RADIAL,
    Then the profile has f, cx, cy, k with f within 2 %.
    """
    output = tmp_path / "testcam.yaml"

    result = calibrate(
        charuco_video,
        output,
        *("--pattern", "charuco", "--marker-mm", str(MARKER_MM), "--model", "SIMPLE_RADIAL"),
    )

    assert result.exit_code == 0, result.output
    profile = load_camera(str(output))
    assert profile.model == "SIMPLE_RADIAL"
    assert profile.intrinsics is not None
    focal, cx, cy, k1 = profile.intrinsics.params
    assert focal == pytest.approx(FOCAL, rel=0.02)
    assert abs(k1) < 0.05


def test_fisheye_model_gives_eight_parameters(chessboard_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "testcam.yaml"

    result = calibrate(chessboard_video, output, "--model", "OPENCV_FISHEYE")

    assert result.exit_code == 0, result.output
    profile = load_camera(str(output))
    assert profile.intrinsics is not None
    assert len(profile.intrinsics.params) == 8
    assert profile.intrinsics.params[0] == pytest.approx(FOCAL, rel=0.02)


def test_video_without_pattern_exits_20_with_hints(tmp_path: Path, require_ffmpeg: None) -> None:  # noqa: F811
    """Given a video that does not show the pattern,
    When `camera calibrate` runs,
    Then it exits 20 with hints how to film the pattern, and writes no profile.
    """
    blank = render_video(tmp_path / "blank.mp4", np.full((10, 10), 128, np.uint8), views=4)
    output = tmp_path / "testcam.yaml"

    result = calibrate(blank, output)

    assert result.exit_code == 20
    assert "nur in 0 von 4" in result.output
    assert "Felder (Quadrate)" in result.output
    assert not output.exists()


def test_too_large_reprojection_error_exits_20(
    chessboard_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a stricter error limit than the calibration reaches,
    When `camera calibrate` runs,
    Then it exits 20 with the measured error and hints, and writes no profile.
    """
    monkeypatch.setenv("DRONESCAN_CALIBRATION__MAX_RMS_ERROR_PX", "0.0001")
    output = tmp_path / "testcam.yaml"

    result = calibrate(chessboard_video, output)

    assert result.exit_code == 20
    assert "Restfehler" in result.output
    assert "Bildecken" in result.output
    assert not output.exists()


def test_failed_optimisation_exits_20(
    chessboard_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def diverge(*_: Any, **__: Any) -> Any:
        raise cv2.error("calibration did not converge")

    monkeypatch.setattr(calibration, "_solve", diverge)

    result = calibrate(chessboard_video, tmp_path / "testcam.yaml")

    assert result.exit_code == 20
    assert "nicht konvergiert" in result.output


def test_unwritable_output_exits_1(chessboard_video: Path, tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")

    result = calibrate(chessboard_video, blocked / "testcam.yaml")

    assert result.exit_code == 1
    assert "Schreibrechte" in result.output


def test_missing_video_exits_10(tmp_path: Path, require_ffmpeg: None) -> None:  # noqa: F811
    result = calibrate(tmp_path / "missing.mp4", tmp_path / "testcam.yaml")

    assert result.exit_code == 10


def test_existing_output_is_kept_without_force(chessboard_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "testcam.yaml"
    output.write_text("keep")

    result = calibrate(chessboard_video, output)

    assert result.exit_code == 2
    assert output.read_text() == "keep"


@pytest.mark.parametrize(
    ("extra", "option"),
    [
        (("--name", "Mein Handy"), "--name"),
        (("--pattern", "circles"), "--pattern"),
        (("--model", "FOV"), "--model"),
    ],
)
def test_invalid_options_exit_2(tmp_path: Path, extra: tuple[str, str], option: str) -> None:
    result = calibrate(tmp_path / "video.mp4", tmp_path / "out.yaml", *extra)

    assert result.exit_code == 2
    assert option in result.output


def test_charuco_without_marker_size_exits_1(chessboard_video: Path, tmp_path: Path) -> None:
    result = calibrate(chessboard_video, tmp_path / "out.yaml", "--pattern", "charuco")

    assert result.exit_code == 1
    assert "--marker-mm" in result.output


def test_missing_opencv_names_the_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_ffmpeg: None,  # noqa: F811
) -> None:
    """Given an installation without the optional extra,
    When `camera calibrate` runs,
    Then it exits 1 and explains how to install OpenCV.
    """
    real_import: Callable[..., Any] = builtins.__import__

    def without_cv2(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cv2":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cv2)

    result = calibrate(tmp_path / "video.mp4", tmp_path / "out.yaml")

    assert result.exit_code == 1
    assert "uv sync --extra calibration" in result.output


@pytest.mark.parametrize(
    ("model", "params", "expected"),
    [
        ("SIMPLE_RADIAL", [700, 480, 270, 0.1], [700, 480.5, 270.5, 0.1]),
        ("PINHOLE", [700, 701, 480, 270], [700, 701, 480.5, 270.5]),
        ("OPENCV_FISHEYE", [700, 701, 480, 270, 1, 2, 3, 4], [700, 701, 480.5, 270.5, 1, 2, 3, 4]),
    ],
)
def test_principal_point_is_converted_to_colmap(
    model: Any, params: list[float], expected: list[float]
) -> None:
    """OpenCV puts the centre of the first pixel at 0, COLMAP at 0.5."""
    assert _to_colmap(model, params) == expected
