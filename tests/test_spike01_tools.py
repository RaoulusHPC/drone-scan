"""Spike KAN-19 tools (COLMAP + gsplat on the GTX 1060): pure-Python helpers.

Synthetic data only: no real recordings in tests (CLAUDE.md rule 3).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest
from tools.spike01 import run_room
from tools.spike01.colmap_text import (
    Camera,
    RegistrationStats,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
    registration_stats,
)
from tools.spike01.frames import (
    analysis_size,
    colmap_option_args,
    display_size,
    ffmpeg_select_expression,
    scale_pinhole,
    select_keyframes,
)

from tests.test_docker import ROOT, require_image

CAMERAS = """\
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 2
1 PINHOLE 1600 900 1200.5 1199.5 800.0 450.0
2 SIMPLE_RADIAL 640 480 500.0 320.0 240.0 0.01
"""

# Two lines per image: pose, then 2D points (the second line may be empty).
IMAGES = """\
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 3, mean observations per image: 1.0
1 1.0 0.0 0.0 0.0 0.5 -0.25 2.0 1 f_0001.jpg
10.0 20.0 7 30.0 40.0 -1
2 0.0 1.0 0.0 0.0 0.0 0.0 1.0 1 f_0002.jpg

3 1.0 0.0 0.0 0.0 1.0 2.0 3.0 1 f_0003.jpg
5.0 6.0 -1
"""

POINTS3D = """\
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 2, mean track length: 2.0
7 1.0 2.0 3.0 255 128 0 0.5 1 0 2 3
8 -1.0 0.5 0.25 10 20 30 1.25 1 1
"""


def test_cameras_txt_is_parsed_with_model_and_parameters() -> None:
    """Given a COLMAP cameras.txt with a PINHOLE and a SIMPLE_RADIAL camera,
    When it is parsed,
    Then ids, models, sizes and parameters are returned and comments are skipped.
    """
    cameras = parse_cameras_txt(CAMERAS)

    assert cameras[1] == Camera(1, "PINHOLE", 1600, 900, (1200.5, 1199.5, 800.0, 450.0))
    assert cameras[2].model == "SIMPLE_RADIAL"
    assert cameras[2].params == (500.0, 320.0, 240.0, 0.01)


def test_images_txt_handles_empty_points_lines() -> None:
    """Given an images.txt where the 2D-points line of one image is empty,
    When it is parsed,
    Then every image keeps its pose and name and the line pairing is not shifted.
    """
    images = parse_images_txt(IMAGES)

    assert [im.name for im in images] == ["f_0001.jpg", "f_0002.jpg", "f_0003.jpg"]
    assert images[0].quat_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert images[0].translation == (0.5, -0.25, 2.0)
    assert images[1].quat_wxyz == (0.0, 1.0, 0.0, 0.0)
    assert [im.camera_id for im in images] == [1, 1, 1]


def test_points3d_txt_is_parsed_with_color_and_error() -> None:
    """Given a points3D.txt,
    When it is parsed,
    Then positions, colours (0-255) and reprojection errors are returned.
    """
    points = parse_points3d_txt(POINTS3D)

    assert len(points) == 2
    assert points[0].xyz == (1.0, 2.0, 3.0)
    assert points[0].rgb == (255, 128, 0)
    assert points[1].error == 1.25


def test_registration_stats_report_percent_of_extracted_frames() -> None:
    """Given 3 registered images out of 4 extracted frames,
    When the registration is evaluated,
    Then 3 of 4 frames, i.e. 75 %, count as registered.
    """
    stats = registration_stats(parse_images_txt(IMAGES), total_frames=4)

    assert stats == RegistrationStats(registered=3, total=4, percent=75.0)


def test_registration_stats_reject_empty_input() -> None:
    """Given zero extracted frames,
    When the registration is evaluated,
    Then it fails with a clear message instead of dividing by zero.
    """
    with pytest.raises(ValueError, match="total_frames"):
        registration_stats([], total_frames=0)


@pytest.mark.parametrize(
    ("size", "rotation", "expected"),
    [
        ((1080, 1920), 0, (1080, 1920)),
        ((1080, 1920), 90, (1920, 1080)),
        ((1080, 1920), -90, (1920, 1080)),
        ((1080, 1920), 270, (1920, 1080)),
        ((1920, 1080), 180, (1920, 1080)),
    ],
)
def test_display_size_applies_the_rotation_flag(
    size: tuple[int, int], rotation: int, expected: tuple[int, int]
) -> None:
    """Given a video with a rotation flag,
    When the size after decoding with autorotate is computed,
    Then width and height are swapped for 90/270 degrees only.
    """
    assert display_size(*size, rotation=rotation) == expected


def test_select_keyframes_takes_the_sharpest_frame_of_each_window() -> None:
    """Given sharpness scores of 6 frames and windows of 3,
    When keyframes are selected,
    Then the sharpest frame of each window is returned in time order.
    """
    scores = [1.0, 9.0, 3.0, 4.0, 2.0, 8.0]

    assert select_keyframes(scores, window=3, min_rel_sharpness=0.0) == [1, 5]


def test_select_keyframes_drops_windows_that_are_all_blurred() -> None:
    """Given a motion-blurred stretch (best score far below the median),
    When keyframes are selected with min_rel_sharpness 0.5,
    Then no frame of that window is kept.
    """
    scores = [10.0, 11.0, 12.0, 1.0, 1.2, 1.1, 10.5, 11.5, 12.5]

    assert select_keyframes(scores, window=3, min_rel_sharpness=0.5) == [2, 8]


def test_select_keyframes_keeps_a_short_last_window() -> None:
    """Given 5 frames and windows of 2,
    When keyframes are selected,
    Then the trailing single frame is a window of its own.
    """
    assert select_keyframes([1.0, 2.0, 3.0, 4.0, 5.0], window=2, min_rel_sharpness=0.0) == [1, 3, 4]


def test_select_keyframes_rejects_a_window_below_one() -> None:
    """Given window=0,
    When keyframes are selected,
    Then a ValueError is raised.
    """
    with pytest.raises(ValueError, match="window"):
        select_keyframes([1.0], window=0, min_rel_sharpness=0.0)


def test_scale_pinhole_scales_focal_length_and_principal_point() -> None:
    """Given an undistorted PINHOLE camera (fx, fy, cx, cy) for 1600x900,
    When the images are downscaled to 800x450,
    Then all four intrinsics are halved and the size follows.
    """
    camera = Camera(1, "PINHOLE", 1600, 900, (1200.0, 1100.0, 800.0, 450.0))

    scaled = scale_pinhole(camera, new_width=800)

    assert (scaled.width, scaled.height) == (800, 450)
    assert scaled.params == (600.0, 550.0, 400.0, 225.0)


def test_scale_pinhole_rejects_other_camera_models() -> None:
    """Given a camera that is not PINHOLE (e.g. SIMPLE_RADIAL with distortion),
    When it is scaled,
    Then it fails: training expects the COLMAP-undistorted pinhole model.
    """
    camera = Camera(1, "SIMPLE_RADIAL", 640, 480, (500.0, 320.0, 240.0, 0.01))

    with pytest.raises(ValueError, match="PINHOLE"):
        scale_pinhole(camera, new_width=320)


def test_ffmpeg_select_expression_lists_the_frame_numbers() -> None:
    """Given keyframe indices 0, 3 and 7 of the analysis frame stream,
    When the ffmpeg select expression is built,
    Then it selects exactly those frame numbers (eq(n,...) joined with +).
    """
    assert ffmpeg_select_expression([0, 3, 7]) == "eq(n\\,0)+eq(n\\,3)+eq(n\\,7)"


def test_ffmpeg_select_expression_rejects_an_empty_selection() -> None:
    """Given no keyframes,
    When the expression is built,
    Then it fails instead of producing a filter that selects nothing.
    """
    with pytest.raises(ValueError, match="no frames"):
        ffmpeg_select_expression([])


@pytest.mark.parametrize(
    ("display", "expected"),
    [((1920, 1080), (480, 270)), ((1080, 1920), (270, 480)), ((1000, 1000), (480, 480))],
)
def test_analysis_size_keeps_the_orientation(
    display: tuple[int, int], expected: tuple[int, int]
) -> None:
    """Given the display size of a video (landscape, portrait, square),
    When the size of the small sharpness-analysis frames is chosen,
    Then the long edge is 480 px and the orientation is kept (even dimensions).
    """
    assert analysis_size(*display, long_edge=480) == expected


def test_colmap_option_args_turns_key_value_pairs_into_flags() -> None:
    """Given Mapper options given as key=value,
    When they are converted into COLMAP command-line arguments,
    Then each becomes `--Mapper.<key> <value>`.
    """
    assert colmap_option_args("Mapper", ["init_min_tri_angle=8", "min_num_matches=10"]) == [
        "--Mapper.init_min_tri_angle",
        "8",
        "--Mapper.min_num_matches",
        "10",
    ]


def test_colmap_option_args_rejects_an_option_without_value() -> None:
    """Given an option without `=value`,
    When it is converted,
    Then a ValueError names the option.
    """
    with pytest.raises(ValueError, match="init_min_tri_angle"):
        colmap_option_args("Mapper", ["init_min_tri_angle"])


def test_work_directory_must_be_private() -> None:
    """Given the repository root,
    When a work directory for private recordings is checked,
    Then a tracked location inside the repo is refused, a git-ignored one (workspace/) and a
    directory outside the repo are accepted.
    """
    with pytest.raises(SystemExit, match="outside the repository"):
        run_room.ensure_private_work_dir(ROOT / "docs" / "spike-work")

    run_room.ensure_private_work_dir(ROOT / "workspace" / "spike-kan19" / "work")
    run_room.ensure_private_work_dir(Path("/tmp/spike-work"))


def test_gpu_stage_lock_serializes_two_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given two jobs that both run a GPU stage and share the GPU lock file,
    When the second one starts while the first one is inside its stage,
    Then it waits until the first has left (GPU jobs never run in parallel).
    """

    class FakeSampler:
        process_peak_mib = 0

        def __enter__(self) -> FakeSampler:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(run_room, "GPU_LOCK", tmp_path / "gpu.lock")
    monkeypatch.setattr(run_room, "GpuSampler", FakeSampler)
    order: list[str] = []
    first_inside = threading.Event()

    def job(name: str, hold: bool) -> None:
        recorder = run_room.Recorder(tmp_path / f"{name}.json")
        with recorder.stage("gpu_stage", gpu=True):
            order.append(f"{name}-in")
            if hold:
                first_inside.set()
                time.sleep(0.4)
            order.append(f"{name}-out")

    first = threading.Thread(target=job, args=("a", True))
    first.start()
    first_inside.wait(timeout=5)
    second = threading.Thread(target=job, args=("b", False))
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert order == ["a-in", "a-out", "b-in", "b-out"]


# --- tests that run the spike tools inside the worker image (marked gpu: image and GTX 1060) ---


def _worker(
    tmp_path: Path, *command: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    """`docker compose run worker ...` with tools/ and a scratch directory mounted."""
    require_image("worker")
    cmd = [
        "docker", "compose", "run", "--rm", "-T",
        "-v", f"{ROOT / 'tools'}:/tools:ro",
        "-v", f"{tmp_path}:/work",
        "-e", "PYTHONPATH=/",
        "-e", "DRONESCAN_VRAM_LIMIT_GB=5.0",
        "worker", *command,
    ]  # fmt: skip
    return subprocess.run(
        cmd, cwd=ROOT, input=stdin, capture_output=True, text=True, timeout=900, check=False
    )


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_psnr_and_ssim_match_known_values(tmp_path: Path) -> None:
    """Given constant images 0.5 and 0.6, identical images and noise,
    When PSNR and SSIM are computed (in the worker image),
    Then PSNR is 20 dB for an error of 0.1, SSIM of identical images is 1 and drops for noise.
    """
    script = """
import torch
from tools.spike01.metrics import psnr, ssim
a = torch.full((64, 64, 3), 0.5)
b = torch.full((64, 64, 3), 0.6)
torch.manual_seed(0)
noise = torch.rand(64, 64, 3)
print(round(psnr(a, b), 3), round(ssim(a, a), 4), ssim(noise, a) < 0.5)
"""
    result = _worker(tmp_path, "python", "-", stdin=script)

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "20.0 1.0 True"


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
@pytest.mark.parametrize("strategy", ["default", "mcmc"])
def test_trainer_learns_a_synthetic_scene_on_the_gtx_1060(tmp_path: Path, strategy: str) -> None:
    """Given a synthetic COLMAP scene (12 views, 400 Gaussians),
    When the spike trainer runs 300 iterations on the GTX 1060 (fp32, 5.0 GiB allocator limit),
    Then it finishes, held-out PSNR/SSIM show that poses, intrinsics and the train/held-out
    split are consistent, and the reported peak memory stays far below the limit.
    """
    scene = _worker(tmp_path, "python", "-m", "tools.spike01.synth_scene", "/work/scene")
    assert scene.returncode == 0, scene.stderr[-2000:]

    result = _worker(
        tmp_path,
        "python", "-m", "tools.spike01.train_gsplat",
        "--scene", "/work/scene", "--raw", "/work/scene/raw", "--out", "/work/out",
        "--width", "320", "--iterations", "300", "--sh-degree", "1", "--holdout-every", "4",
        "--strategy", strategy, "--cap", "800", "--log-every", "100",
    )  # fmt: skip

    assert result.returncode == 0, result.stderr[-2000:]
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["ok"] is True
    assert tuple(metrics["capability"]) >= (6, 1)  # GTX 1060 here, RTX 2060 (7, 5) on target
    assert metrics["images_holdout"] == 3 and metrics["images_train"] == 9
    assert metrics["heldout"]["psnr"] > 20.0
    assert metrics["heldout"]["ssim"] > 0.8
    assert metrics["peak_allocated_mib"] < 5.0 * 1024


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_trainer_memory_stress_mode_uses_the_requested_gaussian_count(tmp_path: Path) -> None:
    """Given a synthetic scene and --init-random 5000 (memory stress mode),
    When the trainer runs 20 iterations,
    Then it starts from exactly 5000 random Gaussians (no neighbour search on the cloud) and
    reports the memory of that size.
    """
    scene = _worker(tmp_path, "python", "-m", "tools.spike01.synth_scene", "/work/scene")
    assert scene.returncode == 0, scene.stderr[-2000:]

    result = _worker(
        tmp_path,
        "python", "-m", "tools.spike01.train_gsplat",
        "--scene", "/work/scene", "--raw", "/work/scene/raw", "--out", "/work/out",
        "--width", "320", "--iterations", "20", "--sh-degree", "1", "--holdout-every", "4",
        "--init-random", "5000", "--log-every", "10",
    )  # fmt: skip

    assert result.returncode == 0, result.stderr[-2000:]
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["init_points"] == 5000 and metrics["final_gaussians"] == 5000
    assert metrics["peak_allocated_mib"] > 0


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_trainer_reports_missing_input_with_exit_code_10(tmp_path: Path) -> None:
    """Given a scene directory without COLMAP files, and a scene without raw frames,
    When the trainer starts,
    Then it exits with code 10 (input missing, stage contract) instead of a traceback.
    """
    no_scene = _worker(
        tmp_path,
        "python", "-m", "tools.spike01.train_gsplat",
        "--scene", "/work/missing", "--raw", "/work/missing",
        "--out", "/work/out", "--width", "320",
    )  # fmt: skip
    assert no_scene.returncode == 10, no_scene.stderr[-1000:]

    scene = _worker(tmp_path, "python", "-m", "tools.spike01.synth_scene", "/work/scene")
    assert scene.returncode == 0, scene.stderr[-2000:]
    no_raw = _worker(
        tmp_path,
        "python", "-m", "tools.spike01.train_gsplat",
        "--scene", "/work/scene", "--raw", "/work/nope", "--out", "/work/out", "--width", "320",
    )  # fmt: skip
    assert no_raw.returncode == 10, no_raw.stderr[-1000:]


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_sharpness_is_higher_for_sharp_than_for_blurred_frames(tmp_path: Path) -> None:
    """Given a checkerboard frame and the same frame averaged with its neighbours,
    When the Laplacian variance is computed (in the worker image, numpy),
    Then the sharp frame scores far higher, so motion blur lowers the score.
    """
    script = """
import numpy as np
from tools.spike01.sharpness import laplacian_variance
y, x = np.mgrid[0:64, 0:64]
sharp = (((x // 4 + y // 4) % 2) * 255).astype(np.uint8)
padded = np.pad(sharp.astype(np.float32), 2, mode="edge")
blurred = sum(
    padded[dy:dy + 64, dx:dx + 64] for dy in range(5) for dx in range(5)
) / 25
scores = laplacian_variance(np.stack([sharp, blurred.astype(np.uint8)]))
print(scores[0] > 10 * scores[1] > 0)
"""
    result = _worker(tmp_path, "python", "-", stdin=script)

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "True"
