# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trainer of stage `recon` on the real GPU (KAN-41): runs in the worker-test container.

`make test-gpu` starts these tests with PyTorch and gsplat on the GTX 1060 (fp32) and the
5.0 GiB allocator limit. The scene is synthetic (random Gaussians rendered by gsplat), never
a real recording.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from dronescan.recon.job import TrainExit, TrainJob, TrainResult
from dronescan.recon.scene import read_ply_header

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.torch,
    pytest.mark.skipif(
        importlib.util.find_spec("gsplat") is None, reason="PyTorch/gsplat not installed"
    ),
]
GIB_MIB = 1024


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """12 views of 400 coloured Gaussians, 320x240, PINHOLE text model + raw RGB."""
    from tools.spike01.synth_scene import write_scene

    out = tmp_path_factory.mktemp("synthetic") / "scene"
    write_scene(out)
    return out


def train(tmp_path: Path, scene: Path, **values: object) -> tuple[int, TrainResult, Path]:
    job = TrainJob(
        profile="low_vram_6gb",
        vram_limit_gb=5.0,
        iterations=300,
        max_gaussians=800,
        max_image_edge_px=320,
        sh_degree=1,
        holdout_every=4,
        log_every=100,
    ).model_copy(update=values)
    (tmp_path / "job.json").write_text(job.model_dump_json())
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "dronescan.recon.train", "--scene", str(scene)]
        + ["--raw", str(scene / "raw"), "--job", str(tmp_path / "job.json"), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    result = TrainResult.model_validate_json((out / "metrics.json").read_text())
    return proc.returncode, result, out


def test_trainer_learns_a_synthetic_scene_within_the_limit(tmp_path: Path, scene: Path) -> None:
    """Given a synthetic scene (12 views, every 4th held out),
    When the trainer runs 300 MCMC iterations on the GPU (fp32, 5.0 GiB limit),
    Then held-out PSNR/SSIM show it learned the scene, the quality curve rises, the PLY holds
    every Gaussian and the allocator peak stays far below the limit.
    """
    code, result, out = train(tmp_path, scene, eval_at=[100, 200])

    assert code == TrainExit.OK, result.error
    assert result.ok and tuple(result.capability) >= (6, 1)
    assert (result.images_train, result.images_holdout) == (9, 3)
    assert result.psnr_db is not None and result.psnr_db > 22.0
    assert result.ssim is not None and result.ssim > 0.85
    assert [point.step for point in result.curve] == [100, 200]
    assert result.curve[0].psnr_db < result.curve[1].psnr_db <= result.psnr_db + 0.5
    vertices, names = read_ply_header(out / "scene.ply")
    assert vertices == result.gaussians > 0
    assert names.count("f_rest_0") == 1 and "f_rest_8" in names  # SH degree 1: 3 x 3 rest
    assert result.peak_reserved_mib is not None and result.peak_reserved_mib < 5.0 * GIB_MIB


def test_trainer_downscales_to_the_requested_edge(tmp_path: Path, scene: Path) -> None:
    """Given 320x240 images and max_image_edge_px=160 (OOM retry size),
    When the trainer runs with packed rendering,
    Then it trains on 160x120 images and still learns.
    """
    code, result, _ = train(tmp_path, scene, max_image_edge_px=160, packed=True)

    assert code == TrainExit.OK, result.error
    assert result.image_edge_px == 160 and result.packed
    assert result.psnr_db is not None and result.psnr_db > 20.0


def test_out_of_memory_exits_30(tmp_path: Path, scene: Path) -> None:
    """Given an allocator limit far below what training needs (0.01 GiB),
    When the trainer runs,
    Then it exits with 30 and metrics.json says out of memory, so the stage can retry.
    """
    code, result, out = train(tmp_path, scene, vram_limit_gb=0.01)

    assert code == TrainExit.OOM
    assert not result.ok and "memory" in (result.error or "").lower()
    assert not (out / "scene.ply").exists()


def test_missing_scene_exits_10(tmp_path: Path) -> None:
    """Given a scene directory without COLMAP files,
    When the trainer runs,
    Then it exits with 10 (input missing) and writes metrics.json with ok=false.
    """
    empty = tmp_path / "empty"
    (empty / "raw").mkdir(parents=True)

    code, result, _ = train(tmp_path, empty)

    assert code == TrainExit.INPUT_MISSING
    assert not result.ok and result.error
