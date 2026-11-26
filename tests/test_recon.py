# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage `recon` (recon.splat): Gaussian splatting in the 6 GB profile (KAN-41, US-026).

CPU tests replace `docker`, `ffmpeg` and `nvidia-smi` by small scripts; the trainer itself is
tested in the worker-test container (tests/test_recon_gpu.py). Private clips are never used.
"""

from __future__ import annotations

import json
import os
import stat
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from dronescan.cli import app
from dronescan.common.config import Settings, load_settings
from dronescan.common.stage import GPU_OOM_HINT, ExitCode, Manifest, StageResult, run_stage
from dronescan.recon.job import TrainJob
from dronescan.recon.scene import (
    SceneError,
    View,
    read_ply_header,
    read_scene,
    split_holdout,
    write_ply,
)
from dronescan.recon.stage import OOM_RETRY_WARNING, ReconStage
from dronescan.recon.worker import WORKER_IMAGE, ContainerVramSampler
from dronescan.sfm.colmap import COLMAP_IMAGE
from tests.test_sfm import make_scan

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()

# ---------- scans ----------


def write_sfm(scan_dir: Path, images: int) -> None:
    """sfm/ after a successful run: manifest and the largest model as COLMAP text."""
    now = datetime.now(UTC)
    manifest = Manifest(
        stage="sfm",
        version="2",
        profile="low_vram_6gb",
        inputs={},
        params={},
        tools={},
        started=now,
        finished=now,
        peak_vram_mb=None,
        status="ok",
        exit_code=0,
    )
    sparse = scan_dir / "sfm" / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (scan_dir / "sfm" / "manifest.json").write_text(manifest.model_dump_json())
    rows = [f"{i} 1 0 0 0 0 0 {i} 1 f01/frames/{i:06d}.jpg\n\n" for i in range(1, images + 1)]
    (sparse / "images.txt").write_text("".join(rows))
    (sparse / "cameras.txt").write_text("1 SIMPLE_RADIAL 1600 900 1200 800 450 0.01\n")
    (sparse / "points3D.txt").write_text("1 0 0 5 10 20 30 0.5 1 0\n")


def make_recon_scan(tmp_path: Path, images: int = 10) -> Path:
    scan = make_scan(tmp_path / "scan", images)
    write_sfm(scan, images)
    return scan


def recon(scan: Path, settings: Settings | None = None, force: bool = False) -> StageResult:
    return run_stage(ReconStage(), scan, settings or load_settings(), force=force)


def with_splat(**values: Any) -> Settings:
    settings = load_settings()
    return settings.model_copy(update={"splat": settings.splat.model_copy(update=values)})


# ---------- fake tools ----------

FAKE_DOCKER = """#!{python}
import csv, json, sys
from pathlib import Path

state = Path({state!r})
config = json.loads((state / "config.json").read_text())
args = sys.argv[1:]
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
if args[0] == "image":
    if args[-1] in config["missing_images"]:
        sys.exit("No such image")
    print("sha256:" + args[-1])
    sys.exit(0)
if args[0] == "kill":
    sys.exit(0)
if args[0] == "top":
    print("PID")
    print("4242")
    sys.exit(0)

mounts = {{}}
for flag, value in zip(args, args[1:]):
    if flag == "--mount":
        fields = dict(f.partition("=")[::2] for f in next(csv.reader([value])))
        mounts[fields["target"]] = Path(fields["source"])
recon = mounts["/recon"]


def host(path):
    return recon / path.removeprefix("/recon/")


if {colmap!r} in args:
    position = args.index("colmap", args.index({colmap!r}))
    command = args[position + 1]
    options = dict(zip(args[position + 2 :: 2], args[position + 3 :: 2]))
    if command == "image_undistorter":
        out = host(options["--output_path"])
        edge = int(options["--max_image_size"])
        width, height = edge, edge * 9 // 16
        names = [
            line.split()[9]
            for line in (mounts["/sparse"] / "images.txt").read_text().splitlines()
            if line.strip()
        ]
        for name in names:
            image = out / "images" / name
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_text(f"{{width}} {{height}}")
        sparse = out / "sparse"
        sparse.mkdir(parents=True)
        rows = "".join(f"{{i}} 1 0 0 0 0 0 0 1 {{n}}\\n\\n" for i, n in enumerate(names, 1))
        (sparse / "images.txt").write_text(rows)
        f = width * 0.8
        (sparse / "cameras.txt").write_text(f"1 PINHOLE {{width}} {{height}} {{f}} {{f}} 1 1\\n")
        (sparse / "points3D.txt").write_text("1 0 0 5 10 20 30 0.5 1 0\\n2 1 0 5 1 2 3 0.5 2 0\\n")
    sys.exit(0)

# worker: ... python -m dronescan.recon.train --scene S --raw R --job J --out O
tail = args[args.index("-m") + 2 :]
options = dict(zip(tail[::2], tail[1::2]))
job = json.loads(host(options["--job"]).read_text())
attempts = config["attempts"]
code, result = attempts.pop(0) if attempts else (0, {{}})
(state / "config.json").write_text(json.dumps(config))
raws = sorted(p.name for p in host(options["--raw"]).rglob("*.rgb"))
with (state / "jobs.jsonl").open("a") as handle:
    handle.write(json.dumps({{"job": job, "raws": raws}}) + "\\n")
if result is None:
    sys.exit(code)
out = host(options["--out"])
out.mkdir(parents=True)
metrics = {{
    "ok": code == 0, "vram_limit_gb": job["vram_limit_gb"], "iterations": job["iterations"],
    "max_gaussians": job["max_gaussians"], "image_edge_px": job["max_image_edge_px"],
    "packed": job["packed"], "images_train": 8, "images_holdout": 2, "gaussians": 1000,
    "psnr_db": 25.5, "ssim": 0.88, "train_psnr_db": 28.0, "train_seconds": 12.0,
    "seconds_per_iteration": 0.01, "peak_allocated_mib": 900.0, "peak_reserved_mib": 1000.0,
    "device": "Fake GPU", "capability": [6, 1],
}}
metrics.update(result)
(out / "metrics.json").write_text(json.dumps(metrics))
if code == 0:
    (out / "scene.ply").write_text("ply")
sys.exit(code)
"""

FAKE_FFMPEG = """#!{python}
import sys
from pathlib import Path

args = sys.argv[1:]
if args == ["-version"]:
    print("ffmpeg version 7.1-fake")
    sys.exit(0)
width, height = map(int, Path(args[args.index("-i") + 1]).read_text().split())
Path(args[-1]).write_bytes(bytes(width * height * 3))
"""

FAKE_NVIDIA_SMI = """#!{python}
import sys
assert sys.argv[1:3] == ["-i", "0"]
assert "--query-compute-apps=pid,used_memory" in sys.argv
print("1, 9000")
print("4242, {mib}")
print("4242, [N/A]")
"""


class FakeDocker:
    def __init__(self, directory: Path) -> None:
        self.state = directory
        self.config: dict[str, Any] = {"missing_images": [], "attempts": []}
        self.save()

    def save(self) -> None:
        (self.state / "config.json").write_text(json.dumps(self.config))

    def trainer(self, *attempts: tuple[int, dict[str, Any] | None]) -> None:
        """Exit code and metrics.json overrides per trainer run (None: no metrics.json)."""
        self.config["attempts"] = [list(attempt) for attempt in attempts]
        self.save()

    def missing(self, image: str) -> None:
        self.config["missing_images"].append(image)
        self.save()

    def calls(self, kind: str = "run") -> list[list[str]]:
        path = self.state / "calls.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        return [call for call in map(json.loads, lines) if call[0] == kind]

    def worker_calls(self) -> list[list[str]]:
        return [call for call in self.calls() if WORKER_IMAGE in call]

    def colmap_commands(self) -> list[str]:
        return [
            c[c.index("colmap", c.index(COLMAP_IMAGE)) + 1]
            for c in self.calls()
            if COLMAP_IMAGE in c
        ]

    def jobs(self) -> list[dict[str, Any]]:
        path = self.state / "jobs.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    directory = tmp_path / "fake"
    directory.mkdir()
    scripts = {
        "docker": FAKE_DOCKER.format(
            python=sys.executable, state=str(directory), colmap=COLMAP_IMAGE
        ),
        "ffmpeg": FAKE_FFMPEG.format(python=sys.executable),
        "nvidia-smi": FAKE_NVIDIA_SMI.format(python=sys.executable, mib=2345),
    }
    for name, text in scripts.items():
        script = directory / name
        script.write_text(text)
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{directory}:{os.environ['PATH']}")
    return FakeDocker(directory)


@pytest.fixture
def scan(tmp_path: Path) -> Path:
    return make_recon_scan(tmp_path)


# ---------- AC 1/2: training in the 6 GB profile, metrics ----------


def test_stage_undistorts_trains_and_reports_psnr_ssim(scan: Path, docker: FakeDocker) -> None:
    """Given a scan with poses from `sfm`,
    When the stage runs in the low_vram_6gb profile,
    Then COLMAP undistorts to 1200 px, the trainer gets every registered frame as raw RGB and
    the profile values (MCMC cap 1.5 M, 15 000 iterations, every 8th frame held out, 5.0 GiB
    limit), and the splat, metrics.json and PSNR/SSIM in the manifest are written.
    """
    result = recon(scan)

    assert result.exit_code is ExitCode.OK, result.manifest.error
    assert docker.colmap_commands() == ["image_undistorter"]
    undistort = docker.calls()[0]
    assert undistort[undistort.index("--max_image_size") + 1] == "1200"
    [run] = docker.jobs()
    assert run["raws"] == [f"{n:06d}.jpg.rgb" for n in range(1, 11)]
    assert TrainJob.model_validate(run["job"]) == TrainJob(
        profile="low_vram_6gb",
        vram_limit_gb=5.0,
        iterations=15000,
        max_gaussians=1_500_000,
        max_image_edge_px=1200,
        sh_degree=3,
        holdout_every=8,
    )
    splat = scan / "recon" / "splat"
    assert {p.name for p in splat.iterdir()} == {"scene.ply", "metrics.json"}
    metrics = result.manifest.metrics
    assert metrics["psnr_db"] == 25.5 and metrics["ssim"] == 0.88
    assert metrics["images"] == 10 and metrics["images_holdout"] == 2  # frames 0 and 8
    assert metrics["oom_retry"] is False
    assert json.loads((splat / "metrics.json").read_text())["psnr_db"] == 25.5


def test_peak_vram_is_the_container_process_including_context(
    scan: Path, docker: FakeDocker
) -> None:
    """Given nvidia-smi reports 2345 MiB for the trainer's process (host PID of the container)
    and 9000 MiB for an unrelated process,
    When the stage runs,
    Then peak_vram_mb is the trainer's process total, not the allocator peak or other processes.
    """
    result = recon(scan)

    assert result.manifest.peak_vram_mb == 2345.0


def test_decoded_and_undistorted_images_are_removed(scan: Path, docker: FakeDocker) -> None:
    """Given a finished run,
    When recon/ is inspected,
    Then no image content is left besides the splat (raw RGB, undistorted JPEGs); logs stay.
    """
    recon(scan)

    work = scan / "recon" / "work"
    assert not (work / "raw").exists() and not (work / "undistorted" / "images").exists()
    assert (work / "logs" / "train-1.log").is_file()


def test_trainer_container_is_offline_unprivileged_and_on_gpu_0(
    scan: Path, docker: FakeDocker
) -> None:
    """Given the default GPU device 0 (GTX 1060),
    When the trainer container starts,
    Then it has no network, runs as the calling user, sees only GPU 0, never pulls an image,
    and runs under the watchdog that ends it with the runner.
    """
    recon(scan)

    [call] = docker.worker_calls()
    assert call[call.index("--network") + 1] == "none"
    assert call[call.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert call[call.index("--gpus") + 1] == "device=0"
    assert call[call.index("--pull") + 1] == "never"
    assert call[call.index(WORKER_IMAGE) + 1 : call.index(WORKER_IMAGE) + 3] == ["sh", "-c"]
    assert call[-8:-6] == ["--scene", "/recon/undistorted/sparse"]


def test_valid_manifest_is_a_cache_hit_and_force_retrains(scan: Path, docker: FakeDocker) -> None:
    """Given a successful run,
    When the stage runs again without and with --force,
    Then it is skipped first and trains again with --force.
    """
    recon(scan)
    again = recon(scan)
    forced = recon(scan, force=True)

    assert again.skipped and not forced.skipped
    assert len(docker.jobs()) == 2


def test_changed_poses_invalidate_the_cache(scan: Path, docker: FakeDocker) -> None:
    """Given a successful run,
    When `sfm` produced a different model,
    Then the stage trains again.
    """
    recon(scan)
    points = scan / "sfm" / "colmap" / "sparse" / "0" / "points3D.txt"
    points.write_text(points.read_text() + "2 1 1 5 1 2 3 0.5 2 0\n")

    assert not recon(scan).skipped


# ---------- AC 3: OOM -> one retry with a smaller profile ----------


def test_oom_retries_once_with_smaller_values(scan: Path, docker: FakeDocker) -> None:
    """Given the trainer runs out of VRAM once,
    When the stage runs,
    Then it retries exactly once with the oom_retry values and packed rendering, succeeds and
    says so in a warning.
    """
    docker.trainer((30, {"error": "OutOfMemoryError: CUDA out of memory"}), (0, {}))

    result = recon(scan)

    assert result.exit_code is ExitCode.OK
    first, second = (TrainJob.model_validate(run["job"]) for run in docker.jobs())
    assert (first.max_gaussians, first.max_image_edge_px, first.packed) == (1_500_000, 1200, False)
    assert (second.max_gaussians, second.max_image_edge_px, second.packed) == (750_000, 960, True)
    assert result.manifest.metrics["oom_retry"] is True
    assert result.manifest.warnings == [OOM_RETRY_WARNING.format(gaussians=750_000, edge=960)]
    assert (scan / "recon" / "splat" / "scene.ply").is_file()


def test_second_oom_ends_with_exit_30(scan: Path, docker: FakeDocker) -> None:
    """Given the trainer runs out of VRAM twice,
    When the stage runs,
    Then it stops after the one retry with exit 30 and the OOM hint.
    """
    docker.trainer((30, {"error": "OOM"}), (30, {"error": "OOM"}), (0, {}))

    result = recon(scan)

    assert result.exit_code is ExitCode.OOM
    assert len(docker.jobs()) == 2
    assert GPU_OOM_HINT in result.manifest.hints
    assert not (scan / "recon" / "splat" / "scene.ply").exists()


# ---------- failures ----------


def test_missing_poses_exit_10(tmp_path: Path, docker: FakeDocker) -> None:
    """Given a scan where `sfm` has not run,
    When the stage runs,
    Then it exits with 10 and no container is started.
    """
    scan = make_scan(tmp_path / "scan", 10)

    result = recon(scan)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert "sfm" in (result.manifest.error or "")
    assert docker.calls() == []


def test_trainer_error_is_reported_with_its_cause(scan: Path, docker: FakeDocker) -> None:
    """Given the trainer fails with an error that is not out-of-memory,
    When the stage runs,
    Then it exits with 1 and names the cause and the log directory; there is no retry.
    """
    docker.trainer((1, {"error": "ValueError: kaputt"}))

    result = recon(scan)

    assert result.exit_code is ExitCode.ERROR
    assert "ValueError: kaputt" in (result.manifest.error or "")
    assert any("recon/work/logs" in hint for hint in result.manifest.hints)
    assert len(docker.jobs()) == 1


def test_trainer_without_metrics_is_an_error(scan: Path, docker: FakeDocker) -> None:
    """Given the trainer dies before writing metrics.json,
    When the stage runs,
    Then it exits with 1 instead of reporting a splat.
    """
    docker.trainer((1, None))

    assert recon(scan).exit_code is ExitCode.ERROR


@pytest.mark.parametrize("image", [WORKER_IMAGE, COLMAP_IMAGE])
def test_missing_image_keeps_previous_result(scan: Path, docker: FakeDocker, image: str) -> None:
    """Given a previous splat and a missing worker or colmap image,
    When the stage runs with --force,
    Then it exits with 1 and a build hint before anything is removed.
    """
    recon(scan)
    docker.missing(image)

    result = recon(scan, force=True)

    assert result.exit_code is ExitCode.ERROR
    assert image in (result.manifest.error or "")
    assert any("docker compose build" in hint for hint in result.manifest.hints)
    assert (scan / "recon" / "splat" / "scene.ply").is_file()


def test_single_view_is_quality_insufficient(tmp_path: Path, docker: FakeDocker) -> None:
    """Given a model with one registered image,
    When the stage runs,
    Then it exits with 20 and hints, without training.
    """
    scan = make_recon_scan(tmp_path, images=1)

    result = recon(scan)

    assert result.exit_code is ExitCode.QUALITY_INSUFFICIENT
    assert result.manifest.hints
    assert docker.jobs() == []


def test_large_train_holdout_gap_warns(scan: Path, docker: FakeDocker) -> None:
    """Given training views 6 dB better than held-out views (spike-01: fragmented model),
    When the stage runs,
    Then it succeeds with a warning and a recording hint.
    """
    docker.trainer((0, {"psnr_db": 24.0, "train_psnr_db": 30.0}))

    result = recon(scan)

    assert result.exit_code is ExitCode.OK
    assert any("30.0 dB" in warning for warning in result.manifest.warnings)
    assert result.manifest.hints


def test_cli_command_uses_environment_overrides(
    scan: Path, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given DRONESCAN_SPLAT__MAX_ITERATIONS=500 (as `make smoke` sets it),
    When `drone-scan recon --scan` runs,
    Then the trainer gets 500 iterations and the command exits with 0.
    """
    monkeypatch.setenv("DRONESCAN_SPLAT__MAX_ITERATIONS", "500")

    outcome = runner.invoke(app, ["recon", "--scan", str(scan)])

    assert outcome.exit_code == 0, outcome.output
    assert docker.jobs()[0]["job"]["iterations"] == 500


def test_worker_image_tag_matches_compose() -> None:
    """Given compose.yaml,
    When the stage starts the worker,
    Then it uses the same image tag as `docker compose build worker`.
    """
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    assert compose["services"]["worker"]["image"] == WORKER_IMAGE


def test_sampler_sums_only_the_containers_processes(
    docker: FakeDocker,
) -> None:
    """Given nvidia-smi lists the container's PID 4242 (twice, once without a value) and others,
    When the sampler takes a sample,
    Then only the container's memory counts.
    """
    sampler = ContainerVramSampler(0, "dronescan-worker-x")
    sampler._sample()

    assert sampler.peak == 2345.0


# ---------- scene and PLY (numpy) ----------


def test_view_scaling_keeps_the_principal_point_relative() -> None:
    """Given a 1600x900 PINHOLE view,
    When it is scaled to a 960 px longest edge,
    Then size, focal lengths and principal point scale by the same factor; smaller views stay.
    """
    view = View("a.jpg", (1, 0, 0, 0), (0, 0, 0), 1200, 1100, 800, 450, 1600, 900)

    scaled = view.scaled(960)

    assert (scaled.width, scaled.height) == (960, 540)
    assert (scaled.fx, scaled.fy, scaled.cx, scaled.cy) == pytest.approx((720, 660, 480, 270))
    assert view.scaled(2000) is view


def test_every_nth_view_is_held_out() -> None:
    """Given 17 views and holdout_every=8,
    When they are split,
    Then views 0, 8 and 16 are held out and the rest trains.
    """
    train, held_out = split_holdout(17, 8)

    assert held_out == [0, 8, 16]
    assert len(train) == 14 and not set(train) & set(held_out)


def test_scene_reader_rejects_distorted_cameras(tmp_path: Path) -> None:
    """Given a model with a SIMPLE_RADIAL camera (not undistorted),
    When it is read as training scene,
    Then a SceneError says so.
    """
    write_sfm(tmp_path, 2)

    with pytest.raises(SceneError, match="SIMPLE_RADIAL"):
        read_scene(tmp_path / "sfm" / "colmap" / "sparse" / "0")


def test_ply_has_the_3dgs_layout(tmp_path: Path) -> None:
    """Given 5 Gaussians with SH degree 1,
    When they are written as PLY,
    Then the header lists the 3DGS properties (9 rest coefficients, channel-major), the data
    has 5 records of float32 and quaternions are normalised.
    """
    rng = np.random.default_rng(0)
    count = 5
    sh_rest = rng.random((count, 3, 3), dtype=np.float32)
    path = tmp_path / "scene.ply"

    write_ply(
        path,
        means=rng.random((count, 3), dtype=np.float32),
        sh0=rng.random((count, 1, 3), dtype=np.float32),
        sh_rest=sh_rest,
        opacity_logits=rng.random(count, dtype=np.float32),
        log_scales=rng.random((count, 3), dtype=np.float32),
        quats_wxyz=np.full((count, 4), 2.0, dtype=np.float32),
    )

    vertices, names = read_ply_header(path)
    assert vertices == count
    assert names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    assert names[9:18] == [f"f_rest_{i}" for i in range(9)]
    assert names[18:] == ["opacity", "scale_0", "scale_1", "scale_2"] + [
        f"rot_{i}" for i in range(4)
    ]
    raw = path.read_bytes()
    body = raw[raw.index(b"end_header\n") + len(b"end_header\n") :]
    data = np.frombuffer(body, dtype="<f4").reshape(count, len(names))
    assert data[:, names.index("rot_0")] == pytest.approx(0.5)
    # channel-major: f_rest_0..2 are the red coefficients of the three rest bands
    assert data[0, 9:12] == pytest.approx(sh_rest[0, :, 0])


def test_binary_and_text_models_give_the_same_scene(tmp_path: Path) -> None:
    """Given the same undistorted model as COLMAP binary (as image_undistorter writes it) and text,
    When both are read,
    Then views (pose, intrinsics, size), points and colours are identical; 2D points and tracks
    are skipped correctly.
    """
    binary, text = tmp_path / "bin", tmp_path / "txt"
    binary.mkdir()
    text.mkdir()
    camera = (1, 1, 640, 360, 300.0, 301.0, 320.0, 180.0)
    (binary / "cameras.bin").write_bytes(struct.pack("<Q", 1) + struct.pack("<iiQQ4d", *camera))
    (text / "cameras.txt").write_text("1 PINHOLE 640 360 300 301 320 180\n")
    images = [(1, "f01/frames/000002.jpg", 0.5), (2, "f01/frames/000001.jpg", -0.25)]
    data = struct.pack("<Q", len(images))
    rows = ""
    for image_id, name, tx in images:
        pose = (0.9, 0.1, -0.2, 0.3, tx, 1.5, -2.0)
        data += struct.pack("<I7dI", image_id, *pose, 1) + name.encode() + b"\0"
        data += struct.pack("<Q", 2) + struct.pack("<2dq", 1.0, 2.0, 7) * 2
        rows += f"{image_id} {' '.join(map(str, pose))} 1 {name}\n1.0 2.0 7 1.0 2.0 7\n"
    (binary / "images.bin").write_bytes(data)
    (text / "images.txt").write_text(rows)
    points = [(7, 1.0, 2.0, 3.0, 10, 20, 30, 3), (9, -1.0, 0.5, 4.0, 200, 100, 0, 1)]
    data = struct.pack("<Q", len(points))
    rows = ""
    for point_id, x, y, z, r, g, b, track in points:
        data += struct.pack("<Q3d3BdQ", point_id, x, y, z, r, g, b, 0.5, track)
        data += struct.pack("<ii", 1, 0) * track
        rows += f"{point_id} {x} {y} {z} {r} {g} {b} 0.5" + " 1 0" * track + "\n"
    (binary / "points3D.bin").write_bytes(data)
    (text / "points3D.txt").write_text(rows)

    from_binary, from_text = read_scene(binary), read_scene(text)

    assert from_binary.views == from_text.views
    assert [view.name for view in from_binary.views] == [images[1][1], images[0][1]]
    assert (from_binary.views[0].fy, from_binary.views[0].height) == (301.0, 360)
    assert np.array_equal(from_binary.points, from_text.points)
    assert np.array_equal(from_binary.colors, from_text.colors)
    assert from_binary.colors[1].tolist() == pytest.approx([200 / 255, 100 / 255, 0.0])
