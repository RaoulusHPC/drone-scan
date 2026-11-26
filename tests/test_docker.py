# SPDX-License-Identifier: AGPL-3.0-or-later
"""Docker images for worker (CUDA) and COLMAP (KAN-18)."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "docker" / "worker.Dockerfile"
COLMAP = ROOT / "docker" / "colmap.Dockerfile"
COMPOSE = ROOT / "compose.yaml"
GSPLAT_PATCH = ROOT / "docker" / "patches" / "gsplat-v1.5.3-pascal.patch"
GSPLAT_CHECK = ROOT / "docker" / "checks" / "gsplat_grads.py"


def compose() -> dict[str, Any]:
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def image_name(service: str) -> str:
    return str(compose()["services"][service]["image"])


def require_image(service: str) -> None:
    """Skip a GPU test if the image is not built (`docker compose run` would silently build it
    and the long timeouts would then look like a test failure)."""
    found = subprocess.run(
        ["docker", "image", "inspect", image_name(service)],
        capture_output=True,
        check=False,
    )
    if found.returncode != 0:
        pytest.skip(f"image {image_name(service)} not built (docker compose build {service})")


def runtime_stage(dockerfile: Path) -> str:
    """Text of the stage that becomes the shipped image (`AS runtime`; the worker's `test`
    stage after it only exists for `make test-gpu`)."""
    stages = dockerfile.read_text(encoding="utf-8").split("\nFROM ")
    return next(stage for stage in stages if stage.split("\n", 1)[0].endswith(" AS runtime"))


def arg_value(dockerfile: Path, name: str) -> str:
    match = re.search(rf"^ARG {name}=(\S+)", dockerfile.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"ARG {name} missing in {dockerfile.name}"
    return match.group(1).strip('"')


def test_worker_uses_cuda_126_and_torch_below_215() -> None:
    """Given the worker Dockerfile,
    When its pinned versions are read,
    Then it uses CUDA 12.6 and PyTorch < 2.15 from the cu126 index (ADR-0010).
    """
    assert arg_value(WORKER, "CUDA_VERSION").startswith("12.6.")
    major, minor = (int(x) for x in arg_value(WORKER, "TORCH_VERSION").split(".")[:2])
    assert (major, minor) < (2, 15)
    assert "download.pytorch.org/whl/cu126" in WORKER.read_text(encoding="utf-8")


@pytest.mark.parametrize("dockerfile", [WORKER, COLMAP])
def test_images_build_for_pascal_and_turing(dockerfile: Path) -> None:
    """Given both images,
    When their CUDA architectures are read,
    Then they target sm_61 (GTX 1060) and sm_75 (RTX 2060).
    """
    text = dockerfile.read_text(encoding="utf-8")
    assert re.search(r'(TORCH_CUDA_ARCH_LIST|CUDA_ARCHITECTURES)="?6\.?1;7\.?5"?', text)
    assert "12.6." in arg_value(dockerfile, "CUDA_VERSION")


def test_sources_are_pinned_to_commits() -> None:
    """Given third-party sources built in the images,
    When they are fetched,
    Then gsplat and COLMAP are pinned to full commit SHAs.
    """
    assert re.fullmatch(r"[0-9a-f]{40}", arg_value(WORKER, "GSPLAT_COMMIT"))
    assert re.fullmatch(r"[0-9a-f]{40}", arg_value(COLMAP, "COLMAP_COMMIT"))


def test_colmap_avoids_proprietary_and_gpl_build_options() -> None:
    """Given the COLMAP image,
    When its build options are read,
    Then it uses OpenBLAS instead of Intel MKL and disables GUI (Qt) and CGAL (GPL parts).
    """
    text = COLMAP.read_text(encoding="utf-8")
    assert "libmkl" not in text and "Intel10" not in text
    assert "-DBLA_VENDOR=OpenBLAS" in text
    assert "-DGUI_ENABLED=OFF" in text
    assert "-DCGAL_ENABLED=OFF" in text


def test_colmap_is_offline_and_without_agpl_lsd() -> None:
    """Given the COLMAP image,
    When its build options are read,
    Then runtime downloads and the FetchContent ONNX runtime are off (CLAUDE.md rule 3)
    and the AGPL-3.0 LSD line detector is not compiled in (license-auditor 2026-11-21).
    """
    text = COLMAP.read_text(encoding="utf-8")
    for flag in ("DOWNLOAD_ENABLED", "ONNX_ENABLED", "FETCH_ONNX", "LSD_ENABLED"):
        assert f"-D{flag}=OFF" in text, flag


@pytest.mark.parametrize(
    ("dockerfile", "notice"),
    [
        (WORKER, "/usr/share/doc/gsplat/"),
        (COLMAP, "/usr/share/doc/colmap/"),
    ],
)
def test_runtime_image_ships_license_notices(dockerfile: Path, notice: str) -> None:
    """Given the runtime image stage,
    When third-party code was compiled into it,
    Then its license texts are copied into the image (ADR-0007).
    """
    assert notice in runtime_stage(dockerfile)


def test_gsplat_patch_is_applied_strictly() -> None:
    """Given the pinned gsplat Pascal patch,
    When the worker image is built,
    Then the patch is checked with `git apply --check` (a stale patch fails the build).
    """
    text = WORKER.read_text(encoding="utf-8")
    assert "docker/patches/gsplat-v1.5.3-pascal.patch" in text
    assert "git apply --check /tmp/gsplat-pascal.patch" in text
    assert arg_value(WORKER, "GSPLAT_VERSION") == "v1.5.3"


def _patch_lines(sign: str) -> list[str]:
    lines = GSPLAT_PATCH.read_text(encoding="utf-8").splitlines()
    return [line[1:] for line in lines if line.startswith(sign) and not line.startswith(sign * 3)]


def test_gsplat_patch_keeps_the_sm70_path_unchanged() -> None:
    """Given the gsplat Pascal patch (ADR-0010),
    When its changes are read,
    Then the kernels only swap cg::labeled_partition for labeled_partition_compat, which
    calls cg::labeled_partition unchanged for __CUDA_ARCH__ >= 700 (sm_75 = RTX 2060), and
    only falls back to single-thread groups below sm_70; besides that only C++17 -> C++20.
    """
    removed, added = _patch_lines("-"), _patch_lines("+")
    kernel_removed = [line for line in removed if "labeled_partition" in line]
    assert len(kernel_removed) == 8
    assert [
        line.replace("cg::labeled_partition(", "labeled_partition_compat(")
        for line in kernel_removed
    ] == [line for line in added if "labeled_partition_compat(warp" in line]
    assert set(removed) - set(kernel_removed) == {
        '    nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]'
    }

    helper = "\n".join(added)
    fallback, sm70 = helper.split("#else", 1)
    assert "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700" in fallback
    assert "cg::tiled_partition<1>(tile)" in fallback
    assert "return cg::labeled_partition(tile, label);" in sm70.split("#endif", 1)[0]


@pytest.mark.parametrize("dockerfile", [WORKER, COLMAP])
def test_runtime_stage_runs_as_non_root(dockerfile: Path) -> None:
    """Given the runtime image stage,
    When a container starts,
    Then it does not run as root.
    """
    for stage in (runtime_stage(dockerfile), dockerfile.read_text(encoding="utf-8")):
        last_user = re.findall(r"^USER (\S+)", stage, re.MULTILINE)[-1]
        assert last_user != "root"  # the shipped stage and the last (test) stage


def test_compose_defines_worker_and_colmap_on_gtx_1060() -> None:
    """Given compose.yaml,
    When the GPU reservations are read,
    Then both services use one GPU, defaulting to device 0 (GTX 1060);
    the RTX 2070 SUPER is never used by default (CLAUDE.md).
    """
    services = compose()["services"]
    assert {"worker", "colmap"} <= set(services)
    for name in ("worker", "colmap"):
        devices = services[name]["deploy"]["resources"]["reservations"]["devices"]
        assert devices[0]["capabilities"] == ["gpu"]
        assert devices[0]["device_ids"] == ["${DRONESCAN_GPU_DEVICE:-0}"]
        assert services[name]["environment"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert services[name]["runtime"] == "nvidia"


def test_compose_sets_vram_limit_and_image_names() -> None:
    """Given compose.yaml,
    When the worker service is read,
    Then GPU code starts with the 5.0 GiB allocator limit by default (CLAUDE.md rule 2) and
    the image names match what the GPU tests and the README use.
    """
    services = compose()["services"]
    assert services["worker"]["environment"]["DRONESCAN_VRAM_LIMIT_GB"] == (
        "${DRONESCAN_VRAM_LIMIT_GB:-5.0}"
    )
    assert image_name("worker").startswith("drone-scan/worker:")
    assert image_name("colmap") == f"drone-scan/colmap:{arg_value(COLMAP, 'COLMAP_VERSION')}"


def test_dockerfiles_check_device_code_for_both_architectures() -> None:
    """Given both Dockerfiles,
    When the images are built,
    Then a build step lists the device code (cuobjdump) and fails without sm_61 and sm_75 (AC 1).
    """
    for dockerfile in (WORKER, COLMAP):
        text = dockerfile.read_text(encoding="utf-8")
        assert "cuobjdump --list-elf" in text
        assert "grep -q 'sm_61'" in text and "grep -q 'sm_75'" in text


def test_compose_mounts_workspace_and_data() -> None:
    """Given the worker service,
    When it runs,
    Then workspace/ is mounted read-write and data/ read-only.
    """
    volumes = compose()["services"]["worker"]["volumes"]
    assert "./workspace:/workspace" in volumes
    assert "./data:/data:ro" in volumes


def test_worker_service_builds_the_runtime_stage() -> None:
    """Given the worker Dockerfile whose last stage is `test` (KAN-41),
    When compose builds the worker service,
    Then it builds the `runtime` stage, so the worker image carries no test tools.
    """
    assert compose()["services"]["worker"]["build"]["target"] == "runtime"
    assert re.search(r"^FROM \S+ AS runtime$", WORKER.read_text(encoding="utf-8"), re.MULTILINE)


def test_worker_test_service_runs_torch_gpu_tests_offline_on_the_source() -> None:
    """Given the worker-test service (make test-gpu, KAN-41),
    When it runs,
    Then it uses the `test` stage on the GTX 1060 with the 5.0 GiB limit, has no network, tests
    the read-only mounted repository from source, hides data/ and workspace/ (private
    recordings) and takes the shared GPU lock; it is not started by `docker compose up`.
    """
    service = compose()["services"]["worker-test"]
    worker = compose()["services"]["worker"]
    assert service["build"]["target"] == "test"
    assert service["profiles"] == ["test"]
    assert service["network_mode"] == "none"
    assert service["deploy"] == worker["deploy"] and service["runtime"] == "nvidia"
    environment = service["environment"]
    assert environment["DRONESCAN_VRAM_LIMIT_GB"] == "${DRONESCAN_VRAM_LIMIT_GB:-5.0}"
    assert (
        environment["DRONESCAN_GPU__LOCK_FILE"] == worker["environment"]["DRONESCAN_GPU__LOCK_FILE"]
    )
    assert environment["PYTHONPATH"].split(":")[0] == "/repo/src"
    assert environment["DRONESCAN_TEST_REQUIRE_GPU"] == "1"
    assert ".:/repo:ro" in service["volumes"]
    assert set(service["tmpfs"]) == {"/repo/data", "/repo/workspace"}
    assert image_name("worker-test").startswith("drone-scan/worker-test:")
    test_stage = WORKER.read_text(encoding="utf-8").rsplit("\nFROM ", 1)[1]
    assert test_stage.startswith("runtime AS test")
    assert "--only-group dev" in test_stage and "--locked" in test_stage
    assert '"gpu and torch"' in test_stage


def test_make_test_gpu_runs_host_and_container_parts() -> None:
    """Given the Makefile,
    When `make test-gpu` runs,
    Then the host runs the GPU tests without PyTorch and the worker-test container the
    PyTorch ones, both on GTX 1060 (device 0) with the 5.0 GiB limit.
    """
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("\ntest-gpu:", 1)[1].split("\n\n", 1)[0]
    assert 'pytest -m "gpu and not torch"' in recipe
    assert "docker compose run --rm -T worker-test" in recipe
    assert "DRONESCAN_GPU_DEVICE=$(GPU_DEVICES)" in recipe
    assert "DRONESCAN_VRAM_LIMIT_GB=$(VRAM_LIMIT_GB)" in recipe
    assert re.search(r"^GPU_DEVICES \?= 0$", makefile, re.MULTILINE)
    assert re.search(r"^VRAM_LIMIT_GB \?= 5\.0$", makefile, re.MULTILINE)


def test_dockerignore_keeps_private_and_large_files_out() -> None:
    """Given .dockerignore,
    When the build context is sent,
    Then scans, data, local venvs and git history are excluded (CLAUDE.md rules 3 and 5).
    """
    entries = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert {"workspace", "data", ".venv", ".git"} <= {e.strip("/") for e in entries}


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_worker_sees_gtx_1060_without_warning() -> None:
    """Given the built worker image and the GTX 1060 (device 0),
    When `docker compose run worker python -c '...get_device_capability()'` runs,
    Then it prints (6, 1) and PyTorch emits no compatibility warning (AC 2).
    """
    require_image("worker")
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "worker",
        "python",
        "-W",
        "always",
        "-c",
        "import torch; print(torch.cuda.get_device_capability())",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("(6, 1)")
    # PyTorch's own compatibility warning; `docker compose` may print unrelated WARN lines
    assert "UserWarning" not in result.stderr
    assert "not compatible" not in result.stderr.lower()
    assert "too old" not in result.stderr.lower()


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_patched_gsplat_gradients_match_reference_on_gtx_1060() -> None:
    """Given the worker image with the Pascal-patched gsplat and the GTX 1060 (device 0),
    When the projection backward pass (3DGS/2DGS, fused/packed) runs on synthetic data,
    Then its gradients match a float64 PyTorch reference (relative L2 error, limits per
    gradient, wider for the approximate 2DGS gradients) and packed matches fused, under the
    5.0 GiB allocator limit (ADR-0010, CLAUDE.md rule 2). `peak_vram_mb` is allocator memory
    only (no CUDA context), the allocator limit is what enforces the budget here.
    """
    require_image("worker")
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-T",
        "-e",
        "DRONESCAN_VRAM_LIMIT_GB=5.0",
        "worker",
        "python",
        "-",
    ]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        input=GSPLAT_CHECK.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["ok"] is True
    assert report["capability"] == [6, 1]
    assert report["peak_vram_mb"] < 5.0 * 1024


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_colmap_binary_starts_with_all_libraries() -> None:
    """Given the built colmap image (needs the image, not the GPU: marked gpu like the others),
    When the colmap binary's dynamic libraries are resolved and `colmap help` runs,
    Then no library is missing and it prints COLMAP 4.2.1 built with CUDA.
    """
    require_image("colmap")
    ldd = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            image_name("colmap"),
            "-c",
            "ldd /usr/local/bin/colmap",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert ldd.returncode == 0, ldd.stderr
    assert "not found" not in ldd.stdout, ldd.stdout

    result = subprocess.run(
        ["docker", "compose", "run", "--rm", "colmap", "colmap", "help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "COLMAP 4.2.1" in result.stdout
    assert "with CUDA" in result.stdout


def _synthetic_png(path: Path, width: int = 640, height: int = 480) -> None:
    """Grey noise image (no real recordings in tests, CLAUDE.md rule 3)."""
    rng = random.Random(0)
    raw = b"".join(b"\x00" + bytes(rng.randrange(256) for _ in range(width)) for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_colmap_extracts_sift_features_on_the_gpu(tmp_path: Path) -> None:
    """Given the colmap image (CUDA 61;75) and the GTX 1060 (device 0),
    When SIFT features are extracted from a synthetic image with use_gpu=1,
    Then the GPU extractor runs and finds features, i.e. the sm_61 kernels work.
    """
    require_image("colmap")
    (tmp_path / "images").mkdir()
    _synthetic_png(tmp_path / "images" / "synth.png")
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-v",
        f"{tmp_path}:/work",
        "colmap",
        "colmap",
        "feature_extractor",
        "--database_path",
        "/work/db.db",
        "--image_path",
        "/work/images",
        "--FeatureExtraction.use_gpu",
        "1",
        "--FeatureExtraction.gpu_index",
        "0",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300, check=False)

    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-2000:]
    assert "Creating SIFT GPU feature extractor" in log
    match = re.search(r"Features:\s+(\d+) \(SIFT\)", log)
    assert match and int(match.group(1)) > 0, log[-2000:]


def test_worker_runtime_stage_installs_ffmpeg() -> None:
    """Given the worker Dockerfile (KAN-81),
    When its runtime stage is read,
    Then ffmpeg (provides ffprobe) is installed from apt, so `flight add` and `ingest` work.
    """
    stage = runtime_stage(WORKER)
    assert re.search(r"apt-get install\b[^\n]*(\\\n[^\n]*)*\bffmpeg\b", stage)


def _drone_scan(args: list[str], *, cwd: Path, image: str | None = None) -> str:
    """Run `drone-scan` on the host (image None) or in the worker image without network."""
    if image is None:
        command = [sys.executable, "-c", "from dronescan.cli import app; app()", *args]
    else:
        command = ["docker", "run", "--rm", "--network", "none"]
        command += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
        command += ["-v", f"{cwd}:/work", "-w", "/work", image, "drone-scan", *args]
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, f"{args[0]} ({image or 'host'}): {result.stderr[-2000:]}"
    return result.stdout


def _flight_and_ingest(root: Path, video: Path, image: str | None) -> Path:
    """`scan create`, `flight add`, `ingest` below ``root``; paths are relative to the cwd so
    the same arguments work on the host and in the container (mounted at /work)."""
    root.mkdir()
    shutil.copy(video, root / "video.mp4")
    out = _drone_scan(["scan", "create", "Synthetik", "--workspace", "ws"], cwd=root, image=image)
    scan = f"ws/{out.strip().rsplit(':', 1)[1].strip()}"
    _drone_scan(["flight", "add", scan, "video.mp4"], cwd=root, image=image)
    _drone_scan(["ingest", "--scan", scan], cwd=root, image=image)
    return root / scan


@pytest.mark.gpu  # needs the image, not the GPU (marker as for the colmap test above)
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="host ffmpeg missing (makes the video)")
def test_flight_add_and_ingest_in_worker_match_host(tmp_path: Path) -> None:
    """Given a synthetic video (ffmpeg testsrc2) and the built worker image (needs no GPU),
    When `scan create`, `flight add` and `ingest` run on the host and in the container
    (`--network none`, host ffmpeg 8.x vs. image ffmpeg 6.1),
    Then the same keyframes are selected (same indices, windows, reasons, files) and
    frames.parquet has the same schema; timestamps agree within 1 microsecond (showinfo
    prints microseconds, so versions may differ only by rounding) and sharpness within 1 %
    (scaling in different ffmpeg versions may round pixels differently).
    """
    require_image("worker")
    video = tmp_path / "synthetic.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12"]
        + ["-t", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    version = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", image_name("worker"), "ffmpeg", "-version"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert version.returncode == 0, version.stderr
    assert version.stdout.startswith("ffmpeg version 6.1"), version.stdout.splitlines()[:1]

    host_scan = _flight_and_ingest(tmp_path / "host", video, None)
    box_scan = _flight_and_ingest(tmp_path / "box", video, image_name("worker"))

    host_table = pq.read_table(host_scan / "ingest" / "f01" / "frames.parquet")
    box_table = pq.read_table(box_scan / "ingest" / "f01" / "frames.parquet")
    assert box_table.schema == host_table.schema
    host_rows, box_rows = host_table.to_pylist(), box_table.to_pylist()
    assert len(box_rows) == len(host_rows) > 0
    for field in ("index", "window", "selected", "reason", "file"):
        assert [r[field] for r in box_rows] == [r[field] for r in host_rows], field
    assert [r["timestamp_s"] for r in box_rows] == pytest.approx(
        [r["timestamp_s"] for r in host_rows], abs=1e-6
    )
    assert [r["sharpness"] for r in box_rows] == pytest.approx(
        [r["sharpness"] for r in host_rows], rel=0.01
    )
    host_jpegs = sorted((host_scan / "ingest" / "f01").rglob("*.jpg"))
    box_jpegs = sorted((box_scan / "ingest" / "f01").rglob("*.jpg"))
    assert [p.name for p in box_jpegs] == [p.name for p in host_jpegs]
    assert len(host_jpegs) == sum(r["selected"] for r in host_rows) > 0
