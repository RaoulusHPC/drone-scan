"""Docker images for worker (CUDA) and COLMAP (KAN-18)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "docker" / "worker.Dockerfile"
COLMAP = ROOT / "docker" / "colmap.Dockerfile"
COMPOSE = ROOT / "compose.yaml"


def compose() -> dict[str, Any]:
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


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
    """Given the final image stage,
    When third-party code was compiled into it,
    Then its license texts are copied into the image (ADR-0007).
    """
    final_stage = dockerfile.read_text(encoding="utf-8").rsplit("\nFROM ", 1)[1]
    assert notice in final_stage


@pytest.mark.parametrize("dockerfile", [WORKER, COLMAP])
def test_runtime_stage_runs_as_non_root(dockerfile: Path) -> None:
    """Given the final image stage,
    When a container starts,
    Then it does not run as root.
    """
    final_stage = dockerfile.read_text(encoding="utf-8").rsplit("\nFROM ", 1)[1]
    assert re.search(r"^USER (?!root)\S+", final_stage, re.MULTILINE)


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


def test_compose_mounts_workspace_and_data() -> None:
    """Given the worker service,
    When it runs,
    Then workspace/ is mounted read-write and data/ read-only.
    """
    volumes = compose()["services"]["worker"]["volumes"]
    assert "./workspace:/workspace" in volumes
    assert "./data:/data:ro" in volumes


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
    assert "warn" not in result.stderr.lower()
