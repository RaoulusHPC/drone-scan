# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage `sfm`: camera poses with COLMAP (KAN-36, US-021).

CPU tests replace `docker` by a small script; GPU tests run the real COLMAP container on a
synthetic room. Private clips are never used.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from numpy.typing import NDArray
from pydantic import ValidationError
from typer.testing import CliRunner

from dronescan.capture.scan import FlightMeta, ScanMeta
from dronescan.cli import app
from dronescan.common.config import Settings, load_settings
from dronescan.common.stage import ExitCode, Manifest, StageResult, run_stage
from dronescan.sfm.colmap import COLMAP_IMAGE, WATCHDOG, Colmap, ColmapUnavailableError, Mount
from dronescan.sfm.model import list_models, read_report
from dronescan.sfm.stage import SfmStage
from tests.synthetic_room import View, aligned_position_errors, views, write_jpegs
from tests.test_docker import require_image

ROOT = Path(__file__).resolve().parents[1]
MAX_PROCESS_MIB = 5.5 * 1024  # Q-2, including the CUDA context

runner = CliRunner()


# ---------- scans ----------


def write_manifest(scan_dir: Path, status: str = "ok") -> None:
    now = datetime.now(UTC)
    manifest = Manifest(
        stage="ingest",
        version="1",
        profile="low_vram_6gb",
        inputs={},
        params={},
        tools={},
        started=now,
        finished=now,
        peak_vram_mb=None,
        status=status,  # type: ignore[arg-type]
        exit_code=0 if status == "ok" else 20,
    )
    (scan_dir / "ingest").mkdir(parents=True, exist_ok=True)
    (scan_dir / "ingest" / "manifest.json").write_text(manifest.model_dump_json())


def write_table(flight_dir: Path, keyframes: int) -> None:
    """frames.parquet as written by `ingest`: rejected frames have no file."""
    rows: list[dict[str, Any]] = []
    for number in range(1, keyframes + 1):
        rows.append({"timestamp_s": number / 2, "file": f"frames/{number:06d}.jpg"})
        rows.append({"timestamp_s": number / 2 + 0.1, "file": None})
    flight_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), flight_dir / "frames.parquet")


def make_scan(scan_dir: Path, *keyframes: int, images: bool = True) -> Path:
    """Scan directory after `ingest` with one flight per entry of ``keyframes``."""
    scan_dir.mkdir(parents=True)
    flights = []
    for index, count in enumerate(keyframes, start=1):
        flight = f"f{index:02d}"
        write_table(scan_dir / "ingest" / flight, count)
        frames = scan_dir / "ingest" / flight / "frames"
        frames.mkdir()
        if images:
            for number in range(1, count + 1):
                (frames / f"{number:06d}.jpg").write_bytes(b"jpg")
        flights.append(
            FlightMeta(
                id=flight,
                source_name="synthetic.mp4",
                sha256="0" * 64,
                added=datetime.now(UTC),
                width=640,
                height=480,
                rotation=0,
                display_width=640,
                display_height=480,
                fps=30,
                duration_s=10,
                codec="h264",
            )
        )
    meta = ScanMeta(id="synthetik", name="Synthetik", created=datetime.now(UTC), flights=flights)
    (scan_dir / "scan.yaml").write_text(yaml.safe_dump(meta.model_dump(mode="json")))
    write_manifest(scan_dir)
    return scan_dir


def with_sfm(**values: Any) -> Settings:
    settings = load_settings()
    return settings.model_copy(update={"sfm": settings.sfm.model_copy(update=values)})


def sfm(scan_dir: Path, settings: Settings | None = None, force: bool = False) -> StageResult:
    return run_stage(SfmStage(), scan_dir, settings or load_settings(), force=force)


# ---------- fake docker ----------

FAKE_DOCKER = """#!{python}
import csv, json, struct, sys, time
from pathlib import Path

state = Path({state!r})
config = json.loads((state / "config.json").read_text())
args = sys.argv[1:]
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
if args[0] == "image":
    if not config["image"]:
        sys.exit("No such image")
    print("sha256:" + config.get("image_id", "1"))
    sys.exit(0)
if args[0] == "kill":
    sys.exit(0)

position = args.index({image!r})
position = args.index("colmap", position)
command = args[position + 1]
options = dict(zip(args[position + 2 :: 2], args[position + 3 :: 2]))
mounts = {{}}
for flag, value in zip(args, args[1:]):
    if flag == "--mount":
        fields = dict(f.partition("=")[::2] for f in next(csv.reader([value])))
        mounts[fields["target"]] = Path(fields["source"])
work = mounts["/sfm"]


def host(path):
    return work / path.removeprefix("/sfm/")


for failure in config["failures"]:
    if failure["command"] == command and failure["times"] > 0:
        failure["times"] -= 1
        (state / "config.json").write_text(json.dumps(config))
        print(failure["output"], file=sys.stderr)
        sys.stderr.flush()
        time.sleep(failure.get("hang_s", 0))
        sys.exit(failure["code"])

if command == "feature_extractor":
    # one call per camera setup; the same image list twice means a stale database
    database = host(options["--database_path"])
    image_list = options.get("--image_list_path", "") + "\\n"
    assert not database.exists() or image_list not in database.read_text()
    with database.open("a") as handle:
        handle.write(image_list)
elif command == "vocab_tree_builder":
    host(options["--vocab_tree_path"]).write_text("tree")
elif command == "sequential_matcher":
    if "--SequentialMatching.loop_detection" in options:
        assert host(options["--SequentialMatching.vocab_tree_path"]).is_file()
elif command == "mapper":
    names = (work / "image_list.txt").read_text().split()
    if not config["models"]:
        sys.exit("E1125 sfm.cc:1] Failed to create any sparse model")
    start = 0
    for index, (size, points) in enumerate(config["models"]):
        model = host(options["--output_path"]) / str(index)
        model.mkdir()
        (model / "images.bin").write_bytes(struct.pack("<Q", size))
        (model / "points3D.bin").write_bytes(struct.pack("<Q", points))
        (model / "names.json").write_text(json.dumps(names[start : start + size]))
        start += size
elif command == "model_converter":
    names = json.loads((host(options["--input_path"]) / "names.json").read_text())
    out = host(options["--output_path"])
    pose = "{{i}} 1 0 0 0 0 0 0 1 {{n}}\\n{{p}}\\n"
    (out / "images.txt").write_text(
        "# comment\\n"
        + "".join(
            pose.format(i=i, n=n, p="" if i % 2 else "1.0 2.0 -1") for i, n in enumerate(names)
        )
    )
    (out / "cameras.txt").write_text("1 SIMPLE_RADIAL 640 480 500 320 240 0\\n")
    (out / "points3D.txt").write_text(
        "# comment\\n1 0 0 0 1 2 3 0.5 1 0 2 0\\n2 0 0 0 1 2 3 1.0 1 1 2 1\\n"
    )
"""

FAKE_NVIDIA_SMI = """#!{python}
import sys
assert sys.argv[1:3] == ["-i", "0"]
print("9000, /usr/bin/other-process")
print("{mib}, colmap")
print("[N/A], /usr/local/bin/colmap")
"""


class FakeDocker:
    def __init__(self, directory: Path) -> None:
        self.state = directory
        self.config: dict[str, Any] = {"image": True, "models": [], "failures": []}
        self.save()

    def save(self) -> None:
        (self.state / "config.json").write_text(json.dumps(self.config))

    def models(self, *sizes: int) -> None:
        self.config["models"] = [[size, 100] for size in sizes]
        self.save()

    def fail(self, command: str, output: str, code: int = 1, times: int = 1) -> None:
        failure = {"command": command, "output": output, "code": code, "times": times}
        self.config["failures"].append(failure)
        self.save()

    def calls(self, kind: str = "run") -> list[list[str]]:
        path = self.state / "calls.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        return [call for call in map(json.loads, lines) if call[0] == kind]

    def commands(self) -> list[str]:
        return [call[call.index("colmap") + 1] for call in self.calls()]

    def options(self, command: str, occurrence: int = 0) -> dict[str, str]:
        call = [c for c in self.calls() if command in c][occurrence]
        tail = call[call.index(command) + 1 :]
        return dict(zip(tail[::2], tail[1::2], strict=True))


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    directory = tmp_path / "fake"
    directory.mkdir()
    scripts = {
        "docker": FAKE_DOCKER.format(
            python=sys.executable, state=str(directory), image=COLMAP_IMAGE
        ),
        "nvidia-smi": FAKE_NVIDIA_SMI.format(python=sys.executable, mib=1234),
    }
    for name, text in scripts.items():
        script = directory / name
        script.write_text(text)
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{directory}:{os.environ['PATH']}")
    return FakeDocker(directory)


@pytest.fixture
def scan(tmp_path: Path) -> Path:
    return make_scan(tmp_path / "scan", 10)


# ---------- AC 1: adapter and pipeline ----------


def test_short_scan_runs_features_exhaustive_matching_and_mapper(
    scan: Path, docker: FakeDocker
) -> None:
    """Given a scan with few keyframes,
    When the stage runs,
    Then COLMAP extracts features, matches all pairs and maps; the largest model is stored as
    COLMAP text in sfm/colmap/sparse/0.
    """
    docker.models(10)

    result = sfm(scan)

    assert result.exit_code is ExitCode.OK
    assert docker.commands() == [
        "feature_extractor",
        "exhaustive_matcher",
        "mapper",
        "model_converter",
    ]
    sparse = scan / "sfm" / "colmap" / "sparse" / "0"
    assert {f.name for f in sparse.iterdir()} == {"cameras.txt", "images.txt", "points3D.txt"}
    images = (scan / "sfm" / "colmap" / "image_list.txt").read_text().split()
    assert images == [f"f01/frames/{n:06d}.jpg" for n in range(1, 11)]
    assert (scan / "sfm" / "colmap" / "database.db").is_file()


@pytest.mark.parametrize(
    ("limit", "matcher"), [(10, "exhaustive_matcher"), (9, "sequential_matcher")]
)
def test_matcher_changes_above_the_keyframe_limit(
    scan: Path, docker: FakeDocker, limit: int, matcher: str
) -> None:
    """Given 10 keyframes and a limit of 10 or 9,
    When the stage runs,
    Then all pairs are matched up to the limit and above it sequential matching with loop
    detection on a vocabulary tree built from this scan (no download).
    """
    docker.models(10)

    result = sfm(scan, with_sfm(exhaustive_max_keyframes=limit))

    assert result.exit_code is ExitCode.OK
    commands = docker.commands()
    assert [c for c in commands if c.endswith("_matcher")] == [matcher]
    if matcher == "sequential_matcher":
        assert commands.index("vocab_tree_builder") < commands.index(matcher)
        tree = docker.options("vocab_tree_builder")
        options = docker.options(matcher)
        assert options["--SequentialMatching.loop_detection"] == "1"
        assert options["--SequentialMatching.vocab_tree_path"] == tree["--vocab_tree_path"]
        assert tree["--database_path"] == options["--database_path"]
    else:
        assert "vocab_tree_builder" not in commands


def test_default_matching_limit_is_250_keyframes() -> None:
    """Given the shipped profiles,
    When they are loaded,
    Then the matching limit and the thresholds follow the PO decision of 2026-11-25.
    """
    for profile in ("low_vram_6gb", "default_8gb"):
        cfg = load_settings(profile).sfm
        assert cfg.exhaustive_max_keyframes == 250
        assert cfg.min_registration_ratio == 0.8
        assert cfg.target_registration_ratio == 0.9


def test_container_is_offline_unprivileged_and_sees_one_gpu(scan: Path, docker: FakeDocker) -> None:
    """Given the COLMAP commands of a run,
    When the containers are started,
    Then they have no network, never pull, run as the calling user, mount ingest/ read-only
    and only GPU commands get a GPU (index 0 = GTX 1060).
    """
    docker.models(10)
    assert sfm(scan).exit_code is ExitCode.OK
    assert len(docker.calls()) == 4

    for call in docker.calls():
        assert call[call.index("--network") + 1] == "none"
        assert call[call.index("--pull") + 1] == "never"
        assert call[call.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
        mounts = [value for flag, value in zip(call, call[1:], strict=False) if flag == "--mount"]
        assert [m for m in mounts if m.endswith("target=/ingest,readonly")]
        gpu = any(name in call for name in ("feature_extractor", "exhaustive_matcher"))
        assert ("--gpus" in call) is gpu
        if gpu:
            assert call[call.index("--gpus") + 1] == "device=0"
            assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in call
        else:
            assert "NVIDIA_VISIBLE_DEVICES=void" in call


def test_gpu_device_uses_the_same_variable_as_compose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given DRONESCAN_GPU_DEVICE (compose.yaml),
    When the settings are loaded,
    Then stages use the same GPU and default to 0.
    """
    assert load_settings().gpu_device == 0
    assert "DRONESCAN_GPU_DEVICE:-0" in (ROOT / "compose.yaml").read_text()

    monkeypatch.setenv("DRONESCAN_GPU_DEVICE", "1")

    assert load_settings().gpu_device == 1


def test_image_tag_matches_compose() -> None:
    """Given the adapter and compose.yaml,
    When their image tags are compared,
    Then `docker compose build colmap` builds the image the adapter uses.
    """
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))

    assert compose["services"]["colmap"]["image"] == COLMAP_IMAGE


@pytest.mark.parametrize("name", ["with space", "a,b", 'quo"te', "umläut"])
def test_mount_paths_may_contain_any_character(
    tmp_path: Path, docker: FakeDocker, name: str
) -> None:
    """Given a scan directory with special characters,
    When the stage runs,
    Then the bind mounts still point to it.
    """
    scan_dir = make_scan(tmp_path / name, 5)
    docker.models(5)

    assert sfm(scan_dir).exit_code is ExitCode.OK


def test_several_flights_are_reconstructed_together(tmp_path: Path, docker: FakeDocker) -> None:
    """Given two flights,
    When the stage runs,
    Then all keyframes are used with one camera per flight and the manifest counts per flight.
    """
    scan_dir = make_scan(tmp_path / "scan", 6, 4)
    docker.models(10)

    result = sfm(scan_dir)

    assert result.exit_code is ExitCode.OK
    options = docker.options("feature_extractor")
    assert options["--ImageReader.single_camera_per_folder"] == "1"
    assert result.manifest.metrics["flights"] == {
        "f01": {"keyframes": 6, "registered": 6},
        "f02": {"keyframes": 4, "registered": 4},
    }
    assert "ingest/f02/frames" in result.manifest.inputs


def test_unchanged_scan_is_skipped_and_force_reruns(scan: Path, docker: FakeDocker) -> None:
    """Given a successful run,
    When the stage runs again without changes or with --force,
    Then it is skipped, or recomputed with the old results replaced.
    """
    docker.models(10)
    sfm(scan)
    stale = scan / "sfm" / "colmap" / "models" / "7"
    stale.mkdir()

    assert sfm(scan).skipped
    assert not sfm(scan, force=True).skipped
    assert not stale.exists()


def test_stopped_docker_keeps_a_valid_result(scan: Path, docker: FakeDocker) -> None:
    """Given a valid result,
    When the stage runs again while docker or the image is not available,
    Then it is skipped, and even with --force the old files are not removed.
    """
    docker.models(10)
    sfm(scan)
    docker.config["image"] = False
    docker.save()

    assert sfm(scan).skipped
    assert sfm(scan, force=True).exit_code is ExitCode.ERROR
    assert (scan / "sfm" / "colmap" / "sparse" / "0" / "images.txt").is_file()


def test_failed_vocab_tree_falls_back_with_a_warning(scan: Path, docker: FakeDocker) -> None:
    """Given a scan with too few features for a vocabulary tree,
    When it has more keyframes than the exhaustive limit,
    Then matching is sequential without loop detection and the manifest warns.
    """
    docker.models(10)
    docker.fail("vocab_tree_builder", "Number of descriptors is too small")

    result = sfm(scan, with_sfm(exhaustive_max_keyframes=5))

    assert result.exit_code is ExitCode.OK
    assert "--SequentialMatching.loop_detection" not in docker.options("sequential_matcher")
    assert result.manifest.metrics["matcher"] == "sequential"
    assert any("Loop-Detection" in warning for warning in result.manifest.warnings)
    assert any("vocab_tree_builder.log" in hint for hint in result.manifest.hints)


@pytest.mark.parametrize(("failures", "matcher"), [(1, "sequential+loop"), (2, "sequential")])
def test_loop_warning_matches_the_final_attempt(
    scan: Path, docker: FakeDocker, failures: int, matcher: str
) -> None:
    """Given a failed tree and then a matcher without enough VRAM,
    When the second attempt builds the tree, or fails again,
    Then matcher and warning describe the final attempt and the warning appears at most once.
    """
    docker.models(10)
    docker.fail("vocab_tree_builder", "failed", times=failures)
    docker.fail("sequential_matcher", "out of memory", code=0)

    manifest = sfm(scan, with_sfm(exhaustive_max_keyframes=5)).manifest

    assert manifest.metrics["matcher"] == matcher
    warnings = [warning for warning in manifest.warnings if "Loop-Detection" in warning]
    hints = [hint for hint in manifest.hints if "Loop-Detection" in hint]
    assert len(warnings) == len(hints) == failures - 1


def test_hanging_vocab_tree_is_stopped(scan: Path, docker: FakeDocker) -> None:
    """Given a vocabulary tree builder that hangs (COLMAP 4.2.1 does so without features),
    When the timeout of the profile is over,
    Then its container is killed and matching continues without loop detection.
    """
    docker.models(10)
    docker.fail("vocab_tree_builder", "", code=0)
    docker.config["failures"][0]["hang_s"] = 30
    docker.save()

    result = sfm(scan, with_sfm(exhaustive_max_keyframes=5, loop_vocab_timeout_s=5))

    assert result.exit_code is ExitCode.OK
    assert result.manifest.metrics["matcher"] == "sequential"
    assert any("LOOP_VOCAB_TIMEOUT_S" in hint for hint in result.manifest.hints)
    (builder,) = [call for call in docker.calls() if "vocab_tree_builder" in call]
    assert docker.calls("kill") == [["kill", builder[builder.index("--name") + 1]]]


def test_oom_in_sequential_matching_is_retried(scan: Path, docker: FakeDocker) -> None:
    """Given a sequential matcher without enough VRAM at first,
    When the stage runs,
    Then features, vocabulary tree and matching are repeated once.
    """
    docker.models(10)
    docker.fail("sequential_matcher", "out of memory", code=0)

    result = sfm(scan, with_sfm(exhaustive_max_keyframes=5))

    assert result.exit_code is ExitCode.OK
    steps = ["feature_extractor", "vocab_tree_builder", "sequential_matcher"]
    assert docker.commands()[:6] == steps * 2


def test_unwritable_output_exits_1_with_a_remedy(
    scan: Path, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given old results that cannot be removed,
    When the stage runs,
    Then it exits 1 and names disk space and permissions.
    """
    (scan / "sfm" / "colmap").mkdir(parents=True)

    def denied(*_: Any, **__: Any) -> None:
        raise PermissionError(13, "denied")

    monkeypatch.setattr("dronescan.sfm.stage.shutil.rmtree", denied)

    result = sfm(scan)

    assert result.exit_code is ExitCode.ERROR
    assert any("Schreibrechte" in hint for hint in result.manifest.hints)


def test_cli_registers_the_stage(
    scan: Path, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given the CLI,
    When `drone-scan sfm --scan <dir> --profile low_vram_6gb` runs,
    Then it follows the stage contract.
    """
    docker.models(10)

    result = runner.invoke(app, ["sfm", "--scan", str(scan), "--profile", "low_vram_6gb"])

    assert result.exit_code == 0, result.output
    assert "sfm: fertig." in result.output


# ---------- AC 1: failures ----------


def test_missing_ingest_exits_10(scan: Path, docker: FakeDocker) -> None:
    """Given a scan whose ingest did not finish successfully,
    When the stage runs,
    Then it exits 10 with a remedy and does not start COLMAP.
    """
    write_manifest(scan, status="quality_insufficient")

    result = sfm(scan)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert result.manifest.hints
    assert docker.calls() == []


def test_flight_added_after_ingest_exits_10(tmp_path: Path, docker: FakeDocker) -> None:
    """Given a second flight without keyframes,
    When the stage runs,
    Then it exits 10 instead of reconstructing only the first flight.
    """
    scan_dir = make_scan(tmp_path / "scan", 5, 5)
    shutil.rmtree(scan_dir / "ingest" / "f02")

    result = sfm(scan_dir)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert "f02" in str(result.manifest.error)


def test_missing_keyframe_file_exits_10(scan: Path, docker: FakeDocker) -> None:
    """Given a keyframe that is listed in frames.parquet but missing on disk,
    When the stage runs,
    Then it exits 10.
    """
    (scan / "ingest" / "f01" / "frames" / "000004.jpg").unlink()

    assert sfm(scan).exit_code is ExitCode.INPUT_MISSING


def test_scan_without_flights_exits_10(tmp_path: Path) -> None:
    """Given a scan without any video,
    When the stage runs,
    Then it exits 10 and explains how to add one.
    """
    result = sfm(make_scan(tmp_path / "scan"))

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert any("flight add" in hint for hint in result.manifest.hints)


def test_missing_image_exits_1_with_build_command(scan: Path, docker: FakeDocker) -> None:
    """Given a machine without the COLMAP image,
    When the stage runs,
    Then it exits 1, names the build command and never pulls an image.
    """
    docker.config["image"] = False
    docker.save()

    result = sfm(scan)

    assert result.exit_code is ExitCode.ERROR
    assert any("docker compose build colmap" in hint for hint in result.manifest.hints)
    assert docker.calls() == []


def test_missing_docker_exits_1(scan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a machine without docker,
    When the stage runs,
    Then it exits 1 with a remedy.
    """
    monkeypatch.setenv("PATH", "/nonexistent")

    result = sfm(scan)

    assert result.exit_code is ExitCode.ERROR
    assert "Docker" in str(result.manifest.error)


def test_container_start_failure_exits_1(scan: Path, docker: FakeDocker) -> None:
    """Given a docker daemon that cannot start GPU containers (exit 125),
    When the stage runs,
    Then it exits 1 and points to the NVIDIA container setup.
    """
    docker.fail("feature_extractor", "unknown or invalid runtime name: nvidia", code=125)

    result = sfm(scan)

    assert result.exit_code is ExitCode.ERROR
    assert any("make docker-check" in hint for hint in result.manifest.hints)


def test_colmap_failure_exits_1_without_stale_result(scan: Path, docker: FakeDocker) -> None:
    """Given an earlier result and a COLMAP command that fails,
    When the stage runs again,
    Then it exits 1 and the old model is gone.
    """
    docker.models(10)
    sfm(scan)
    docker.fail("mapper", "Check failed", code=134)

    result = sfm(scan, force=True)

    assert result.exit_code is ExitCode.ERROR
    assert "mapper" in str(result.manifest.error)
    assert not (scan / "sfm" / "colmap" / "sparse").exists()


@pytest.mark.parametrize(
    ("command", "code", "output"),
    [
        ("exhaustive_matcher", 1, "Not enough GPU memory to match 8192 features. Reduce ..."),
        # Real output of COLMAP 4.2.1 on the GTX 1060: it skips the image and exits with 0.
        ("feature_extractor", 0, "PyramidCU::GenerateFeatureList:\tout of memory"),
    ],
)
def test_gpu_oom_is_retried_once_with_smaller_values(
    scan: Path, docker: FakeDocker, command: str, code: int, output: str
) -> None:
    """Given a first attempt without enough VRAM (reported by exit code or only in the log),
    When the stage runs,
    Then features and matching are repeated once with the smaller values of the profile.
    """
    docker.models(10)
    docker.fail(command, output, code=code)
    cfg = load_settings().sfm

    result = sfm(scan)

    assert result.exit_code is ExitCode.OK
    commands = docker.commands()
    assert commands.count(command) == 2
    assert commands[-4:] == ["feature_extractor", "exhaustive_matcher", "mapper", "model_converter"]
    second = docker.options("feature_extractor", occurrence=1)
    edge, features = cfg.oom_retry_image_edge_px, cfg.oom_retry_num_features
    assert second["--FeatureExtraction.max_image_size"] == str(edge)
    assert second["--SiftExtraction.max_num_features"] == str(features)
    matching = docker.options("exhaustive_matcher", occurrence=-1)
    assert matching["--FeatureMatching.max_num_matches"] == str(cfg.oom_retry_num_matches)
    assert result.manifest.metrics["oom_retry"] is True
    assert result.manifest.metrics["limits"]["image_edge_px"] == edge
    logs = [log.name for log in (scan / "sfm" / "colmap" / "logs").iterdir()]
    assert len([name for name in logs if name.endswith(f"{command}.log")]) == 2
    assert any("Grafikspeicher" in warning for warning in result.manifest.warnings)


def test_second_gpu_oom_exits_30(scan: Path, docker: FakeDocker) -> None:
    """Given a GPU that is too small for both attempts,
    When the stage runs,
    Then it exits 30 with the GPU memory hint.
    """
    docker.fail("feature_extractor", "CUDA_ERROR_OUT_OF_MEMORY", times=2)

    result = sfm(scan)

    assert result.exit_code is ExitCode.OOM
    assert any("Grafikspeicher" in hint for hint in result.manifest.hints)
    assert result.manifest.peak_vram_mb == 1234  # also measured for failed commands


@pytest.mark.parametrize(("code", "output"), [(137, ""), (134, "std::bad_alloc")])
def test_host_oom_exits_30(scan: Path, docker: FakeDocker, code: int, output: str) -> None:
    """Given a mapper that runs out of RAM,
    When the stage runs,
    Then it exits 30 with the RAM hint.
    """
    docker.fail("mapper", output, code=code)

    result = sfm(scan)

    assert result.exit_code is ExitCode.OOM
    assert any("Arbeitsspeicher" in hint for hint in result.manifest.hints)


def test_interrupt_stops_the_container(
    tmp_path: Path, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a running COLMAP container,
    When the user presses Ctrl+C,
    Then a container of the adapter is killed (it must not keep the GPU busy).
    """
    original = subprocess.Popen.wait
    interrupted = False

    def wait(self: subprocess.Popen[bytes], timeout: float | None = None) -> int:
        nonlocal interrupted
        if not interrupted and "run" in list(self.args):  # type: ignore[arg-type]
            interrupted = True
            raise KeyboardInterrupt
        return original(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    colmap = Colmap([Mount(tmp_path, "/sfm")], gpu_device=0, log_dir=tmp_path / "logs")

    with pytest.raises(KeyboardInterrupt):
        colmap.run("feature_extractor", {"database_path": "/sfm/database.db"}, gpu=True)

    killed = docker.calls("kill")
    assert len(killed) == 1
    assert killed[0][1].startswith("dronescan-colmap-")


def test_interrupted_stage_writes_an_error_manifest(
    scan: Path, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a stage whose mapper is interrupted with Ctrl+C,
    When run_stage handles it,
    Then the manifest says error, there is no sparse/0 and the started container is killed.
    """
    docker.models(10)
    original = subprocess.Popen.wait
    interrupted = False

    def wait(self: subprocess.Popen[bytes], timeout: float | None = None) -> int:
        nonlocal interrupted
        if not interrupted and "mapper" in list(self.args):  # type: ignore[arg-type]
            interrupted = True
            original(self, timeout)  # the fake has recorded its call
            raise KeyboardInterrupt
        return original(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)

    with pytest.raises(KeyboardInterrupt):
        sfm(scan)

    manifest = Manifest.model_validate_json((scan / "sfm" / "manifest.json").read_text())
    assert manifest.status == "error"
    assert not (scan / "sfm" / "colmap" / "sparse").exists()
    (mapper,) = [call for call in docker.calls() if "mapper" in call]
    assert docker.calls("kill") == [["kill", mapper[mapper.index("--name") + 1]]]


@pytest.mark.parametrize(("script", "code"), [("exit 0", 0), ("exit 3", 3)])
def test_watchdog_passes_the_exit_code(script: str, code: int) -> None:
    """Given the script that wraps COLMAP in the container,
    When the command ends while the runner is alive (stdin open),
    Then its exit code is returned.
    """
    command = ["sh", "-c", WATCHDOG, "sh", "sh", "-c", script]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        assert proc.wait(timeout=10) == code


def test_watchdog_kills_the_command_when_the_runner_is_gone() -> None:
    """Given a long-running command,
    When stdin closes because the runner died,
    Then the command is killed at once.
    """
    command = ["sh", "-c", WATCHDOG, "sh", "sleep", "60"]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=0.5)
        assert proc.stdin is not None
        proc.stdin.close()
        assert proc.wait(timeout=10) == 137


def test_every_container_runs_under_the_watchdog(scan: Path, docker: FakeDocker) -> None:
    """Given a run,
    When containers are started,
    Then each has an open stdin and the watchdog as its command.
    """
    docker.models(10)
    assert sfm(scan).exit_code is ExitCode.OK
    assert len(docker.calls()) == 4

    for call in docker.calls():
        assert "--interactive" in call and WATCHDOG in call


def test_adapter_reports_unavailable_container(tmp_path: Path, docker: FakeDocker) -> None:
    """Given a missing image,
    When the adapter is used directly,
    Then it raises ColmapUnavailableError.
    """
    docker.config["image"] = False
    docker.save()

    with pytest.raises(ColmapUnavailableError):
        Colmap([], gpu_device=0, log_dir=tmp_path).run("mapper", {})


# ---------- AC 2: registration ratio and reprojection error ----------


def test_manifest_contains_ratio_error_and_peak_vram(scan: Path, docker: FakeDocker) -> None:
    """Given a complete reconstruction,
    When the stage finishes,
    Then manifest.json on disk has the registration ratio, the mean reprojection error, the
    model sizes, run times and the peak VRAM of the COLMAP process (not of other processes).
    """
    docker.models(10)
    sfm(scan)

    manifest = Manifest.model_validate_json((scan / "sfm" / "manifest.json").read_text())
    metrics = manifest.metrics
    assert metrics["keyframes"] == 10
    assert metrics["registered_images"] == 10
    assert metrics["registration_ratio"] == 1.0
    assert metrics["mean_reprojection_error_px"] == pytest.approx(0.75)
    assert metrics["points"] == 2
    assert metrics["model_sizes"] == [10]
    assert metrics["matcher"] == "exhaustive"
    assert metrics["colmap_image_id"] == "sha256:1"
    assert set(metrics["seconds"]) == {"features", "matching", "mapper"}
    assert manifest.peak_vram_mb == 1234
    assert manifest.warnings == [] and manifest.hints == []


@pytest.mark.parametrize(
    ("registered", "code", "warned"),
    [
        (79, ExitCode.QUALITY_INSUFFICIENT, False),
        (80, ExitCode.OK, True),
        (89, ExitCode.OK, True),
        (90, ExitCode.OK, False),
    ],
)
def test_registration_thresholds(
    tmp_path: Path, docker: FakeDocker, registered: int, code: ExitCode, warned: bool
) -> None:
    """Given 100 keyframes,
    When 79, 80, 89 or 90 of them are in the largest model,
    Then < 80 % exits 20 with hints and no sparse/0, 80 to < 90 % continues with a warning and
    hints, and >= 90 % is fine (PO decision 2026-11-25). The metrics are always recorded.
    """
    scan_dir = make_scan(tmp_path / "scan", 100)
    docker.models(registered, 100 - registered)

    result = sfm(scan_dir)

    assert result.exit_code is code
    assert result.manifest.metrics["registration_ratio"] == registered / 100
    assert result.manifest.metrics["mean_reprojection_error_px"] == pytest.approx(0.75)
    assert (scan_dir / "sfm" / "colmap" / "sparse" / "0").exists() is (code is ExitCode.OK)
    assert bool(result.manifest.warnings) is warned
    if warned or code is not ExitCode.OK:
        hints = " ".join(result.manifest.hints)
        assert "Licht" in hints and "AprilTags" in hints
        assert "2 nicht verbundene Teile" in hints
        message = str(result.manifest.error) if not warned else result.manifest.warnings[0]
        assert f"{registered} von 100" in message and f"{registered} %" in message
    else:
        assert result.manifest.hints == []


def test_largest_model_wins_and_empty_models_are_ignored(
    tmp_path: Path, docker: FakeDocker
) -> None:
    """Given several partial models, one of them without points,
    When the stage evaluates them,
    Then the largest model with points is used, whatever its number.
    """
    scan_dir = make_scan(tmp_path / "scan", 100)
    docker.config["models"] = [[3, 0], [5, 100], [92, 100]]
    docker.save()

    result = sfm(scan_dir)

    assert result.exit_code is ExitCode.OK
    assert result.manifest.metrics["model_sizes"] == [92, 5]
    assert docker.options("model_converter")["--input_path"] == "/sfm/models/2"


@pytest.mark.parametrize("models", [[], [[4, 0]]])
def test_no_model_exits_20_with_hints(
    scan: Path, docker: FakeDocker, models: list[list[int]]
) -> None:
    """Given images COLMAP cannot reconstruct (no model, or only one without points),
    When the stage runs,
    Then it exits 20 with hints and a ratio of 0.
    """
    docker.config["models"] = models
    docker.save()

    result = sfm(scan)

    assert result.exit_code is ExitCode.QUALITY_INSUFFICIENT
    assert result.manifest.metrics["registration_ratio"] == 0.0
    assert any("Licht" in hint for hint in result.manifest.hints)
    assert not (scan / "sfm" / "colmap" / "sparse").exists()


def test_hint_names_the_flight_that_is_not_connected(tmp_path: Path, docker: FakeDocker) -> None:
    """Given two flights of which only the first is in the largest model,
    When the stage fails,
    Then a hint names the second flight.
    """
    scan_dir = make_scan(tmp_path / "scan", 60, 40)
    docker.models(60, 40)

    result = sfm(scan_dir)

    assert result.exit_code is ExitCode.QUALITY_INSUFFICIENT
    flights = [hint for hint in result.manifest.hints if hint.startswith("Flug ")]
    assert len(flights) == 1 and flights[0].startswith("Flug f02: nur 0 von 40")


# ---------- AC 3: profile ----------


@pytest.mark.parametrize("profile", ["low_vram_6gb", "default_8gb"])
def test_profile_limits_reach_colmap(scan: Path, docker: FakeDocker, profile: str) -> None:
    """Given a VRAM profile,
    When the stage runs,
    Then image edge, features and match capacity are limited by the profile and the mapper
    does not use the GPU.
    """
    docker.models(10)
    settings = load_settings(profile)

    sfm(scan, settings)

    features = docker.options("feature_extractor")
    assert features["--FeatureExtraction.max_image_size"] == str(settings.sfm.max_image_edge_px)
    assert features["--SiftExtraction.max_num_features"] == str(settings.sfm.max_num_features)
    matching = docker.options("exhaustive_matcher")
    assert matching["--FeatureMatching.max_num_matches"] == str(settings.sfm.max_num_matches)
    assert docker.options("mapper")["--Mapper.ba_use_gpu"] == "0"


def test_low_vram_profile_limits_the_image_edge() -> None:
    """Given the low_vram_6gb profile,
    When it is loaded,
    Then the image edge is at most 1600 px and the retry after OOM uses smaller values.
    """
    cfg = load_settings("low_vram_6gb").sfm

    assert cfg.max_image_edge_px <= 1600
    assert cfg.oom_retry_image_edge_px < cfg.max_image_edge_px
    assert cfg.oom_retry_num_features < cfg.max_num_features


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MIN_REGISTRATION_RATIO", "0.95"),
        ("MAX_NUM_MATCHES", "100"),
        ("CAMERA_MODEL", "x y"),
        ("OOM_RETRY_NUM_FEATURES", "8192"),
    ],
)
def test_inconsistent_profile_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """Given contradicting or unsafe sfm values,
    When the settings are loaded,
    Then validation fails.
    """
    monkeypatch.setenv(f"DRONESCAN_SFM__{name}", value)

    with pytest.raises(ValidationError):
        load_settings()


# ---------- model files ----------


def test_read_report_handles_images_without_points(tmp_path: Path) -> None:
    """Given a COLMAP text model in which an image has no 2D points (empty second line),
    When it is read,
    Then all image names and the mean point error are returned.
    """
    (tmp_path / "images.txt").write_text(
        "# h\n1 1 0 0 0 0 0 0 1 f01/frames/000001.jpg\n\n"
        "2 1 0 0 0 0 0 0 1 f01/frames/000002.jpg\n5 5 -1\n"
    )
    (tmp_path / "points3D.txt").write_text("# h\n1 0 0 0 1 2 3 0.25 1 0\n2 0 0 0 1 2 3 0.75\n")

    report = read_report(tmp_path)

    assert report.image_names == ["f01/frames/000001.jpg", "f01/frames/000002.jpg"]
    assert report.points == 2
    assert report.mean_reprojection_error_px == pytest.approx(0.5)


def test_list_models_ignores_incomplete_directories(tmp_path: Path) -> None:
    """Given a model directory without binary files,
    When the models are listed,
    Then it is ignored.
    """
    (tmp_path / "0").mkdir()

    assert list_models(tmp_path) == []


# ---------- real COLMAP on the GPU (synthetic room) ----------

needs_colmap = pytest.mark.skipif(
    shutil.which("docker") is None or shutil.which("ffmpeg") is None,
    reason="docker or ffmpeg not installed",
)
SMALL = (640, 480)


def room_scan(scan_dir: Path, poses: list[View], size: tuple[int, int] = SMALL) -> Path:
    require_image("colmap")
    make_scan(scan_dir, len(poses), images=False)
    write_jpegs(scan_dir / "ingest" / "f01" / "frames", poses, size)
    return scan_dir


def wait_for[T](check: Callable[[], T], timeout_s: float) -> T:
    """Poll ``check`` until it is truthy or the time is up; returns the last value."""
    deadline = time.monotonic() + timeout_s
    while not (value := check()) and time.monotonic() < deadline:
        time.sleep(0.2)
    return value


def position_errors(scan_dir: Path, poses: list[View]) -> NDArray[np.float64]:
    """Camera position errors (m) of the stored model against the true poses."""
    lines = (scan_dir / "sfm" / "colmap" / "sparse" / "0" / "images.txt").read_text().splitlines()
    estimated, truth = [], []
    for line in [line for line in lines if not line.startswith("#")][0::2]:
        fields = line.split()
        w, x, y, z = (float(v) for v in fields[1:5])
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        estimated.append(-rotation.T @ np.array([float(v) for v in fields[5:8]]))
        truth.append(poses[int(Path(fields[9]).stem) - 1].centre)
    return aligned_position_errors(np.array(estimated), np.array(truth))


@pytest.mark.gpu
@needs_colmap
def test_synthetic_room_is_reconstructed_with_correct_poses(tmp_path: Path) -> None:
    """Given 30 views of a synthetic room,
    When the stage runs with the low_vram_6gb profile on the GTX 1060,
    Then all views are registered, the poses match the ground truth within 1 cm (after a
    similarity transform) and the COLMAP process stays within the VRAM budget.
    """
    poses = views(30)
    scan_dir = room_scan(tmp_path / "scan", poses)

    result = sfm(scan_dir, load_settings("low_vram_6gb"))

    assert result.exit_code is ExitCode.OK, result.manifest.error
    metrics = result.manifest.metrics
    assert metrics["registration_ratio"] == 1.0
    assert 0 < metrics["mean_reprojection_error_px"] < 1.0
    assert position_errors(scan_dir, poses).max() < 0.01
    assert result.manifest.peak_vram_mb is not None
    assert 0 < result.manifest.peak_vram_mb <= MAX_PROCESS_MIB


@pytest.mark.gpu
@needs_colmap
def test_sequential_matching_closes_the_loop(tmp_path: Path) -> None:
    """Given a camera that returns to its start and more keyframes than the limit,
    When the stage matches sequentially with loop detection,
    Then the first view is verified against the last ones, which are far outside the overlap
    window, and the poses are correct.
    """
    poses = views(40)
    scan_dir = room_scan(tmp_path / "scan", poses)
    settings = with_sfm(exhaustive_max_keyframes=10, sequential_overlap=4, loop_vocab_words=512)

    result = sfm(scan_dir, settings)

    assert result.exit_code is ExitCode.OK, result.manifest.error
    assert result.manifest.metrics["matcher"] == "sequential+loop"
    database = scan_dir / "sfm" / "colmap" / "database.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        names = dict(connection.execute("SELECT image_id, name FROM images"))
        rows = connection.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0")
        pairs = [divmod(pair, 2**31 - 1) for (pair,) in rows]
    numbers = [sorted(int(Path(names[i]).stem) for i in pair) for pair in pairs]
    assert [pair for pair in numbers if pair[0] == 1 and pair[1] >= 36]
    assert position_errors(scan_dir, poses).max() < 0.02


@pytest.mark.gpu
@needs_colmap
def test_full_profile_resolution_stays_within_the_vram_budget(tmp_path: Path) -> None:
    """Given feature-rich views at the full image edge of low_vram_6gb (1600 px),
    When the stage runs,
    Then the peak of the COLMAP process including the CUDA context is <= 5.5 GiB (Q-2).
    """
    scan_dir = room_scan(tmp_path / "scan", views(120)[:8], size=(1600, 1200))

    result = sfm(scan_dir, load_settings("low_vram_6gb"))

    assert result.exit_code is ExitCode.OK, result.manifest.error
    assert result.manifest.warnings == []  # no retry with smaller values
    assert result.manifest.peak_vram_mb is not None
    assert 0 < result.manifest.peak_vram_mb <= MAX_PROCESS_MIB


@pytest.mark.gpu
@needs_colmap
def test_unrelated_images_exit_20(tmp_path: Path) -> None:
    """Given images that show nothing in common,
    When the stage runs with the real COLMAP,
    Then it exits 20 with hints instead of failing.
    """
    require_image("colmap")
    scan_dir = make_scan(tmp_path / "scan", 4, images=False)
    frames = scan_dir / "ingest" / "f01" / "frames"
    for number in range(1, 5):
        one = tmp_path / str(number)
        write_jpegs(one, views(4)[number - 1 : number], SMALL, seed=number)
        (one / "000001.jpg").rename(frames / f"{number:06d}.jpg")

    result = sfm(scan_dir)

    assert result.exit_code is ExitCode.QUALITY_INSUFFICIENT, result.manifest.error
    assert result.manifest.metrics["registration_ratio"] == 0.0
    assert result.manifest.hints


@pytest.mark.gpu
@needs_colmap
def test_killed_runner_does_not_leave_a_container(tmp_path: Path) -> None:
    """Given a running COLMAP container,
    When the runner is killed hard (SIGKILL, no chance to clean up),
    Then the container ends by itself, because the GPU lock is already free again.
    """
    scan_dir = room_scan(tmp_path / "scan", views(30))
    code = "from dronescan.cli import app; app()"

    def containers() -> set[str]:
        command = ["docker", "ps", "--filter", "name=dronescan-colmap-", "--format", "{{.Names}}"]
        output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
        return set(output.split())

    before = containers()
    with subprocess.Popen([sys.executable, "-c", code, "sfm", "--scan", str(scan_dir)]) as runner_:
        started = wait_for(lambda: containers() - before, timeout_s=60)
        runner_.kill()
    assert started
    assert wait_for(lambda: not containers() & started, timeout_s=15)
