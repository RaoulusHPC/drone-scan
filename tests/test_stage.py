# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage framework: manifest, caching and exit codes (KAN-21)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

from dronescan.common.config import Settings, load_settings
from dronescan.common.stage import (
    ExitCode,
    InputMissingError,
    Manifest,
    QualityError,
    Stage,
    StageContext,
    hash_path,
    register_stage,
    run_stage,
)

GIB = 1024**3


class EchoStage(Stage):
    """Copies flights/f1/video.mp4 to <scan>/echo/out.txt and counts runs."""

    name: ClassVar[str] = "echo"
    version: ClassVar[str] = "1"

    def __init__(self, behaviour: str = "ok") -> None:
        self.behaviour = behaviour
        self.runs = 0

    def inputs(self, scan_dir: Path) -> list[Path]:
        return [scan_dir / "flights" / "f1" / "video.mp4"]

    def params(self, settings: Settings) -> dict[str, Any]:
        return {"max_image_edge_px": settings.sfm.max_image_edge_px}

    def tool_versions(self) -> dict[str, str]:
        return {"echo-tool": "0.1"}

    def run(self, ctx: StageContext) -> None:
        self.runs += 1
        if self.behaviour == "quality":
            raise QualityError(
                "Zu wenige scharfe Bilder.", hints=["Langsamer bewegen (≤ 0,5 m/s)."]
            )
        if self.behaviour == "input":
            raise InputMissingError("Telemetrie fehlt.", hints=["Flug erneut importieren."])
        if self.behaviour == "boom":
            raise ValueError("unexpected")
        if self.behaviour == "oom_text":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if self.behaviour == "ram":
            raise MemoryError()
        if self.behaviour == "interrupt":
            raise KeyboardInterrupt
        if self.behaviour == "path_error":
            raise FileNotFoundError(f"missing {ctx.scan_dir.resolve()}/secret/name.txt")
        ctx.warn("Belichtung schwankt.")
        ctx.hint("Licht einschalten.")
        (ctx.out_dir / "out.txt").write_bytes(ctx.inputs[0].read_bytes())


@pytest.fixture
def scan(tmp_path: Path) -> Path:
    video = tmp_path / "flights" / "f1" / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"frame-data")
    return tmp_path


@pytest.fixture
def settings() -> Settings:
    return load_settings()


def read_manifest(scan_dir: Path, stage: str = "echo") -> Manifest:
    return Manifest.model_validate_json((scan_dir / stage / "manifest.json").read_text())


def test_successful_run_writes_complete_manifest(scan: Path, settings: Settings) -> None:
    """Given a scan with all inputs,
    When a stage runs,
    Then it exits 0 and writes manifest.json with hashes, params, tools, times,
    peak_vram_mb, warnings and hints.
    """
    result = run_stage(EchoStage(), scan, settings)

    assert result.exit_code is ExitCode.OK
    assert not result.skipped
    manifest = read_manifest(scan)
    video_hash = hashlib.sha256(b"frame-data").hexdigest()
    assert manifest.stage == "echo"
    assert manifest.version == "1"
    assert manifest.inputs == {"flights/f1/video.mp4": video_hash}
    assert manifest.params == {"max_image_edge_px": settings.sfm.max_image_edge_px}
    assert manifest.profile == settings.profile
    assert manifest.tools["echo-tool"] == "0.1"
    assert "python" in manifest.tools and "drone-scan" in manifest.tools
    assert manifest.started <= manifest.finished
    assert manifest.peak_vram_mb is None  # CPU stage
    assert manifest.status == "ok"
    assert manifest.exit_code == 0
    assert manifest.warnings == ["Belichtung schwankt."]
    assert manifest.hints == ["Licht einschalten."]
    assert (scan / "echo" / "out.txt").read_bytes() == b"frame-data"


def test_unchanged_inputs_and_params_skip_the_stage(scan: Path, settings: Settings) -> None:
    """Given a valid manifest from a previous run,
    When the stage runs again with the same inputs and parameters,
    Then it is skipped (cache) and exits 0.
    """
    stage = EchoStage()
    run_stage(stage, scan, settings)

    result = run_stage(stage, scan, settings)

    assert result.skipped
    assert result.exit_code is ExitCode.OK
    assert stage.runs == 1


def test_force_reruns_the_stage(scan: Path, settings: Settings) -> None:
    """Given a valid manifest,
    When the stage runs with force=True (--force),
    Then it runs again.
    """
    stage = EchoStage()
    run_stage(stage, scan, settings)

    result = run_stage(stage, scan, settings, force=True)

    assert not result.skipped
    assert stage.runs == 2


def test_changed_input_invalidates_cache(scan: Path, settings: Settings) -> None:
    """Given a valid manifest,
    When an input file changes,
    Then the stage runs again.
    """
    stage = EchoStage()
    run_stage(stage, scan, settings)
    (scan / "flights" / "f1" / "video.mp4").write_bytes(b"new-frames")

    assert not run_stage(stage, scan, settings).skipped
    assert stage.runs == 2


def test_changed_params_invalidate_cache(scan: Path, settings: Settings) -> None:
    """Given a valid manifest,
    When a stage parameter changes (other profile values),
    Then the stage runs again.
    """
    stage = EchoStage()
    run_stage(stage, scan, settings)
    other = settings.model_copy(
        update={"sfm": settings.sfm.model_copy(update={"max_image_edge_px": 1})}
    )

    assert not run_stage(stage, scan, other).skipped


def test_changed_stage_version_invalidates_cache(scan: Path, settings: Settings) -> None:
    """Given a valid manifest written by stage version 1,
    When version 2 of the stage runs,
    Then the cache is not used.
    """
    run_stage(EchoStage(), scan, settings)

    class EchoV2(EchoStage):
        version: ClassVar[str] = "2"

    stage = EchoV2()
    assert not run_stage(stage, scan, settings).skipped


def test_corrupt_manifest_is_ignored(scan: Path, settings: Settings) -> None:
    """Given a manifest.json that cannot be parsed,
    When the stage runs,
    Then it reruns instead of crashing.
    """
    stage = EchoStage()
    run_stage(stage, scan, settings)
    (scan / "echo" / "manifest.json").write_text("{not json", encoding="utf-8")

    assert not run_stage(stage, scan, settings).skipped


def test_missing_input_exits_10_with_german_hint(tmp_path: Path, settings: Settings) -> None:
    """Given a scan without the required video,
    When the stage runs,
    Then it exits 10, does not call run() and the manifest names the missing input.
    """
    stage = EchoStage()

    result = run_stage(stage, tmp_path, settings)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert stage.runs == 0
    manifest = read_manifest(tmp_path)
    assert manifest.status == "input_missing"
    assert "flights/f1/video.mp4" in (manifest.error or "")
    assert manifest.hints


def test_input_missing_error_raised_by_stage_exits_10(scan: Path, settings: Settings) -> None:
    """Given a stage that detects a missing input itself,
    When it raises InputMissingError,
    Then the exit code is 10.
    """
    assert run_stage(EchoStage("input"), scan, settings).exit_code is ExitCode.INPUT_MISSING


def test_quality_error_exits_20_with_hints_and_is_not_cached(
    scan: Path, settings: Settings
) -> None:
    """Given input of insufficient quality,
    When the stage raises QualityError,
    Then it exits 20, the hints land in the manifest and the next run is not skipped.
    """
    stage = EchoStage("quality")

    result = run_stage(stage, scan, settings)

    assert result.exit_code is ExitCode.QUALITY_INSUFFICIENT
    manifest = read_manifest(scan)
    assert manifest.status == "quality_insufficient"
    assert manifest.hints == ["Langsamer bewegen (≤ 0,5 m/s)."]
    assert not run_stage(stage, scan, settings).skipped


def test_quality_error_requires_hints() -> None:
    """Given exit code 20 must always come with user hints,
    When QualityError is created without hints,
    Then it is rejected.
    """
    with pytest.raises(ValueError):
        QualityError("schlecht", hints=[])


def test_unexpected_exception_exits_1(scan: Path, settings: Settings) -> None:
    """Given a stage that crashes with an unexpected exception,
    When it runs,
    Then it exits 1 and the manifest records the error.
    """
    result = run_stage(EchoStage("boom"), scan, settings)

    assert result.exit_code is ExitCode.ERROR
    manifest = read_manifest(scan)
    assert manifest.status == "error"
    assert "ValueError" in (manifest.error or "")


def test_cuda_oom_runtime_error_exits_30(scan: Path, settings: Settings) -> None:
    """Given a CUDA extension that raises RuntimeError('CUDA out of memory ...'),
    When the stage runs,
    Then the OOM is recognised and the stage exits 30 with a German remedy.
    """
    result = run_stage(EchoStage("oom_text"), scan, settings)

    assert result.exit_code is ExitCode.OOM
    manifest = read_manifest(scan)
    assert manifest.status == "oom"
    assert any("Grafikspeicher" in hint for hint in manifest.hints)


@dataclass
class FakeCuda:
    peak_bytes: int = 3 * GIB
    calls: list[str] = field(default_factory=list)

    class OutOfMemoryError(RuntimeError):
        pass

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, device: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(total_memory=8 * GIB)

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None:
        self.calls.append("limit")

    def reset_peak_memory_stats(self, device: int | None = None) -> None:
        self.calls.append("reset")

    def max_memory_reserved(self, device: int | None = None) -> int:
        return self.peak_bytes


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
    cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return cuda


class GpuStage(EchoStage):
    name: ClassVar[str] = "gpu_echo"
    uses_gpu: ClassVar[bool] = True

    def __init__(self, cuda: FakeCuda, oom: bool = False) -> None:
        super().__init__()
        self.cuda = cuda
        self.oom = oom

    def run(self, ctx: StageContext) -> None:
        self.cuda.calls.append("run")
        if self.oom:
            raise FakeCuda.OutOfMemoryError("CUDA out of memory")
        super().run(ctx)


def test_gpu_stage_applies_vram_limit_and_records_peak(
    scan: Path, settings: Settings, fake_cuda: FakeCuda
) -> None:
    """Given a GPU stage,
    When it runs,
    Then the VRAM limit is applied before run() and peak_vram_mb is recorded.
    """
    result = run_stage(GpuStage(fake_cuda), scan, settings)

    assert result.exit_code is ExitCode.OK
    assert fake_cuda.calls[:3] == ["limit", "reset", "run"]
    assert read_manifest(scan, "gpu_echo").peak_vram_mb == pytest.approx(3 * 1024)


def test_torch_oom_exception_exits_30(scan: Path, settings: Settings, fake_cuda: FakeCuda) -> None:
    """Given a GPU stage that raises torch.cuda.OutOfMemoryError,
    When it runs,
    Then it exits 30 (OOM).
    """
    result = run_stage(GpuStage(fake_cuda, oom=True), scan, settings)

    assert result.exit_code is ExitCode.OOM


def test_gpu_stage_without_cuda_exits_1(
    scan: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a machine without PyTorch,
    When a GPU stage runs,
    Then it exits 1 with a remedy and run() is not called.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    stage = GpuStage(FakeCuda())

    result = run_stage(stage, scan, settings)

    assert result.exit_code is ExitCode.ERROR
    assert stage.runs == 0
    assert any("PyTorch" in hint for hint in read_manifest(scan, "gpu_echo").hints)


def test_directory_hash_tracks_content(tmp_path: Path) -> None:
    """Given a directory input,
    When a file inside changes, is added or renamed,
    Then its hash changes; identical content gives an identical hash.
    """
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "0001.jpg").write_bytes(b"a")
    first = hash_path(frames)

    assert hash_path(frames) == first
    (frames / "0001.jpg").write_bytes(b"b")
    assert hash_path(frames) != first
    (frames / "0001.jpg").write_bytes(b"a")
    (frames / "0002.jpg").write_bytes(b"c")
    assert hash_path(frames) != first
    two_files = hash_path(frames)
    (frames / "0002.jpg").rename(frames / "0003.jpg")
    assert hash_path(frames) != two_files


def test_manifest_json_is_valid_json_on_disk(scan: Path, settings: Settings) -> None:
    """Given a finished run,
    When manifest.json is read by other tools (server, UI),
    Then it is plain JSON without leftover temporary files.
    """
    run_stage(EchoStage(), scan, settings)

    data = json.loads((scan / "echo" / "manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert sorted(p.name for p in (scan / "echo").iterdir()) == ["manifest.json", "out.txt"]


@pytest.mark.parametrize(
    ("behaviour", "code"),
    [("ok", 0), ("input", 10), ("quality", 20), ("oom_text", 30), ("boom", 1)],
)
def test_cli_command_returns_contract_exit_codes(scan: Path, behaviour: str, code: int) -> None:
    """Given a stage registered as CLI command,
    When `drone-scan echo --scan <dir>` runs,
    Then the process exit code follows the stage contract (0/10/20/30/1).
    """
    app = typer.Typer()
    register_stage(app, EchoStage, lambda: EchoStage(behaviour))

    @app.callback()
    def main() -> None:
        """Test app."""

    result = CliRunner().invoke(app, ["echo", "--scan", str(scan), "--profile", "low_vram_6gb"])

    assert result.exit_code == code


def test_cli_force_flag_reruns(scan: Path) -> None:
    """Given a cached stage,
    When `drone-scan echo --scan <dir> --force` runs,
    Then the stage is executed again.
    """
    stage = EchoStage()
    app = typer.Typer()
    register_stage(app, EchoStage, lambda: stage)

    @app.callback()
    def main() -> None:
        """Test app."""

    runner = CliRunner()
    runner.invoke(app, ["echo", "--scan", str(scan)])
    runner.invoke(app, ["echo", "--scan", str(scan)])
    runner.invoke(app, ["echo", "--scan", str(scan), "--force"])

    assert stage.runs == 2


def test_host_memory_error_exits_30_with_ram_hint(scan: Path, settings: Settings) -> None:
    """Given a stage that runs out of host memory (MemoryError),
    When it runs,
    Then it exits 30 with a hint about RAM, not GPU memory.
    """
    result = run_stage(EchoStage("ram"), scan, settings)

    assert result.exit_code is ExitCode.OOM
    assert any("Arbeitsspeicher" in hint for hint in read_manifest(scan).hints)


def test_unexpected_error_has_german_remedy_and_no_absolute_paths(
    scan: Path, settings: Settings
) -> None:
    """Given an exception whose message contains the absolute scan path,
    When the stage fails with exit 1,
    Then the manifest has a German remedy and the absolute path is redacted.
    """
    result = run_stage(EchoStage("path_error"), scan, settings)

    assert result.exit_code is ExitCode.ERROR
    manifest = read_manifest(scan)
    assert any("Abhilfe" in hint for hint in manifest.hints)
    assert str(scan.resolve()) not in (manifest.error or "")
    assert "<scan>/secret/name.txt" in (manifest.error or "")


def test_interrupt_records_status_and_reraises(scan: Path, settings: Settings) -> None:
    """Given a run interrupted with Ctrl+C,
    When the KeyboardInterrupt propagates,
    Then the manifest says 'error' (no valid cache) and the interrupt is re-raised.
    """
    stage = EchoStage("interrupt")

    with pytest.raises(KeyboardInterrupt):
        run_stage(stage, scan, settings)

    assert read_manifest(scan).status == "error"
    stage.behaviour = "ok"
    assert not run_stage(stage, scan, settings).skipped


def test_missing_scan_dir_exits_10_without_creating_directories(
    tmp_path: Path, settings: Settings
) -> None:
    """Given a mistyped --scan path,
    When the stage runs,
    Then it exits 10 and creates nothing on disk.
    """
    missing = tmp_path / "typo"

    result = run_stage(EchoStage(), missing, settings)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert not missing.exists()
    assert result.manifest.hints


def test_input_outside_scan_dir_is_rejected(tmp_path: Path, settings: Settings) -> None:
    """Given a stage whose input lies outside the scan directory,
    When it runs,
    Then it exits 1 and the manifest names only the file name, not its absolute path.
    """
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(b"x")

    class Outside(EchoStage):
        def inputs(self, scan_dir: Path) -> list[Path]:
            return [outside]

    result = run_stage(Outside(), scan_dir, settings)

    assert result.exit_code is ExitCode.ERROR
    manifest = read_manifest(scan_dir)
    assert "elsewhere.mp4" in (manifest.error or "")
    assert str(tmp_path) not in manifest.model_dump_json()


def test_tool_version_change_invalidates_cache(scan: Path, settings: Settings) -> None:
    """Given a valid manifest,
    When an external tool version changes (e.g. COLMAP upgrade),
    Then the stage runs again.
    """
    run_stage(EchoStage(), scan, settings)

    class Upgraded(EchoStage):
        def tool_versions(self) -> dict[str, str]:
            return {"echo-tool": "0.2"}

    assert not run_stage(Upgraded(), scan, settings).skipped


def test_non_json_params_still_hit_the_cache(scan: Path, settings: Settings) -> None:
    """Given params with tuples and paths,
    When the stage runs twice,
    Then the second run is a cache hit (params are normalised like the stored manifest).
    """

    class TupleParams(EchoStage):
        def params(self, settings: Settings) -> dict[str, Any]:
            return {"size": (640, 480), "model": Path("weights/aliked.pt")}

    stage = TupleParams()
    run_stage(stage, scan, settings)

    assert run_stage(stage, scan, settings).skipped
    assert read_manifest(scan).params == {"model": "weights/aliked.pt", "size": [640, 480]}


def test_cli_prints_german_status_and_hints(scan: Path) -> None:
    """Given a stage registered as CLI command,
    When it succeeds, is cached and fails,
    Then the output says 'fertig', 'übersprungen' and shows 'Hinweis:' lines.
    """
    app = typer.Typer()
    stage = EchoStage()
    register_stage(app, EchoStage, lambda: stage)

    @app.callback()
    def main() -> None:
        """Test app."""

    runner = CliRunner()
    first = runner.invoke(app, ["echo", "--scan", str(scan)])
    second = runner.invoke(app, ["echo", "--scan", str(scan)])
    stage.behaviour = "quality"
    failed = runner.invoke(app, ["echo", "--scan", str(scan), "--force"])

    assert "echo: fertig." in first.output
    assert "Hinweis: Licht einschalten." in first.output
    assert "übersprungen" in second.output
    assert failed.exit_code == 20
    assert "Hinweis: Langsamer bewegen" in failed.output
