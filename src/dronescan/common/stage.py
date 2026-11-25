# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage framework: every pipeline stage follows the same contract (architecture §5.2).

``drone-scan <stage> --scan <dir> [--profile] [--force]`` reads defined inputs,
writes ``<scan>/<stage>/`` plus ``manifest.json`` and exits with
0 ok · 10 input missing · 20 quality insufficient · 30 OOM · 1 other error.
A stage is skipped when its manifest is valid: status ok and the same stage
version, input hashes, parameters and tool versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import structlog
import typer
from pydantic import BaseModel, ConfigDict, ValidationError

from dronescan import __version__
from dronescan.common import gpu
from dronescan.common.config import ProfileError, Settings, describe_validation_error, load_settings

MANIFEST_NAME = "manifest.json"
_HASH_CHUNK = 1024 * 1024
_MANIFEST_MODE = 0o644  # readable by server/worker containers running as another uid

log = structlog.get_logger(__name__)

Status = Literal["ok", "input_missing", "quality_insufficient", "oom", "error"]


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1
    INPUT_MISSING = 10
    QUALITY_INSUFFICIENT = 20
    OOM = 30


_STATUS: dict[ExitCode, Status] = {
    ExitCode.OK: "ok",
    ExitCode.ERROR: "error",
    ExitCode.INPUT_MISSING: "input_missing",
    ExitCode.QUALITY_INSUFFICIENT: "quality_insufficient",
    ExitCode.OOM: "oom",
}

GPU_OOM_HINT = (
    "Der Grafikspeicher hat nicht gereicht. Abhilfe: Profil 'low_vram_6gb' verwenden, andere "
    "GPU-Programme schließen oder ein kürzeres Video bzw. weniger Räume pro Scan verarbeiten."
)
RAM_OOM_HINT = (
    "Der Arbeitsspeicher hat nicht gereicht. Abhilfe: andere Programme schließen oder ein "
    "kürzeres Video bzw. weniger Räume pro Scan verarbeiten."
)
ERROR_HINT = (
    "Unerwarteter Fehler. Abhilfe: mit --force erneut ausführen; tritt er wieder auf, "
    "Log-Ausgabe und manifest.json an das Entwicklungsteam geben."
)
INTERRUPTED = "Abgebrochen (Strg+C bzw. Prozess beendet)."
INTERRUPTED_HINT = "Abhilfe: Stufe erneut starten; bereits erzeugte Dateien werden neu berechnet."
INPUT_MISSING_HINT = (
    "Abhilfe: vorherige Stufe bzw. den Import für diesen Scan ausführen und den Pfad mit "
    "--scan prüfen."
)


class StageError(Exception):
    """Expected stage failure with a user-facing (German) message and hints."""

    exit_code: ClassVar[ExitCode] = ExitCode.ERROR

    def __init__(self, message: str, hints: list[str] | None = None) -> None:
        super().__init__(message)
        self.hints = list(hints or [])


class InputMissingError(StageError):
    exit_code: ClassVar[ExitCode] = ExitCode.INPUT_MISSING


class QualityError(StageError):
    """Input quality is insufficient; always carries concrete hints (exit 20)."""

    exit_code: ClassVar[ExitCode] = ExitCode.QUALITY_INSUFFICIENT

    def __init__(self, message: str, hints: list[str]) -> None:
        if not hints:
            raise ValueError("QualityError needs at least one hint for the user")
        super().__init__(message, hints)


class OutOfMemoryError(StageError):
    """An external GPU tool ran out of VRAM (exit 30)."""

    exit_code: ClassVar[ExitCode] = ExitCode.OOM


class Manifest(BaseModel):
    """Content of ``<scan>/<stage>/manifest.json``."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    version: str
    profile: str
    inputs: dict[str, str]
    params: dict[str, Any]
    tools: dict[str, str]
    started: datetime
    finished: datetime
    peak_vram_mb: float | None
    status: Status
    exit_code: int
    error: str | None = None
    warnings: list[str] = []
    hints: list[str] = []
    metrics: dict[str, Any] = {}  # measured results of the stage (numbers, no image content)


@dataclass
class StageContext:
    """What a stage may use while running; collects warnings and hints."""

    scan_dir: Path
    out_dir: Path
    settings: Settings
    inputs: list[Path]
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    peak_vram_mb: float | None = None  # only set by GPU stages without PyTorch

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def hint(self, message: str) -> None:
        self.hints.append(message)

    def record_peak_vram(self, mib: float | None) -> None:
        """Keep the highest peak of the external GPU processes of this stage."""
        if mib is not None:
            self.peak_vram_mb = max(mib, self.peak_vram_mb or 0.0)


class Stage(ABC):
    """Base class for pipeline stages. Subclasses set ``name`` and ``version``.

    Constructors must be cheap (no heavy imports); work belongs in `run`.
    """

    name: ClassVar[str]
    version: ClassVar[str]
    uses_gpu: ClassVar[bool] = False
    # False for GPU stages that run external tools (COLMAP): no allocator limit, and the stage
    # reports its own peak via `StageContext.record_peak_vram`.
    uses_torch: ClassVar[bool] = True

    @abstractmethod
    def inputs(self, scan_dir: Path) -> list[Path]:
        """Input files/directories read by this stage; must lie inside ``scan_dir``."""

    def params(self, settings: Settings) -> dict[str, Any]:
        """Parameters that influence the output (part of the cache key, JSON-serialisable)."""
        return {}

    def tool_versions(self) -> dict[str, str]:
        """Versions of external tools (COLMAP, gsplat, ...); a change invalidates the cache."""
        return {}

    @abstractmethod
    def run(self, ctx: StageContext) -> None:
        """Produce all outputs in ``ctx.out_dir``; raise StageError subclasses on failure.

        The framework does not clear ``out_dir`` (stages may keep per-flight
        subfolders), so a stage must overwrite or remove its own old outputs.
        """


@dataclass(frozen=True)
class StageResult:
    exit_code: ExitCode
    skipped: bool
    manifest: Manifest


def hash_path(path: Path) -> str:
    """SHA-256 of a file, or of a directory tree (relative paths + file contents)."""
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(hash_path(child).encode())
        return digest.hexdigest()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def is_oom(exc: BaseException) -> bool:
    """True for GPU/host out-of-memory errors (torch, CUDA extensions, MemoryError)."""
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    """Normalise params the way they look after a manifest round trip (tuples → lists, ...)."""
    data: dict[str, Any] = json.loads(json.dumps(params, default=str, sort_keys=True))
    return data


def _tools(stage: Stage) -> dict[str, str]:
    return {"python": platform.python_version(), "drone-scan": __version__, **stage.tool_versions()}


def _redact(text: str, scan_dir: Path) -> str:
    """Remove absolute paths that may contain personal names (CLAUDE.md)."""
    for prefix in {str(scan_dir.resolve()), str(scan_dir), str(Path.home())}:
        if prefix and prefix != ".":
            text = text.replace(prefix, "<scan>" if prefix != str(Path.home()) else "~")
    return text


def load_manifest(path: Path) -> Manifest | None:
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


def _write_manifest(manifest: Manifest, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(manifest.model_dump_json(indent=2))
        os.chmod(tmp, _MANIFEST_MODE)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class _OutsideScanError(Exception):
    pass


@dataclass
class _Run:
    """Mutable state of one stage run, turned into the manifest at the end."""

    stage: Stage
    scan_dir: Path
    settings: Settings
    started: datetime = field(default_factory=lambda: datetime.now(UTC))
    inputs: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    peak: float | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def out_dir(self) -> Path:
        return self.scan_dir / self.stage.name

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.scan_dir.resolve()).as_posix()
        except ValueError as exc:
            raise _OutsideScanError(path.name) from exc

    def manifest(self, code: ExitCode) -> Manifest:
        return Manifest(
            stage=self.stage.name,
            version=self.stage.version,
            profile=self.settings.profile,
            inputs=self.inputs,
            params=self.params,
            tools=_tools(self.stage),
            started=self.started,
            finished=datetime.now(UTC),
            peak_vram_mb=self.peak,
            status=_STATUS[code],
            exit_code=int(code),
            error=None if self.error is None else _redact(self.error, self.scan_dir),
            warnings=self.warnings,
            hints=self.hints,
            metrics=self.metrics,
        )

    def finish(self, code: ExitCode) -> StageResult:
        manifest = self.manifest(code)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(manifest, self.out_dir / MANIFEST_NAME)
        event = "stage_finished" if code is ExitCode.OK else "stage_failed"
        log.info(event, stage=self.stage.name, exit_code=int(code), peak_vram_mb=self.peak)
        return StageResult(exit_code=code, skipped=False, manifest=manifest)


def _is_cache_hit(previous: Manifest | None, run: _Run) -> bool:
    return (
        previous is not None
        and previous.status == "ok"
        and previous.version == run.stage.version
        and previous.inputs == run.inputs
        and previous.params == run.params
        and previous.tools == _tools(run.stage)
    )


def run_stage(stage: Stage, scan_dir: Path, settings: Settings, force: bool = False) -> StageResult:
    """Run ``stage`` for ``scan_dir`` following the stage contract."""
    run = _Run(stage=stage, scan_dir=scan_dir, settings=settings)
    if not scan_dir.is_dir():
        # Never create directories for a mistyped --scan path.
        run.error = f"Scan-Verzeichnis '{scan_dir.name}' existiert nicht."
        run.hints.append("Abhilfe: Pfad mit --scan prüfen bzw. den Scan zuerst anlegen.")
        log.info("stage_failed", stage=stage.name, exit_code=int(ExitCode.INPUT_MISSING))
        return StageResult(
            ExitCode.INPUT_MISSING, skipped=False, manifest=run.manifest(ExitCode.INPUT_MISSING)
        )

    gpu_started = False
    code = ExitCode.OK
    ctx: StageContext | None = None
    try:
        input_paths = stage.inputs(scan_dir)
        run.params = _jsonable(stage.params(settings))
        missing = [p for p in input_paths if not p.exists()]
        if missing:
            names = ", ".join(run.relative(p) for p in missing)
            raise InputMissingError(f"Eingabe fehlt: {names}", [INPUT_MISSING_HINT])
        run.inputs = {run.relative(p): hash_path(p) for p in input_paths}

        manifest_path = run.out_dir / MANIFEST_NAME
        if not force and _is_cache_hit(load_manifest(manifest_path), run):
            log.info("stage_skipped", stage=stage.name)
            cached = load_manifest(manifest_path)
            assert cached is not None
            return StageResult(exit_code=ExitCode.OK, skipped=True, manifest=cached)

        run.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.unlink(missing_ok=True)  # a crash must not leave a valid cache behind
        log.info("stage_started", stage=stage.name, force=force)
        if stage.uses_gpu and stage.uses_torch:
            gpu.apply_vram_limit(settings)  # before the first allocation (KAN-20)
            gpu.reset_peak_memory()
            gpu_started = True
        ctx = StageContext(
            scan_dir=scan_dir, out_dir=run.out_dir, settings=settings, inputs=input_paths
        )
        stage.run(ctx)
    except KeyboardInterrupt:
        run.error = INTERRUPTED
        run.hints.append(INTERRUPTED_HINT)
        _collect(run, ctx, gpu_started)
        run.finish(ExitCode.ERROR)
        raise
    except StageError as exc:
        code, run.error = exc.exit_code, str(exc)
        run.hints.extend(exc.hints)
    except _OutsideScanError as exc:
        code = ExitCode.ERROR
        run.error = f"Eingabe '{exc}' liegt außerhalb des Scan-Verzeichnisses."
        run.hints.append("Abhilfe: Dateien in das Scan-Verzeichnis importieren (Stufe 'import').")
    except gpu.GpuUnavailableError as exc:
        code, run.error = ExitCode.ERROR, exc.cause
        run.hints.append(f"Abhilfe: {exc.remedy}")
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
        if is_oom(exc):
            code = ExitCode.OOM
            run.hints.append(RAM_OOM_HINT if isinstance(exc, MemoryError) else GPU_OOM_HINT)
        else:
            code = ExitCode.ERROR
            run.hints.append(ERROR_HINT)
    _collect(run, ctx, gpu_started)
    return run.finish(code)


def _collect(run: _Run, ctx: StageContext | None, gpu_started: bool) -> None:
    """Copy warnings/hints from the stage context and read the VRAM peak."""
    if ctx is not None:
        run.warnings = ctx.warnings + run.warnings
        run.hints = ctx.hints + run.hints
        run.metrics = ctx.metrics
        run.peak = ctx.peak_vram_mb
    if gpu_started:
        try:
            run.peak = gpu.peak_reserved_mib()
        except Exception:  # the measurement must never hide the stage result
            run.peak = None


def register_stage(
    app: typer.Typer, stage_cls: type[Stage], factory: Callable[[], Stage] | None = None
) -> None:
    """Add ``drone-scan <stage_cls.name> --scan --profile --force`` to ``app``.

    The stage is only instantiated when the command runs (``factory`` defaults to
    ``stage_cls``), so ``drone-scan --help`` stays fast.
    """
    make = factory or stage_cls
    name = stage_cls.name

    @app.command(name)
    def command(
        scan: Annotated[
            Path, typer.Option("--scan", help="Scan-Verzeichnis, z. B. workspace/<id>.")
        ],
        profile: Annotated[
            str | None, typer.Option("--profile", help="VRAM-Profil, z. B. low_vram_6gb.")
        ] = None,
        force: Annotated[
            bool, typer.Option("--force", help="Cache ignorieren und neu berechnen.")
        ] = False,
    ) -> None:
        try:
            settings = load_settings(profile)
        except ProfileError as exc:
            raise typer.BadParameter(str(exc), param_hint="--profile / DRONESCAN_PROFILE") from exc
        except ValidationError as exc:
            typer.echo(describe_validation_error(exc, profile), err=True)
            raise typer.Exit(code=int(ExitCode.ERROR)) from exc
        result = run_stage(make(), scan, settings, force=force)
        manifest = result.manifest
        failed = result.exit_code is not ExitCode.OK
        if result.skipped:
            typer.echo(f"{name}: unverändert, übersprungen (Cache). Neu rechnen mit --force.")
        elif failed:
            typer.echo(f"{name}: {manifest.error}", err=True)
        else:
            typer.echo(f"{name}: fertig.")
        for hint in manifest.hints:
            typer.echo(f"Hinweis: {hint}", err=failed)
        raise typer.Exit(code=int(result.exit_code))
