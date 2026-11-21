"""Stage framework: every pipeline stage follows the same contract (architecture §5.2).

``drone-scan <stage> --scan <dir> [--profile] [--force]`` reads defined inputs,
writes ``<scan>/<stage>/`` plus ``manifest.json`` and exits with
0 ok · 10 input missing · 20 quality insufficient · 30 OOM · 1 other error.
A stage is skipped when its manifest is valid (status ok, same stage version,
input hashes and parameters).
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
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
MIB = 1024**2
_HASH_CHUNK = 1024 * 1024

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

OOM_HINTS = [
    "Der Grafikspeicher hat nicht gereicht. Abhilfe: Profil 'low_vram_6gb' verwenden, "
    "andere GPU-Programme schließen oder ein kürzeres Video bzw. weniger Räume pro Scan "
    "verarbeiten."
]


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


@dataclass
class StageContext:
    """What a stage may use while running; collects warnings and hints."""

    scan_dir: Path
    out_dir: Path
    settings: Settings
    inputs: list[Path]
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def hint(self, message: str) -> None:
        self.hints.append(message)


class Stage(ABC):
    """Base class for pipeline stages. Subclasses set ``name`` and ``version``."""

    name: ClassVar[str]
    version: ClassVar[str]
    uses_gpu: ClassVar[bool] = False

    @abstractmethod
    def inputs(self, scan_dir: Path) -> list[Path]:
        """Input files/directories read by this stage (inside ``scan_dir``)."""

    def params(self, settings: Settings) -> dict[str, Any]:
        """Parameters that influence the output (part of the cache key)."""
        return {}

    def tool_versions(self) -> dict[str, str]:
        """Versions of external tools (COLMAP, gsplat, ...) used by the stage."""
        return {}

    @abstractmethod
    def run(self, ctx: StageContext) -> None:
        """Produce the outputs in ``ctx.out_dir``; raise StageError subclasses on failure."""


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


def _relative(path: Path, scan_dir: Path) -> str:
    try:
        return path.resolve().relative_to(scan_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _tools(stage: Stage) -> dict[str, str]:
    return {"python": platform.python_version(), "drone-scan": __version__, **stage.tool_versions()}


def _load_manifest(path: Path) -> Manifest | None:
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


def _write_manifest(manifest: Manifest, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(manifest.model_dump_json(indent=2))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _is_cache_hit(
    previous: Manifest | None, stage: Stage, inputs: dict[str, str], params: dict[str, Any]
) -> bool:
    return (
        previous is not None
        and previous.status == "ok"
        and previous.version == stage.version
        and previous.inputs == inputs
        and previous.params == params
    )


def _peak_vram_mb(torch_cuda: Any) -> float | None:
    if torch_cuda is None:
        return None
    return float(torch_cuda.max_memory_reserved()) / MIB


def run_stage(stage: Stage, scan_dir: Path, settings: Settings, force: bool = False) -> StageResult:
    """Run ``stage`` for ``scan_dir`` following the stage contract."""
    out_dir = scan_dir / stage.name
    manifest_path = out_dir / MANIFEST_NAME
    started = datetime.now(UTC)
    input_paths = stage.inputs(scan_dir)
    params = stage.params(settings)
    warnings: list[str] = []
    hints: list[str] = []
    error: str | None = None
    peak: float | None = None
    hashes: dict[str, str] = {}

    def finish(code: ExitCode) -> StageResult:
        manifest = Manifest(
            stage=stage.name,
            version=stage.version,
            profile=settings.profile,
            inputs=hashes,
            params=params,
            tools=_tools(stage),
            started=started,
            finished=datetime.now(UTC),
            peak_vram_mb=peak,
            status=_STATUS[code],
            exit_code=int(code),
            error=error,
            warnings=warnings,
            hints=hints,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(manifest, manifest_path)
        event = "stage_finished" if code is ExitCode.OK else "stage_failed"
        log.info(event, stage=stage.name, exit_code=int(code), peak_vram_mb=peak)
        return StageResult(exit_code=code, skipped=False, manifest=manifest)

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        error = "Eingabe fehlt: " + ", ".join(_relative(p, scan_dir) for p in missing)
        hints.append(
            "Abhilfe: vorherige Stufe bzw. den Import für diesen Scan ausführen "
            "und den Pfad mit --scan prüfen."
        )
        return finish(ExitCode.INPUT_MISSING)

    hashes = {_relative(p, scan_dir): hash_path(p) for p in input_paths}
    previous = _load_manifest(manifest_path)
    if not force and previous is not None and _is_cache_hit(previous, stage, hashes, params):
        log.info("stage_skipped", stage=stage.name)
        return StageResult(exit_code=ExitCode.OK, skipped=True, manifest=previous)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)  # a crash must not leave a valid cache behind
    log.info("stage_started", stage=stage.name, force=force)

    torch_cuda: Any = None
    if stage.uses_gpu:
        try:
            gpu.apply_vram_limit(settings)
        except gpu.GpuUnavailableError as exc:
            error = str(exc)
            hints.append(str(exc))
            return finish(ExitCode.ERROR)
        torch_cuda = sys.modules["torch"].cuda
        torch_cuda.reset_peak_memory_stats()

    ctx = StageContext(scan_dir=scan_dir, out_dir=out_dir, settings=settings, inputs=input_paths)
    code = ExitCode.OK
    try:
        stage.run(ctx)
    except StageError as exc:
        code, error = exc.exit_code, str(exc)
        ctx.hints.extend(exc.hints)
    except Exception as exc:
        if is_oom(exc):
            code, error = ExitCode.OOM, f"{type(exc).__name__}: {exc}"
            ctx.hints.extend(OOM_HINTS)
        else:
            code, error = ExitCode.ERROR, f"{type(exc).__name__}: {exc}"
    finally:
        peak = _peak_vram_mb(torch_cuda)
    warnings, hints = ctx.warnings, ctx.hints
    return finish(code)


def register_stage(app: typer.Typer, factory: Callable[[], Stage]) -> None:
    """Add ``drone-scan <stage.name> --scan --profile --force`` to ``app``.

    ``factory`` is called once at registration (to read the name) and once per run,
    so stage constructors must be cheap.
    """
    name = factory().name

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
        result = run_stage(factory(), scan, settings, force=force)
        manifest = result.manifest
        if result.skipped:
            typer.echo(f"{name}: unverändert, übersprungen (Cache). Neu rechnen mit --force.")
        elif result.exit_code is ExitCode.OK:
            typer.echo(f"{name}: fertig.")
        else:
            typer.echo(f"{name}: {manifest.error}", err=True)
        for hint in manifest.hints:
            typer.echo(f"Hinweis: {hint}", err=result.exit_code is not ExitCode.OK)
        raise typer.Exit(code=int(result.exit_code))
