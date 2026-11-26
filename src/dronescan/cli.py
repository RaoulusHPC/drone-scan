# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line interface `drone-scan`.

Pipeline stage commands (`ingest`, `sfm`, `recon`, ...) are added by their own stories
via `dronescan.common.stage.register_stage(app, StageClass)`.
User-facing texts are German, code and identifiers English (CLAUDE.md).
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Annotated, cast

import typer
import yaml
from pydantic import ValidationError

from dronescan import __version__
from dronescan.capture.camera import (
    CameraError,
    available_cameras,
    is_camera_name,
    load_camera,
    write_camera,
)
from dronescan.capture.scan import ScanError, add_flight, create_scan, resolve_scan_dir
from dronescan.common import error_catalog as catalog
from dronescan.common.camera_models import CAMERA_MODELS
from dronescan.common.config import ProfileError, describe_validation_error, load_settings
from dronescan.common.logging import configure_logging
from dronescan.common.models_lock import ModelsLock, load_models_lock, model_status
from dronescan.common.stage import ExitCode, register_stage
from dronescan.ingest.stage import IngestStage
from dronescan.recon.stage import ReconStage
from dronescan.sfm.stage import SfmStage

app = typer.Typer(
    name="drone-scan",
    help="Drohnen- oder Smartphone-Video → 3D-Modell eines Innenraums (Gaussian Splatting).",
    no_args_is_help=True,
    add_completion=False,
)


def _echo_code(code: str) -> None:
    """Print the catalog code of a failure (docs/fehlercodes.md explains cause and remedy)."""
    typer.echo(f"Fehlercode: {code} (docs/fehlercodes.md)", err=True)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"drone-scan {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Version anzeigen und beenden.",
        ),
    ] = False,
) -> None:
    """Lokale Pipeline für 3D-Scans von Innenräumen."""
    configure_logging()


@app.command("config")
def show_config(
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help="VRAM-Profil, z. B. low_vram_6gb oder default_8gb (Standard: DRONESCAN_PROFILE).",
        ),
    ] = None,
) -> None:
    """Wirksame Konfiguration (Profil + Umgebungsvariablen) als JSON anzeigen."""
    try:
        settings = load_settings(profile)
    except ProfileError as exc:
        _echo_code("cli.usage")
        raise typer.BadParameter(str(exc), param_hint="--profile / DRONESCAN_PROFILE") from exc
    except ValidationError as exc:
        typer.echo(describe_validation_error(exc, profile), err=True)
        _echo_code("cli.config_invalid")
        raise typer.Exit(code=1) from exc
    typer.echo(settings.model_dump_json(indent=2))


WorkspaceOption = Annotated[
    Path,
    typer.Option(
        "--workspace",
        envvar="DRONESCAN_WORKSPACE_DIR",
        help="Workspace-Verzeichnis mit allen Scans.",
    ),
]

scan_app = typer.Typer(help="Scans anlegen.", no_args_is_help=True)
flight_app = typer.Typer(help="Flüge (Videos) zu einem Scan hinzufügen.", no_args_is_help=True)
camera_app = typer.Typer(help="Kameraprofile anzeigen und kalibrieren.", no_args_is_help=True)
models_app = typer.Typer(
    help="Modellgewichte aus models.lock (einmaliger Download, danach offline).",
    no_args_is_help=True,
)
app.add_typer(scan_app, name="scan")
app.add_typer(flight_app, name="flight")
app.add_typer(camera_app, name="camera")
app.add_typer(models_app, name="models")


@scan_app.command("create")
def scan_create(
    name: Annotated[str, typer.Argument(help="Name des Scans, z. B. 'Haus Muster EG'.")],
    workspace: WorkspaceOption = Path("workspace"),
) -> None:
    """Neuen Scan anlegen (workspace/<id>/scan.yaml)."""
    try:
        meta = create_scan(workspace, name)
    except ScanError as exc:
        raise typer.BadParameter(str(exc), param_hint="NAME") from exc
    typer.echo(f"Scan angelegt: {meta.id}")


@flight_app.command("add")
def flight_add(
    scan: Annotated[str, typer.Argument(help="Scan-ID oder Pfad zum Scan-Verzeichnis.")],
    video: Annotated[Path, typer.Argument(help="Video eines Akkuflugs (MP4/MOV).")],
    workspace: WorkspaceOption = Path("workspace"),
    camera: Annotated[
        str | None,
        typer.Option(
            "--camera",
            help="Kameraprofil: Name (siehe `drone-scan camera list`) oder Pfad zu einer "
            "Profildatei. Ohne Angabe schätzt COLMAP die Kamera.",
        ),
    ] = None,
) -> None:
    """Video kopieren, Metadaten lesen und eine Flug-ID vergeben."""
    try:
        profile = load_camera(camera) if camera is not None else None
    except CameraError as exc:
        _echo_code("capture.camera_profile_invalid")
        raise typer.BadParameter(str(exc), param_hint="--camera") from exc
    try:
        flight = add_flight(resolve_scan_dir(workspace, scan), video, camera=profile)
    except ScanError as exc:
        typer.echo(str(exc), err=True)
        _echo_code("capture.scan_invalid")
        raise typer.Exit(code=int(ExitCode.INPUT_MISSING)) from exc
    typer.echo(
        f"Flug {flight.id} hinzugefügt: {flight.display_width}x{flight.display_height}, "
        f"{flight.fps:.2f} fps, {flight.duration_s:.1f} s, {flight.codec}"
        + (f", Kamera {profile.name}" if profile else "")
    )


@camera_app.command("list")
def camera_list() -> None:
    """Mitgelieferte Kameraprofile anzeigen."""
    for name in available_cameras():
        profile = load_camera(name)
        values = "mit Intrinsik" if profile.intrinsics else "Intrinsik schätzt COLMAP"
        typer.echo(f"{name}: {profile.model}, {values}. {profile.description}")


@camera_app.command("calibrate")
def camera_calibrate(
    video: Annotated[Path, typer.Argument(help="Video des Kalibriermusters.")],
    name: Annotated[
        str, typer.Option("--name", help="Name des Profils, z. B. 'mein_handy' (a-z, 0-9, _ -).")
    ],
    cols: Annotated[int, typer.Option("--cols", min=3, help="Felder (Quadrate) waagrecht.")],
    rows: Annotated[int, typer.Option("--rows", min=3, help="Felder (Quadrate) senkrecht.")],
    square_mm: Annotated[
        float, typer.Option("--square-mm", min=0.1, help="Gemessene Kantenlänge eines Felds (mm).")
    ],
    pattern: Annotated[
        str, typer.Option("--pattern", help="Muster: chessboard oder charuco.")
    ] = "chessboard",
    marker_mm: Annotated[
        float | None,
        typer.Option("--marker-mm", min=0.1, help="ChArUco: Kantenlänge eines Markers (mm)."),
    ] = None,
    dictionary: Annotated[
        str, typer.Option("--dictionary", help="ChArUco: ArUco-Wörterbuch.")
    ] = "DICT_5X5_100",
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=f"COLMAP-Kameramodell: {', '.join(CAMERA_MODELS)} (Fisheye: OPENCV_FISHEYE).",
        ),
    ] = "OPENCV",
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Zieldatei (Standard: <name>.yaml im aktuellen Ordner)."),
    ] = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="Konfigurationsprofil, z. B. low_vram_6gb.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Vorhandene Zieldatei ersetzen.")] = False,
) -> None:
    """Kamera mit einem Schachbrett- oder ChArUco-Video kalibrieren (optional)."""
    from dronescan.capture.calibration import Board, CalibrationError, Pattern, calibrate

    if not is_camera_name(name):
        raise typer.BadParameter(
            "Nur Kleinbuchstaben, Ziffern, '_' und '-' verwenden.", param_hint="--name"
        )
    if pattern not in ("chessboard", "charuco"):
        raise typer.BadParameter("chessboard oder charuco angeben.", param_hint="--pattern")
    if model not in CAMERA_MODELS:
        raise typer.BadParameter(f"Eines von {', '.join(CAMERA_MODELS)}.", param_hint="--model")
    target = output or Path(f"{name}.yaml")
    if target.exists() and not force:
        raise typer.BadParameter(
            f"'{target.name}' existiert bereits. Abhilfe: --force oder eine andere --output.",
            param_hint="--output",
        )
    try:
        settings = load_settings(profile)
    except ProfileError as exc:
        _echo_code("cli.usage")
        raise typer.BadParameter(str(exc), param_hint="--profile / DRONESCAN_PROFILE") from exc
    except ValidationError as exc:
        typer.echo(describe_validation_error(exc, profile), err=True)
        _echo_code("cli.config_invalid")
        raise typer.Exit(code=int(ExitCode.ERROR)) from exc
    board = Board(cast(Pattern, pattern), cols, rows, square_mm, marker_mm, dictionary)
    try:
        result = calibrate(video, board, model, name, settings.calibration)
    except CalibrationError as exc:
        typer.echo(f"Kalibrierung: {exc}", err=True)
        for hint in exc.hints:
            typer.echo(f"Hinweis: {hint}", err=True)
        _echo_code(exc.code)
        raise typer.Exit(code=int(exc.exit_code)) from exc
    try:
        write_camera(result.profile, target)
    except OSError as exc:
        typer.echo(
            f"Kameraprofil '{target.name}' konnte nicht geschrieben werden. Abhilfe: Zielordner "
            "und Schreibrechte prüfen oder mit --output einen anderen Pfad angeben.",
            err=True,
        )
        _echo_code("capture.calibration_write_failed")
        raise typer.Exit(code=int(ExitCode.ERROR)) from exc
    typer.echo(
        f"Kameraprofil gespeichert: {target.name} ({result.profile.model}, "
        f"{result.views} Ansichten, Restfehler {result.profile.rms_error_px:.3f} px). "
        f"Verwenden mit `drone-scan flight add <scan> <video> --camera {target}`."
    )


LockOption = Annotated[
    Path,
    typer.Option("--lock", envvar="DRONESCAN_MODELS_LOCK", help="Modell-Registry models.lock."),
]
ModelsDirOption = Annotated[
    Path,
    typer.Option(
        "--models-dir",
        envvar="DRONESCAN_MODELS_DIR",
        help="Verzeichnis der heruntergeladenen Modellgewichte.",
    ),
]


def _read_lock(lock: Path) -> ModelsLock:
    if not lock.is_file():
        typer.echo(
            f"'{lock}' nicht gefunden. Abhilfe: im Repository-Verzeichnis ausführen oder den "
            "Pfad mit --lock bzw. DRONESCAN_MODELS_LOCK angeben.",
            err=True,
        )
        _echo_code("models.lock_missing")
        raise typer.Exit(code=int(ExitCode.INPUT_MISSING))
    try:
        return load_models_lock(lock)
    except (ValueError, OSError, yaml.YAMLError) as exc:  # ValidationError is a ValueError
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        typer.echo(
            f"'{lock.name}' ist ungültig ({detail}). Abhilfe: jedes Modell braucht name, "
            "version, source, url (https), sha256, weights_license und code_license.",
            err=True,
        )
        _echo_code("models.lock_invalid")
        raise typer.Exit(code=int(ExitCode.ERROR)) from exc


@models_app.command("list")
def models_list(
    lock: LockOption = Path("models.lock"), models_dir: ModelsDirOption = Path("models")
) -> None:
    """Modelle aus models.lock mit Lizenzen und lokalem Status anzeigen (offline)."""
    registry = _read_lock(lock)
    if not registry.models:
        typer.echo("models.lock enthält keine Modelle.")
    for entry in registry.models:
        typer.echo(
            f"{entry.name} {entry.version}: {model_status(entry, models_dir)}, Gewichte "
            f"{entry.weights_license}, Code {entry.code_license}, Quelle {entry.source}"
        )


@models_app.command("pull")
def models_pull(
    names: Annotated[
        list[str] | None,
        typer.Argument(help="Modellnamen aus models.lock (Standard: alle)."),
    ] = None,
    lock: LockOption = Path("models.lock"),
    models_dir: ModelsDirOption = Path("models"),
    force: Annotated[
        bool, typer.Option("--force", help="Auch gültige, vorhandene Dateien neu laden.")
    ] = False,
) -> None:
    """Gelistete Modelle einmalig herunterladen und per SHA-256 prüfen.

    Einziger Netzwerkzugriff von drone-scan; eine Hash-Abweichung bricht ab.
    """
    from dronescan.common.models_pull import PullError, pull_model

    registry = _read_lock(lock)
    unknown = [name for name in names or [] if registry.get(name) is None]
    if unknown:
        _echo_code("models.unknown_name")
        known = ", ".join(m.name for m in registry.models) or "keine"
        raise typer.BadParameter(
            f"Nicht in models.lock: {', '.join(unknown)}. Gelistet: {known}. Abhilfe: nur "
            "gelistete Modelle laden; neue Einträge prüft der PO (Lizenz der Gewichte).",
            param_hint="NAMES",
        )
    entries = [e for e in registry.models if not names or e.name in names]
    if not entries:
        typer.echo("models.lock enthält keine Modelle, nichts zu laden.")
        return
    for entry in entries:
        typer.echo(f"{entry.name} {entry.version} ({entry.weights_license}) …")
        try:
            result = pull_model(entry, models_dir, force=force, progress=typer.echo)
        except PullError as exc:
            typer.echo(str(exc), err=True)
            _echo_code(exc.code)
            raise typer.Exit(code=int(ExitCode.ERROR)) from exc
        typer.echo(f"{entry.name}: {result}, SHA-256 geprüft.")


def is_loopback(host: str) -> bool:
    """True for addresses only reachable from this machine (127.0.0.0/8, ::1, localhost)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


@app.command("serve")
def serve(
    workspace: WorkspaceOption = Path("workspace"),
    host: Annotated[
        str | None,
        typer.Option("--host", help="Adresse (Standard: server.host, nur localhost erlaubt)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", min=1, max=65535, help="Port (Standard: server.port)."),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Profil des Servers und Standardprofil der Jobs."),
    ] = None,
) -> None:
    """Lokalen API-Server mit Job-Queue starten (REST + Server-Sent Events, nur localhost)."""
    try:
        settings = load_settings(profile)
    except ProfileError as exc:
        _echo_code("cli.usage")
        raise typer.BadParameter(str(exc), param_hint="--profile / DRONESCAN_PROFILE") from exc
    except ValidationError as exc:
        typer.echo(describe_validation_error(exc, profile), err=True)
        _echo_code("cli.config_invalid")
        raise typer.Exit(code=int(ExitCode.ERROR)) from exc
    bind = host or settings.server.host
    if not is_loopback(bind):
        _echo_code("server.not_loopback")
        raise typer.BadParameter(
            f"'{bind}' ist keine localhost-Adresse. {catalog.hint('server.not_loopback')}",
            param_hint="--host / DRONESCAN_SERVER__HOST",
        )
    try:
        import uvicorn

        from dronescan.server.api import create_app
    except ImportError as exc:
        typer.echo(
            f"{catalog.cause('server.extra_missing')} {catalog.hint('server.extra_missing')}",
            err=True,
        )
        _echo_code("server.extra_missing")
        raise typer.Exit(code=int(ExitCode.ERROR)) from exc
    listen = port or settings.server.port
    typer.echo(f"drone-scan-Server: http://{bind}:{listen}/api (Strg+C beendet)")
    # No access log: request paths contain scan ids, which may be names of people (CLAUDE.md).
    uvicorn.run(
        create_app(workspace, settings),
        host=bind,
        port=listen,
        access_log=False,
        log_level="warning",
    )


register_stage(app, IngestStage)
register_stage(app, SfmStage)
register_stage(app, ReconStage)
