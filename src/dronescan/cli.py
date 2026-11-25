# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line interface `drone-scan`.

Pipeline stage commands (`ingest`, `sfm`, `recon`, ...) are added by their own stories
via `dronescan.common.stage.register_stage(app, StageClass)`.
User-facing texts are German, code and identifiers English (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from dronescan import __version__
from dronescan.capture.scan import ScanError, add_flight, create_scan, resolve_scan_dir
from dronescan.common.config import ProfileError, describe_validation_error, load_settings
from dronescan.common.logging import configure_logging
from dronescan.common.stage import ExitCode, register_stage
from dronescan.ingest.stage import IngestStage
from dronescan.sfm.stage import SfmStage

app = typer.Typer(
    name="drone-scan",
    help="Drohnen- oder Smartphone-Video → 3D-Modell eines Innenraums (Gaussian Splatting).",
    no_args_is_help=True,
    add_completion=False,
)


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
        raise typer.BadParameter(str(exc), param_hint="--profile / DRONESCAN_PROFILE") from exc
    except ValidationError as exc:
        typer.echo(describe_validation_error(exc, profile), err=True)
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
app.add_typer(scan_app, name="scan")
app.add_typer(flight_app, name="flight")


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
) -> None:
    """Video kopieren, Metadaten lesen und eine Flug-ID vergeben."""
    try:
        flight = add_flight(resolve_scan_dir(workspace, scan), video)
    except ScanError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=int(ExitCode.INPUT_MISSING)) from exc
    typer.echo(
        f"Flug {flight.id} hinzugefügt: {flight.display_width}x{flight.display_height}, "
        f"{flight.fps:.2f} fps, {flight.duration_s:.1f} s, {flight.codec}"
    )


register_stage(app, IngestStage)
register_stage(app, SfmStage)
