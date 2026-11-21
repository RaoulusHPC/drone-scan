"""Command line interface `drone-scan`.

Pipeline stage commands (`ingest`, `sfm`, `recon`, ...) are added by their own stories.
User-facing texts are German, code and identifiers English (CLAUDE.md).
"""

from __future__ import annotations

from typing import Annotated

import typer
from pydantic import ValidationError

from dronescan import __version__
from dronescan.common.config import ProfileError, load_settings

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
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    except ValidationError as exc:
        typer.echo(f"Ungültige Konfiguration:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(settings.model_dump_json(indent=2))
