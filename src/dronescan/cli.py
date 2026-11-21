"""Command line interface `drone-scan`.

Pipeline stage commands (`ingest`, `sfm`, `recon`, ...) are added by their own stories.
User-facing texts are German, code and identifiers English (CLAUDE.md).
"""

from __future__ import annotations

from typing import Annotated

import typer

from dronescan import __version__

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
