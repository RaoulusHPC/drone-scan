"""Package structure and CLI entry point (KAN-16, AC 3)."""

from __future__ import annotations

import importlib
import shutil
import subprocess

import pytest
from typer.testing import CliRunner

import dronescan
from dronescan.cli import app

SUBPACKAGES = [
    "common",
    "capture",
    "ingest",
    "sfm",
    "recon",
    "search",
    "privacy",
    "export",
    "server",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable(name: str) -> None:
    """Given the package layout from CLAUDE.md,
    When a pipeline subpackage is imported,
    Then it exists and documents its responsibility.
    """
    module = importlib.import_module(f"dronescan.{name}")

    assert module.__doc__


def test_cli_help_exits_successfully() -> None:
    """Given the Typer app,
    When `--help` is requested,
    Then it exits with code 0 and shows the program name.
    """
    result = CliRunner().invoke(app, ["--help"], prog_name="drone-scan")

    assert result.exit_code == 0
    assert "drone-scan" in result.output


def test_cli_version_option() -> None:
    """Given the Typer app,
    When `--version` is requested,
    Then it prints the package version and exits with code 0.
    """
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert dronescan.__version__ in result.output


def test_console_script_is_installed() -> None:
    """Given the environment created by `uv sync`,
    When the `drone-scan --help` console script is executed,
    Then it exits with code 0.
    """
    executable = shutil.which("drone-scan")
    assert executable is not None, "console script missing - run `make setup`"

    result = subprocess.run(
        [executable, "--help"], capture_output=True, text=True, timeout=30, check=False
    )

    assert result.returncode == 0
    assert "drone-scan" in result.stdout
