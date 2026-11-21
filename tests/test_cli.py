"""Package structure and CLI entry point (KAN-16, AC 3)."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize("args", [[], ["does-not-exist"]])
def test_cli_usage_error_exits_with_2(args: list[str]) -> None:
    """Given the Typer app,
    When it is called without a command or with an unknown command,
    Then it shows usage information and exits with code 2.
    """
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2


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
    executable = Path(sys.executable).parent / "drone-scan"
    assert executable.is_file(), "console script missing - run `make setup`"

    result = subprocess.run(
        [str(executable), "--help"], capture_output=True, text=True, timeout=30, check=False
    )

    assert result.returncode == 0
    assert "drone-scan" in result.stdout


def test_config_command_shows_effective_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given the default_8gb profile and a VRAM override in the environment,
    When `drone-scan config --profile default_8gb` runs,
    Then the effective settings are printed as JSON including the override.
    """
    monkeypatch.setenv("DRONESCAN_VRAM_LIMIT_GB", "5.5")

    result = CliRunner().invoke(app, ["config", "--profile", "default_8gb"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["profile"] == "default_8gb"
    assert data["vram_limit_gb"] == 5.5


def test_config_command_invalid_env_value_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given DRONESCAN_VRAM_LIMIT_GB=-1,
    When `drone-scan config` runs,
    Then it exits with code 1 and names the field, the env variable and a remedy in German.
    """
    monkeypatch.setenv("DRONESCAN_VRAM_LIMIT_GB", "-1")

    result = CliRunner().invoke(app, ["config"])

    assert result.exit_code == 1
    assert "Ungültige Konfiguration" in result.output
    assert "DRONESCAN_VRAM_LIMIT_GB" in result.output
    assert "Abhilfe" in result.output


def test_config_command_unknown_profile_exits_2() -> None:
    """Given an unknown profile name,
    When `drone-scan config --profile nope` runs,
    Then it exits with code 2 and names the available profiles.
    """
    result = CliRunner().invoke(app, ["config", "--profile", "nope"])

    assert result.exit_code == 2
    assert "low_vram_6gb" in result.output
