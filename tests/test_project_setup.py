"""Project setup artefacts required by `make setup` (KAN-16, AC 1 and AC 2)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import dronescan

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_requires_python_312() -> None:
    """Given the project metadata,
    When the Python requirement is read,
    Then it pins the 3.12 series used on all machines.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.12,<3.13"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_lockfile_is_committed() -> None:
    """Given a fresh clone,
    When `make setup` runs `uv sync`,
    Then a committed uv.lock makes the environment reproducible.
    """
    assert (ROOT / "uv.lock").is_file()


def test_pre_commit_blocks_large_files() -> None:
    """Given the pre-commit configuration,
    When a file larger than 20 MB is staged,
    Then the large-file hook rejects it (CLAUDE.md rule 5).
    """
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "check-added-large-files" in config
    assert "--maxkb=20480" in config


def test_package_exposes_version() -> None:
    """Given the installed package,
    When it is imported,
    Then it exposes a version string (placeholder test for `make check`).
    """
    assert isinstance(dronescan.__version__, str)
    assert dronescan.__version__
