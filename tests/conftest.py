# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared test fixtures and hooks."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Typer forces a coloured terminal on GitHub Actions (GITHUB_ACTIONS, FORCE_COLOR, PY_COLORS):
# the ANSI codes split option names like "--name" in CliRunner output, so CLI tests would pass
# locally and fail in CI. Typer reads this switch once on import, hence before any dronescan
# import (tests/test_cli.py checks that it took effect).
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

from dronescan.common.config import load_settings  # noqa: E402
from dronescan.common.gpu_lock import stage_gpu_lock  # noqa: E402

SITL_ENV_VAR = "DRONESCAN_SITL"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `sitl` tests unless ArduPilot SITL is available (DRONESCAN_SITL=1).

    Keeps `make check` and CI green without a flight simulator; `make test-sitl`
    sets the variable (PO decision 2026-11-21, KAN-17).
    """
    if os.environ.get(SITL_ENV_VAR) == "1":
        return
    skip = pytest.mark.skip(
        reason=f"ArduPilot-SITL nicht verfügbar – mit `make test-sitl` ({SITL_ENV_VAR}=1) ausführen"
    )
    for item in items:
        if item.get_closest_marker("sitl"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolate_dronescan_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove DRONESCAN_* variables so the developer's shell cannot change test results.

    GPU and SITL tests keep them: `make test-gpu` sets DRONESCAN_VRAM_LIMIT_GB and
    `make test-sitl` sets DRONESCAN_SITL.
    """
    if request.node.get_closest_marker("gpu") or request.node.get_closest_marker("sitl"):
        return
    for name in list(os.environ):
        if name.startswith("DRONESCAN_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="session")
def gpu_session_lock() -> Iterator[None]:
    """Hold the real machine-wide GPU lock while GPU tests run (CLAUDE.md rule 2).

    The stages under test lock a file in their temporary state directory, so they cannot
    deadlock with it.
    """
    with stage_gpu_lock(load_settings()):
        yield


@pytest.fixture(autouse=True)
def isolate_state_dir(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point XDG_STATE_HOME at a temp dir: tests must never touch the real GPU lock (KAN-80).

    Exception: the session as a whole holds the real lock while it runs `gpu` tests.
    """
    if request.node.get_closest_marker("gpu"):
        request.getfixturevalue("gpu_session_lock")  # before the path is redirected
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
