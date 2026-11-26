# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared test fixtures and hooks."""

from __future__ import annotations

import os
import socket
from collections.abc import Generator, Iterator
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
REQUIRE_GPU_ENV_VAR = "DRONESCAN_TEST_REQUIRE_GPU"  # set by the worker-test container (KAN-41)


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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """In the worker-test container a skipped `torch` test is a failure (KAN-41).

    There PyTorch, gsplat and the GPU must be present, so a skip ("no CUDA device", "not
    enough free VRAM", ...) would hide that the test never ran.
    """
    report: pytest.TestReport = yield
    if (
        report.skipped
        and os.environ.get(REQUIRE_GPU_ENV_VAR) == "1"
        and item.get_closest_marker("torch")
    ):
        report.outcome = "failed"
        report.longrepr = f"übersprungen, aber im worker-test-Container Pflicht: {report.longrepr}"
    return report


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


class NetworkBlockedError(OSError):
    """Raised by `block_network` for any attempt to reach another host."""


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block all IP network access of this process (CLAUDE.md rule 3, KAN-24).

    Name resolution and every IPv4/IPv6 connect fail, including loopback; Unix sockets
    (subprocess pipes, Docker's socket in fake setups) keep working. Subprocesses are not
    covered: containers started by stages run with `--network none` (tests in test_sfm.py
    and test_recon.py).
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise NetworkBlockedError("Netzwerkzugriff im Test blockiert (CLAUDE.md Regel 3)")

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: object) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            refuse()
        original_connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: socket.socket, address: object) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            refuse()
        return original_connect_ex(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "gethostbyname", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
