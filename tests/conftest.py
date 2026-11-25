# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared test fixtures and hooks."""

from __future__ import annotations

import os

import pytest

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
