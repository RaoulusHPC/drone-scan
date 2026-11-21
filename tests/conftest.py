"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_dronescan_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove DRONESCAN_* variables so the developer's shell cannot change test results.

    GPU tests keep them: `make test-gpu` deliberately sets DRONESCAN_VRAM_LIMIT_GB.
    """
    if request.node.get_closest_marker("gpu"):
        return
    for name in list(os.environ):
        if name.startswith("DRONESCAN_"):
            monkeypatch.delenv(name)
