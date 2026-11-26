# SPDX-License-Identifier: AGPL-3.0-or-later
"""No network access at runtime (CLAUDE.md rule 3, KAN-24 AC 3).

Two layers: a static check that only `drone-scan models pull` imports network modules, and
runs of the CLI and stages with all IP network access of the test process blocked
(`block_network` in conftest.py). Containers started by `sfm`/`recon` run with
`--network none`, checked in test_sfm.py and test_recon.py, which also run those stages
offline.
"""

from __future__ import annotations

import ast
import json
import shutil
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dronescan.cli import app
from dronescan.common.licenses import run_gate
from dronescan.common.models_lock import ModelsLock, resolve_model
from dronescan.common.stage import ExitCode
from tests.conftest import NetworkBlockedError
from tests.test_ingest import needs_ffmpeg, panning, require_ffmpeg, write_video  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "dronescan"
runner = CliRunner()

# Modules that open network connections or download files (incl. implicit weight downloads).
NETWORK_MODULES = frozenset(
    {
        "socket",
        "ssl",
        "http.client",
        "urllib.request",
        "urllib3",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "websocket",
        "websockets",
        "huggingface_hub",
        "torch.hub",
    }
)
# The one sanctioned exception: the explicit, user-started model download.
ALLOWED = {"common/models_pull.py": {"urllib.request", "http.client", "ssl"}}


def network_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    found = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        for name in names:
            parts = name.split(".")
            prefixes = {".".join(parts[: i + 1]) for i in range(len(parts))}
            found |= prefixes & NETWORK_MODULES
    return found


def test_only_models_pull_imports_network_modules() -> None:
    """Given all Python modules in src/,
    When their imports are scanned,
    Then only models_pull.py uses a network module (the explicit `models pull`).
    """
    offenders = {
        path.relative_to(SRC).as_posix(): imports
        for path in sorted(SRC.rglob("*.py"))
        if (imports := network_imports(path) - ALLOWED.get(path.relative_to(SRC).as_posix(), set()))
    }

    assert offenders == {}
    assert "urllib.request" in network_imports(SRC / "common" / "models_pull.py")


def test_scanner_finds_hidden_network_imports(tmp_path: Path) -> None:
    """Given imports in the forms `from urllib import request`, `import http.client as c`
    and `from torch.hub import load_state_dict_from_url`,
    When the file is scanned,
    Then each one is found.
    """
    module = tmp_path / "m.py"
    module.write_text(
        "from urllib import request\nimport http.client as c\n"
        "from torch.hub import load_state_dict_from_url\nimport json\n"
    )

    assert network_imports(module) == {"urllib.request", "http.client", "torch.hub"}


def test_cli_does_not_load_the_downloader() -> None:
    """Given the CLI module,
    When it is imported (every stage run does this),
    Then the downloader is only imported inside `models pull`.
    """
    tree = ast.parse((SRC / "cli.py").read_text(encoding="utf-8"))
    top_level = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "dronescan.common.models_pull" not in top_level


@pytest.mark.usefixtures("block_network")
def test_network_block_is_effective() -> None:
    """Given the blocked network,
    When anything tries to reach a host (even loopback),
    Then it fails, so the tests below would notice a network access.
    """
    for url in ("http://127.0.0.1:9/", "https://example.org/"):
        with pytest.raises(urllib.error.URLError) as caught:
            urllib.request.urlopen(url, timeout=1)
        assert isinstance(caught.value.reason, NetworkBlockedError)
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.org", 443))


@needs_ffmpeg
@pytest.mark.usefixtures("block_network")
def test_import_and_ingest_run_without_network(tmp_path: Path) -> None:
    """Given a blocked network,
    When a scan is created, a synthetic video added and `ingest` runs via the CLI,
    Then everything succeeds (exit 0).
    """
    video = write_video(tmp_path / "clip.mp4", panning(48))
    workspace = tmp_path / "workspace"

    created = runner.invoke(app, ["scan", "create", "Offline", "--workspace", str(workspace)])
    assert created.exit_code == 0, created.output
    scan_id = created.output.strip().rsplit(" ", 1)[-1]
    added = runner.invoke(
        app, ["flight", "add", scan_id, str(video), "--workspace", str(workspace)]
    )
    assert added.exit_code == 0, added.output

    result = runner.invoke(app, ["ingest", "--scan", str(workspace / scan_id)])

    assert result.exit_code == int(ExitCode.OK), result.output
    manifest = json.loads((workspace / scan_id / "ingest" / "manifest.json").read_text())
    assert manifest["status"] == "ok"


@pytest.mark.usefixtures("block_network")
def test_models_and_license_gate_run_without_network(tmp_path: Path) -> None:
    """Given a blocked network,
    When models are listed and resolved and the license gate runs,
    Then they work offline; a missing model points to `models pull` instead of downloading.
    """
    lock = tmp_path / "models.lock"
    shutil.copy(ROOT / "models.lock", lock)
    listed = runner.invoke(app, ["models", "list", "--lock", str(lock)])
    assert listed.exit_code == 0, listed.output

    registry = ModelsLock.model_validate(
        {
            "models": [
                {
                    "name": "tiny",
                    "version": "1",
                    "source": "https://example.org/card",
                    "url": "https://example.org/tiny.pth",
                    "sha256": "0" * 64,
                    "weights_license": "MIT",
                    "code_license": "MIT",
                }
            ]
        }
    )
    with pytest.raises(LookupError, match="models pull tiny"):
        resolve_model("tiny", registry, tmp_path / "models")

    assert [f for f in run_gate(ROOT) if f.severity == "error"] == []
