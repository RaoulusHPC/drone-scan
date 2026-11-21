"""Large and private artefacts stay out of Git (KAN-16, AC 4)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=ROOT,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "workspace/scan-001/ingest/frames/0001.jpg",
        "data/raw/wohnung.mp4",
        "models/aliked/weights.bin",
        "checkpoints/splat.pt",
        "checkpoints/detector.pth",
        "checkpoints/encoder.safetensors",
        "checkpoints/model.onnx",
        "checkpoints/model.ckpt",
        "checkpoints/model.engine",
    ],
)
def test_private_and_large_artefacts_are_ignored(path: str) -> None:
    """Given a scan, raw data or model weight file,
    When Git evaluates .gitignore,
    Then the file is ignored.
    """
    assert is_ignored(path)


@pytest.mark.parametrize("path", ["data/samples/kurz.mp4", "src/dronescan/cli.py"])
def test_samples_and_sources_are_tracked(path: str) -> None:
    """Given a released sample clip or a source file,
    When Git evaluates .gitignore,
    Then the file is not ignored.
    """
    assert not is_ignored(path)
