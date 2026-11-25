# SPDX-License-Identifier: AGPL-3.0-or-later
"""Golden-data smoke test (KAN-26): golden thresholds, evaluation and the real run."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dronescan.common.smoke import (
    GOLDEN_DEFAULT,
    SAMPLE_CLIP,
    Golden,
    Measured,
    SmokeError,
    evaluate,
    load_golden,
    run_smoke,
)
from dronescan.sfm.colmap import image_id

ROOT = Path(__file__).resolve().parent.parent


def good() -> Measured:
    return Measured(
        seconds=50.0,
        registered_frames=30,
        reproj_error_px=0.61,
        points=8752,
        peak_vram_mib=844.0,
        psnr_db=None,
    )


def golden() -> Golden:
    return load_golden(ROOT / GOLDEN_DEFAULT)


def test_golden_file_is_versioned_and_valid() -> None:
    """Given tests/golden/kurz.yaml,
    When it is loaded,
    Then it has a version, a clip path inside data/samples and sane thresholds (AC 3).
    """
    g = golden()
    assert g.version >= 1
    assert g.clip == SAMPLE_CLIP
    assert (ROOT / g.clip).exists()
    assert g.max_seconds <= 120
    assert g.poses.max_peak_vram_mib <= 5.5 * 1024


def test_good_run_passes() -> None:
    """Given measurements inside all thresholds, Then there are no violations."""
    assert evaluate(golden(), good()) == []


@pytest.mark.parametrize(
    ("field", "value", "word"),
    [
        ("seconds", 121.0, "Laufzeit"),
        ("registered_frames", 20, "registriert"),
        ("reproj_error_px", 1.5, "Reprojektionsfehler"),
        ("points", 100, "Punkte"),
        ("peak_vram_mib", 6000.0, "VRAM"),
    ],
)
def test_violation_is_reported(field: str, value: float, word: str) -> None:
    """Given one measurement outside its threshold,
    When the run is evaluated,
    Then exactly that violation is named.
    """
    measured = good().model_copy(update={field: value})

    violations = evaluate(golden(), measured)

    assert len(violations) == 1
    assert word in violations[0]


def test_env_vram_limit_tightens_golden_ceiling() -> None:
    """Given DRONESCAN_VRAM_LIMIT_GB=0.5 (512 MiB) and a peak of 844 MiB,
    When the run is evaluated,
    Then it fails, while the golden ceiling alone would pass it (AC 2).
    """
    assert evaluate(golden(), good()) == []

    violations = evaluate(golden(), good(), vram_limit_gb=0.5)

    assert len(violations) == 1
    assert "VRAM" in violations[0]


def test_lfs_pointer_instead_of_clip_is_reported(tmp_path: Path) -> None:
    """Given a clip file that is only an LFS pointer, Then the run stops with a hint."""
    (tmp_path / "data/samples").mkdir(parents=True)
    (tmp_path / SAMPLE_CLIP).write_text("version https://git-lfs.github.com/spec/v1\n")

    with pytest.raises(SmokeError, match="git lfs pull"):
        run_smoke(tmp_path, golden(), tmp_path / "ws")


def test_missing_vram_peak_is_a_violation() -> None:
    """Given no VRAM measurement, Then the run fails (the VRAM limit must be provable)."""
    violations = evaluate(golden(), good().model_copy(update={"peak_vram_mib": None}))

    assert any("VRAM" in v for v in violations)


def test_psnr_is_checked_only_when_threshold_is_set() -> None:
    """Given a PSNR threshold and a splat result below it, Then it is a violation;
    without a threshold the missing splat is not.
    """
    g = golden()
    assert evaluate(g, good()) == []
    strict = g.model_copy(update={"splat": g.splat.model_copy(update={"min_psnr_db": 20.0})})

    assert evaluate(strict, good().model_copy(update={"psnr_db": 18.0}))
    assert evaluate(strict, good().model_copy(update={"psnr_db": 22.0})) == []


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_smoke_run_on_sample_clip(tmp_path: Path) -> None:
    """Given data/samples/kurz.mp4 and the COLMAP image on the GTX 1060,
    When the smoke run executes ingest and sfm,
    Then the golden thresholds hold and it finishes within 2 min (AC 1).
    """
    if image_id() == "missing":
        pytest.skip("COLMAP image missing: docker compose build colmap")

    result = run_smoke(ROOT, golden(), tmp_path)

    assert result.violations == []
    assert result.measured.seconds < 120
