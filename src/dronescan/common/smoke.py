# SPDX-License-Identifier: AGPL-3.0-or-later
"""Golden-data smoke test: sample clip -> poses (-> mini splat once `recon` exists) (KAN-26).

``python -m dronescan.common.smoke`` runs the real CLI stages in a temporary workspace and
compares the manifests with the thresholds in ``tests/golden/kurz.yaml``. Exit 0 = ok,
1 = a threshold was violated or a stage failed. Only ``data/samples/kurz.mp4`` is used.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel, ConfigDict

from dronescan.common.stage import MANIFEST_NAME, load_manifest

SAMPLE_CLIP = "data/samples/kurz.mp4"
GOLDEN_DEFAULT = "tests/golden/kurz.yaml"
SPLAT_STAGE = "recon"  # KAN-41 registers the CLI command `recon` (stage recon.splat)

log = structlog.get_logger(__name__)


class PoseGolden(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_registered_frames: int
    max_reproj_error_px: float
    min_points: int
    max_peak_vram_mib: float


class SplatGolden(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_psnr_db: float | None = None


class Golden(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    clip: str
    profile: str
    max_seconds: float
    poses: PoseGolden
    splat: SplatGolden


class Measured(BaseModel):
    seconds: float
    registered_frames: int
    reproj_error_px: float
    points: int
    peak_vram_mib: float | None
    psnr_db: float | None = None


@dataclass(frozen=True)
class SmokeResult:
    measured: Measured
    violations: list[str]
    splat_skipped: bool


class SmokeError(Exception):
    """A stage failed or its manifest is missing; the message is user-facing (German)."""


def load_golden(path: Path) -> Golden:
    return Golden.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def evaluate(golden: Golden, measured: Measured, vram_limit_gb: float | None = None) -> list[str]:
    """Violations of the golden thresholds (empty = ok).

    ``vram_limit_gb`` (DRONESCAN_VRAM_LIMIT_GB) tightens the golden VRAM ceiling. COLMAP runs in a
    container and ignores the PyTorch allocator limit, so its sampled peak is checked against it.
    """
    poses = golden.poses
    vram_ceiling = poses.max_peak_vram_mib
    if vram_limit_gb is not None:
        vram_ceiling = min(vram_ceiling, vram_limit_gb * 1024)
    violations: list[str] = []
    if measured.seconds > golden.max_seconds:
        violations.append(
            f"Laufzeit {measured.seconds:.0f} s über dem Limit von {golden.max_seconds:.0f} s."
        )
    if measured.registered_frames < poses.min_registered_frames:
        violations.append(
            f"Nur {measured.registered_frames} Bilder registriert, "
            f"mindestens {poses.min_registered_frames} erwartet."
        )
    if measured.reproj_error_px > poses.max_reproj_error_px:
        violations.append(
            f"Reprojektionsfehler {measured.reproj_error_px:.2f} px über dem Limit "
            f"von {poses.max_reproj_error_px:.2f} px."
        )
    if measured.points < poses.min_points:
        violations.append(f"Nur {measured.points} Punkte, mindestens {poses.min_points} erwartet.")
    if measured.peak_vram_mib is None:
        violations.append("Kein VRAM-Spitzenwert gemessen: nvidia-smi muss erreichbar sein.")
    elif measured.peak_vram_mib > vram_ceiling:
        violations.append(
            f"VRAM-Spitze {measured.peak_vram_mib:.0f} MiB über dem Limit "
            f"von {vram_ceiling:.0f} MiB."
        )
    threshold = golden.splat.min_psnr_db
    if threshold is not None and (measured.psnr_db is None or measured.psnr_db < threshold):
        violations.append(f"Splat-PSNR unter {threshold:.1f} dB (gemessen: {measured.psnr_db}).")
    return violations


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    code = "from dronescan.cli import app; app()"
    return subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, check=False
    )


def _registered_commands() -> set[str]:
    from dronescan.cli import app

    commands = {command.name or "" for command in app.registered_commands}
    return commands | {group.name or "" for group in app.registered_groups}


def _stage(name: str, scan: Path, profile: str) -> None:
    proc = _cli(name, "--scan", str(scan), "--profile", profile)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
        hints = load_manifest(scan / name / MANIFEST_NAME)
        extra = " ".join(hints.hints) if hints else ""
        raise SmokeError(
            f"Stufe '{name}' endete mit Code {proc.returncode}: {' '.join(detail)} {extra}".strip()
        )


def run_smoke(root: Path, golden: Golden, workspace: Path) -> SmokeResult:
    """Run ingest + sfm (+ recon if registered) on the sample clip in ``workspace``."""
    clip = root / golden.clip
    if not clip.is_file() or clip.stat().st_size < 1024:  # an LFS pointer is a few bytes
        raise SmokeError(
            f"Testclip fehlt oder ist nur ein LFS-Zeiger: {golden.clip} (git lfs pull)."
        )
    started = time.monotonic()
    created = _cli("scan", "create", "smoke", "--workspace", str(workspace))
    scan = workspace / "smoke"
    if created.returncode != 0:
        raise SmokeError("Scan konnte nicht angelegt werden.")
    added = _cli("flight", "add", str(scan), str(clip), "--workspace", str(workspace))
    if added.returncode != 0:
        raise SmokeError("Video konnte nicht hinzugefügt werden.")
    _stage("ingest", scan, golden.profile)
    _stage("sfm", scan, golden.profile)
    splat_skipped = SPLAT_STAGE not in _registered_commands()
    if not splat_skipped:
        _stage(SPLAT_STAGE, scan, golden.profile)
    seconds = time.monotonic() - started

    manifest = load_manifest(scan / "sfm" / MANIFEST_NAME)
    if manifest is None or manifest.status != "ok":
        raise SmokeError("Manifest der Stufe 'sfm' fehlt oder ist nicht ok.")
    metrics = manifest.metrics
    peak = manifest.peak_vram_mb
    psnr: float | None = None
    if not splat_skipped:
        splat = load_manifest(scan / SPLAT_STAGE / MANIFEST_NAME)
        if splat is not None:
            psnr = splat.metrics.get("psnr_db")
            if splat.peak_vram_mb is not None:
                peak = max(peak or 0.0, splat.peak_vram_mb)
    measured = Measured(
        seconds=round(seconds, 1),
        registered_frames=int(metrics.get("registered_images", 0)),
        reproj_error_px=float(metrics.get("mean_reprojection_error_px", float("inf"))),
        points=int(metrics.get("points", 0)),
        peak_vram_mib=peak,
        psnr_db=psnr,
    )
    limit = os.environ.get("DRONESCAN_VRAM_LIMIT_GB")
    violations = evaluate(golden, measured, float(limit) if limit else None)
    return SmokeResult(measured, violations, splat_skipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Golden-data smoke test (make smoke).")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--golden", type=Path, default=None)
    args = parser.parse_args(argv)
    root: Path = args.root
    golden = load_golden(args.golden or root / GOLDEN_DEFAULT)
    workspace = Path(tempfile.mkdtemp(prefix="dronescan-smoke-"))
    try:
        result = run_smoke(root, golden, workspace)
    except SmokeError as exc:
        print(f"Smoke-Test fehlgeschlagen: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    m = result.measured
    print(
        f"Posen: {m.registered_frames} Bilder registriert, {m.reproj_error_px:.2f} px, "
        f"{m.points} Punkte, VRAM-Spitze {m.peak_vram_mib} MiB, Laufzeit {m.seconds:.0f} s."
    )
    if result.splat_skipped:
        print("Splat: übersprungen (wartet auf KAN-41, Stufe recon.splat).")
    else:
        print(f"Splat: PSNR {m.psnr_db} dB.")
    for violation in result.violations:
        print(f"Abweichung: {violation}", file=sys.stderr)
    return 1 if result.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
