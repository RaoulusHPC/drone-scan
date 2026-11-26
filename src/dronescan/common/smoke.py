# SPDX-License-Identifier: AGPL-3.0-or-later
"""Golden-data smoke test: sample clip -> poses -> mini splat (KAN-26, KAN-41).

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
SPLAT_STAGE = "recon"  # stage recon.splat (KAN-41)

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
    # Profile values of the mini splat (`splat` section of the profile), passed to the stage as
    # DRONESCAN_SPLAT__<NAME>; they keep poses + splat below max_seconds (KAN-41).
    overrides: dict[str, int | float | str] = {}

    def environment(self) -> dict[str, str]:
        return {
            f"DRONESCAN_SPLAT__{name.upper()}": str(value) for name, value in self.overrides.items()
        }


class Golden(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    clip: str
    profile: str
    max_seconds: float
    poses: PoseGolden
    splat: SplatGolden


class Measured(BaseModel):
    seconds: float  # poses + splat
    poses_seconds: float | None = None
    registered_frames: int
    reproj_error_px: float
    points: int
    peak_vram_mib: float | None  # highest process peak of sfm and recon
    psnr_db: float | None = None
    ssim: float | None = None
    gaussians: int | None = None
    splat_peak_vram_mib: float | None = None


@dataclass(frozen=True)
class SmokeResult:
    measured: Measured
    violations: list[str]


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


def _cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    code = "from dronescan.cli import app; app()"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )


def _stage(name: str, scan: Path, profile: str, env: dict[str, str] | None = None) -> None:
    proc = _cli(name, "--scan", str(scan), "--profile", profile, env=env)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
        hints = load_manifest(scan / name / MANIFEST_NAME)
        extra = " ".join(hints.hints) if hints else ""
        raise SmokeError(
            f"Stufe '{name}' endete mit Code {proc.returncode}: {' '.join(detail)} {extra}".strip()
        )


def run_smoke(root: Path, golden: Golden, workspace: Path) -> SmokeResult:
    """Run ingest, sfm and recon (mini splat) on the sample clip in ``workspace``."""
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
    poses_seconds = time.monotonic() - started
    _stage(SPLAT_STAGE, scan, golden.profile, env=golden.splat.environment())
    seconds = time.monotonic() - started

    manifest = load_manifest(scan / "sfm" / MANIFEST_NAME)
    if manifest is None or manifest.status != "ok":
        raise SmokeError("Manifest der Stufe 'sfm' fehlt oder ist nicht ok.")
    splat = load_manifest(scan / SPLAT_STAGE / MANIFEST_NAME)
    if splat is None or splat.status != "ok":
        raise SmokeError(f"Manifest der Stufe '{SPLAT_STAGE}' fehlt oder ist nicht ok.")
    metrics = manifest.metrics
    # only a process peak (incl. CUDA context) counts as "gesamt"; the allocator fallback is low
    splat_peak = splat.peak_vram_mb if splat.metrics.get("peak_vram_source") == "process" else None
    peaks = [p for p in (manifest.peak_vram_mb, splat_peak) if p is not None]
    measured = Measured(
        seconds=round(seconds, 1),
        poses_seconds=round(poses_seconds, 1),
        registered_frames=int(metrics.get("registered_images", 0)),
        reproj_error_px=float(metrics.get("mean_reprojection_error_px", float("inf"))),
        points=int(metrics.get("points", 0)),
        peak_vram_mib=max(peaks) if len(peaks) == 2 else None,  # both stages must be measured
        psnr_db=splat.metrics.get("psnr_db"),
        ssim=splat.metrics.get("ssim"),
        gaussians=splat.metrics.get("gaussians"),
        splat_peak_vram_mib=splat_peak,
    )
    limit = os.environ.get("DRONESCAN_VRAM_LIMIT_GB")
    violations = evaluate(golden, measured, float(limit) if limit else None)
    return SmokeResult(measured, violations)


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
        f"{m.points} Punkte; VRAM-Spitze gesamt {m.peak_vram_mib} MiB."
    )
    print(
        f"Splat: PSNR {m.psnr_db} dB, SSIM {m.ssim}, {m.gaussians} Gaussians, "
        f"VRAM-Spitze {m.splat_peak_vram_mib} MiB (Prozess inkl. CUDA-Kontext); "
        f"Posen {m.poses_seconds:.0f} s, Posen + Splat {m.seconds:.0f} s."
    )
    for violation in result.violations:
        print(f"Abweichung: {violation}", file=sys.stderr)
    return 1 if result.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
