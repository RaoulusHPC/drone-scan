# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage `compress`: SPZ scene with at most 1.5 M Gaussians for phones (KAN-42, US-027).

Synthetic splats only (no GPU, no private clips). AC 2 (≥ 30 fps on a mid-range Android,
Q-7) needs a real device and the viewer (KAN-43) and is measured by hand.
"""

from __future__ import annotations

import gzip
import struct
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from dronescan.cli import app
from dronescan.common.config import Settings, load_settings
from dronescan.common.stage import ExitCode, Manifest, StageResult, run_stage
from dronescan.export.prune import cap, drop_transparent, reduce_sh
from dronescan.export.spz import decode, encode, read_spz, to_rub
from dronescan.export.stage import SCENE_NAME, CompressStage
from dronescan.recon.scene import Splats, read_ply, write_ply

runner = CliRunner()
POSITION_STEP = 1 / 4096


def make_splats(count: int, sh_degree: int = 3, seed: int = 0) -> Splats:
    rng = np.random.default_rng(seed)
    rest = (sh_degree + 1) ** 2 - 1
    quats = rng.normal(size=(count, 4)).astype(np.float32)
    return Splats(
        means=rng.uniform(-5, 5, (count, 3)).astype(np.float32),
        sh0=rng.uniform(-1.5, 1.5, (count, 1, 3)).astype(np.float32),
        sh_rest=rng.uniform(-0.5, 0.5, (count, rest, 3)).astype(np.float32),
        opacity_logits=rng.uniform(-4, 4, count).astype(np.float32),
        log_scales=rng.uniform(-6, -1, (count, 3)).astype(np.float32),
        quats_wxyz=quats / np.linalg.norm(quats, axis=1, keepdims=True),
    )


def save(path: Path, splats: Splats) -> None:
    write_ply(
        path,
        means=splats.means,
        sh0=splats.sh0,
        sh_rest=splats.sh_rest,
        opacity_logits=splats.opacity_logits,
        log_scales=splats.log_scales,
        quats_wxyz=splats.quats_wxyz,
    )


def rotation(q: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def sh_basis(d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Real SH basis of degree 1-3 without the DC term, order and signs as in 3DGS."""
    x, y, z = d
    xx, yy, zz = x * x, y * y, z * z
    return np.array(
        [
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2 * zz - xx - yy),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (xx - yy),
            -0.5900435899266435 * y * (3 * xx - yy),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (4 * zz - xx - yy),
            0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
            -0.4570457994644658 * x * (4 * zz - xx - yy),
            1.445305721320277 * z * (xx - yy),
            -0.5900435899266435 * x * (xx - 3 * yy),
        ]
    )


# ---------- AC 1: SPZ format ----------


def test_spz_has_the_reference_header_and_layout() -> None:
    """Given one Gaussian with known values (SH degree 1),
    When it is encoded,
    Then the gzip content is the SPZ v2 header followed by the quantised attributes.
    """
    splats = Splats(
        means=np.array([[1.0, 0.5, -2.0]], np.float32),  # COLMAP frame
        sh0=np.zeros((1, 1, 3), np.float32),
        sh_rest=np.zeros((1, 3, 3), np.float32),
        opacity_logits=np.zeros(1, np.float32),  # sigmoid = 0.5
        log_scales=np.full((1, 3), -2.0, np.float32),
        quats_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
    )

    raw = gzip.decompress(encode(splats))

    assert struct.unpack_from("<IIIBBBB", raw) == (0x5053474E, 2, 1, 1, 12, 0, 0)
    body = raw[16:]
    rub = [round(v * 4096) & 0xFFFFFF for v in (1.0, -0.5, 2.0)]  # y and z flipped
    assert body[:9] == b"".join(v.to_bytes(3, "little") for v in rub)
    assert body[9] == 128  # alpha
    assert body[10:13] == bytes([128] * 3)  # colour of SH DC 0
    assert body[13:16] == bytes([128] * 3)  # (-2 + 10) * 16
    assert body[16:19] == bytes([128] * 3)  # identity rotation: x = y = z = 0
    assert body[19:28] == bytes([128] * 9)  # SH degree 1, zero
    assert len(body) == 28


def test_encoding_is_deterministic() -> None:
    """Given the same splats twice, When encoded, Then the files are byte-identical (cache)."""
    splats = make_splats(100)

    assert encode(splats) == encode(splats)


def test_round_trip_stays_within_the_quantisation() -> None:
    """Given random splats of SH degree 3,
    When they are written as SPZ and read back,
    Then every attribute is within its 8/24-bit quantisation step.
    """
    splats = make_splats(2000)

    back = decode(encode(splats))

    assert len(back) == 2000 and back.sh_degree == 3
    np.testing.assert_allclose(back.means, splats.means, atol=POSITION_STEP)
    np.testing.assert_allclose(back.sh0, splats.sh0, atol=0.5 / (0.15 * 255) + 1e-6)
    np.testing.assert_allclose(back.log_scales, splats.log_scales, atol=0.5 / 16 + 1e-6)
    alpha = 1 / (1 + np.exp(-splats.opacity_logits))
    np.testing.assert_allclose(1 / (1 + np.exp(-back.opacity_logits)), alpha, atol=0.5 / 255)
    np.testing.assert_allclose(back.sh_rest[:, :3], splats.sh_rest[:, :3], atol=4.5 / 128)
    np.testing.assert_allclose(back.sh_rest[:, 3:], splats.sh_rest[:, 3:], atol=8.5 / 128)
    # q and -q are the same rotation
    dots = np.abs(np.sum(back.quats_wxyz * splats.quats_wxyz, axis=1))
    assert dots.min() > 0.99


def test_conversion_to_rub_keeps_the_scene_geometry() -> None:
    """Given splats in the COLMAP frame (x right, y down, z forward),
    When converted to SPZ's RUB frame,
    Then positions, covariances and view-dependent colours are those of the scene
    rotated by 180° about x.
    """
    splats = make_splats(50)
    flip = np.diag([1.0, -1.0, -1.0])

    rub = to_rub(splats)

    np.testing.assert_allclose(rub.means, splats.means @ flip, atol=1e-6)
    rng = np.random.default_rng(1)
    for i in range(10):
        scales = np.diag(np.exp(splats.log_scales[i].astype(np.float64)) ** 2)
        r = rotation(splats.quats_wxyz[i].astype(np.float64))
        r_rub = rotation(rub.quats_wxyz[i].astype(np.float64))
        np.testing.assert_allclose(
            r_rub @ scales @ r_rub.T, flip @ r @ scales @ r.T @ flip, atol=1e-6
        )
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        colour = sh_basis(d) @ splats.sh_rest[i].astype(np.float64)
        colour_rub = sh_basis(flip @ d) @ rub.sh_rest[i].astype(np.float64)
        np.testing.assert_allclose(colour_rub, colour, atol=1e-5)
    np.testing.assert_allclose(to_rub(rub).means, splats.means)


def test_invalid_spz_is_rejected() -> None:
    """Given broken content, When decoded, Then a ValueError names the problem."""
    with pytest.raises(ValueError, match="gzip"):
        decode(b"not gzip")
    with pytest.raises(ValueError, match="size"):
        decode(gzip.compress(gzip.decompress(encode(make_splats(3)))[:-1]))


def test_ply_round_trip(tmp_path: Path) -> None:
    """Given splats written by the trainer's PLY writer, When read back, Then they are equal."""
    splats = make_splats(20, sh_degree=2)
    save(tmp_path / "scene.ply", splats)

    back = read_ply(tmp_path / "scene.ply")

    assert back.sh_degree == 2
    np.testing.assert_allclose(back.sh_rest, splats.sh_rest)
    np.testing.assert_allclose(back.means, splats.means)
    np.testing.assert_allclose(back.quats_wxyz, splats.quats_wxyz, atol=1e-6)


# ---------- AC 1: pruning to <= 1.5 M Gaussians ----------


def test_cap_keeps_the_most_important_gaussians_in_order() -> None:
    """Given 1000 Gaussians of which 100 are large and opaque,
    When capped at 100,
    Then exactly those 100 remain, in their original order.
    """
    splats = make_splats(1000)
    important = np.sort(np.random.default_rng(2).choice(1000, 100, replace=False))
    splats.opacity_logits[important] = 6.0
    splats.log_scales[important] = 0.0

    kept = cap(splats, 100)

    assert len(kept) == 100
    np.testing.assert_array_equal(kept.means, splats.means[important])


def test_cap_and_sh_reduction_are_no_ops_when_within_limits() -> None:
    splats = make_splats(10, sh_degree=1)

    assert cap(splats, 10) is splats
    assert reduce_sh(splats, 3) is splats


def test_transparent_gaussians_are_dropped() -> None:
    """Given Gaussians with opacity 0.001 and 0.5, When filtered at 0.005, Then only 0.5 stays."""
    splats = make_splats(4)
    splats.opacity_logits[:] = [-6.9, 0.0, -6.9, 0.0]

    assert len(drop_transparent(splats, 0.005)) == 2


def test_sh_reduction_keeps_the_low_degrees() -> None:
    splats = make_splats(5, sh_degree=3)

    reduced = reduce_sh(splats, 1)

    assert reduced.sh_degree == 1
    np.testing.assert_array_equal(reduced.sh_rest, splats.sh_rest[:, :3])
    assert reduce_sh(splats, 0).sh_rest.shape == (5, 0, 3)


# ---------- AC 1: stage contract ----------


def write_recon(scan: Path, splats: Splats, status: str = "ok") -> None:
    now = datetime.now(UTC)
    manifest = Manifest(
        stage="recon",
        version="1",
        profile="low_vram_6gb",
        inputs={},
        params={},
        tools={},
        started=now,
        finished=now,
        peak_vram_mb=None,
        status=status,  # type: ignore[arg-type]
        exit_code=0 if status == "ok" else 1,
    )
    (scan / "recon" / "splat").mkdir(parents=True, exist_ok=True)
    (scan / "recon" / "manifest.json").write_text(manifest.model_dump_json())
    save(scan / "recon" / "splat" / "scene.ply", splats)


def settings(**compress: object) -> Settings:
    base = load_settings("low_vram_6gb")
    return base.model_copy(
        update={"compress": base.compress.model_copy(update=compress)}, deep=True
    )


def compress(scan: Path, force: bool = False, **values: object) -> StageResult:
    return run_stage(CompressStage(), scan, settings(**values), force=force)


@pytest.fixture
def scan(tmp_path: Path) -> Path:
    scan = tmp_path / "scan"
    scan.mkdir()
    write_recon(scan, make_splats(3000, sh_degree=3))
    return scan


def test_profiles_cap_at_1_5_million_gaussians() -> None:
    """Given the shipped profiles, Then the scene for phones has at most 1.5 M Gaussians (Q-7)."""
    for name in ("low_vram_6gb", "default_8gb"):
        assert load_settings(name).compress.max_gaussians <= 1_500_000


def test_stage_writes_a_capped_spz_and_reports_metrics(scan: Path) -> None:
    """Given a trained scene with 3000 Gaussians,
    When `compress` runs with a cap of 1000 and SH degree 1,
    Then compress/scene.spz holds 1000 Gaussians of degree 1 and the manifest the counts.
    """
    result = compress(scan, max_gaussians=1000, sh_degree=1)

    assert result.exit_code is ExitCode.OK, result.manifest.error
    scene = read_spz(scan / "compress" / SCENE_NAME)
    assert len(scene) == 1000 and scene.sh_degree == 1
    metrics = result.manifest.metrics
    assert metrics["gaussians_in"] == 3000 and metrics["gaussians_out"] == 1000
    assert metrics["gaussians_in"] == (
        metrics["gaussians_transparent"] + metrics["gaussians_over_cap"] + 1000
    )
    assert metrics["sh_degree_in"] == 3 and metrics["sh_degree_out"] == 1
    assert metrics["spz_bytes"] == (scan / "compress" / SCENE_NAME).stat().st_size
    assert metrics["compression_ratio"] > 4
    assert result.manifest.peak_vram_mb is None  # CPU only


def test_valid_manifest_is_a_cache_hit_and_changed_values_recompress(scan: Path) -> None:
    """Given a successful run,
    When it runs again, with --force and with another cap,
    Then it is skipped first and recomputed in the other two cases.
    """
    compress(scan)

    assert compress(scan).skipped
    assert not compress(scan, force=True).skipped
    assert not compress(scan, max_gaussians=10).skipped


def test_retrained_scene_invalidates_the_cache(scan: Path) -> None:
    compress(scan)
    write_recon(scan, make_splats(3000, seed=1))

    assert not compress(scan).skipped


@pytest.mark.parametrize("status", [None, "error"])
def test_missing_scene_exit_10(tmp_path: Path, status: str | None) -> None:
    """Given a scan where `recon` has not run successfully,
    When `compress` runs,
    Then it exits with 10 and tells the user to run `recon` first.
    """
    scan = tmp_path / "scan"
    scan.mkdir()
    if status:
        write_recon(scan, make_splats(10), status=status)

    result = compress(scan)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert result.manifest.error_code == "compress.recon_missing"
    assert any("recon" in hint for hint in result.manifest.hints)


def test_scene_without_visible_gaussians_is_quality_insufficient(tmp_path: Path) -> None:
    scan = tmp_path / "scan"
    scan.mkdir()
    splats = make_splats(10)
    splats.opacity_logits[:] = -10.0
    write_recon(scan, splats)

    result = compress(scan)

    assert result.exit_code is ExitCode.QUALITY_INSUFFICIENT
    assert result.manifest.error_code == "compress.empty"
    assert result.manifest.hints


def test_broken_ply_is_reported(scan: Path) -> None:
    ply = scan / "recon" / "splat" / "scene.ply"
    ply.write_bytes(ply.read_bytes()[:500])

    result = compress(scan)

    assert result.exit_code is ExitCode.ERROR
    assert result.manifest.error_code == "compress.scene_unreadable"


@pytest.mark.usefixtures("block_network")
def test_stage_runs_without_network(scan: Path) -> None:
    assert compress(scan).exit_code is ExitCode.OK


def test_cli_command_uses_environment_overrides(
    scan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given DRONESCAN_COMPRESS__MAX_GAUSSIANS=500,
    When `drone-scan compress --scan` runs,
    Then the SPZ has 500 Gaussians and the command exits with 0.
    """
    monkeypatch.setenv("DRONESCAN_COMPRESS__MAX_GAUSSIANS", "500")

    outcome = runner.invoke(app, ["compress", "--scan", str(scan)])

    assert outcome.exit_code == 0, outcome.output
    assert len(read_spz(scan / "compress" / SCENE_NAME)) == 500
