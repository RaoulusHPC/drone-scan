"""Scan creation and flight import (KAN-28, US-013).

All videos are synthetic (ffmpeg ``testsrc``); private PO clips are never used.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from dronescan.capture.probe import ProbeError, _rotation, probe_video
from dronescan.cli import app

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe nicht installiert",
)

runner = CliRunner()


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("synthetic") / "clip.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x180:rate=10:duration=2",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )
    return path


@pytest.fixture(scope="module")
def rotated_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Portrait-encoded clip (180x320) with a 90 degree display rotation flag."""
    base = tmp_path_factory.mktemp("synthetic_rot")
    raw = base / "raw.mp4"
    out = base / "rotated.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=180x320:rate=10:duration=2",
        "-pix_fmt",
        "yuv420p",
        str(raw),
    )
    _ffmpeg("-display_rotation", "90", "-i", str(raw), "-c", "copy", str(out))
    return out


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


def _create(ws: Path, name: str = "Haus Test") -> str:
    result = runner.invoke(app, ["scan", "create", name, "--workspace", str(ws)])
    assert result.exit_code == 0, result.output
    return result.output.strip().splitlines()[-1].split()[-1]


def test_scan_create_writes_scan_yaml(ws: Path) -> None:
    """Given an empty workspace,
    When `scan create` runs,
    Then workspace/<id>/scan.yaml exists with the name.
    """
    scan_id = _create(ws)

    data = yaml.safe_load((ws / scan_id / "scan.yaml").read_text(encoding="utf-8"))
    assert data["name"] == "Haus Test"
    assert data["id"] == scan_id
    assert data["flights"] == []


def test_scan_create_same_name_gets_unique_id(ws: Path) -> None:
    """Given an existing scan, When the same name is used again, Then a new directory is made."""
    first = _create(ws)
    second = _create(ws)

    assert first != second
    assert (ws / first / "scan.yaml").exists()
    assert (ws / second / "scan.yaml").exists()


def test_scan_create_rejects_empty_name(ws: Path) -> None:
    """Given a name without usable characters, When `scan create` runs, Then it fails."""
    result = runner.invoke(app, ["scan", "create", "///", "--workspace", str(ws)])

    assert result.exit_code != 0
    assert not ws.exists() or not any(ws.iterdir())


def test_flight_add_copies_video_and_stores_metadata(ws: Path, video: Path) -> None:
    """Given a scan, When a video is added, Then it is copied and ffprobe metadata is stored."""
    scan_id = _create(ws)

    result = runner.invoke(app, ["flight", "add", scan_id, str(video), "--workspace", str(ws)])

    assert result.exit_code == 0, result.output
    scan_dir = ws / scan_id
    assert (scan_dir / "flights" / "f01" / "video.mp4").read_bytes() == video.read_bytes()
    flight = yaml.safe_load((scan_dir / "scan.yaml").read_text(encoding="utf-8"))["flights"][0]
    assert flight["id"] == "f01"
    assert (flight["display_width"], flight["display_height"]) == (320, 180)
    assert flight["rotation"] == 0
    assert flight["fps"] == pytest.approx(10.0)
    assert flight["duration_s"] == pytest.approx(2.0, abs=0.2)
    assert flight["codec"] == "h264"
    assert len(flight["sha256"]) == 64


def test_second_flight_gets_next_id(ws: Path, video: Path) -> None:
    """Given a scan with one flight, When another is added, Then it becomes f02."""
    scan_id = _create(ws)
    for _ in range(2):
        result = runner.invoke(app, ["flight", "add", scan_id, str(video), "--workspace", str(ws)])
        assert result.exit_code == 0, result.output

    flights = yaml.safe_load((ws / scan_id / "scan.yaml").read_text(encoding="utf-8"))["flights"]
    assert [f["id"] for f in flights] == ["f01", "f02"]
    assert (ws / scan_id / "flights" / "f02" / "video.mp4").exists()


def test_rotation_flag_is_applied_to_display_size(rotated_video: Path) -> None:
    """Given a portrait-encoded video with a 90 degree flag (smartphone),
    When it is probed,
    Then encoded size is kept and the display size is landscape.
    """
    meta = probe_video(rotated_video)

    assert (meta.width, meta.height) == (180, 320)
    assert meta.rotation == 270  # ffmpeg -display_rotation 90 is counter-clockwise
    assert (meta.display_width, meta.display_height) == (320, 180)


def test_rotation_180_keeps_size(tmp_path: Path) -> None:
    """Given a 180 degree clip, When probed, Then width and height are not swapped."""
    raw, out = tmp_path / "raw.mp4", tmp_path / "out.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=180x320:rate=10:duration=1",
        "-pix_fmt",
        "yuv420p",
        str(raw),
    )
    _ffmpeg("-display_rotation", "180", "-i", str(raw), "-c", "copy", str(out))

    meta = probe_video(out)

    assert meta.rotation == 180
    assert (meta.display_width, meta.display_height) == (180, 320)


@pytest.mark.parametrize(("tag", "expected"), [("90", 90), ("270", 270), ("0", 0), ("x", 0)])
def test_legacy_rotate_tag(tag: str, expected: int) -> None:
    """Given an older ffprobe output with tags.rotate, Then it is read as clockwise degrees."""
    assert _rotation({"tags": {"rotate": tag}}) == expected


def test_flight_add_stores_display_size_for_rotated_video(ws: Path, rotated_video: Path) -> None:
    """Given a rotated smartphone clip, When imported, Then scan.yaml holds the display size."""
    scan_id = _create(ws)

    result = runner.invoke(
        app, ["flight", "add", scan_id, str(rotated_video), "--workspace", str(ws)]
    )

    assert result.exit_code == 0, result.output
    flight = yaml.safe_load((ws / scan_id / "scan.yaml").read_text(encoding="utf-8"))["flights"][0]
    assert (flight["width"], flight["height"]) == (180, 320)
    assert (flight["display_width"], flight["display_height"]) == (320, 180)


def test_flight_add_rejects_non_video(ws: Path, tmp_path: Path) -> None:
    """Given a text file named .mp4, When imported, Then exit 10 with a clear message."""
    scan_id = _create(ws)
    fake = tmp_path / "fake.mp4"
    fake.write_text("kein video", encoding="utf-8")

    result = runner.invoke(app, ["flight", "add", scan_id, str(fake), "--workspace", str(ws)])

    assert result.exit_code == 10
    assert "Video" in result.output
    assert not (ws / scan_id / "flights").exists()


def test_flight_add_rejects_truncated_video(ws: Path, video: Path, tmp_path: Path) -> None:
    """Given a truncated video, When imported, Then exit 10 and nothing is left behind."""
    scan_id = _create(ws)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(video.read_bytes()[:200])

    result = runner.invoke(app, ["flight", "add", scan_id, str(broken), "--workspace", str(ws)])

    assert result.exit_code == 10
    assert not (ws / scan_id / "flights").exists()
    flights = yaml.safe_load((ws / scan_id / "scan.yaml").read_text(encoding="utf-8"))["flights"]
    assert flights == []


def test_flight_add_rejects_audio_only_file(ws: Path, tmp_path: Path) -> None:
    """Given a file without a video stream, When imported, Then exit 10."""
    scan_id = _create(ws)
    audio = tmp_path / "ton.m4a"
    _ffmpeg("-f", "lavfi", "-i", "sine=duration=1", str(audio))

    result = runner.invoke(app, ["flight", "add", scan_id, str(audio), "--workspace", str(ws)])

    assert result.exit_code == 10


def test_flight_add_missing_file_and_missing_scan(ws: Path, video: Path, tmp_path: Path) -> None:
    """Given a missing video or a missing scan, When imported, Then exit 10."""
    scan_id = _create(ws)

    missing_video = runner.invoke(
        app, ["flight", "add", scan_id, str(tmp_path / "nope.mp4"), "--workspace", str(ws)]
    )
    missing_scan = runner.invoke(
        app, ["flight", "add", "gibt-es-nicht", str(video), "--workspace", str(ws)]
    )

    assert missing_video.exit_code == 10
    assert missing_scan.exit_code == 10
    assert "scan create" in missing_scan.output


def test_probe_error_for_garbage(tmp_path: Path) -> None:
    """Given garbage bytes, When probed, Then ProbeError is raised."""
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"\x00" * 64)

    with pytest.raises(ProbeError):
        probe_video(bad)


def test_flight_add_copy_failure_gives_exit_10(
    ws: Path, video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a failing copy (disk full), When imported, Then exit 10 and no leftovers."""
    scan_id = _create(ws)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("dronescan.capture.scan.shutil.copyfile", boom)

    result = runner.invoke(app, ["flight", "add", scan_id, str(video), "--workspace", str(ws)])

    assert result.exit_code == 10
    assert not (ws / scan_id / "flights").exists()


def test_flight_add_skips_id_taken_by_parallel_import(ws: Path, video: Path) -> None:
    """Given f01 already exists on disk but not in scan.yaml, Then the next import gets f02."""
    scan_id = _create(ws)
    (ws / scan_id / "flights" / "f01").mkdir(parents=True)

    result = runner.invoke(app, ["flight", "add", scan_id, str(video), "--workspace", str(ws)])

    assert result.exit_code == 0, result.output
    assert (ws / scan_id / "flights" / "f02" / "video.mp4").exists()
    assert (ws / scan_id / "flights" / "f01").exists()
