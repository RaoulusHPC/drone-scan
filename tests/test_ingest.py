# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage `ingest`: frame extraction with sharpness and duplicate filter (KAN-33, US-018).

All videos are synthetic (numpy frames encoded with ffmpeg); private clips are never used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from dronescan.capture.scan import add_flight, create_scan
from dronescan.cli import app
from dronescan.common.config import Settings, load_settings
from dronescan.common.stage import ExitCode, Manifest, StageResult, run_stage
from dronescan.ingest.ffmpeg import FfmpegError, decode_gray, extract_jpegs, fit_size
from dronescan.ingest.selection import Reason, Sample, Selection, select_keyframes
from dronescan.ingest.sharpness import laplacian_variance, similarity, thumbnail
from dronescan.ingest.stage import IngestStage

Frame = NDArray[np.uint8]

WIDTH, HEIGHT, FPS = 320, 180, 12
SETTINGS_FPS = 6.0

runner = CliRunner()


@pytest.fixture(scope="session")
def require_ffmpeg() -> None:
    """Skip locally without ffmpeg/ffprobe; fail in CI so the tests cannot silently vanish."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if os.environ.get("CI") == "true":
        pytest.fail("ffmpeg/ffprobe fehlen in der CI (ci.yml: Schritt 'Install ffmpeg').")
    pytest.skip("ffmpeg/ffprobe nicht installiert")


needs_ffmpeg = pytest.mark.usefixtures("require_ffmpeg")


# ---------- synthetic data ----------


def blur(image: NDArray[Any], passes: int) -> NDArray[np.float32]:
    data: NDArray[np.float32] = image.astype(np.float32)
    for _ in range(passes):
        for axis in (0, 1):
            total = np.roll(data, 1, axis) + data + np.roll(data, -1, axis)
            data = (total / 3).astype(np.float32)
    return data


def texture(width: int = 4 * WIDTH, height: int = HEIGHT, seed: int = 0) -> Frame:
    """Smooth random texture (survives H.264), 0..255."""
    noise = np.random.default_rng(seed).random((height, width))
    smooth = blur(noise, 4)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return (smooth * 255).astype(np.uint8)


def panning(count: int, step: int = 6) -> list[Frame]:
    """Camera moving along a textured wall."""
    wall = texture(width=WIDTH + count * step)
    return [wall[:, i * step : i * step + WIDTH] for i in range(count)]


def write_video(path: Path, frames: list[Frame], fps: float = FPS, rotation: int = 0) -> Path:
    """Encode grey frames losslessly; ``rotation`` sets the (counter-clockwise) display flag."""
    height, width = frames[0].shape
    raw = path.with_name(f"raw_{path.name}")
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray"]
    command += ["-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
    command += ["-c:v", "libx264", "-qp", "4", "-pix_fmt", "yuv420p", str(raw)]
    subprocess.run(command, input=b"".join(f.tobytes() for f in frames), check=True)
    if not rotation:
        return raw.rename(path)
    flag = ["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotation)]
    subprocess.run([*flag, "-i", str(raw), "-c", "copy", str(path)], check=True)
    return path


def read_gray(image: Path) -> Frame:
    """Decode a JPEG to grey (size from ffprobe)."""
    probe = ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0"]
    size = subprocess.run([*probe, str(image)], capture_output=True, text=True, check=True)
    width, height = (int(v) for v in size.stdout.strip().split(","))
    command = ["ffmpeg", "-v", "error", "-i", str(image), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


def make_scan(tmp_path: Path, *videos: Path) -> Path:
    workspace = tmp_path / "workspace"
    scan_dir = workspace / create_scan(workspace, "Synthetik").id
    for video in videos:
        add_flight(scan_dir, video)
    return scan_dir


def ingest(scan_dir: Path, settings: Settings | None = None, force: bool = False) -> StageResult:
    return run_stage(IngestStage(), scan_dir, settings or load_settings(), force=force)


def with_ingest(**values: Any) -> Settings:
    settings = load_settings()
    return settings.model_copy(update={"ingest": settings.ingest.model_copy(update=values)})


def rows(scan_dir: Path, flight: str = "f01") -> list[dict[str, Any]]:
    table = pq.read_table(scan_dir / "ingest" / flight / "frames.parquet")
    result: list[dict[str, Any]] = table.to_pylist()
    return result


def images(scan_dir: Path, flight: str = "f01") -> list[str]:
    return sorted(p.name for p in (scan_dir / "ingest" / flight / "frames").iterdir())


@pytest.fixture(scope="session")
def moving_video(require_ffmpeg: None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3 s of steady camera motion."""
    return write_video(tmp_path_factory.mktemp("moving") / "moving.mp4", panning(3 * FPS))


# ---------- AC1: sharpness ----------


def test_blur_lowers_the_sharpness() -> None:
    """Given a frame and blurred copies, When measured, Then more blur means a lower score."""
    frame = texture()
    scores = [laplacian_variance(blur(frame, n).astype(np.uint8)) for n in (0, 2, 6)]

    assert scores[0] > scores[1] > scores[2]


def test_laplacian_variance_of_known_image() -> None:
    """Given a single bright pixel, When measured, Then the variance matches the kernel."""
    frame = np.zeros((7, 7), dtype=np.uint8)
    frame[3, 3] = 10
    expected = np.zeros((5, 5))
    expected[2, 2] = 40
    for row, col in ((1, 2), (3, 2), (2, 1), (2, 3)):
        expected[row, col] = -10

    assert laplacian_variance(frame) == pytest.approx(expected.var())


def test_sharpness_does_not_overflow() -> None:
    """Given extreme contrast, When measured, Then uint8 arithmetic does not wrap around."""
    checker = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)

    assert laplacian_variance(checker) == pytest.approx((8 * 255 / 2) ** 2, rel=0.01)


def samples_of(scores: list[float], fps: float = SETTINGS_FPS) -> list[Sample]:
    """Samples with the given scores; every thumbnail is a different view."""
    rng = np.random.default_rng(1)
    thumbs = [rng.random((8, 8), dtype=np.float32) * 255 for _ in scores]
    return [Sample(i / fps, s, t) for i, (s, t) in enumerate(zip(scores, thumbs, strict=True))]


def select(samples: list[Sample], rate: float = 2.0) -> Selection:
    return select_keyframes(
        samples, keyframes_per_s=rate, min_rel_sharpness=0.4, max_similarity=0.99
    )


def test_sharpest_frame_of_each_window_is_selected() -> None:
    """Given three samples per 0.5 s window, When selected, Then the sharpest of each wins."""
    selection = select(samples_of([10, 30, 20, 25, 11, 12, 9, 8, 19]))

    assert [f.index for f in selection.keyframes] == [1, 3, 8]
    assert Counter(f.reason for f in selection.frames)[Reason.NOT_SHARPEST] == 6
    assert [f.window for f in selection.frames] == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_blurry_window_is_dropped() -> None:
    """Given a window whose best frame is below 40 % of the median, Then it has no keyframe."""
    selection = select(samples_of([20, 20, 20, 5, 7, 6, 20, 20, 20]))

    assert [f.index for f in selection.keyframes] == [0, 6]
    assert selection.frames[4].reason is Reason.BLURRY
    assert selection.sharpness_threshold == pytest.approx(8.0)


def test_threshold_adapts_to_the_video() -> None:
    """Given the same scene at a lower contrast (scores scaled),
    When selected,
    Then the same frames are chosen (threshold relative to the median of the video).
    """
    scores = [20.0, 21, 22, 5, 7, 6, 23, 20, 20]
    bright, dark = select(samples_of(scores)), select(samples_of([s * 0.01 for s in scores]))

    assert [f.reason for f in bright.frames] == [f.reason for f in dark.frames]
    assert dark.sharpness_threshold == pytest.approx(bright.sharpness_threshold * 0.01)


def test_windows_follow_timestamps_not_frame_numbers() -> None:
    """Given a gap in the timestamps (variable frame rate),
    When selected,
    Then frames after the gap belong to later windows.
    """
    base = samples_of([1.0, 2.0, 3.0])
    late = [Sample(t, s.sharpness, s.thumbnail) for t, s in zip((0.0, 0.1, 5.0), base, strict=True)]

    assert [f.window for f in select(late).frames] == [0, 0, 10]


def test_empty_video_gives_empty_selection() -> None:
    """Given no samples, When selected, Then nothing is returned (no crash)."""
    assert select([]).frames == []


# ---------- AC2: duplicates and rate ----------


def test_similarity_recognises_views() -> None:
    """Given thumbnails, When compared, Then same view ≈ 1 (also at another exposure) and a
    shifted or different view is lower.
    """
    wall = texture()
    same = thumbnail(wall[:, :WIDTH], 64)
    darker = thumbnail((wall[:, :WIDTH] * 0.6).astype(np.uint8), 64)
    shifted = thumbnail(wall[:, 20 : 20 + WIDTH], 64)

    assert same.shape == (36, 64)
    assert similarity(same, same) == pytest.approx(1.0)
    assert similarity(same, darker) > 0.995
    assert similarity(same, shifted) < 0.9


def test_similarity_of_flat_views() -> None:
    """Given structureless frames, When compared, Then flat = flat and flat ≠ structured."""
    flat = thumbnail(np.full((HEIGHT, WIDTH), 200, dtype=np.uint8), 64)
    structured = thumbnail(texture(WIDTH), 64)

    assert similarity(flat, flat) == 1.0
    assert similarity(flat, structured) == 0.0


def test_duplicates_are_compared_with_the_last_keyframe() -> None:
    """Given a camera standing still and then a new view,
    When selected,
    Then repeated views are duplicates and their similarity is recorded.
    """
    a, _, _, b = samples_of([10, 10, 10, 10], fps=2)
    samples = [Sample(i / 2, 10, t) for i, t in enumerate([a.thumbnail] * 3 + [b.thumbnail])]
    frames = select(samples).frames

    kept, duplicate = Reason.KEYFRAME, Reason.DUPLICATE
    assert [f.reason for f in frames] == [kept, duplicate, duplicate, kept]
    assert frames[0].similarity is None
    assert frames[1].similarity == pytest.approx(1.0)


@needs_ffmpeg
def test_standing_still_yields_no_redundant_keyframes(tmp_path: Path) -> None:
    """Given a video: 2 s standing still, then 2 s of motion,
    When ingested,
    Then the still part contributes one keyframe and the moving part one per window.
    """
    moving = panning(2 * FPS)
    video = write_video(tmp_path / "still.mp4", [moving[0]] * (2 * FPS) + moving)
    scan_dir = make_scan(tmp_path, video)

    assert ingest(scan_dir).exit_code is ExitCode.OK

    by_window = {r["window"]: r["reason"] for r in rows(scan_dir) if r["reason"] != "not_sharpest"}
    assert [by_window[w] for w in range(8)] == ["keyframe"] + ["duplicate"] * 3 + ["keyframe"] * 4


@needs_ffmpeg
@pytest.mark.parametrize(("rate", "expected"), [(1.0, 3), (2.0, 6), (4.0, 12)])
def test_keyframe_rate_is_configurable(
    tmp_path: Path, moving_video: Path, rate: float, expected: int
) -> None:
    """Given 3 s of motion, When ingested with 1, 2 or 4 keyframes/s, Then 3, 6 or 12 result."""
    scan_dir = make_scan(tmp_path, moving_video)

    assert ingest(scan_dir, with_ingest(keyframes_per_s=rate, analysis_fps=12)).exit_code == 0

    assert len(images(scan_dir)) == expected


# ---------- AC1: decoding and rotation ----------


@needs_ffmpeg
@pytest.mark.parametrize(
    ("flag", "size", "corner"),
    [  # ffmpeg's display_rotation is counter-clockwise; marker is top left in the encoded frame
        (0, (180, 320), "top left"),
        (90, (320, 180), "bottom left"),
        (180, (180, 320), "bottom right"),
        (270, (320, 180), "top right"),
    ],
)
def test_rotation_is_applied_exactly_once(
    tmp_path: Path, flag: int, size: tuple[int, int], corner: str
) -> None:
    """Given a portrait-encoded clip with a rotation flag and a white corner marker,
    When ingested,
    Then size and marker position match the display orientation (not rotated twice).
    """
    frames = []
    for index in range(12):
        frame = np.full((320, 180), 40, dtype=np.uint8)
        frame[:80, :60] = 250
        frame[150 + 8 * index : 170 + 8 * index, 60:120] = 120  # motion, so frames differ
        frames.append(frame)
    scan_dir = make_scan(tmp_path, write_video(tmp_path / "v.mp4", frames, 6, rotation=flag))

    assert ingest(scan_dir, with_ingest(max_similarity=2.0)).exit_code is ExitCode.OK

    image = read_gray(scan_dir / "ingest" / "f01" / "frames" / "000001.jpg")
    assert (image.shape[1], image.shape[0]) == size
    corners = {
        "top left": image[10, 10],
        "top right": image[10, -10],
        "bottom left": image[-10, 10],
        "bottom right": image[-10, -10],
    }
    assert max(corners, key=lambda name: int(corners[name])) == corner
    assert sum(int(value) > 200 for value in corners.values()) == 1


@pytest.mark.parametrize(
    ("size", "edge", "expected"),
    [
        ((1920, 1080), 1600, (1600, 900)),
        ((1080, 1920), 480, (270, 480)),
        ((320, 180), 1600, (320, 180)),
        ((1921, 3), 64, (64, 2)),
    ],
)
def test_fit_size(size: tuple[int, int], edge: int, expected: tuple[int, int]) -> None:
    """Given a frame size, When fitted, Then it is even, keeps its aspect and never grows."""
    assert fit_size(*size, edge) == expected


@needs_ffmpeg
def test_jpegs_are_limited_to_the_configured_edge(tmp_path: Path, moving_video: Path) -> None:
    """Given max_image_edge_px = 160, When ingested, Then the JPEGs are 160 x 90."""
    scan_dir = make_scan(tmp_path, moving_video)

    ingest(scan_dir, with_ingest(max_image_edge_px=160))

    image = read_gray(scan_dir / "ingest" / "f01" / "frames" / "000001.jpg")
    assert image.shape == (90, 160)


@needs_ffmpeg
def test_jpegs_show_the_selected_frames(tmp_path: Path) -> None:
    """Given frames whose brightness encodes their number,
    When some analysis frames are extracted,
    Then every JPEG shows exactly the selected frame.
    """
    frames = [np.full((HEIGHT, WIDTH), 20 + 8 * i, dtype=np.uint8) for i in range(24)]
    video = write_video(tmp_path / "v.mp4", frames, fps=12)
    seen: list[float] = []

    decode_gray(video, (WIDTH, HEIGHT), 6, lambda frame: seen.append(float(frame.mean())))
    out = tmp_path / "100% out"  # "%" must not be read as an ffmpeg image pattern
    out.mkdir()
    paths = extract_jpegs(video, [0, 3, 11], size=(WIDTH, HEIGHT), fps=6, qscale=2, out_dir=out)

    assert [p.name for p in paths] == ["000001.jpg", "000002.jpg", "000003.jpg"]
    assert len(seen) == 12
    for path, index in zip(paths, (0, 3, 11), strict=True):
        assert float(read_gray(path).mean()) == pytest.approx(seen[index], abs=3)


@needs_ffmpeg
def test_many_keyframes_fit_into_one_ffmpeg_call(tmp_path: Path) -> None:
    """Given thousands of keyframe numbers (very long video),
    When extracted,
    Then the select expression is still accepted by ffmpeg.
    """
    video = write_video(tmp_path / "v.mp4", panning(12), fps=12)
    indices = [0, 1, 2, *range(1000, 7000)]

    with pytest.raises(FfmpegError):  # later frames do not exist, but the filter was parsed
        extract_jpegs(video, indices, size=(WIDTH, HEIGHT), fps=12, qscale=2, out_dir=tmp_path)

    assert (tmp_path / "000003.jpg").is_file()


@needs_ffmpeg
def test_timestamps_of_variable_frame_rate(tmp_path: Path) -> None:
    """Given a video with a variable frame rate (1 s at 12 fps, then 2 s at 2 fps),
    When decoded,
    Then the timestamps are the real presentation times.
    """
    base = write_video(tmp_path / "cfr.mp4", panning(16), fps=12)
    video = tmp_path / "vfr.mp4"
    expression = "setpts='if(lt(N,12),N/12,1+(N-12)/2)/TB'"
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(base), "-vf", expression]
    command += ["-fps_mode", "vfr", "-c:v", "libx264", "-enc_time_base", "1/1200", str(video)]
    subprocess.run(command, check=True)

    times = decode_gray(video, (WIDTH, HEIGHT), 6, lambda frame: None)

    assert times == pytest.approx([i / 6 for i in range(6)] + [1.0, 1.5, 2.0, 2.5], abs=0.002)


@needs_ffmpeg
@pytest.mark.parametrize("fps", [29.97, 30, 25, 24])
def test_sampling_keeps_the_requested_rate(tmp_path: Path, fps: float) -> None:
    """Given usual camera frame rates, When 4 s are decoded at 6 fps, Then ≈ 24 frames result
    (no drift by rounding).
    """
    video = write_video(tmp_path / "v.mp4", panning(round(4 * fps), step=1), fps=fps)

    times = decode_gray(video, (WIDTH, HEIGHT), 6, lambda frame: None)

    assert len(times) in (24, 25)


# ---------- AC3: frames.parquet and layout ----------


@needs_ffmpeg
def test_table_has_timestamp_sharpness_and_reason_per_frame(
    tmp_path: Path, moving_video: Path
) -> None:
    """Given an ingested flight,
    When frames.parquet is read,
    Then each analysed frame has timestamp, sharpness and reason, and each keyframe a file.
    """
    scan_dir = make_scan(tmp_path, moving_video)

    result = ingest(scan_dir)

    assert result.exit_code is ExitCode.OK, result.manifest.error
    table = rows(scan_dir)
    assert len(table) == 18  # 3 s at 6 analysed frames per second
    assert [r["index"] for r in table] == list(range(18))
    assert [r["timestamp_s"] for r in table] == pytest.approx([i / 6 for i in range(18)], abs=0.001)
    assert all(r["sharpness"] > 0 for r in table)
    assert {r["reason"] for r in table} <= {reason.value for reason in Reason}
    selected = [r for r in table if r["selected"]]
    assert [r["file"] for r in selected] == [f"frames/{name}" for name in images(scan_dir)]
    assert all(r["reason"] == "keyframe" for r in selected)
    assert all(r["file"] is None for r in table if not r["selected"])


@needs_ffmpeg
def test_flight_manifest_summarises_the_flight(tmp_path: Path, moving_video: Path) -> None:
    """Given an ingested flight, When its manifest is read, Then counts and sizes match."""
    scan_dir = make_scan(tmp_path, moving_video)
    ingest(scan_dir)

    summary = json.loads((scan_dir / "ingest" / "f01" / "manifest.json").read_text())

    assert summary["video"] == "flights/f01/video.mp4"
    assert (summary["frame_width"], summary["frame_height"]) == (WIDTH, HEIGHT)
    assert summary["analysed_frames"] == sum(summary["counts"].values()) == 18
    assert summary["counts"]["keyframe"] == len(images(scan_dir))
    assert summary["sharpness_threshold"] == pytest.approx(0.4 * summary["sharpness_median"])
    assert summary["decoded_s"] == pytest.approx(3.0, abs=0.01)


# ---------- stage contract ----------


@needs_ffmpeg
def test_several_flights_and_stage_manifest(tmp_path: Path, moving_video: Path) -> None:
    """Given two flights, When ingested, Then each has its folder and one stage manifest exists."""
    scan_dir = make_scan(tmp_path, moving_video, moving_video)

    assert ingest(scan_dir).exit_code is ExitCode.OK

    assert images(scan_dir, "f01") and images(scan_dir, "f01") == images(scan_dir, "f02")
    manifest = Manifest.model_validate_json((scan_dir / "ingest" / "manifest.json").read_text())
    assert set(manifest.inputs) == {
        "scan.yaml",
        "flights/f01/video.mp4",
        "flights/f02/video.mp4",
    }
    assert manifest.params["keyframes_per_s"] == 2
    assert {"ffmpeg", "numpy", "pyarrow"} <= set(manifest.tools)
    assert manifest.peak_vram_mb is None
    assert manifest.warnings == []


@needs_ffmpeg
def test_cache_force_and_stale_outputs(tmp_path: Path, moving_video: Path) -> None:
    """Given an ingested scan,
    When it runs again unchanged, with --force and with other parameters,
    Then it is skipped, recomputed, and old files do not survive.
    """
    scan_dir = make_scan(tmp_path, moving_video)
    ingest(scan_dir, with_ingest(keyframes_per_s=4))
    many = images(scan_dir)
    assert ingest(scan_dir, with_ingest(keyframes_per_s=4)).skipped
    (scan_dir / "ingest" / "f07" / "frames").mkdir(parents=True)  # flight no longer in the scan

    assert not ingest(scan_dir, with_ingest(keyframes_per_s=4), force=True).skipped
    assert images(scan_dir) == many
    assert not (scan_dir / "ingest" / "f07").exists()
    assert not ingest(scan_dir).skipped
    assert len(images(scan_dir)) < len(many)


def test_scan_without_flight_exits_10(tmp_path: Path) -> None:
    """Given a scan without videos, When ingested, Then exit 10 names the import command."""
    result = ingest(make_scan(tmp_path))

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert any("flight add" in hint for hint in result.manifest.hints)


def test_directory_without_scan_yaml_exits_10(tmp_path: Path) -> None:
    """Given a directory that is no scan, When ingested, Then exit 10."""
    result = ingest(tmp_path)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert "scan.yaml" in (result.manifest.error or "")


def test_unreadable_scan_yaml_exits_10(tmp_path: Path) -> None:
    """Given a broken scan.yaml, When ingested, Then exit 10 instead of a crash."""
    (tmp_path / "scan.yaml").write_text("id: [", encoding="utf-8")

    assert ingest(tmp_path).exit_code is ExitCode.INPUT_MISSING


@needs_ffmpeg
@pytest.mark.parametrize("flight_id", ["../../outside", "f1/..", ""])
def test_manipulated_flight_id_is_rejected(
    tmp_path: Path, moving_video: Path, flight_id: str
) -> None:
    """Given a scan.yaml with a path as flight id,
    When ingested,
    Then exit 10 and nothing outside the stage directory is touched.
    """
    scan_dir = make_scan(tmp_path, moving_video)
    scan_file = scan_dir / "scan.yaml"
    scan_file.write_text(scan_file.read_text().replace("id: f01", f"id: '{flight_id}'"))

    assert ingest(scan_dir).exit_code is ExitCode.INPUT_MISSING


@needs_ffmpeg
def test_missing_video_exits_10(tmp_path: Path, moving_video: Path) -> None:
    """Given a flight whose video was deleted, When ingested, Then exit 10 names the file."""
    scan_dir = make_scan(tmp_path, moving_video)
    (scan_dir / "flights" / "f01" / "video.mp4").unlink()

    result = ingest(scan_dir)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert "flights/f01/video.mp4" in (result.manifest.error or "")


@needs_ffmpeg
def test_broken_video_exits_10_with_remedy(tmp_path: Path, moving_video: Path) -> None:
    """Given a video that was damaged after the import, When ingested, Then exit 10 with remedy."""
    scan_dir = make_scan(tmp_path, moving_video)
    (scan_dir / "flights" / "f01" / "video.mp4").write_bytes(b"kein video")

    result = ingest(scan_dir)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert "f01" in (result.manifest.error or "")
    assert any("neu importieren" in hint for hint in result.manifest.hints)


@needs_ffmpeg
def test_partially_decodable_video_is_reported(tmp_path: Path, moving_video: Path) -> None:
    """Given a video that decodes only partly (shorter than registered),
    When ingested,
    Then it succeeds with a warning and a hint.
    """
    scan_dir = make_scan(tmp_path, moving_video)
    scan_file = scan_dir / "scan.yaml"
    scan_file.write_text(scan_file.read_text().replace("duration_s: 3.0", "duration_s: 60.0"))

    result = ingest(scan_dir)

    assert result.exit_code is ExitCode.OK
    assert any("f01" in warning for warning in result.manifest.warnings)
    assert result.manifest.hints


@needs_ffmpeg
def test_too_few_keyframes_exit_20_with_hints(tmp_path: Path) -> None:
    """Given a video of a camera standing still,
    When ingested,
    Then exit 20 with concrete hints (one view cannot be reconstructed).
    """
    video = write_video(tmp_path / "still.mp4", [texture(WIDTH)] * (3 * FPS))

    result = ingest(make_scan(tmp_path, video))

    assert result.exit_code is ExitCode.QUALITY_INSUFFICIENT
    assert any("bewegen" in hint for hint in result.manifest.hints)


@needs_ffmpeg
def test_missing_ffmpeg_exits_1_with_remedy(
    tmp_path: Path, moving_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a system without ffmpeg, When ingested, Then the remedy names the package."""
    scan_dir = make_scan(tmp_path, moving_video)
    monkeypatch.setattr("dronescan.ingest.ffmpeg.shutil.which", lambda name: None)

    result = ingest(scan_dir)

    assert result.exit_code is ExitCode.ERROR
    assert any("apt install ffmpeg" in hint for hint in result.manifest.hints)
    assert result.manifest.tools["ffmpeg"] == "missing"


@needs_ffmpeg
def test_failed_extraction_is_reported(
    tmp_path: Path, moving_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given JPEGs that cannot be written, When ingested, Then exit 1 names the remedy."""
    scan_dir = make_scan(tmp_path, moving_video)

    def fail(*args: Any, **kwargs: Any) -> list[Path]:
        raise FfmpegError("Die Einzelbilder konnten nicht geschrieben werden.")

    monkeypatch.setattr("dronescan.ingest.stage.extract_jpegs", fail)

    result = ingest(scan_dir)

    assert result.exit_code is ExitCode.ERROR
    assert any("Speicherplatz" in hint for hint in result.manifest.hints)


@needs_ffmpeg
def test_decoding_stops_when_the_consumer_fails(moving_video: Path) -> None:
    """Given an error while frames are processed, Then it propagates and ffmpeg is stopped."""

    def fail(frame: Frame) -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        decode_gray(moving_video, (WIDTH, HEIGHT), 6, fail)


@needs_ffmpeg
def test_cli_ingest(tmp_path: Path, moving_video: Path) -> None:
    """Given a scan, When `drone-scan ingest --scan` runs, Then exit 0 and frames exist."""
    scan_dir = make_scan(tmp_path, moving_video)

    result = runner.invoke(app, ["ingest", "--scan", str(scan_dir)])

    assert result.exit_code == 0, result.output
    assert len(images(scan_dir)) == 6


def test_cli_start_does_not_import_numpy() -> None:
    """Given the CLI module, When imported, Then heavy packages stay unloaded (fast --help)."""
    code = "import sys, dronescan.cli; print({'numpy', 'pyarrow'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.stdout.strip() == "set()", result.stderr
