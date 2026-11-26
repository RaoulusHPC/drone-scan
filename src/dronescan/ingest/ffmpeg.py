# SPDX-License-Identifier: AGPL-3.0-or-later
"""ffmpeg adapter: decode analysis frames and extract keyframes as JPEG.

ffmpeg applies the display rotation itself (autorotate), so callers pass sizes in display
orientation and never rotate again. Both passes use the same sampler, so frame numbers of
the analysis pass address exactly the same source frames in the extraction pass, also for
videos with a variable frame rate.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_PTS = re.compile(r"showinfo.*\bn:\s*\d+\s+pts:\s*(\S+)")
_US_PER_S = 1_000_000
_GRID_EPS = 1e-3  # keeps frames that lie exactly on the sampling grid (float rounding)
_VERSION = re.compile(r"^ffmpeg version (\S+)")
_JPEG_PATTERN = "%06d.jpg"

log = structlog.get_logger(__name__)


class FfmpegError(Exception):
    """ffmpeg failed; the message is user-facing (German) and contains no paths."""


class FfmpegMissingError(FfmpegError):
    """ffmpeg is not installed."""


def ffmpeg_version() -> str:
    if shutil.which("ffmpeg") is None:
        return "missing"
    proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
    match = _VERSION.match(proc.stdout)
    return match.group(1) if match else "unknown"


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegMissingError("ffmpeg ist nicht installiert.")


def fit_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """Even size with the aspect ratio of ``width`` x ``height``; never scales up."""
    scale = min(1.0, max_edge / max(width, height))
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def _sampler(fps: float) -> str:
    """Select the first frame in every cell of a ``1/fps`` time grid."""
    cell = f"floor({{}}*{fps:g}+{_GRID_EPS:g})"
    previous = cell.format("prev_selected_t")
    return f"select='isnan(prev_selected_t)+gt({cell.format('t')},{previous})'"


def _input_args(video: Path) -> list[str]:
    args = ["-nostdin", "-hide_banner", "-nostats"]
    # A change of resolution inside the video must not restart the filters: the frame numbers
    # of the two passes would no longer match.
    args += ["-reinit_filter", "0", "-i", str(video.resolve())]
    return [*args, "-map", "0:V:0"]  # "V" skips cover art, like probe_video


def _environment() -> dict[str, str]:
    return {**os.environ, "AV_LOG_FORCE_NOCOLOR": "1"}  # the showinfo output is parsed


def _timestamps(output: IO[str]) -> list[float]:
    output.seek(0)
    times = []
    for line in output:
        if match := _PTS.search(line):
            try:
                times.append(int(match.group(1)) / _US_PER_S)
            except ValueError as exc:
                raise FfmpegError("Das Video enthält Bilder ohne Zeitstempel.") from exc
    return times


def decode_gray(
    video: Path,
    size: tuple[int, int],
    fps: float,
    on_frame: Callable[[NDArray[np.uint8]], None],
) -> list[float]:
    """Decode ``fps`` grey frames per second; returns the timestamp (s) of each frame.

    ``on_frame`` is called with one [height, width] array per frame (streaming; frames are
    not kept here).
    """
    import numpy as np

    _require_ffmpeg()
    width, height = size
    # AVTB (microseconds) keeps timestamps exact: showinfo prints seconds with 6 digits only.
    filters = f"{_sampler(fps)},scale={width}:{height},format=gray,settb=AVTB,showinfo"
    command = ["ffmpeg", "-loglevel", "info", *_input_args(video), "-vf", filters]
    command += ["-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    frame_bytes = width * height
    count = 0
    # stderr goes to a file: a full pipe would block ffmpeg while we read stdout.
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as output:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=output, env=_environment()
        ) as proc:
            assert proc.stdout is not None
            try:
                while len(data := proc.stdout.read(frame_bytes)) == frame_bytes:
                    on_frame(np.frombuffer(data, dtype=np.uint8).reshape(height, width))
                    count += 1
            except BaseException:
                proc.kill()
                raise
        if proc.returncode != 0 or count == 0:
            log.error("ffmpeg_decode_failed", returncode=proc.returncode, frames=count)
            raise FfmpegError("Das Video kann nicht dekodiert werden.")
        times = _timestamps(output)
    if len(times) != count:
        raise FfmpegError("ffmpeg lieferte nicht für jedes Bild einen Zeitstempel.")
    return times


def extract_jpegs(
    video: Path,
    indices: Sequence[int],
    *,
    size: tuple[int, int],
    fps: float,
    qscale: int,
    out_dir: Path,
) -> list[Path]:
    """Write analysis frames ``indices`` (ascending) as ``000001.jpg``, ... to empty ``out_dir``."""
    _require_ffmpeg()
    if not indices:
        return []
    width, height = size
    wanted = "+".join(f"eq(n,{index})" for index in indices)
    # JPEG (JFIF) means full range and BT.601; HD video is usually limited range BT.709.
    scale = f"scale={width}:{height}:flags=lanczos:out_color_matrix=bt601:out_range=pc"
    command = ["ffmpeg", "-loglevel", "error", *_input_args(video)]
    command += ["-vf", f"{_sampler(fps)},select='{wanted}',{scale}", "-fps_mode", "passthrough"]
    command += ["-map_metadata", "-1", "-c:v", "mjpeg", "-pix_fmt", "yuvj420p"]
    # Relative pattern + cwd: a "%" in the scan path would break the image pattern.
    command += ["-qscale:v", str(qscale), "-f", "image2", "-y", _JPEG_PATTERN]
    proc = subprocess.run(command, cwd=out_dir, capture_output=True, check=False)
    paths = [out_dir / (_JPEG_PATTERN % number) for number in range(1, len(indices) + 1)]
    if proc.returncode != 0 or sorted(out_dir.glob("*.jpg")) != paths:
        log.error("ffmpeg_extract_failed", returncode=proc.returncode, frames=len(paths))
        raise FfmpegError("Die Einzelbilder konnten nicht geschrieben werden.")
    return paths


def image_to_rgb(image: Path, raw: Path, width: int, height: int) -> None:
    """Decode one image file to raw 8-bit RGB (row-major) and check its size (stage recon)."""
    _require_ffmpeg()
    raw.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(image)]
    command += ["-f", "rawvideo", "-pix_fmt", "rgb24", str(raw)]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        log.error("ffmpeg_failed", step="image_to_rgb", returncode=proc.returncode)
        raise FfmpegError("Ein entzerrtes Einzelbild konnte nicht gelesen werden.")
    if raw.stat().st_size != width * height * 3:
        raise FfmpegError(
            f"Ein entzerrtes Einzelbild hat nicht die erwartete Größe {width}×{height}."
        )
