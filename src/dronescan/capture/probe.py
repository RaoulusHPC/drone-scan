# SPDX-License-Identifier: AGPL-3.0-or-later
"""ffprobe adapter: reads video metadata and applies the display rotation flag.

Smartphone clips are often stored in portrait (e.g. 1080x1920) with a rotation flag
of 90 degrees; ffmpeg displays them in landscape (spike-01, section 3). We keep both
the encoded and the display size.
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_PROBE_TIMEOUT_S = 60


class ProbeError(Exception):
    """The file is missing, unreadable or not a usable video (user-facing, German)."""


class VideoMeta(BaseModel):
    width: int
    height: int
    rotation: int  # degrees clockwise, normalised to 0, 90, 180 or 270
    display_width: int
    display_height: int
    fps: float
    duration_s: float
    codec: str


def _rotation(stream: dict[str, Any]) -> int:
    """Clockwise display rotation in degrees (0, 90, 180, 270)."""
    try:
        degrees = int(round(float((stream.get("tags") or {}).get("rotate", 0))))  # legacy tag
        for side in stream.get("side_data_list") or []:
            if "rotation" in side:
                # Display matrix: ffprobe reports counter-clockwise degrees.
                degrees = -int(round(float(side["rotation"])))
    except (TypeError, ValueError):
        return 0
    return degrees % 360


def _fps(stream: dict[str, Any]) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = float(Fraction(str(stream.get(key, "0/0"))))
        except (ZeroDivisionError, ValueError):
            continue
        if rate > 0:
            return rate
    return 0.0


def probe_video(path: Path) -> VideoMeta:
    """Return metadata of the first video stream or raise ``ProbeError``."""
    name = path.name
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ProbeError(
            "ffprobe ist nicht installiert. "
            "Abhilfe: ffmpeg installieren (z. B. `sudo apt install ffmpeg`)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"'{name}' konnte nicht rechtzeitig gelesen werden.") from exc
    if proc.returncode != 0:
        raise ProbeError(
            f"'{name}' ist kein lesbares Video (defekt oder nicht unterstütztes Format). "
            "Abhilfe: Datei prüfen oder neu exportieren (MP4/MOV mit H.264/H.265)."
        )
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"'{name}': ffprobe lieferte keine lesbare Ausgabe.") from exc

    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    # Cover art in audio files shows up as a one-frame "video" stream; skip it.
    streams = [s for s in streams if not (s.get("disposition") or {}).get("attached_pic")]
    if not streams:
        raise ProbeError(f"'{name}' enthält keine Videospur. Abhilfe: eine Videodatei verwenden.")
    stream = streams[0]
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    try:
        duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    fps = _fps(stream)
    if width <= 0 or height <= 0 or duration <= 0 or fps <= 0:
        raise ProbeError(
            f"'{name}' ist unvollständig (keine Größe, Dauer oder Bildrate). "
            "Abhilfe: Aufnahme erneut übertragen oder exportieren."
        )
    rotation = _rotation(stream)
    swap = rotation in (90, 270)
    return VideoMeta(
        width=width,
        height=height,
        rotation=rotation,
        display_width=height if swap else width,
        display_height=width if swap else height,
        fps=fps,
        duration_s=duration,
        codec=str(stream.get("codec_name", "unknown")),
    )
