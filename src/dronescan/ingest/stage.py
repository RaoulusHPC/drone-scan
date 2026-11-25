# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage ``ingest``: sharp, non-redundant keyframes from the flight videos (KAN-33).

Per flight: ``ingest/<flight_id>/frames/*.jpg``, ``frames.parquet`` (one row per analysed
frame) and ``manifest.json`` (summary). ``ingest/manifest.json`` is the stage manifest.
"""

from __future__ import annotations

import re
import shutil
import statistics
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import structlog
from pydantic import BaseModel

from dronescan.capture.scan import SCAN_FILE, FlightMeta, ScanError, flight_video, load_scan
from dronescan.common.config import IngestSettings, Settings
from dronescan.common.stage import (
    INPUT_MISSING_HINT,
    MANIFEST_NAME,
    InputMissingError,
    QualityError,
    Stage,
    StageContext,
    StageError,
)
from dronescan.ingest.ffmpeg import (
    FfmpegError,
    FfmpegMissingError,
    decode_gray,
    extract_jpegs,
    ffmpeg_version,
    fit_size,
)

if TYPE_CHECKING:
    from dronescan.ingest.selection import Selection

FRAMES_DIR = "frames"
FRAMES_TABLE = "frames.parquet"
_FLIGHT_DIR = re.compile(r"^f\d{2,}$")

NO_FLIGHT_HINT = "Abhilfe: Video importieren mit `drone-scan flight add <scan> <video>`."
FFMPEG_HINT = "Abhilfe: ffmpeg installieren (z. B. `sudo apt install ffmpeg`)."
BROKEN_VIDEO_HINT = (
    "Abhilfe: Video am Rechner abspielen und neu importieren, ggf. vorher neu exportieren "
    "(MP4/MOV mit H.264/H.265)."
)
WRITE_HINT = "Abhilfe: freien Speicherplatz und Schreibrechte im Workspace prüfen."
TOO_FEW_HINTS = [
    "Kamera während der Aufnahme langsam und stetig durch den Raum bewegen (≤ 0,5 m/s).",
    "Länger aufnehmen: mindestens 30 Sekunden pro Raum.",
    "Licht einschalten, damit die Bilder nicht verwackeln.",
]

log = structlog.get_logger(__name__)


class FlightSummary(BaseModel):
    """Content of ``ingest/<flight_id>/manifest.json`` (no timestamps: reproducible)."""

    flight_id: str
    video: str
    video_sha256: str
    params: dict[str, Any]
    tools: dict[str, str]
    frame_width: int
    frame_height: int
    analysed_frames: int
    decoded_s: float
    sharpness_min: float
    sharpness_median: float
    sharpness_max: float
    sharpness_threshold: float
    counts: dict[str, int]  # per Reason

    @property
    def keyframes(self) -> int:
        return self.counts["keyframe"]


class IngestStage(Stage):
    name: ClassVar[str] = "ingest"
    version: ClassVar[str] = "1"

    def inputs(self, scan_dir: Path) -> list[Path]:
        scan_file = scan_dir / SCAN_FILE
        if not scan_file.exists():
            return [scan_file]
        return [scan_file, *(flight_video(scan_dir, f) for f in _flights(scan_dir))]

    def params(self, settings: Settings) -> dict[str, Any]:
        return settings.ingest.model_dump()

    def tool_versions(self) -> dict[str, str]:
        return {
            "ffmpeg": ffmpeg_version(),
            "numpy": version("numpy"),
            "pyarrow": version("pyarrow"),
        }

    def run(self, ctx: StageContext) -> None:
        flights = _flights(ctx.scan_dir)
        # Also removes results of flights that are no longer part of the scan.
        for old in ctx.out_dir.iterdir():
            if old.is_dir() and _FLIGHT_DIR.match(old.name):
                shutil.rmtree(old)
        total = 0
        for flight in flights:
            summary = self._process(ctx, flight)
            total += summary.keyframes
            log.info("flight_ingested", flight=flight.id, **summary.counts)
        if total < ctx.settings.ingest.min_keyframes:
            raise QualityError(
                f"Nur {total} brauchbare Einzelbilder in der Aufnahme "
                f"(mindestens {ctx.settings.ingest.min_keyframes} nötig).",
                TOO_FEW_HINTS,
            )

    def _process(self, ctx: StageContext, flight: FlightMeta) -> FlightSummary:
        cfg = ctx.settings.ingest
        video = flight_video(ctx.scan_dir, flight)
        selection = _analyse(video, flight, cfg)
        keyframes = selection.keyframes

        flight_dir = ctx.out_dir / flight.id
        frames_dir = flight_dir / FRAMES_DIR
        frames_dir.mkdir(parents=True)
        size = fit_size(flight.display_width, flight.display_height, cfg.max_image_edge_px)
        try:
            images = extract_jpegs(
                video,
                [frame.index for frame in keyframes],
                size=size,
                fps=cfg.analysis_fps,
                qscale=cfg.jpeg_qscale,
                out_dir=frames_dir,
            )
        except FfmpegError as exc:
            raise StageError(f"Flug {flight.id}: {exc}", [WRITE_HINT]) from exc
        files = {
            frame.index: f"{FRAMES_DIR}/{image.name}"
            for frame, image in zip(keyframes, images, strict=True)
        }
        _write_table(selection, files, flight_dir / FRAMES_TABLE)

        summary = self._summarise(ctx, flight, video, selection, size)
        (flight_dir / MANIFEST_NAME).write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        if summary.decoded_s < cfg.min_decoded_ratio * flight.duration_s:
            ctx.warn(
                f"Flug {flight.id}: nur {summary.decoded_s:.1f} s von {flight.duration_s:.1f} s "
                "konnten gelesen werden, das Video ist vermutlich beschädigt."
            )
            ctx.hint(f"Flug {flight.id}: {BROKEN_VIDEO_HINT}")
        return summary

    def _summarise(
        self,
        ctx: StageContext,
        flight: FlightMeta,
        video: Path,
        selection: Selection,
        size: tuple[int, int],
    ) -> FlightSummary:
        from dronescan.ingest.selection import Reason

        scores = [frame.sharpness for frame in selection.frames]
        times = [frame.timestamp_s for frame in selection.frames]
        counts = Counter(frame.reason for frame in selection.frames)
        return FlightSummary(
            flight_id=flight.id,
            video=video.relative_to(ctx.scan_dir).as_posix(),
            video_sha256=flight.sha256,
            params=self.params(ctx.settings),
            tools=self.tool_versions(),
            frame_width=size[0],
            frame_height=size[1],
            analysed_frames=len(scores),
            decoded_s=times[-1] - times[0] + 1 / ctx.settings.ingest.analysis_fps,
            sharpness_min=min(scores),
            sharpness_median=statistics.median(scores),
            sharpness_max=max(scores),
            sharpness_threshold=selection.sharpness_threshold,
            counts={reason.value: counts[reason] for reason in Reason},
        )


def _flights(scan_dir: Path) -> list[FlightMeta]:
    try:
        flights = load_scan(scan_dir).flights
    except ScanError as exc:
        raise InputMissingError(str(exc), [INPUT_MISSING_HINT]) from exc
    if not flights:
        raise InputMissingError("Der Scan enthält noch kein Video.", [NO_FLIGHT_HINT])
    return flights


def _analyse(video: Path, flight: FlightMeta, cfg: IngestSettings) -> Selection:
    # Heavy imports (numpy) only when the stage runs, so `drone-scan --help` stays fast.
    import numpy as np
    from numpy.typing import NDArray

    from dronescan.ingest.selection import Sample, select_keyframes
    from dronescan.ingest.sharpness import Thumbnail, laplacian_variance, thumbnail

    measured: list[tuple[float, Thumbnail]] = []

    def measure(frame: NDArray[np.uint8]) -> None:
        measured.append((laplacian_variance(frame), thumbnail(frame, cfg.similarity_edge_px)))

    size = fit_size(flight.display_width, flight.display_height, cfg.analysis_edge_px)
    try:
        times = decode_gray(video, size, cfg.analysis_fps, measure)
    except FfmpegMissingError as exc:
        raise StageError(str(exc), [FFMPEG_HINT]) from exc
    except FfmpegError as exc:
        raise InputMissingError(f"Flug {flight.id}: {exc}", [BROKEN_VIDEO_HINT]) from exc
    samples = [
        Sample(timestamp_s=time, sharpness=score, thumbnail=thumb)
        for time, (score, thumb) in zip(times, measured, strict=True)
    ]
    return select_keyframes(
        samples,
        keyframes_per_s=cfg.keyframes_per_s,
        min_rel_sharpness=cfg.min_rel_sharpness,
        max_similarity=cfg.max_similarity,
    )


def _write_table(selection: Selection, files: dict[int, str], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("index", pa.int32(), nullable=False),
            pa.field("timestamp_s", pa.float64(), nullable=False),
            pa.field("sharpness", pa.float64(), nullable=False),
            pa.field("window", pa.int32(), nullable=False),
            pa.field("similarity", pa.float64()),
            pa.field("selected", pa.bool_(), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
            pa.field("file", pa.string()),  # relative to the flight directory
        ]
    )
    rows = [
        {
            "index": frame.index,
            "timestamp_s": frame.timestamp_s,
            "sharpness": frame.sharpness,
            "window": frame.window,
            "similarity": frame.similarity,
            "selected": frame.index in files,
            "reason": frame.reason.value,
            "file": files.get(frame.index),
        }
        for frame in selection.frames
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
