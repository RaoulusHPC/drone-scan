# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage ``ingest``: sharp, non-redundant keyframes from the flight videos (KAN-33, KAN-34).

Per flight: ``ingest/<flight_id>/frames/*.jpg``, ``frames.parquet`` (one row per analysed
frame), ``manifest.json`` (summary) and ``quality.json`` (sharpness/exposure/motion/coverage
ampel, US-019). ``ingest/manifest.json`` is the stage manifest; its ``metrics.quality`` is the
per-scan summary (overall ampel, every flight's ampel, deduplicated hints).
"""

from __future__ import annotations

import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import structlog
from pydantic import BaseModel

from dronescan.capture.scan import SCAN_FILE, FlightMeta, ScanError, flight_video, load_scan
from dronescan.common import error_catalog as catalog
from dronescan.common.config import IngestSettings, Settings
from dronescan.common.stage import (
    INPUT_MISSING_HINT,
    MANIFEST_NAME,
    NO_FLIGHT_HINT,
    WRITE_HINT,
    InputMissingError,
    QualityError,
    Stage,
    StageContext,
    StageError,
    hash_path,
)
from dronescan.ingest.ffmpeg import (
    FfmpegError,
    FfmpegMissingError,
    decode_gray,
    extract_jpegs,
    ffmpeg_version,
    fit_size,
)
from dronescan.ingest.hints import HINT_LIGHTS_ON, HINT_MOVE_SLOWLY, HINT_RECORD_LONGER
from dronescan.ingest.quality import (
    QUALITY_FILE,
    FlightQuality,
    TrafficLight,
    aggregate_scan_quality,
    assess_flight,
)

if TYPE_CHECKING:
    from dronescan.ingest.selection import Selection

FRAMES_DIR = "frames"
FRAMES_TABLE = "frames.parquet"
_FLIGHT_DIR = re.compile(r"^f\d{2,}$")

FFMPEG_HINT = catalog.hint("ingest.ffmpeg_missing")
BROKEN_VIDEO_HINT = catalog.hint("ingest.video_broken")
TOO_FEW_HINTS = [HINT_MOVE_SLOWLY, HINT_RECORD_LONGER, HINT_LIGHTS_ON]
_CRITERION_LABELS = (
    ("Schärfe", "sharpness"),
    ("Belichtung", "exposure"),
    ("Bewegung", "motion"),
    ("Abdeckung", "coverage"),
)

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
        qualities: list[FlightQuality] = []
        for flight in flights:
            summary, quality = self._process(ctx, flight)
            total += summary.keyframes
            qualities.append(quality)
            log.info("flight_ingested", flight=flight.id, **summary.counts)

        scan_quality = aggregate_scan_quality(qualities)
        ctx.metrics["quality"] = scan_quality.model_dump(mode="json")
        for quality in qualities:
            if quality.overall is not TrafficLight.GREEN:
                ctx.warn(f"Flug {quality.flight_id}: {_describe(quality)}.")

        if total < ctx.settings.ingest.min_keyframes:
            raise QualityError(
                f"Nur {total} brauchbare Einzelbilder in der Aufnahme "
                f"(mindestens {ctx.settings.ingest.min_keyframes} nötig).",
                TOO_FEW_HINTS,
                code="ingest.too_few_frames",
            )
        if scan_quality.overall is TrafficLight.RED:
            raise QualityError(
                "Die Aufnahmequalität reicht nicht für eine brauchbare Rekonstruktion.",
                scan_quality.hints,
                code="ingest.quality_red",
            )
        for hint in scan_quality.hints:
            ctx.hint(hint)

    def _process(
        self, ctx: StageContext, flight: FlightMeta
    ) -> tuple[FlightSummary, FlightQuality]:
        cfg = ctx.settings.ingest
        video = flight_video(ctx.scan_dir, flight)
        analysis = _analyse(video, flight, cfg)
        selection = analysis.selection
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
            raise StageError(
                f"Flug {flight.id}: {exc}", [WRITE_HINT], code="common.write_failed"
            ) from exc
        files = {
            frame.index: f"{FRAMES_DIR}/{image.name}"
            for frame, image in zip(keyframes, images, strict=True)
        }
        summary = self._summarise(ctx, flight, video, selection, size)
        quality = assess_flight(flight.id, selection, analysis.brightness, cfg)
        try:
            _write_table(selection, files, flight_dir / FRAMES_TABLE)
            (flight_dir / MANIFEST_NAME).write_text(
                summary.model_dump_json(indent=2), encoding="utf-8"
            )
            (flight_dir / QUALITY_FILE).write_text(
                quality.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise StageError(
                f"Flug {flight.id}: Ergebnisse konnten nicht geschrieben werden.",
                [WRITE_HINT],
                code="common.write_failed",
            ) from exc
        if summary.decoded_s < cfg.min_decoded_ratio * flight.duration_s:
            ctx.warn(
                f"Flug {flight.id}: nur {summary.decoded_s:.1f} s von {flight.duration_s:.1f} s "
                "konnten gelesen werden, das Video ist vermutlich beschädigt."
            )
            ctx.hint(f"Flug {flight.id}: {BROKEN_VIDEO_HINT}")
        return summary, quality

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
            video_sha256=hash_path(video),
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
        raise InputMissingError(
            str(exc), [INPUT_MISSING_HINT], code="common.input_missing"
        ) from exc
    if not flights:
        raise InputMissingError(
            "Der Scan enthält noch kein Video.", [NO_FLIGHT_HINT], code="common.no_flight"
        )
    return flights


def _describe(quality: FlightQuality) -> str:
    """Readable CLI line naming the non-green criteria of one flight (US-019 AC3)."""
    bad = [
        label
        for label, field in _CRITERION_LABELS
        if getattr(quality, field).light is not TrafficLight.GREEN
    ]
    return f"Aufnahmequalität {quality.overall.value} ({', '.join(bad)})"


@dataclass(frozen=True)
class _Analysis:
    """Keyframe selection plus the raw measures the quality assessment needs (KAN-34)."""

    selection: Selection
    brightness: list[float]  # grey level (0..255) per analysed frame, same order as selection


def _analyse(video: Path, flight: FlightMeta, cfg: IngestSettings) -> _Analysis:
    # Heavy imports (numpy) only when the stage runs, so `drone-scan --help` stays fast.
    import numpy as np
    from numpy.typing import NDArray

    from dronescan.ingest.selection import Sample, select_keyframes
    from dronescan.ingest.sharpness import Thumbnail, brightness, laplacian_variance, thumbnail

    measured: list[tuple[float, Thumbnail, float]] = []

    def measure(frame: NDArray[np.uint8]) -> None:
        measured.append(
            (laplacian_variance(frame), thumbnail(frame, cfg.similarity_edge_px), brightness(frame))
        )

    size = fit_size(flight.display_width, flight.display_height, cfg.analysis_edge_px)
    try:
        times = decode_gray(video, size, cfg.analysis_fps, measure)
    except FfmpegMissingError as exc:
        raise StageError(str(exc), [FFMPEG_HINT], code="ingest.ffmpeg_missing") from exc
    except FfmpegError as exc:
        raise InputMissingError(
            f"Flug {flight.id}: {exc}", [BROKEN_VIDEO_HINT], code="ingest.video_broken"
        ) from exc
    samples = [
        Sample(timestamp_s=time, sharpness=score, thumbnail=thumb)
        for time, (score, thumb, _) in zip(times, measured, strict=True)
    ]
    selection = select_keyframes(
        samples,
        keyframes_per_s=cfg.keyframes_per_s,
        min_rel_sharpness=cfg.min_rel_sharpness,
        max_similarity=cfg.max_similarity,
    )
    return _Analysis(selection=selection, brightness=[bright for _, _, bright in measured])


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
