# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-flight and per-scan quality assessment: sharpness, exposure, motion, coverage (KAN-34).

Three of the four dimensions are a share of the keyframe windows already decided by
`select_keyframes` (KAN-33, ``Reason``): how many were lost to blur (sharpness), how many
were duplicates of the last keyframe (motion: the camera stood still) and how many produced
a new keyframe at all (coverage, (grob) per US-019). Ratios are self-calibrating per video
(the underlying ``Reason`` already compares each frame against that video's own median), so
the profile's thresholds stay meaningful across very different cameras and lighting; only
exposure needs the raw brightness, measured during decoding. ``Criterion`` carries the German
hints that become either a warning (yellow) or, if the whole scan is red, exit 20 (Q-9).

``FlightQuality`` is the content of ``ingest/<flight_id>/quality.json``; ``ScanQuality`` is
the per-scan summary stored under ``metrics.quality`` in ``ingest/manifest.json`` (viewer
ampel, KAN-43).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from dronescan.common.config import IngestSettings
from dronescan.ingest.hints import (
    HINT_AVOID_BACKLIGHT,
    HINT_DAYLIGHT,
    HINT_KEEP_MOVING,
    HINT_LIGHTS_ON,
    HINT_MOVE_SLOWLY,
    HINT_RECORD_LONGER,
)

if TYPE_CHECKING:
    # `selection` imports `sharpness` (numpy) at module level; keep the CLI's fast start
    # (`drone-scan --help`, CLAUDE.md) by only importing it for real inside `assess_flight`.
    from dronescan.ingest.selection import Selection

QUALITY_FILE = "quality.json"
_NEUTRAL_BRIGHTNESS = 128.0  # assumed when nothing could be measured (no decoded frames)


class TrafficLight(StrEnum):
    """Overall assessment of one criterion, a flight or a scan (viewer ampel, KAN-43)."""

    GREEN = "grün"
    YELLOW = "gelb"
    RED = "rot"


_ORDER = {TrafficLight.GREEN: 0, TrafficLight.YELLOW: 1, TrafficLight.RED: 2}


def worst(lights: Sequence[TrafficLight]) -> TrafficLight:
    """The least favourable light; green for an empty sequence (nothing to report)."""
    return max(lights, key=_ORDER.__getitem__, default=TrafficLight.GREEN)


class Criterion(BaseModel):
    """One assessed dimension; ``hints`` is empty exactly when ``light`` is green."""

    light: TrafficLight
    value: float
    hints: list[str] = []


class FlightQuality(BaseModel):
    """Content of ``ingest/<flight_id>/quality.json`` (US-019 AC1)."""

    flight_id: str
    overall: TrafficLight
    sharpness: Criterion
    exposure: Criterion
    motion: Criterion
    coverage: Criterion

    @property
    def hints(self) -> list[str]:
        """All hints of the non-green criteria, deduplicated, in a stable order."""
        seen: dict[str, None] = {}
        for criterion in (self.sharpness, self.exposure, self.motion, self.coverage):
            for hint in criterion.hints:
                seen.setdefault(hint, None)
        return list(seen)


class ScanQuality(BaseModel):
    """Per-scan summary: overall ampel, each flight's ampel, deduplicated hints."""

    overall: TrafficLight
    flights: dict[str, TrafficLight]
    hints: list[str]


def _ratio_light(value: float, warn: float, bad: float) -> TrafficLight:
    """A loss ratio (share of windows lost to some problem): higher is worse."""
    if value >= bad:
        return TrafficLight.RED
    if value >= warn:
        return TrafficLight.YELLOW
    return TrafficLight.GREEN


def _coverage_light(value: float, min_ratio: float, target_ratio: float) -> TrafficLight:
    """A usable-share ratio: lower is worse."""
    if value < min_ratio:
        return TrafficLight.RED
    if value < target_ratio:
        return TrafficLight.YELLOW
    return TrafficLight.GREEN


def _exposure_light(value: float, cfg: IngestSettings) -> TrafficLight:
    if value < cfg.exposure_dark_bad or value > cfg.exposure_bright_bad:
        return TrafficLight.RED
    if value < cfg.exposure_dark_warn or value > cfg.exposure_bright_warn:
        return TrafficLight.YELLOW
    return TrafficLight.GREEN


def assess_flight(
    flight_id: str, selection: Selection, brightness: Sequence[float], cfg: IngestSettings
) -> FlightQuality:
    """Assess one flight from its keyframe `Selection` (KAN-33) and measured brightness."""
    from dronescan.ingest.selection import Reason  # see the TYPE_CHECKING import above

    windows = len({frame.window for frame in selection.frames})
    counts = {reason: sum(1 for f in selection.frames if f.reason is reason) for reason in Reason}

    blurry_ratio = counts[Reason.BLURRY] / windows if windows else 0.0
    sharpness_light = _ratio_light(blurry_ratio, cfg.blurry_warn_ratio, cfg.blurry_bad_ratio)
    sharpness = Criterion(
        light=sharpness_light,
        value=blurry_ratio,
        hints=[] if sharpness_light is TrafficLight.GREEN else [HINT_MOVE_SLOWLY, HINT_LIGHTS_ON],
    )

    exposure_value = statistics.median(brightness) if brightness else _NEUTRAL_BRIGHTNESS
    exposure_light = _exposure_light(exposure_value, cfg)
    exposure_hint = (
        HINT_DAYLIGHT if exposure_value < cfg.exposure_dark_warn else HINT_AVOID_BACKLIGHT
    )
    exposure = Criterion(
        light=exposure_light,
        value=exposure_value,
        hints=[] if exposure_light is TrafficLight.GREEN else [exposure_hint],
    )

    duplicate_ratio = counts[Reason.DUPLICATE] / windows if windows else 0.0
    motion_light = _ratio_light(duplicate_ratio, cfg.duplicate_warn_ratio, cfg.duplicate_bad_ratio)
    motion = Criterion(
        light=motion_light,
        value=duplicate_ratio,
        hints=[] if motion_light is TrafficLight.GREEN else [HINT_KEEP_MOVING],
    )

    coverage_value = len(selection.keyframes) / windows if windows else 0.0
    coverage_light = _coverage_light(
        coverage_value, cfg.min_coverage_ratio, cfg.target_coverage_ratio
    )
    coverage = Criterion(
        light=coverage_light,
        value=coverage_value,
        hints=[] if coverage_light is TrafficLight.GREEN else [HINT_RECORD_LONGER],
    )

    overall = worst([sharpness_light, exposure_light, motion_light, coverage_light])
    return FlightQuality(
        flight_id=flight_id,
        overall=overall,
        sharpness=sharpness,
        exposure=exposure,
        motion=motion,
        coverage=coverage,
    )


def aggregate_scan_quality(flights: Sequence[FlightQuality]) -> ScanQuality:
    """Combine all flights of a scan: the worst light wins, hints are deduplicated in order."""
    hints: dict[str, None] = {}
    for flight in flights:
        for hint in flight.hints:
            hints.setdefault(hint, None)
    return ScanQuality(
        overall=worst([flight.overall for flight in flights]),
        flights={flight.flight_id: flight.overall for flight in flights},
        hints=list(hints),
    )
