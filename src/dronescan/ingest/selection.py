# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keyframe selection: sharpest frame per time window, adaptive threshold, no duplicates."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby

from dronescan.ingest.sharpness import Thumbnail, similarity

_WINDOW_EPS = 1e-3  # frames on a window border belong to the new window (timestamps are µs)


class Reason(StrEnum):
    """Why an analysed frame was selected or rejected (column ``reason`` in frames.parquet)."""

    KEYFRAME = "keyframe"  # sharpest frame of its window, sharp enough, new view
    NOT_SHARPEST = "not_sharpest"  # another frame of the window is sharper
    BLURRY = "blurry"  # sharpest of its window, but below the threshold of this video
    DUPLICATE = "duplicate"  # (nearly) the same view as the previous keyframe


@dataclass(frozen=True)
class Sample:
    """One analysed frame."""

    timestamp_s: float
    sharpness: float
    thumbnail: Thumbnail


@dataclass(frozen=True)
class FrameDecision:
    index: int
    timestamp_s: float
    sharpness: float
    window: int
    similarity: float | None  # to the previous keyframe; None if not compared
    reason: Reason


@dataclass(frozen=True)
class Selection:
    sharpness_threshold: float
    frames: list[FrameDecision]

    @property
    def keyframes(self) -> list[FrameDecision]:
        return [frame for frame in self.frames if frame.reason is Reason.KEYFRAME]


def select_keyframes(
    samples: Sequence[Sample],
    *,
    keyframes_per_s: float,
    min_rel_sharpness: float,
    max_similarity: float,
) -> Selection:
    """Decide for every sample; ``samples`` must be in temporal order."""
    if not samples:
        return Selection(sharpness_threshold=0.0, frames=[])
    threshold = min_rel_sharpness * statistics.median(s.sharpness for s in samples)
    start = samples[0].timestamp_s

    def window_of(index: int) -> int:
        elapsed = samples[index].timestamp_s - start
        return math.floor(elapsed * keyframes_per_s + _WINDOW_EPS)

    decisions: list[FrameDecision] = []
    last_keyframe: Sample | None = None
    for window, members in groupby(range(len(samples)), key=window_of):
        indices = list(members)
        best = max(indices, key=lambda i: samples[i].sharpness)
        for index in indices:
            sample = samples[index]
            score: float | None = None
            if index != best:
                reason = Reason.NOT_SHARPEST
            elif sample.sharpness < threshold:
                reason = Reason.BLURRY
            else:
                reason = Reason.KEYFRAME
                if last_keyframe is not None:
                    score = similarity(last_keyframe.thumbnail, sample.thumbnail)
                    if score >= max_similarity:
                        reason = Reason.DUPLICATE
                if reason is Reason.KEYFRAME:
                    last_keyframe = sample
            decisions.append(
                FrameDecision(
                    index=index,
                    timestamp_s=sample.timestamp_s,
                    sharpness=sample.sharpness,
                    window=window,
                    similarity=score,
                    reason=reason,
                )
            )
    return Selection(sharpness_threshold=threshold, frames=decisions)
