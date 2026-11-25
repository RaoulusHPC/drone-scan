# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage ``sfm``: camera poses and sparse point cloud with COLMAP (KAN-36).

All flights are reconstructed together (one camera per flight). ``sfm/colmap/models/`` keeps
every partial model, ``sfm/colmap/sparse/0`` is the largest one as COLMAP text. It only exists
if enough keyframes are registered, so later stages never build on a fragment.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import structlog

from dronescan.capture.scan import SCAN_FILE, ScanError, load_scan
from dronescan.common.config import Settings, SfmSettings
from dronescan.common.stage import (
    GPU_OOM_HINT,
    INPUT_MISSING_HINT,
    MANIFEST_NAME,
    InputMissingError,
    OutOfMemoryError,
    QualityError,
    Stage,
    StageContext,
    StageError,
    load_manifest,
)
from dronescan.ingest.stage import FRAMES_DIR, FRAMES_TABLE, NO_FLIGHT_HINT, IngestStage
from dronescan.sfm.colmap import (
    COLMAP_IMAGE,
    Colmap,
    ColmapError,
    ColmapOutOfMemoryError,
    ColmapUnavailableError,
    Mount,
    StepResult,
    image_id,
)
from dronescan.sfm.model import ModelReport, SparseModel, list_models, read_report

COLMAP_DIR = "colmap"
SPARSE_DIR = "sparse/0"
_MODELS_DIR = "models"
_DATABASE = "database.db"
_IMAGE_LIST = "image_list.txt"
_VOCAB_TREE = "vocab_tree.bin"
_NO_MODEL = "failed to create any sparse model"

UNAVAILABLE_HINTS = [
    "Abhilfe: Docker starten und das Image bauen mit `docker compose build colmap`.",
    "GPU im Container prüfen mit `make docker-check` (NVIDIA Container Toolkit, siehe README).",
]
COLMAP_HINT = "Abhilfe: Logdateien in sfm/colmap/logs/ prüfen und mit --force erneut ausführen."
QUALITY_HINTS = [
    "Licht einschalten und möglichst bei Tageslicht aufnehmen.",
    "Langsamer bewegen (≤ 0,5 m/s) und 1 bis 1,5 m Abstand zu den Wänden halten.",
    "Türdurchgänge doppelt aufnehmen (hin und zurück).",
    "Bei weißen Wänden, Glas oder Spiegeln 2 bis 4 AprilTags aufhängen.",
]
SPLIT_HINT = (
    "Die Aufnahme zerfällt in {count} nicht verbundene Teile: an den Übergängen erneut langsam "
    "und mit viel Überlappung aufnehmen."
)
FLIGHT_HINT = (
    "Flug {flight}: nur {registered} von {keyframes} Einzelbildern verortet. Diesen Bereich "
    "erneut aufnehmen und mit einem bereits erfassten Bereich beginnen."
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _Limits:
    """Parameters that determine the VRAM of features and matching."""

    image_edge_px: int
    num_features: int
    num_matches: int


class SfmStage(Stage):
    name: ClassVar[str] = "sfm"
    version: ClassVar[str] = "1"
    uses_gpu: ClassVar[bool] = True
    applies_vram_limit: ClassVar[bool] = False  # COLMAP runs in its own container

    def inputs(self, scan_dir: Path) -> list[Path]:
        scan_file = scan_dir / SCAN_FILE
        if not scan_file.exists():
            return [scan_file]
        try:
            flights = load_scan(scan_dir).flights
        except ScanError as exc:
            raise InputMissingError(str(exc), [INPUT_MISSING_HINT]) from exc
        if not flights:
            raise InputMissingError("Der Scan enthält noch kein Video.", [NO_FLIGHT_HINT])
        ingest = scan_dir / IngestStage.name
        manifest = load_manifest(ingest / MANIFEST_NAME)
        if manifest is None or manifest.status != "ok":
            raise InputMissingError(
                "Die Einzelbilder fehlen: Stufe 'ingest' ist nicht erfolgreich gelaufen.",
                [INPUT_MISSING_HINT],
            )
        return [
            ingest / flight.id / part for flight in flights for part in (FRAMES_TABLE, FRAMES_DIR)
        ]

    def params(self, settings: Settings) -> dict[str, Any]:
        return settings.sfm.model_dump()

    def tool_versions(self) -> dict[str, str]:
        return {"colmap": COLMAP_IMAGE, "colmap_image_id": image_id()}

    def run(self, ctx: StageContext) -> None:
        work = ctx.out_dir / COLMAP_DIR
        shutil.rmtree(work, ignore_errors=True)
        (work / _MODELS_DIR).mkdir(parents=True)
        images = _keyframes(ctx.inputs)
        (work / _IMAGE_LIST).write_text("\n".join(images) + "\n", encoding="utf-8")
        colmap = Colmap(
            mounts=[
                Mount(ctx.scan_dir / IngestStage.name, "/ingest", read_only=True),
                Mount(work, "/sfm"),
            ],
            gpu_device=ctx.settings.gpu_device,
            log_dir=work / "logs",
        )
        try:
            _Reconstruction(ctx, colmap, work, images).run()
        except ColmapUnavailableError as exc:
            raise StageError(str(exc), UNAVAILABLE_HINTS) from exc
        except ColmapOutOfMemoryError as exc:
            raise OutOfMemoryError(str(exc), [GPU_OOM_HINT]) from exc
        except ColmapError as exc:
            raise StageError(str(exc), [COLMAP_HINT]) from exc


class _Reconstruction:
    """One run of the COLMAP steps; collects metrics and peak VRAM in the stage context."""

    def __init__(self, ctx: StageContext, colmap: Colmap, work: Path, images: list[str]) -> None:
        self._ctx = ctx
        self._cfg: SfmSettings = ctx.settings.sfm
        self._colmap = colmap
        self._work = work
        self._images = images
        self._exhaustive = len(images) <= self._cfg.exhaustive_max_keyframes
        self._seconds: dict[str, float] = {}

    def run(self) -> None:
        self._ctx.metrics["keyframes"] = len(self._images)
        self._ctx.metrics["matcher"] = "exhaustive" if self._exhaustive else "sequential+loop"
        self._ctx.metrics["seconds"] = self._seconds
        self._features_and_matches()
        self._map()
        models = list_models(self._work / _MODELS_DIR)
        self._ctx.metrics["model_sizes"] = [model.images for model in models]
        if not models:
            self._ctx.metrics.update(registered_images=0, registration_ratio=0.0)
            raise QualityError("Aus den Einzelbildern entstand kein 3D-Modell.", QUALITY_HINTS)
        self._evaluate(models)

    def _record(self, step: str, result: StepResult) -> None:
        self._seconds[step] = round(self._seconds.get(step, 0.0) + result.seconds, 1)
        self._ctx.record_peak_vram(result.peak_vram_mib)

    def _features_and_matches(self) -> None:
        cfg = self._cfg
        limits = _Limits(cfg.max_image_edge_px, cfg.max_num_features, cfg.max_num_matches)
        try:
            self._extract(limits)
            self._match(limits)
        except ColmapOutOfMemoryError:
            edge = min(cfg.oom_retry_image_edge_px, cfg.max_image_edge_px)
            count = min(cfg.oom_retry_num_features, cfg.max_num_features)
            log.warning("sfm_oom_retry", image_edge_px=edge, num_features=count)
            self._ctx.warn(
                "Der Grafikspeicher reichte nicht, zweiter Versuch mit kleineren Werten "
                f"({edge} px, {count} Merkmale je Bild)."
            )
            for leftover in self._work.glob(f"{_DATABASE}*"):
                leftover.unlink()
            reduced = _Limits(edge, count, count)
            self._extract(reduced)
            self._match(reduced)

    def _extract(self, limits: _Limits) -> None:
        options: dict[str, str | int | float] = {
            "database_path": f"/sfm/{_DATABASE}",
            "image_path": "/ingest",
            "image_list_path": f"/sfm/{_IMAGE_LIST}",
            "ImageReader.camera_model": self._cfg.camera_model,
            "ImageReader.single_camera_per_folder": 1,  # one folder per flight
            "FeatureExtraction.use_gpu": 1,
            "FeatureExtraction.gpu_index": 0,  # the container sees a single GPU
            "FeatureExtraction.max_image_size": limits.image_edge_px,
            "SiftExtraction.max_num_features": limits.num_features,
        }
        self._record("features", self._colmap.run("feature_extractor", options, gpu=True))

    def _match(self, limits: _Limits) -> None:
        options: dict[str, str | int | float] = {
            "database_path": f"/sfm/{_DATABASE}",
            "FeatureMatching.use_gpu": 1,
            "FeatureMatching.gpu_index": 0,
            "FeatureMatching.max_num_matches": limits.num_matches,
        }
        if self._exhaustive:
            result = self._colmap.run("exhaustive_matcher", options, gpu=True)
        else:
            self._build_vocab_tree()
            options["SequentialMatching.overlap"] = self._cfg.sequential_overlap
            options["SequentialMatching.loop_detection"] = 1
            options["SequentialMatching.vocab_tree_path"] = f"/sfm/{_VOCAB_TREE}"
            result = self._colmap.run("sequential_matcher", options, gpu=True)
            (self._work / _VOCAB_TREE).unlink(missing_ok=True)
        self._record("matching", result)

    def _build_vocab_tree(self) -> None:
        """Vocabulary tree from the scan's own features: no download, no foreign weights."""
        options: dict[str, str | int | float] = {
            "database_path": f"/sfm/{_DATABASE}",
            "vocab_tree_path": f"/sfm/{_VOCAB_TREE}",
            "num_visual_words": self._cfg.loop_vocab_words,
            "num_iterations": self._cfg.loop_vocab_iterations,
            "num_rounds": self._cfg.loop_vocab_rounds,
            "max_num_descriptors": self._cfg.loop_vocab_max_descriptors,
        }
        self._record("vocab_tree", self._colmap.run("vocab_tree_builder", options))

    def _map(self) -> None:
        options: dict[str, str | int | float] = {
            "database_path": f"/sfm/{_DATABASE}",
            "image_path": "/ingest",
            "output_path": f"/sfm/{_MODELS_DIR}",
            "Mapper.ba_use_gpu": 0,
        }
        try:
            self._record("mapper", self._colmap.run("mapper", options))
        except ColmapUnavailableError:
            raise
        except ColmapError as exc:
            if _NO_MODEL not in exc.output.lower():
                raise  # a model that could not be built is a quality problem, not an error

    def _evaluate(self, models: list[SparseModel]) -> None:
        best = models[0]
        text_dir = self._work / "best"
        text_dir.mkdir()
        options: dict[str, str | int | float] = {
            "input_path": f"/sfm/{_MODELS_DIR}/{best.path.name}",
            "output_path": f"/sfm/{text_dir.name}",
            "output_type": "TXT",
        }
        self._colmap.run("model_converter", options)
        report = read_report(text_dir)
        total, registered = len(self._images), len(report.image_names)
        ratio = registered / total
        flights = self._per_flight(report)
        self._ctx.metrics.update(
            registered_images=registered,
            registration_ratio=round(ratio, 4),
            mean_reprojection_error_px=round(report.mean_reprojection_error_px, 4),
            points=report.points,
            flights=flights,
        )
        log.info("sfm_registered", registered=registered, keyframes=total, models=len(models))
        if ratio < self._cfg.target_registration_ratio:
            message = (
                f"Nur {registered} von {total} Einzelbildern ({_percent(ratio)}) konnten im "
                "größten Teilmodell verortet werden"
            )
            hints = self._hints(len(models), flights)
            if ratio < self._cfg.min_registration_ratio:
                shutil.rmtree(text_dir)
                needed = _percent(self._cfg.min_registration_ratio)
                raise QualityError(f"{message} (mindestens {needed} nötig).", hints)
            goal = _percent(self._cfg.target_registration_ratio)
            self._ctx.warn(f"{message} (Ziel: {goal}). Teile des Modells können fehlen.")
            for hint in hints:
                self._ctx.hint(hint)
        sparse = self._work / SPARSE_DIR
        sparse.parent.mkdir()
        text_dir.rename(sparse)

    def _per_flight(self, report: ModelReport) -> dict[str, dict[str, int]]:
        flights: dict[str, dict[str, int]] = {}
        for image in self._images:
            entry = flights.setdefault(_flight(image), {"keyframes": 0, "registered": 0})
            entry["keyframes"] += 1
        for image in report.image_names:
            flights[_flight(image)]["registered"] += 1
        return flights

    def _hints(self, models: int, flights: dict[str, dict[str, int]]) -> list[str]:
        hints = list(QUALITY_HINTS)
        if models > 1:
            hints.append(SPLIT_HINT.format(count=models))
        if len(flights) > 1:
            hints += [
                FLIGHT_HINT.format(flight=flight, **counts)
                for flight, counts in flights.items()
                if counts["registered"] < self._cfg.min_registration_ratio * counts["keyframes"]
            ]
        return hints


def _flight(image: str) -> str:
    return image.split("/", 1)[0]


def _percent(ratio: float) -> str:
    """Rounded down, so a value below a threshold never reads like the threshold."""
    value = math.floor(ratio * 1000 + 1e-9) / 10
    return f"{value:g} %".replace(".", ",")


def _keyframes(inputs: list[Path]) -> list[str]:
    """Image names relative to ``ingest/`` in temporal order per flight."""
    import pyarrow.parquet as pq

    images = []
    for table_path in (path for path in inputs if path.name == FRAMES_TABLE):
        flight_dir = table_path.parent
        rows = pq.read_table(table_path, columns=["timestamp_s", "file"]).to_pylist()
        rows = [row for row in rows if row["file"] is not None]
        for row in sorted(rows, key=lambda row: row["timestamp_s"]):
            if not (flight_dir / row["file"]).is_file():
                raise InputMissingError(
                    f"Flug {flight_dir.name}: Einzelbilder fehlen.", [INPUT_MISSING_HINT]
                )
            images.append(f"{flight_dir.name}/{row['file']}")
    return images
