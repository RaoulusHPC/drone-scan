# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage ``sfm``: camera poses and sparse point cloud with COLMAP (KAN-36).

All flights are reconstructed together (one camera per flight). ``sfm/colmap/models/`` keeps
every partial model, ``sfm/colmap/sparse/0`` is the largest one as COLMAP text. It only exists
if enough keyframes are registered, so later stages never build on a fragment.

The camera model of a flight comes from its camera profile in ``scan.yaml`` (KAN-29), without
one from ``sfm.camera_model``. Intrinsics of a profile are start values that COLMAP refines.
The estimated intrinsics are stored per flight as a camera profile in ``sfm/cameras/``.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import structlog

from dronescan.capture.camera import CameraProfile, Intrinsics, write_camera
from dronescan.capture.scan import SCAN_FILE, FlightMeta, ScanError, load_scan
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
from dronescan.ingest.stage import (
    FRAMES_DIR,
    FRAMES_TABLE,
    NO_FLIGHT_HINT,
    WRITE_HINT,
    FlightSummary,
    IngestStage,
)
from dronescan.sfm.colmap import (
    COLMAP_IMAGE,
    Colmap,
    ColmapError,
    ColmapOutOfMemoryError,
    ColmapTimeoutError,
    ColmapUnavailableError,
    Mount,
    StepResult,
    image_id,
)
from dronescan.sfm.model import ModelReport, SparseModel, list_models, read_cameras, read_report

COLMAP_DIR = "colmap"
SPARSE_DIR = "sparse/0"
CAMERAS_DIR = "cameras"
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
LOOP_WARNING = (
    "Die Erkennung bereits besuchter Orte (Loop-Detection) war nicht möglich. Das Modell kann "
    "sich über lange Wege verziehen oder in Teile zerfallen."
)
LOOP_TIMEOUT_HINT = (
    "Die Loop-Detection brauchte zu lange, meist wegen zu weniger Bildmerkmale: mit mehr Licht "
    "und Struktur neu aufnehmen. Auf langsamen Rechnern DRONESCAN_SFM__LOOP_VOCAB_TIMEOUT_S "
    "erhöhen und mit --force wiederholen."
)
LOOP_FAILED_HINT = (
    "Loop-Detection: Logdatei *_vocab_tree_builder.log in sfm/colmap/logs/ und freien "
    "Speicherplatz prüfen, dann mit --force wiederholen."
)
SPLIT_HINT = (
    "Die Aufnahme zerfällt in {count} nicht verbundene Teile: an den Übergängen erneut langsam "
    "und mit viel Überlappung aufnehmen."
)
FLIGHT_HINT = (
    "Flug {flight}: nur {registered} von {keyframes} Einzelbildern verortet. Diesen Bereich "
    "erneut aufnehmen und mit einem bereits erfassten Bereich beginnen."
)

CAMERA_ASPECT_WARNING = (
    "Flug {flight}: Die Werte des Kameraprofils '{profile}' gelten für ein anderes "
    "Seitenverhältnis ({profile_size}) als die Einzelbilder ({frame_size}). COLMAP schätzt die "
    "Intrinsik deshalb selbst."
)
CAMERA_ASPECT_HINT = (
    "Ein Kameraprofil gilt nur für den Videomodus, in dem kalibriert wurde: im selben Modus "
    "aufnehmen oder für diesen Modus neu kalibrieren (`drone-scan camera calibrate`)."
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _CameraSetup:
    """What COLMAP is told about a camera: model and optional start values."""

    model: str
    params: tuple[float, ...] | None = None


@dataclass
class _CameraGroup:
    """Flights with the same camera setup share one feature extraction call."""

    setup: _CameraSetup
    images: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Limits:
    """Parameters that determine the VRAM of features and matching."""

    image_edge_px: int
    num_features: int
    num_matches: int


class SfmStage(Stage):
    name: ClassVar[str] = "sfm"
    version: ClassVar[str] = "2"  # 2: camera profiles, sfm/cameras/ (KAN-29)
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
        # scan.yaml holds the camera profiles: a changed profile invalidates the result.
        return [scan_file] + [
            ingest / flight.id / part for flight in flights for part in (FRAMES_TABLE, FRAMES_DIR)
        ]

    def params(self, settings: Settings) -> dict[str, Any]:
        return settings.sfm.model_dump()

    def tool_versions(self) -> dict[str, str]:
        # Only the tag (= COLMAP version): a stopped docker must not invalidate a valid result.
        return {"colmap": COLMAP_IMAGE}

    def run(self, ctx: StageContext) -> None:
        work = ctx.out_dir / COLMAP_DIR
        images = _keyframes(ctx.inputs)
        groups = _camera_groups(ctx, load_scan(ctx.scan_dir).flights, images)
        ctx.metrics["colmap_image_id"] = image_id()
        if ctx.metrics["colmap_image_id"] == "missing":  # before old results are removed
            raise StageError(
                f"Das Docker-Image '{COLMAP_IMAGE}' fehlt oder Docker ist nicht erreichbar.",
                UNAVAILABLE_HINTS,
            )
        try:
            for old in (work, ctx.out_dir / CAMERAS_DIR):
                if old.exists():
                    shutil.rmtree(old)
            (work / _MODELS_DIR).mkdir(parents=True)
            (work / _IMAGE_LIST).write_text("\n".join(images) + "\n", encoding="utf-8")
            for index, group in enumerate(groups):
                if len(groups) > 1:
                    text = "\n".join(group.images) + "\n"
                    (work / _group_list(index)).write_text(text, encoding="utf-8")
        except OSError as exc:
            raise StageError("Ergebnisse konnten nicht geschrieben werden.", [WRITE_HINT]) from exc
        colmap = Colmap(
            mounts=[
                Mount(ctx.scan_dir / IngestStage.name, "/ingest", read_only=True),
                Mount(work, "/sfm"),
            ],
            gpu_device=ctx.settings.gpu_device,
            log_dir=work / "logs",
        )
        try:
            _Reconstruction(ctx, colmap, work, images, groups).run()
        except ColmapUnavailableError as exc:
            raise StageError(str(exc), UNAVAILABLE_HINTS) from exc
        except ColmapOutOfMemoryError as exc:
            raise OutOfMemoryError(str(exc), [GPU_OOM_HINT]) from exc
        except ColmapError as exc:
            raise StageError(str(exc), [COLMAP_HINT]) from exc
        finally:
            ctx.record_peak_vram(colmap.peak_vram_mib)


class _Reconstruction:
    """One run of the COLMAP steps; collects metrics in the stage context."""

    def __init__(
        self,
        ctx: StageContext,
        colmap: Colmap,
        work: Path,
        images: list[str],
        groups: list[_CameraGroup],
    ) -> None:
        self._ctx = ctx
        self._cfg: SfmSettings = ctx.settings.sfm
        self._colmap = colmap
        self._work = work
        self._images = images
        self._groups = groups
        self._exhaustive = len(images) <= self._cfg.exhaustive_max_keyframes
        self._seconds: dict[str, float] = {}

    def run(self) -> None:
        self._ctx.metrics["keyframes"] = len(self._images)
        self._ctx.metrics["matcher"] = "exhaustive" if self._exhaustive else "sequential+loop"
        self._ctx.metrics["seconds"] = self._seconds
        self._ctx.metrics["oom_retry"] = False
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

    def _features_and_matches(self) -> None:
        cfg = self._cfg
        limits = _Limits(cfg.max_image_edge_px, cfg.max_num_features, cfg.max_num_matches)
        self._ctx.metrics["limits"] = asdict(limits)
        try:
            self._extract(limits)
            self._match(limits)
        except ColmapOutOfMemoryError:
            edge = min(cfg.oom_retry_image_edge_px, cfg.max_image_edge_px)
            count = min(cfg.oom_retry_num_features, cfg.max_num_features)
            matches = min(cfg.oom_retry_num_matches, cfg.max_num_matches)
            log.warning("sfm_oom_retry", image_edge_px=edge, num_features=count)
            self._ctx.metrics["oom_retry"] = True
            self._ctx.warn(
                "Der Grafikspeicher reichte nicht, zweiter Versuch mit kleineren Werten "
                f"({edge} px, {count} Merkmale je Bild). Für die volle Qualität andere "
                "GPU-Programme schließen und mit --force wiederholen."
            )
            for leftover in self._work.glob(f"{_DATABASE}*"):
                leftover.unlink()
            reduced = _Limits(edge, count, matches)
            self._ctx.metrics["limits"] = asdict(reduced)
            self._extract(reduced)
            self._match(reduced)

    def _extract(self, limits: _Limits) -> None:
        """One call per camera setup (the ImageReader options apply to all listed images)."""
        for index, group in enumerate(self._groups):
            image_list = _IMAGE_LIST if len(self._groups) == 1 else _group_list(index)
            options: dict[str, str | int | float] = {
                "database_path": f"/sfm/{_DATABASE}",
                "image_path": "/ingest",
                "image_list_path": f"/sfm/{image_list}",
                "ImageReader.camera_model": group.setup.model,
                "ImageReader.single_camera_per_folder": 1,  # one folder per flight
                "FeatureExtraction.use_gpu": 1,
                "FeatureExtraction.gpu_index": 0,  # the container sees a single GPU
                "FeatureExtraction.max_image_size": limits.image_edge_px,
                "SiftExtraction.max_num_features": limits.num_features,
            }
            if group.setup.params is not None:
                options["ImageReader.camera_params"] = ",".join(
                    f"{value:.17g}" for value in group.setup.params
                )
            result = self._colmap.run("feature_extractor", options, gpu=True)
            self._record("features", result)

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
            options["SequentialMatching.overlap"] = self._cfg.sequential_overlap
            if self._build_vocab_tree():
                options["SequentialMatching.loop_detection"] = 1
                options["SequentialMatching.vocab_tree_path"] = f"/sfm/{_VOCAB_TREE}"
            result = self._colmap.run("sequential_matcher", options, gpu=True)
            (self._work / _VOCAB_TREE).unlink(missing_ok=True)
        self._record("matching", result)

    def _build_vocab_tree(self) -> bool:
        """Vocabulary tree from the scan's own features: no download, no foreign weights."""
        options: dict[str, str | int | float] = {
            "database_path": f"/sfm/{_DATABASE}",
            "vocab_tree_path": f"/sfm/{_VOCAB_TREE}",
            "num_visual_words": self._cfg.loop_vocab_words,
            "num_iterations": self._cfg.loop_vocab_iterations,
            "num_rounds": self._cfg.loop_vocab_rounds,
            "max_num_descriptors": self._cfg.loop_vocab_max_descriptors,
        }
        try:
            result = self._colmap.run(
                "vocab_tree_builder", options, timeout_s=self._cfg.loop_vocab_timeout_s
            )
            self._record("vocab_tree", result)
        except ColmapUnavailableError:
            raise
        except ColmapError as exc:
            # Without features COLMAP 4.2.1 hangs. The scan stays usable, but drift is not
            # corrected and the registration thresholds cannot detect it: always warn.
            self._ctx.metrics["matcher"] = "sequential"
            if LOOP_WARNING not in self._ctx.warnings:
                self._ctx.warn(LOOP_WARNING)
                timeout = isinstance(exc, ColmapTimeoutError)
                self._ctx.hint(LOOP_TIMEOUT_HINT if timeout else LOOP_FAILED_HINT)
            return False
        self._ctx.metrics["matcher"] = "sequential+loop"
        if LOOP_WARNING in self._ctx.warnings:  # second attempt after an OOM was successful
            self._ctx.warnings.remove(LOOP_WARNING)
            loop_hints = (LOOP_TIMEOUT_HINT, LOOP_FAILED_HINT)
            self._ctx.hints[:] = [hint for hint in self._ctx.hints if hint not in loop_hints]
        return True

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
            registration_ratio=math.floor(ratio * 10000 + 1e-9) / 10000,
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
        self._store_cameras(report, sparse)

    def _store_cameras(self, report: ModelReport, sparse: Path) -> None:
        """Estimated intrinsics per flight as camera profiles (reusable with --camera)."""
        cameras = read_cameras(sparse)
        flight_cameras: dict[str, int] = {}
        for image, camera_id in report.image_cameras.items():
            flight_cameras.setdefault(_flight(image), camera_id)
        metrics = self._ctx.metrics["cameras"]
        for flight, camera_id in sorted(flight_cameras.items()):
            camera = cameras[camera_id]
            intrinsics = Intrinsics(width=camera.width, height=camera.height, params=camera.params)
            profile = CameraProfile(
                name=flight,
                description=f"Von COLMAP geschätzt (Flug {flight}).",
                model=camera.model,  # type: ignore[arg-type]  # one of the models we passed
                intrinsics=intrinsics,
                source="colmap",
            )
            try:
                write_camera(profile, self._ctx.out_dir / CAMERAS_DIR / f"{flight}.yaml")
            except OSError as exc:
                raise StageError(
                    "Ergebnisse konnten nicht geschrieben werden.", [WRITE_HINT]
                ) from exc
            metrics[flight]["estimated"] = intrinsics.model_dump()

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


def _group_list(index: int) -> str:
    return f"image_list_{index}.txt"


def _frame_size(scan_dir: Path, flight: str) -> tuple[int, int]:
    path = scan_dir / IngestStage.name / flight / MANIFEST_NAME
    try:
        summary = FlightSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputMissingError(
            f"Flug {flight}: Angaben zu den Einzelbildern fehlen.", [INPUT_MISSING_HINT]
        ) from exc
    return summary.frame_width, summary.frame_height


def _camera_setup(ctx: StageContext, flight: FlightMeta) -> _CameraSetup:
    profile = flight.camera
    if profile is None:
        return _CameraSetup(ctx.settings.sfm.camera_model)
    if profile.intrinsics is None:
        return _CameraSetup(profile.model)
    width, height = _frame_size(ctx.scan_dir, flight.id)
    params = profile.params_for(width, height)
    if params is None:
        ctx.warn(
            CAMERA_ASPECT_WARNING.format(
                flight=flight.id,
                profile=profile.name,
                profile_size=f"{profile.intrinsics.width}x{profile.intrinsics.height}",
                frame_size=f"{width}x{height}",
            )
        )
        if CAMERA_ASPECT_HINT not in ctx.hints:
            ctx.hint(CAMERA_ASPECT_HINT)
        return _CameraSetup(profile.model)
    return _CameraSetup(profile.model, tuple(params))


def _camera_groups(
    ctx: StageContext, flights: list[FlightMeta], images: list[str]
) -> list[_CameraGroup]:
    """Group the keyframes by camera setup, in the order of the flights."""
    setups = {flight.id: _camera_setup(ctx, flight) for flight in flights}
    ctx.metrics["cameras"] = {
        flight.id: {
            "profile": flight.camera.name if flight.camera else None,
            "model": setups[flight.id].model,
            "start_params": list(setups[flight.id].params or []) or None,
        }
        for flight in flights
    }
    groups: dict[_CameraSetup, _CameraGroup] = {}
    for image in images:
        setup = setups[_flight(image)]
        groups.setdefault(setup, _CameraGroup(setup)).images.append(image)
    return list(groups.values())


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
