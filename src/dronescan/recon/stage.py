# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage ``recon`` (recon.splat): Gaussian splatting with gsplat in the 6 GB profile (KAN-41).

1. COLMAP ``image_undistorter`` (CPU, colmap container) turns the largest model of ``sfm``
   into PINHOLE images whose longest edge is ``splat.max_image_edge_px`` (binary model, read
   directly: no second container start); ffmpeg (host) decodes them to raw RGB.
2. The trainer (`dronescan.recon.train`) runs in the worker container under the PyTorch
   allocator limit of the profile (fp32 on Pascal). Every ``splat.holdout_every``-th image
   is held out; PSNR/SSIM on them go to ``recon/splat/metrics.json`` and the manifest.
3. Out of VRAM: one automatic retry with the smaller ``splat.oom_retry_*`` values and
   gsplat's packed mode. A second OOM ends the stage with exit 30.

Output: ``recon/splat/scene.ply`` (3DGS PLY) and ``recon/splat/metrics.json``. Decoded and
undistorted images are removed afterwards; logs stay in ``recon/work/logs/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

import structlog
from pydantic import ValidationError

from dronescan.capture.scan import ScanError, load_scan
from dronescan.common.config import Settings, SplatSettings
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
from dronescan.ingest.ffmpeg import FfmpegError, ffmpeg_version, image_to_rgb
from dronescan.ingest.stage import FRAMES_DIR, WRITE_HINT, IngestStage
from dronescan.recon.job import (
    JOB_NAME,
    PLY_NAME,
    RESULT_NAME,
    TrainExit,
    TrainJob,
    TrainResult,
)
from dronescan.recon.worker import (
    WORKER_IMAGE,
    Worker,
    WorkerError,
    WorkerUnavailableError,
    docker_image_id,
)
from dronescan.sfm.colmap import (
    COLMAP_IMAGE,
    Colmap,
    ColmapError,
    ColmapUnavailableError,
    Mount,
)
from dronescan.sfm.stage import COLMAP_DIR, SPARSE_DIR, SfmStage

SPLAT_DIR = "splat"
WORK_DIR = "work"
_LOGS = "logs"
_UNDISTORTED = "undistorted"
_SCENE = f"{_UNDISTORTED}/sparse"  # undistorted model (COLMAP binary)
_IMAGES = f"{_UNDISTORTED}/images"
_RAW = "raw"
_TRAIN_MODULE = "dronescan.recon.train"
_MIN_IMAGES = 2  # one for training, one held out

UNAVAILABLE_HINTS = [
    "Abhilfe: Docker starten und die Images bauen mit `docker compose build worker colmap`.",
    "GPU im Container prüfen mit `make docker-check` (NVIDIA Container Toolkit, siehe README).",
]
LOG_HINT = "Abhilfe: Logdateien in recon/work/logs/ prüfen und mit --force erneut ausführen."
TIMEOUT_HINT = (
    "Abhilfe: andere GPU-Programme schließen; auf langsamen Rechnern "
    "DRONESCAN_SPLAT__TIMEOUT_S erhöhen oder DRONESCAN_SPLAT__MAX_ITERATIONS senken."
)
FEW_IMAGES_HINTS = [
    "Mehr Einzelbilder aufnehmen: langsamer und länger filmen.",
    "Die Stufe 'sfm' muss mindestens zwei Einzelbilder verorten.",
]
OOM_RETRY_WARNING = (
    "Der Grafikspeicher hat nicht gereicht. Das Training wurde einmal mit kleineren Werten "
    "wiederholt ({gaussians} Gaussians, {edge} px): Die Szene ist weniger detailliert."
)
OVERFIT_WARNING = (
    "Die Szene passt deutlich besser zu den Trainingsbildern ({train:.1f} dB) als zu den "
    "zurückgehaltenen Bildern ({held:.1f} dB): Bereiche zwischen den Aufnahmepositionen "
    "können unscharf oder verzerrt aussehen."
)
PEAK_FALLBACK_WARNING = (
    "Der Grafikspeicher des Trainings konnte nicht mit nvidia-smi gemessen werden. "
    "peak_vram_mb ist nur der PyTorch-Wert ohne CUDA-Kontext (zu niedrig)."
)
OVERFIT_HINT = (
    "Den Raum vollständiger aufnehmen: mehr Blickwinkel, Türdurchgänge doppelt, langsam gehen."
)

log = structlog.get_logger(__name__)


class ReconStage(Stage):
    name: ClassVar[str] = "recon"
    version: ClassVar[str] = "1"
    uses_gpu: ClassVar[bool] = True
    # The trainer runs in the worker container and applies the PyTorch limit there itself;
    # this process only orchestrates and samples the container's VRAM.
    applies_vram_limit: ClassVar[bool] = False

    def inputs(self, scan_dir: Path) -> list[Path]:
        sfm = scan_dir / SfmStage.name
        manifest = load_manifest(sfm / MANIFEST_NAME)
        if manifest is None or manifest.status != "ok":
            raise InputMissingError(
                "Die Kameraposen fehlen: Stufe 'sfm' ist nicht erfolgreich gelaufen.",
                [INPUT_MISSING_HINT],
            )
        try:
            flights = load_scan(scan_dir).flights
        except ScanError as exc:
            raise InputMissingError(str(exc), [INPUT_MISSING_HINT]) from exc
        ingest = scan_dir / IngestStage.name
        return [sfm / COLMAP_DIR / SPARSE_DIR] + [ingest / f.id / FRAMES_DIR for f in flights]

    def params(self, settings: Settings) -> dict[str, Any]:
        # The timeout does not change the result.
        return settings.splat.model_dump(exclude={"timeout_s"})

    def tool_versions(self) -> dict[str, str]:
        # Only the tags: a stopped docker must not invalidate a valid result.
        return {"worker": WORKER_IMAGE, "colmap": COLMAP_IMAGE, "ffmpeg": ffmpeg_version()}

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.settings.splat
        for image in (WORKER_IMAGE, COLMAP_IMAGE):  # before old results are removed
            if docker_image_id(image) == "missing":
                raise StageError(
                    f"Das Docker-Image '{image}' fehlt oder Docker ist nicht erreichbar.",
                    UNAVAILABLE_HINTS,
                )
        ctx.metrics["worker_image_id"] = docker_image_id(WORKER_IMAGE)
        work, splat = ctx.out_dir / WORK_DIR, ctx.out_dir / SPLAT_DIR
        try:
            for old in (work, splat):
                if old.exists():
                    shutil.rmtree(old)
            work.mkdir(parents=True)
            splat.mkdir()
        except OSError as exc:
            raise StageError("Ergebnisse konnten nicht geschrieben werden.", [WRITE_HINT]) from exc
        try:
            self._undistort(ctx, work, cfg)
            result = self._train(ctx, work, cfg)
            for name in (PLY_NAME, RESULT_NAME):
                shutil.move(work / f"attempt-{result.attempt}" / name, splat / name)
        finally:
            for heavy in (work / _IMAGES, work / _RAW):  # image content, gigabytes
                shutil.rmtree(heavy, ignore_errors=True)
        self._report(ctx, result.metrics, cfg)

    # ---------- steps ----------

    def _undistort(self, ctx: StageContext, work: Path, cfg: SplatSettings) -> None:
        from dronescan.recon.scene import SceneError, read_scene, split_holdout  # numpy: lazy

        colmap = Colmap(
            mounts=[
                Mount(ctx.scan_dir / IngestStage.name, "/ingest", read_only=True),
                Mount(ctx.inputs[0], "/sparse", read_only=True),
                Mount(work, "/recon"),
            ],
            gpu_device=ctx.settings.gpu_device,
            log_dir=work / _LOGS,
        )
        try:
            colmap.run(
                "image_undistorter",
                {
                    "image_path": "/ingest",
                    "input_path": "/sparse",
                    "output_path": f"/recon/{_UNDISTORTED}",
                    "output_type": "COLMAP",
                    "max_image_size": cfg.max_image_edge_px,
                },
            )
        except ColmapUnavailableError as exc:
            raise StageError(str(exc), UNAVAILABLE_HINTS) from exc
        except ColmapError as exc:
            raise StageError(str(exc), [LOG_HINT]) from exc
        try:
            scene = read_scene(work / _SCENE)
        except SceneError as exc:
            raise StageError(str(exc), [LOG_HINT]) from exc
        if len(scene.views) < _MIN_IMAGES:
            raise QualityError(
                f"Nur {len(scene.views)} verortetes Einzelbild: zu wenig für das Training.",
                FEW_IMAGES_HINTS,
            )
        try:
            for view in scene.views:
                image_to_rgb(
                    work / _IMAGES / view.name,
                    work / _RAW / view.raw_name,
                    view.width,
                    view.height,
                )
        except FfmpegError as exc:
            raise StageError(str(exc), [LOG_HINT]) from exc
        train, held_out = split_holdout(len(scene.views), cfg.holdout_every)
        ctx.metrics.update(images=len(scene.views), images_train=len(train))
        ctx.metrics["images_holdout"] = len(held_out)

    def _train(self, ctx: StageContext, work: Path, cfg: SplatSettings) -> _Attempt:
        worker = Worker(mounts=[Mount(work, "/recon")], gpu_device=ctx.settings.gpu_device)
        job = TrainJob(
            profile=ctx.settings.profile,
            vram_limit_gb=ctx.settings.vram_limit_gb,
            iterations=cfg.max_iterations,
            max_gaussians=cfg.max_gaussians,
            max_image_edge_px=cfg.max_image_edge_px,
            sh_degree=cfg.sh_degree,
            holdout_every=cfg.holdout_every,
        )
        try:
            first = self._attempt(ctx, worker, work, job, 1)
            if first.code != TrainExit.OOM:
                return first.checked()
            retry = job.model_copy(
                update={
                    "max_gaussians": cfg.oom_retry_max_gaussians,
                    "max_image_edge_px": cfg.oom_retry_image_edge_px,
                    "packed": True,
                }
            )
            log.warning("recon_oom_retry", gaussians=retry.max_gaussians)
            ctx.warn(
                OOM_RETRY_WARNING.format(
                    gaussians=retry.max_gaussians, edge=retry.max_image_edge_px
                )
            )
            ctx.metrics["oom_retry"] = True
            return self._attempt(ctx, worker, work, retry, 2).checked()
        except WorkerUnavailableError as exc:
            raise StageError(str(exc), UNAVAILABLE_HINTS) from exc
        except WorkerError as exc:
            raise StageError(str(exc), [TIMEOUT_HINT]) from exc
        finally:
            # nvidia-smi per container PID includes the CUDA context; without it only the
            # allocator peak recorded by _attempt remains (a lower bound, flagged).
            if worker.peak_vram_mib is not None:
                ctx.record_peak_vram(worker.peak_vram_mib)
                ctx.metrics["peak_vram_source"] = "process"
            else:
                ctx.metrics["peak_vram_source"] = "allocator"
                ctx.warn(PEAK_FALLBACK_WARNING)

    def _attempt(
        self, ctx: StageContext, worker: Worker, work: Path, job: TrainJob, attempt: int
    ) -> _Attempt:
        job_file = work / f"{attempt}-{JOB_NAME}"
        job_file.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        out = f"/recon/attempt-{attempt}"
        code = worker.run(
            _TRAIN_MODULE,
            ["--scene", f"/recon/{_SCENE}", "--raw", f"/recon/{_RAW}"]
            + ["--job", f"/recon/{job_file.name}", "--out", out],
            log_file=work / _LOGS / f"train-{attempt}.log",
            timeout_s=ctx.settings.splat.timeout_s,
        )
        ctx.metrics.setdefault("peak_vram_mib_attempts", []).append(worker.last_peak_vram_mib)
        result: TrainResult | None = None
        try:
            text = (work / f"attempt-{attempt}" / RESULT_NAME).read_text(encoding="utf-8")
            result = TrainResult.model_validate_json(text)
        except (OSError, ValidationError):
            log.error("recon_no_metrics", attempt=attempt, returncode=code)
        if result is not None:
            # allocator peak without CUDA context: fallback if nvidia-smi missed the process
            ctx.record_peak_vram(result.peak_reserved_mib)
        return _Attempt(attempt, code, result)

    def _report(self, ctx: StageContext, result: TrainResult, cfg: SplatSettings) -> None:
        ctx.metrics.update(
            psnr_db=result.psnr_db,
            ssim=result.ssim,
            train_psnr_db=result.train_psnr_db,
            gaussians=result.gaussians,
            iterations=result.iterations,
            image_edge_px=result.image_edge_px,
            train_seconds=result.train_seconds,
            seconds_per_iteration=result.seconds_per_iteration,
            peak_allocated_mib=result.peak_allocated_mib,
            peak_reserved_mib=result.peak_reserved_mib,
            device=result.device,
            init_points=result.init_points,
            init_points_used=result.init_points_used,
        )
        ctx.metrics.setdefault("oom_retry", False)
        held, train = result.psnr_db, result.train_psnr_db
        if held is not None and train is not None and train - held > cfg.overfit_warn_gap_db:
            ctx.warn(OVERFIT_WARNING.format(train=train, held=held))
            ctx.hint(OVERFIT_HINT)


class _Attempt:
    """Outcome of one trainer run."""

    def __init__(self, attempt: int, code: int, result: TrainResult | None) -> None:
        self.attempt = attempt
        self.code = code
        self._result = result

    @property
    def metrics(self) -> TrainResult:
        assert self._result is not None
        return self._result

    def checked(self) -> _Attempt:
        """Self if the training succeeded, else the matching StageError."""
        detail = self._result.error if self._result and self._result.error else None
        if self.code == TrainExit.OOM:
            raise OutOfMemoryError(
                "Der Grafikspeicher hat auch mit kleineren Werten nicht gereicht.",
                [GPU_OOM_HINT],
            )
        if self.code != TrainExit.OK or self._result is None or not self._result.ok:
            message = f"Das Training ist fehlgeschlagen (Code {self.code})"
            raise StageError(f"{message}: {detail}." if detail else f"{message}.", [LOG_HINT])
        return self
