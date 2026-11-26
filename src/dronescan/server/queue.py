# SPDX-License-Identifier: AGPL-3.0-or-later
"""Job queue: runs the stages of one job after another, one job at a time (KAN-52).

* Every stage runs as its own process ``python -m dronescan <stage> --scan ...`` in a new
  process group. The server process never touches the GPU: allocator limit, VRAM peak and
  structlog stay per stage process, and a crash of a stage cannot take the server down.
* GPU concurrency 1: the queue has exactly one worker thread, and the stage runner takes the
  machine-wide GPU lock (KAN-80) as before. That lock stays the only GPU serialisation, so
  CLI runs next to the server are covered as well.
* Cancel: a queued job ends at once; a running stage gets SIGINT (the runner writes its
  manifest and releases the GPU lock, COLMAP/worker containers end via their watchdog) and
  SIGKILL after ``server.cancel_grace_s``.
* Resume: failed, cancelled and interrupted jobs are queued again and continue with the
  first unfinished stage. Jobs still ``running`` when the server starts were cut off by a
  crash and become ``interrupted``.
* Exit 30 (OOM): one retry of the stage with ``server.oom_retry_profile`` of the job's
  profile; the remaining stages then use that profile too.

Stage output goes to ``<scan>/jobs/<job_id>.log``; progress and manifest results are
stored as events (`dronescan.server.store`).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog

from dronescan.common import error_catalog as catalog
from dronescan.common.config import ProfileError, ServerSettings, load_settings
from dronescan.common.stage import MANIFEST_NAME, ExitCode, Manifest, load_manifest
from dronescan.server.store import Job, JobEvent, JobStatus, JobStore, StageStatus, new_job_id

JOBS_DIR = "jobs"
StageName = Literal["ingest", "sfm", "recon"]
PIPELINE: tuple[StageName, ...] = ("ingest", "sfm", "recon")
"""Stage commands in pipeline order (search index follows with EP-07)."""

StageCommand = Callable[[str, Path, str | None, bool], list[str]]

log = structlog.get_logger(__name__)


class JobStateError(Exception):
    """The job is in a state that does not allow the operation; ``code`` names the reason."""

    def __init__(self, code: str) -> None:
        super().__init__(catalog.cause(code))
        self.code = code


def stage_command(stage: str, scan_dir: Path, profile: str | None, force: bool) -> list[str]:
    """``drone-scan <stage>`` with the interpreter of the server (same environment)."""
    command = [sys.executable, "-m", "dronescan", stage, "--scan", str(scan_dir)]
    if profile is not None:
        command += ["--profile", profile]
    return command + (["--force"] if force else [])


def _percent(done: int, total: int) -> float:
    return round(100.0 * done / total, 1) if total else 100.0


class JobQueue:
    """Single worker thread over a `JobStore`; all public methods are thread-safe."""

    def __init__(
        self,
        store: JobStore,
        workspace: Path,
        settings: ServerSettings,
        command: StageCommand = stage_command,
    ) -> None:
        self.store = store
        self.workspace = workspace
        self.settings = settings
        self.command = command
        self._lock = threading.Lock()  # guards the fields below and every job state change
        self._wake = threading.Event()
        self._stopping = False
        self._cancelled: set[str] = set()
        self._running: tuple[str, subprocess.Popen[bytes]] | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        self._interrupt_leftovers()
        self._thread = threading.Thread(target=self._loop, name="job-queue", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker; a running stage is interrupted and its job becomes resumable."""
        with self._lock:
            self._stopping = True
            running = self._running
        self._wake.set()
        if running is not None:
            self._signal(running[1], signal.SIGINT)
        if self._thread is not None:
            self._thread.join()

    def _interrupt_leftovers(self) -> None:
        for job in self.store.jobs():
            if job.status == "running":
                self._interrupted(job)

    # --- API operations ----------------------------------------------------------------
    def submit(self, scan_id: str, stages: list[str], profile: str | None, force: bool) -> Job:
        with self._lock:
            if any(j.status in ("queued", "running") for j in self.store.jobs(scan_id)):
                raise JobStateError("server.job_conflict")
            job = Job(id=new_job_id(), scan_id=scan_id, stages=stages, profile=profile, force=force)
            self.store.add(job, _job_event(job))
        self._wake.set()
        log.info("job_queued", job_id=job.id, stages=stages)
        return job

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._get(job_id)
            if job.status == "queued":
                cause = catalog.cause("common.interrupted")
                return self._finish(job, "cancelled", error=cause)
            if job.status != "running":
                raise JobStateError("server.job_not_cancellable")
            self._cancelled.add(job_id)
            running = self._running
        if running is not None and running[0] == job_id:
            self._signal(running[1], signal.SIGINT)
        log.info("job_cancel_requested", job_id=job_id)
        return job

    def resume(self, job_id: str) -> Job:
        with self._lock:
            job = self._get(job_id)
            if job.status not in ("failed", "cancelled", "interrupted"):
                raise JobStateError("server.job_not_resumable")
            if any(
                j.status in ("queued", "running") and j.id != job_id
                for j in self.store.jobs(job.scan_id)
            ):
                raise JobStateError("server.job_conflict")
            job = job.model_copy(
                update={
                    "status": "queued",
                    "exit_code": None,
                    "error": None,
                    "error_code": None,
                    "hints": [],
                    "finished": None,
                }
            )
            self.store.save(job, _job_event(job))
        self._wake.set()
        log.info("job_resumed", job_id=job_id, stages_done=job.stages_done)
        return job

    def _get(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise JobStateError("server.job_not_found")
        return job

    # --- worker ------------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            with self._lock:
                if self._stopping:
                    return
                job = self.store.claim_next()
                if job is not None:
                    self.store.save(job, _job_event(job))
            if job is None:
                self._wake.wait(self.settings.queue_poll_s)
                self._wake.clear()
                continue
            try:
                self._run(job)
            except Exception as exc:  # the worker must survive any single job
                log.exception("job_crashed", job_id=job.id)
                error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._finish(self._get(job.id), "failed", error=error, code="common.unexpected")

    def _run(self, job: Job) -> None:
        scan_dir = self.workspace / job.scan_id
        log_path = scan_dir / JOBS_DIR / f"{job.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("job_started", job_id=job.id, stages_done=job.stages_done)
        while job.stages_done < len(job.stages):
            index = job.stages_done
            stage = job.current_stage = job.stages[index]
            self._stage_event(job, index, None, "running")
            code, manifest, skipped = self._run_stage(job, stage, scan_dir, log_path)
            with self._lock:
                stopping, cancelled = self._stopping, job.id in self._cancelled
                self._cancelled.discard(job.id)
            if cancelled or stopping:
                self._stage_event(job, index, manifest, "error")
                if cancelled:
                    self._finish(job, "cancelled", error=catalog.cause("common.interrupted"))
                else:
                    self._interrupted(job)
                return
            retry = self._retry_profile(job) if code == ExitCode.OOM else None
            if retry is not None:
                job.warnings.append(
                    f"Stufe {stage}: zu wenig Speicher, Wiederholung mit Profil {retry}."
                )
                self._stage_event(job, index, manifest, "oom")
                job.profile, job.oom_retried = retry, True
                continue
            if code != ExitCode.OK:
                self._stage_event(job, index, manifest, None)
                self._fail(job, code, manifest)
                return
            job.stages_done += 1
            job.progress = _percent(job.stages_done, len(job.stages))
            if manifest is not None:
                job.warnings.extend(manifest.warnings)
            self._stage_event(job, index, manifest, "skipped" if skipped else "ok")
        job.current_stage = None
        self._finish(job, "succeeded")

    def _run_stage(
        self, job: Job, stage: str, scan_dir: Path, log_path: Path
    ) -> tuple[int, Manifest | None, bool]:
        """Run one stage process; returns exit code, its manifest and whether it was cached."""
        manifest_path = scan_dir / stage / MANIFEST_NAME
        before = load_manifest(manifest_path)
        command = self.command(stage, scan_dir, job.profile, job.force)
        with log_path.open("ab") as output:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group: SIGINT reaches the whole stage
            )
            with self._lock:
                self._running = (job.id, proc)
                interrupt = self._stopping or job.id in self._cancelled
            if interrupt:  # cancelled between claiming the job and starting the process
                self._signal(proc, signal.SIGINT)
            try:
                code = proc.wait()
            finally:
                with self._lock:
                    self._running = None
        manifest = load_manifest(manifest_path)
        skipped = (
            code == 0
            and before is not None
            and manifest is not None
            and before.finished == manifest.finished
        )
        return code, manifest, skipped

    def _signal(self, proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        """Send ``sig`` to the stage's process group; escalate to SIGKILL after the grace time."""
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return

        def kill_late() -> None:
            try:
                proc.wait(timeout=self.settings.cancel_grace_s)
            except subprocess.TimeoutExpired:
                log.warning("stage_killed", pid=proc.pid)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)

        threading.Thread(target=kill_late, name="stage-kill", daemon=True).start()

    def _retry_profile(self, job: Job) -> str | None:
        if job.oom_retried:
            return None
        try:
            retry = load_settings(job.profile).server.oom_retry_profile
        except (ProfileError, ValueError):
            return None
        return retry if retry != job.profile else None

    # --- state changes -----------------------------------------------------------------
    def _fail(self, job: Job, code: int, manifest: Manifest | None) -> None:
        job.exit_code = code
        if manifest is None or manifest.status == "ok":  # process died before its manifest
            crashed = "server.stage_crashed"
            self._finish(job, "failed", catalog.cause(crashed), crashed, catalog.hints(crashed))
            return
        self._finish(job, "failed", manifest.error, manifest.error_code, list(manifest.hints))

    def _finish(
        self,
        job: Job,
        status: JobStatus,
        error: str | None = None,
        code: str | None = None,
        hints: list[str] | None = None,
    ) -> Job:
        """Final state of ``job`` plus its event (caller holds no lock or ``self._lock``)."""
        job.status, job.finished = status, datetime.now(UTC)
        job.error, job.error_code, job.hints = error, code, list(hints or [])
        self.store.save(job, _job_event(job))
        log.info("job_finished", job_id=job.id, status=status, error_code=code)
        return job

    def _interrupted(self, job: Job) -> None:
        code = "server.job_interrupted"
        self._finish(job, "interrupted", catalog.cause(code), code, catalog.hints(code))

    def _stage_event(
        self, job: Job, index: int, manifest: Manifest | None, status: StageStatus | None
    ) -> None:
        """Save ``job`` with a stage event; ``status`` None: the manifest's status."""
        event = JobEvent(
            job_id=job.id,
            type="stage",
            status=job.status,
            stage=job.stages[index],
            stage_status=status or (manifest.status if manifest else "error"),
            stage_index=index,
            stage_count=len(job.stages),
            percent=job.progress,
        )
        if manifest is not None:
            event.warnings, event.hints = list(manifest.warnings), list(manifest.hints)
            event.error, event.error_code = manifest.error, manifest.error_code
        self.store.save(job, event)


def _job_event(job: Job) -> JobEvent:
    return JobEvent(
        job_id=job.id,
        type="job",
        status=job.status,
        stage=job.current_stage,
        stage_count=len(job.stages),
        percent=job.progress,
        warnings=list(job.warnings),
        hints=list(job.hints),
        error=job.error,
        error_code=job.error_code,
    )
