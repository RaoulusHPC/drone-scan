# SPDX-License-Identifier: AGPL-3.0-or-later
"""REST API of the local server (KAN-52): scans, flights, jobs, results, progress via SSE.

OpenAPI 3.1 at ``/api/openapi.json``. The interactive docs pages are off: they load their
scripts from a CDN, and the server makes no network access (CLAUDE.md rule 3; FastAPI's
OpenTelemetry hooks are disabled as well). Errors are JSON ``{code, cause, hints}`` with a
code from ``error_catalog.yaml``. The app only serves what lies in the workspace.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import anyio
import structlog
from fastapi import Depends, FastAPI, Header, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from dronescan import __version__
from dronescan.capture.camera import CameraError, CameraProfile, load_camera
from dronescan.capture.scan import (
    SCAN_FILE,
    FlightMeta,
    ScanError,
    ScanMeta,
    add_flight,
    create_scan,
    load_scan,
)
from dronescan.common import error_catalog as catalog
from dronescan.common.config import Settings, available_profiles
from dronescan.common.stage import MANIFEST_NAME, Manifest, load_manifest
from dronescan.server.queue import (
    PIPELINE,
    JobQueue,
    JobStateError,
    StageCommand,
    StageName,
    stage_command,
)
from dronescan.server.store import DB_NAME, FINAL, Job, JobEvent, JobStore

API_PREFIX = "/api"
GIB = 1024**3
UPLOAD_DIR = ".uploads"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".avi"})
RESULT_FILES: dict[str, tuple[str, ...]] = {
    "recon": ("splat/scene.ply", "splat/metrics.json"),
}
"""Downloadable artifacts per stage (relative to ``<scan>/<stage>/``); nothing else is served."""

_SCAN_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_FILENAME = re.compile(r"^[\w][\w .-]*$")
_CHUNK = 1024 * 1024

log = structlog.get_logger(__name__)


class ErrorResponse(BaseModel):
    code: str = Field(description="Code from docs/fehlercodes.md, e.g. server.scan_not_found.")
    cause: str
    hints: list[str]


class ApiError(Exception):
    """Answer with ``status_code`` and catalog ``code``; ``cause`` overrides the catalog text."""

    def __init__(self, status_code: int, code: str, cause: str | None = None) -> None:
        super().__init__(code)
        self.status_code, self.code = status_code, code
        self.cause = cause or catalog.cause(code)


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ScanCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200, examples=["Haus Muster EG"])


class ScanSummary(BaseModel):
    id: str
    name: str
    created: datetime
    flights: int


class StageResult(BaseModel):
    stage: str
    status: str
    exit_code: int
    error_code: str | None
    finished: datetime
    warnings: list[str]
    hints: list[str]
    files: list[str] = Field(description="Downloadable via .../results/{stage}/files/{path}.")


class JobCreate(BaseModel):
    scan_id: str
    stages: list[StageName] = Field(
        default=list(PIPELINE), min_length=1, description="In pipeline order, no duplicates."
    )
    profile: str | None = Field(default=None, description="Default: the server's profile.")
    force: bool = Field(default=False, description="Recompute even if a stage is cached.")

    @field_validator("stages")
    @classmethod
    def _pipeline_order(cls, stages: list[StageName]) -> list[StageName]:
        positions = [PIPELINE.index(stage) for stage in stages]
        if positions != sorted(set(positions)):
            raise ValueError(f"Stufen ohne Wiederholung in der Reihenfolge {', '.join(PIPELINE)}")
        return stages


_ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (400, 404, 409, 413, 422)
}


def existing_job(request: Request, job_id: str) -> Job:
    """Dependency: the job ``job_id`` or 404 (checked before a stream starts)."""
    store: JobStore = request.app.state.store
    job = store.get(job_id)
    if job is None:
        raise ApiError(404, "server.job_not_found")
    return job


def create_app(
    workspace: Path,
    settings: Settings,
    command: StageCommand = stage_command,
    run_queue: bool = True,
) -> FastAPI:
    """App for ``workspace``; ``run_queue`` False leaves jobs queued (tests of the API only)."""
    workspace.mkdir(parents=True, exist_ok=True)
    store = JobStore(workspace / DB_NAME)
    queue = JobQueue(store, workspace, settings.server, command)
    server = settings.server

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if run_queue:
            queue.start()
        try:
            yield
        finally:
            if run_queue:
                await run_in_threadpool(queue.stop)

    app = FastAPI(
        title="drone-scan",
        version=__version__,
        description="Lokale API: Scans, Flüge, Pipeline-Jobs, Ergebnisse, Fortschritt (SSE).",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
        responses=_ERRORS,
        telemetry={
            "tracing": False,
            "metrics": False,
            "logs": False,
            "operation_spans": False,
            "auto_configure": False,
        },
    )
    app.state.queue, app.state.store = queue, store

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        body = ErrorResponse(code=exc.code, cause=exc.cause, hints=catalog.hints(exc.code))
        return JSONResponse(body.model_dump(), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'][1:]) or e['loc'][0]}: {e['msg']}"
            for e in exc.errors()
        )
        code = "server.invalid_request"
        cause = f"{catalog.cause(code)} {details}"
        body = ErrorResponse(code=code, cause=cause, hints=catalog.hints(code))
        return JSONResponse(body.model_dump(), status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    def scan_dir(scan_id: str) -> Path:
        path = workspace / scan_id
        if not _SCAN_ID.fullmatch(scan_id) or not (path / SCAN_FILE).is_file():
            raise ApiError(404, "server.scan_not_found")
        return path

    def read_scan(scan_id: str) -> ScanMeta:
        path = scan_dir(scan_id)
        try:
            return load_scan(path)
        except ScanError as exc:
            raise ApiError(422, "capture.scan_invalid", str(exc)) from exc

    def get_job(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise ApiError(404, "server.job_not_found")
        return job

    def manifest_of(scan_id: str, stage: str) -> Manifest:
        manifest = load_manifest(scan_dir(scan_id) / stage / MANIFEST_NAME)
        if manifest is None:
            raise ApiError(404, "server.result_not_found")
        return manifest

    # --- health ------------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/health", tags=["system"])
    def health() -> Health:
        return Health(version=__version__)

    # --- scans and flights -------------------------------------------------------------
    @app.get(f"{API_PREFIX}/scans", tags=["scans"])
    def list_scans() -> list[ScanSummary]:
        scans: list[ScanSummary] = []
        for path in sorted(p for p in workspace.iterdir() if (p / SCAN_FILE).is_file()):
            try:
                meta = load_scan(path)
            except ScanError:
                log.warning("scan_unreadable")
                continue
            scans.append(
                ScanSummary(
                    id=meta.id, name=meta.name, created=meta.created, flights=len(meta.flights)
                )
            )
        return scans

    @app.post(f"{API_PREFIX}/scans", status_code=201, tags=["scans"])
    def new_scan(body: ScanCreate) -> ScanMeta:
        try:
            return create_scan(workspace, body.name)
        except ScanError as exc:
            raise ApiError(422, "server.invalid_request", str(exc)) from exc

    @app.get(f"{API_PREFIX}/scans/{{scan_id}}", tags=["scans"])
    def get_scan(scan_id: str) -> ScanMeta:
        return read_scan(scan_id)

    @app.get(f"{API_PREFIX}/scans/{{scan_id}}/flights", tags=["flights"])
    def list_flights(scan_id: str) -> list[FlightMeta]:
        return read_scan(scan_id).flights

    @app.post(
        f"{API_PREFIX}/scans/{{scan_id}}/flights",
        status_code=201,
        tags=["flights"],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/octet-stream": {"schema": {"type": "string"}}},
            }
        },
    )
    async def upload_flight(
        scan_id: str,
        request: Request,
        filename: Annotated[str, Query(description="Original file name, e.g. flug1.mp4.")],
        camera: Annotated[str | None, Query(description="Camera profile name.")] = None,
    ) -> FlightMeta:
        """Video of one battery flight as raw request body (streamed, not multipart)."""
        target_dir = scan_dir(scan_id)
        name = Path(filename).name
        if not _FILENAME.fullmatch(name) or Path(name).suffix.lower() not in VIDEO_SUFFIXES:
            raise ApiError(
                422,
                "server.invalid_request",
                f"Ungültiger Dateiname. Erlaubt: {', '.join(sorted(VIDEO_SUFFIXES))}.",
            )
        profile: CameraProfile | None = None
        if camera is not None:
            try:
                profile = load_camera(camera)
            except CameraError as exc:
                raise ApiError(422, "capture.camera_profile_invalid", str(exc)) from exc
        limit = int(server.max_upload_gb * GIB)
        uploads = target_dir / UPLOAD_DIR
        uploads.mkdir(exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=uploads))
        try:
            video = tmp / name
            size = 0
            try:
                with video.open("wb") as handle:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > limit:
                            raise ApiError(413, "server.upload_too_large")
                        await run_in_threadpool(handle.write, chunk)
            except OSError as exc:
                raise ApiError(500, "server.upload_failed") from exc
            if size == 0:
                raise ApiError(422, "server.invalid_request", "Der Upload enthält keine Daten.")
            try:
                return await run_in_threadpool(add_flight, target_dir, video, profile)
            except ScanError as exc:
                raise ApiError(422, "capture.scan_invalid", str(exc)) from exc
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # --- results -----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/scans/{{scan_id}}/results", tags=["results"])
    def list_results(scan_id: str) -> list[StageResult]:
        path = scan_dir(scan_id)
        results: list[StageResult] = []
        for stage in PIPELINE:
            manifest = load_manifest(path / stage / MANIFEST_NAME)
            if manifest is None:
                continue
            files = [f for f in RESULT_FILES.get(stage, ()) if (path / stage / f).is_file()]
            results.append(
                StageResult(
                    stage=stage,
                    status=manifest.status,
                    exit_code=manifest.exit_code,
                    error_code=manifest.error_code,
                    finished=manifest.finished,
                    warnings=manifest.warnings,
                    hints=manifest.hints,
                    files=files if manifest.status == "ok" else [],
                )
            )
        return results

    @app.get(f"{API_PREFIX}/scans/{{scan_id}}/results/{{stage}}", tags=["results"])
    def get_result(scan_id: str, stage: StageName) -> Manifest:
        return manifest_of(scan_id, stage)

    @app.get(
        f"{API_PREFIX}/scans/{{scan_id}}/results/{{stage}}/files/{{file_path:path}}",
        tags=["results"],
        response_class=FileResponse,
    )
    def get_result_file(scan_id: str, stage: StageName, file_path: str) -> FileResponse:
        manifest = manifest_of(scan_id, stage)
        target = scan_dir(scan_id) / stage / file_path
        if (
            manifest.status != "ok"
            or file_path not in RESULT_FILES.get(stage, ())
            or not target.is_file()
        ):
            raise ApiError(404, "server.result_not_found")
        return FileResponse(target, filename=Path(file_path).name)

    # --- jobs --------------------------------------------------------------------------
    @app.post(f"{API_PREFIX}/jobs", status_code=201, tags=["jobs"])
    def new_job(body: JobCreate) -> Job:
        scan_dir(body.scan_id)
        if body.profile is not None and body.profile not in available_profiles():
            raise ApiError(422, "server.profile_unknown")
        try:
            profile = body.profile or settings.profile
            return queue.submit(body.scan_id, list(body.stages), profile, body.force)
        except JobStateError as exc:
            raise ApiError(409, exc.code) from exc

    @app.get(f"{API_PREFIX}/jobs", tags=["jobs"])
    def list_jobs(scan_id: str | None = None) -> list[Job]:
        return store.jobs(scan_id)

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}", tags=["jobs"])
    def job_details(job_id: str) -> Job:
        return get_job(job_id)

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/cancel", tags=["jobs"])
    def cancel_job(job_id: str) -> Job:
        get_job(job_id)
        try:
            return queue.cancel(job_id)
        except JobStateError as exc:
            raise ApiError(409, exc.code) from exc

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/resume", tags=["jobs"])
    def resume_job(job_id: str) -> Job:
        get_job(job_id)
        try:
            return queue.resume(job_id)
        except JobStateError as exc:
            raise ApiError(409, exc.code) from exc

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/events",
        tags=["jobs"],
        response_class=EventSourceResponse,
        responses={200: {"model": JobEvent, "description": "text/event-stream of JobEvent"}},
    )
    async def job_events(
        job: Annotated[Job, Depends(existing_job)],
        last_event_id: Annotated[int | None, Header(ge=0)] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """Progress as Server-Sent Events (event = job|stage, id for Last-Event-ID).

        Ends after the event of the final job status.
        """
        job_id = job.id
        seq = last_event_id or 0
        while True:
            job = await run_in_threadpool(get_job, job_id)
            events = await run_in_threadpool(store.events_after, job_id, seq)
            for event in events:
                seq = event.seq
                yield ServerSentEvent(data=event, event=event.type, id=str(event.seq))
            if job.status in FINAL and not events:
                return
            await anyio.sleep(server.events_poll_s)

    return app
