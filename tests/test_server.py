# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local API server with job queue (KAN-52, US-037).

AC 1: FastAPI with OpenAPI 3.1; endpoints for scans, flights, jobs and results.
AC 2: SQLite job queue with GPU concurrency 1; cancelling and resuming jobs.
AC 3: progress via Server-Sent Events (stage, percent, warnings, hints).

The stages are replaced by `tests/fake_stage.py` (a real subprocess per stage), except in
the end-to-end test that runs the real `ingest` on a synthetic video.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from dronescan.cli import app as cli_app
from dronescan.cli import is_loopback
from dronescan.common import error_catalog as catalog
from dronescan.common.config import Settings, load_settings
from dronescan.server.api import create_app
from dronescan.server.queue import stage_command
from dronescan.server.store import FINAL, Job, JobStore
from tests.test_ingest import needs_ffmpeg, panning, require_ffmpeg, write_video  # noqa: F401

FAKE_STAGE = Path(__file__).with_name("fake_stage.py")
TIMEOUT_S = 20.0
runner = CliRunner()


def fake_command(stage: str, scan_dir: Path, profile: str | None, force: bool) -> list[str]:
    return [sys.executable, str(FAKE_STAGE), stage, str(scan_dir), profile or "-", str(int(force))]


def server_settings(**server: Any) -> Settings:
    settings = load_settings()
    values = {"events_poll_s": 0.05, "queue_poll_s": 0.05, "cancel_grace_s": 5.0, **server}
    return settings.model_copy(update={"server": settings.server.model_copy(update=values)})


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture
def make_client(workspace: Path) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(settings: Settings | None = None, **kwargs: Any) -> TestClient:
        app = create_app(workspace, settings or server_settings(), command=fake_command, **kwargs)
        client = TestClient(app)
        client.__enter__()  # runs the lifespan: starts the queue
        clients.append(client)
        return client

    yield make
    for client in reversed(clients):
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def new_scan(client: TestClient, workspace: Path, fake: dict[str, Any] | None = None) -> str:
    response = client.post("/api/scans", json={"name": "Testraum"})
    assert response.status_code == 201, response.text
    scan_id: str = response.json()["id"]
    (workspace / scan_id / "fake.json").write_text(json.dumps(fake or {}))
    return scan_id


def submit(client: TestClient, scan_id: str, **body: Any) -> dict[str, Any]:
    response = client.post("/api/jobs", json={"scan_id": scan_id, **body})
    assert response.status_code == 201, response.text
    job: dict[str, Any] = response.json()
    return job


def wait_for(
    client: TestClient, job_id: str, statuses: frozenset[str] | set[str] = FINAL
) -> dict[str, Any]:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {statuses}: {job}")


def wait_for_process(workspace: Path, scan_id: str, stage: str) -> None:
    """Until the fake stage process of ``stage`` runs (imports done, handlers installed)."""
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        if any(c.startswith(f"{stage} ") for c in calls(workspace, scan_id)):
            return
        time.sleep(0.02)
    raise AssertionError(f"stage process {stage} never started")


def calls(workspace: Path, scan_id: str) -> list[str]:
    path = workspace / scan_id / "calls.log"
    return path.read_text().split("\n")[:-1] if path.exists() else []


def sse(client: TestClient, job_id: str, last_id: int | None = None) -> list[dict[str, Any]]:
    """All events of the stream (it ends after the final job status)."""
    headers = {"Last-Event-ID": str(last_id)} if last_id is not None else {}
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=headers) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line:
                if "data" in current:
                    events.append(current)
                current = {}
            elif line.startswith(("event:", "id:", "data:")):
                key, _, value = line.partition(":")
                current[key] = json.loads(value) if key == "data" else value.strip()
    return events


# --- AC 1: API, OpenAPI 3.1, scans, flights, results ---------------------------------------
def test_openapi_is_3_1_and_lists_all_endpoints(client: TestClient) -> None:
    """Given the running server
    When /api/openapi.json is fetched
    Then it is OpenAPI 3.1 with endpoints for scans, flights, jobs, results and events,
    and the CDN-based docs pages are off (offline).
    """
    spec = client.get("/api/openapi.json").json()

    assert spec["openapi"].startswith("3.1")
    paths = set(spec["paths"])
    assert {
        "/api/health",
        "/api/scans",
        "/api/scans/{scan_id}",
        "/api/scans/{scan_id}/flights",
        "/api/scans/{scan_id}/results",
        "/api/scans/{scan_id}/results/{stage}",
        "/api/scans/{scan_id}/results/{stage}/files/{file_path}",
        "/api/jobs",
        "/api/jobs/{job_id}",
        "/api/jobs/{job_id}/cancel",
        "/api/jobs/{job_id}/resume",
        "/api/jobs/{job_id}/events",
    } <= paths
    assert client.get("/docs").status_code == 404
    assert client.get("/api/docs").status_code == 404


def test_scans_can_be_created_listed_and_read(client: TestClient) -> None:
    """Given an empty workspace
    When a scan is created via POST /api/scans
    Then it appears in the list and can be read by its id.
    """
    created = client.post("/api/scans", json={"name": "Haus Muster EG"})

    assert created.status_code == 201
    scan_id = created.json()["id"]
    assert scan_id == "haus-muster-eg"
    assert [s["id"] for s in client.get("/api/scans").json()] == [scan_id]
    assert client.get(f"/api/scans/{scan_id}").json()["name"] == "Haus Muster EG"


@pytest.mark.parametrize("scan_id", ["unbekannt", "..", "%2E%2E", "a/../b"])
def test_unknown_scan_is_404_with_catalog_code(client: TestClient, scan_id: str) -> None:
    """Given no such scan (or a path traversal attempt)
    When it is requested
    Then the answer is 404 with code, cause and hints from the error catalog.
    """
    response = client.get(f"/api/scans/{scan_id}")

    assert response.status_code == 404
    body = response.json()
    if "code" in body:  # paths with '/' do not reach a route at all
        assert body["code"] == "server.scan_not_found"
        assert body["hints"] == catalog.hints("server.scan_not_found")


def test_invalid_request_uses_catalog_code(client: TestClient) -> None:
    """Given a job request with stages out of pipeline order
    When it is posted
    Then the answer is 422 with server.invalid_request.
    """
    response = client.post("/api/jobs", json={"scan_id": "x", "stages": ["sfm", "ingest"]})

    assert response.status_code == 422
    assert response.json()["code"] == "server.invalid_request"


@needs_ffmpeg
def test_flight_upload_imports_the_video(
    client: TestClient, workspace: Path, tmp_path: Path
) -> None:
    """Given a scan and a synthetic video
    When it is uploaded as request body to /api/scans/{id}/flights
    Then the flight is registered (f01) and listed; no upload leftovers remain.
    """
    scan_id = new_scan(client, workspace)
    video = write_video(tmp_path / "clip.mp4", panning(12))

    response = client.post(
        f"/api/scans/{scan_id}/flights",
        params={"filename": "flug 1.mp4"},
        content=video.read_bytes(),
        headers={"content-type": "application/octet-stream"},
    )

    assert response.status_code == 201, response.text
    assert response.json()["id"] == "f01"
    assert response.json()["source_name"] == "flug 1.mp4"
    assert [f["id"] for f in client.get(f"/api/scans/{scan_id}/flights").json()] == ["f01"]
    assert list((workspace / scan_id / ".uploads").iterdir()) == []


def test_flight_upload_rejects_bad_names_and_too_large_files(
    make_client: Callable[..., TestClient], workspace: Path
) -> None:
    """Given a server with a tiny upload limit
    When a video with a path in its name, a non-video name or a too large body is uploaded
    Then the answers are 422 resp. 413 with catalog codes and no file is kept.
    """
    client = make_client(server_settings(max_upload_gb=1e-6))  # ~1 KiB
    scan_id = new_scan(client, workspace)
    url = f"/api/scans/{scan_id}/flights"

    bad = client.post(url, params={"filename": "../x.txt"}, content=b"x")
    large = client.post(url, params={"filename": "a.mp4"}, content=b"x" * 4096)

    assert bad.status_code == 422 and bad.json()["code"] == "server.invalid_request"
    assert large.status_code == 413 and large.json()["code"] == "server.upload_too_large"
    assert not (workspace / scan_id / "flights").exists()
    assert list((workspace / scan_id / ".uploads").iterdir()) == []


def test_results_list_manifests_and_serve_only_listed_files(
    client: TestClient, workspace: Path
) -> None:
    """Given a finished job whose recon stage produced scene.ply
    When the results are requested
    Then each stage's manifest summary is listed, scene.ply can be downloaded,
    and other files or paths are refused.
    """
    fake = {
        "ingest": {"warnings": ["Flug f01: unscharf"]},
        "recon": {"files": {"splat/scene.ply": "ply", "splat/secret.txt": "x"}},
    }
    scan_id = new_scan(client, workspace, fake)
    job = wait_for(client, submit(client, scan_id)["id"])
    assert job["status"] == "succeeded"

    results = client.get(f"/api/scans/{scan_id}/results").json()
    base = f"/api/scans/{scan_id}/results/recon"

    assert [r["stage"] for r in results] == ["ingest", "sfm", "recon"]
    assert results[0]["warnings"] == ["Flug f01: unscharf"]
    assert results[2]["files"] == ["splat/scene.ply"]
    assert client.get(base).json()["status"] == "ok"
    assert client.get(f"{base}/files/splat/scene.ply").content == b"ply"
    assert client.get(f"{base}/files/splat/secret.txt").status_code == 404
    assert client.get(f"{base}/files/../fake.json").status_code == 404
    assert client.get(f"/api/scans/{scan_id}/results/search").status_code == 422


def test_missing_result_is_404(client: TestClient, workspace: Path) -> None:
    """Given a scan without any stage run
    When a stage result is requested
    Then the answer is 404 server.result_not_found.
    """
    scan_id = new_scan(client, workspace)

    response = client.get(f"/api/scans/{scan_id}/results/sfm")

    assert response.status_code == 404
    assert response.json()["code"] == "server.result_not_found"


def test_serve_refuses_non_loopback_hosts() -> None:
    """Given `drone-scan serve --host 0.0.0.0`
    When it starts
    Then it refuses (exit 2, server.not_loopback); LAN access comes with KAN-54.
    """
    result = runner.invoke(cli_app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 2
    assert "server.not_loopback" in result.output
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("192.168.1.5") and not is_loopback("example.org")


# --- AC 2: queue, concurrency 1, cancel, resume -------------------------------------------
def test_job_runs_all_stages_in_order(client: TestClient, workspace: Path) -> None:
    """Given a scan
    When a job without explicit stages is submitted
    Then ingest, sfm and recon run in order with the server's profile, and the job succeeds
    with 100 % progress.
    """
    scan_id = new_scan(client, workspace)

    job = wait_for(client, submit(client, scan_id)["id"])

    assert job["status"] == "succeeded"
    assert job["progress"] == 100.0
    assert job["stages_done"] == 3
    assert calls(workspace, scan_id) == [
        "ingest low_vram_6gb 0",
        "sfm low_vram_6gb 0",
        "recon low_vram_6gb 0",
    ]
    assert (workspace / scan_id / "jobs" / f"{job['id']}.log").is_file()


def test_jobs_never_run_stages_in_parallel(client: TestClient, workspace: Path) -> None:
    """Given three scans with slow stages
    When a job is submitted for each at the same time
    Then all succeed one after another: never two stage processes at once (GPU = 1).
    """
    fake = {stage: {"sleep": 0.15} for stage in ("ingest", "sfm", "recon")}
    jobs = [submit(client, new_scan(client, workspace, fake))["id"] for _ in range(3)]

    finished = [wait_for(client, job_id) for job_id in jobs]

    assert [job["status"] for job in finished] == ["succeeded"] * 3
    assert not (workspace / "overlap.log").exists()
    starts = [job["started"] for job in finished]
    assert starts == sorted(starts)  # first in, first out


def test_second_job_for_the_same_scan_is_a_conflict(client: TestClient, workspace: Path) -> None:
    """Given a scan with a running job
    When another job for that scan is submitted
    Then the answer is 409 server.job_conflict.
    """
    scan_id = new_scan(client, workspace, {"ingest": {"sleep": 1.0}})
    job_id = submit(client, scan_id)["id"]

    response = client.post("/api/jobs", json={"scan_id": scan_id})

    assert response.status_code == 409
    assert response.json()["code"] == "server.job_conflict"
    client.post(f"/api/jobs/{job_id}/cancel")
    wait_for(client, job_id)


def test_unknown_profile_is_refused(client: TestClient, workspace: Path) -> None:
    """Given a job request with a profile that does not exist
    When it is posted
    Then the answer is 422 server.profile_unknown.
    """
    scan_id = new_scan(client, workspace)

    response = client.post("/api/jobs", json={"scan_id": scan_id, "profile": "riesig"})

    assert response.status_code == 422
    assert response.json()["code"] == "server.profile_unknown"


@pytest.mark.parametrize(
    ("exit_code", "error_code"),
    [(10, "common.input_missing"), (20, "ingest.quality_red"), (1, "common.unexpected")],
)
def test_failing_stage_fails_the_job_with_code_and_hints(
    client: TestClient, workspace: Path, exit_code: int, error_code: str
) -> None:
    """Given a stage that ends with exit 10, 20 or 1 and hints in its manifest
    When the job runs
    Then it stops there: failed, with exit code, error code and hints of the manifest.
    """
    fake = {"sfm": {"exit": exit_code, "error_code": error_code, "hints": ["Licht an"]}}
    scan_id = new_scan(client, workspace, fake)

    job = wait_for(client, submit(client, scan_id)["id"])

    assert job["status"] == "failed"
    assert (job["exit_code"], job["error_code"]) == (exit_code, error_code)
    assert job["hints"] == ["Licht an"]
    assert job["current_stage"] == "sfm"
    assert [c.split()[0] for c in calls(workspace, scan_id)] == ["ingest", "sfm"]


def test_stage_without_manifest_is_reported_as_crash(client: TestClient, workspace: Path) -> None:
    """Given a stage process that dies without writing a manifest
    When the job runs
    Then it fails with server.stage_crashed and its hints.
    """
    scan_id = new_scan(client, workspace, {"ingest": {"exit": 1, "no_manifest": True}})

    job = wait_for(client, submit(client, scan_id)["id"])

    assert job["status"] == "failed"
    assert job["error_code"] == "server.stage_crashed"
    assert job["hints"] == catalog.hints("server.stage_crashed")


def test_oom_is_retried_once_with_the_smaller_profile(client: TestClient, workspace: Path) -> None:
    """Given a job with profile default_8gb whose recon stage runs out of memory there
    When the job runs
    Then recon is repeated once with low_vram_6gb (server.oom_retry_profile), with a warning,
    and the job succeeds.
    """
    fake = {"recon": {"exit_by_profile": {"default_8gb": 30}, "error_code": "common.gpu_oom"}}
    scan_id = new_scan(client, workspace, fake)

    job = wait_for(client, submit(client, scan_id, profile="default_8gb")["id"])

    assert job["status"] == "succeeded"
    assert calls(workspace, scan_id)[-2:] == ["recon default_8gb 0", "recon low_vram_6gb 0"]
    assert any("low_vram_6gb" in w for w in job["warnings"])


def test_second_oom_fails_the_job(client: TestClient, workspace: Path) -> None:
    """Given a stage that runs out of memory with every profile
    When the job (default_8gb) runs
    Then it fails with exit 30 after exactly one retry.
    """
    scan_id = new_scan(client, workspace, {"sfm": {"exit": 30, "error_code": "common.gpu_oom"}})

    job = wait_for(client, submit(client, scan_id, profile="default_8gb")["id"])

    assert (job["status"], job["exit_code"]) == ("failed", 30)
    assert [c for c in calls(workspace, scan_id) if c.startswith("sfm")] == [
        "sfm default_8gb 0",
        "sfm low_vram_6gb 0",
    ]


def test_cancel_queued_job(client: TestClient, workspace: Path) -> None:
    """Given a job waiting behind a running one
    When the waiting job is cancelled
    Then it ends as cancelled without running any stage.
    """
    first = submit(client, new_scan(client, workspace, {"ingest": {"sleep": 1.0}}))["id"]
    waiting_scan = new_scan(client, workspace)
    waiting = submit(client, waiting_scan)["id"]

    response = client.post(f"/api/jobs/{waiting}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    client.post(f"/api/jobs/{first}/cancel")
    wait_for(client, first)
    assert calls(workspace, waiting_scan) == []
    assert client.post(f"/api/jobs/{waiting}/cancel").json()["code"] == (
        "server.job_not_cancellable"
    )


def test_cancel_running_job_interrupts_the_stage(client: TestClient, workspace: Path) -> None:
    """Given a job whose sfm stage runs for a long time
    When it is cancelled
    Then the stage process gets SIGINT, writes its manifest and the job ends as cancelled
    long before the stage would have finished.
    """
    scan_id = new_scan(client, workspace, {"sfm": {"sleep": 60}})
    job_id = submit(client, scan_id)["id"]
    wait_for_process(workspace, scan_id, "sfm")
    started = time.monotonic()

    client.post(f"/api/jobs/{job_id}/cancel")
    job = wait_for(client, job_id)

    assert job["status"] == "cancelled"
    assert time.monotonic() - started < 5
    assert job["stages_done"] == 1
    manifest = client.get(f"/api/scans/{scan_id}/results/sfm").json()
    assert manifest["error_code"] == "common.interrupted"


def test_cancel_kills_a_stage_that_ignores_sigint(
    make_client: Callable[..., TestClient], workspace: Path
) -> None:
    """Given a stage that ignores SIGINT and a grace time of 0.3 s
    When the job is cancelled
    Then the stage is killed and the job ends as cancelled.
    """
    client = make_client(server_settings(cancel_grace_s=0.3))
    scan_id = new_scan(client, workspace, {"ingest": {"sleep": 60, "ignore_sigint": True}})
    job_id = submit(client, scan_id)["id"]
    wait_for_process(workspace, scan_id, "ingest")

    client.post(f"/api/jobs/{job_id}/cancel")

    assert wait_for(client, job_id)["status"] == "cancelled"


def test_resume_continues_with_the_interrupted_stage(client: TestClient, workspace: Path) -> None:
    """Given a job cancelled during sfm (ingest finished)
    When it is resumed after the cause is gone
    Then it continues with sfm (ingest does not run again) and succeeds.
    """
    scan_id = new_scan(client, workspace, {"sfm": {"sleep": 60}})
    job_id = submit(client, scan_id, force=True)["id"]
    wait_for_process(workspace, scan_id, "sfm")
    client.post(f"/api/jobs/{job_id}/cancel")
    wait_for(client, job_id)
    (workspace / scan_id / "fake.json").write_text("{}")

    response = client.post(f"/api/jobs/{job_id}/resume")
    job = wait_for(client, job_id)

    assert response.status_code == 200 and response.json()["status"] == "queued"
    assert job["status"] == "succeeded"
    assert [c.split()[0] for c in calls(workspace, scan_id)] == ["ingest", "sfm", "sfm", "recon"]
    assert all(c.endswith(" 1") for c in calls(workspace, scan_id))  # --force is kept


def test_resume_of_a_succeeded_job_is_refused(client: TestClient, workspace: Path) -> None:
    """Given a succeeded job
    When it is resumed
    Then the answer is 409 server.job_not_resumable.
    """
    job_id = wait_for(client, submit(client, new_scan(client, workspace))["id"])["id"]

    response = client.post(f"/api/jobs/{job_id}/resume")

    assert response.status_code == 409
    assert response.json()["code"] == "server.job_not_resumable"


def test_unknown_job_is_404(client: TestClient) -> None:
    """Given no such job
    When it is read, cancelled, resumed or streamed
    Then each answer is 404 server.job_not_found.
    """
    for method, url in [
        ("GET", "/api/jobs/nope"),
        ("POST", "/api/jobs/nope/cancel"),
        ("POST", "/api/jobs/nope/resume"),
        ("GET", "/api/jobs/nope/events"),
    ]:
        response = client.request(method, url)
        assert response.status_code == 404, url
        assert response.json()["code"] == "server.job_not_found"


def test_jobs_survive_a_restart_and_running_ones_become_interrupted(
    make_client: Callable[..., TestClient], workspace: Path
) -> None:
    """Given a job stored as running (server crashed) and a finished one in jobs.sqlite3
    When the server starts again
    Then both are listed; the running one is interrupted and can be resumed.
    """
    client = make_client(run_queue=False)
    scan_id = new_scan(client, workspace)
    store = JobStore(workspace / "jobs.sqlite3")
    store.add(
        Job(
            id="crashed",
            scan_id=scan_id,
            stages=["ingest"],
            profile=None,
            force=False,
            status="running",
        )
    )
    store.add(
        Job(
            id="done",
            scan_id=scan_id,
            stages=["ingest"],
            profile=None,
            force=False,
            status="succeeded",
        )
    )

    restarted = make_client()
    jobs = {j["id"]: j for j in restarted.get("/api/jobs").json()}

    assert jobs["done"]["status"] == "succeeded"
    assert jobs["crashed"]["status"] == "interrupted"
    assert jobs["crashed"]["error_code"] == "server.job_interrupted"
    restarted.post("/api/jobs/crashed/resume")
    assert wait_for(restarted, "crashed")["status"] == "succeeded"


def test_stopping_the_server_interrupts_the_running_job(
    make_client: Callable[..., TestClient], workspace: Path
) -> None:
    """Given a running job
    When the server shuts down
    Then the stage is interrupted and the job is stored as interrupted (resumable).
    """
    client = make_client()
    scan_id = new_scan(client, workspace, {"ingest": {"sleep": 60}})
    job_id = submit(client, scan_id)["id"]
    wait_for_process(workspace, scan_id, "ingest")

    client.__exit__(None, None, None)

    job = JobStore(workspace / "jobs.sqlite3").get(job_id)
    assert job is not None and job.status == "interrupted"


# --- AC 3: progress via SSE -----------------------------------------------------------
def test_events_report_stage_percent_warnings_and_hints(
    client: TestClient, workspace: Path
) -> None:
    """Given a job whose ingest warns and whose sfm fails with hints
    When its event stream is read
    Then it contains per stage a running and a finished event with percent, the warnings and
    hints of the manifests, and it ends with the final job event.
    """
    fake = {
        "ingest": {"warnings": ["Flug f01: zu dunkel"]},
        "sfm": {"exit": 20, "error_code": "sfm.no_model", "hints": ["AprilTags platzieren"]},
    }
    scan_id = new_scan(client, workspace, fake)
    job_id = submit(client, scan_id)["id"]

    events = sse(client, job_id)

    data = [e["data"] for e in events]
    stages = [(d["stage"], d["stage_status"], d["percent"]) for d in data if d["type"] == "stage"]
    assert stages == [
        ("ingest", "running", 0.0),
        ("ingest", "ok", 33.3),
        ("sfm", "running", 33.3),
        ("sfm", "quality_insufficient", 33.3),
    ]
    ingest_done = next(d for d in data if d["stage"] == "ingest" and d["stage_status"] == "ok")
    assert ingest_done["warnings"] == ["Flug f01: zu dunkel"]
    sfm_done = data[-2]
    assert (sfm_done["hints"], sfm_done["error_code"]) == (["AprilTags platzieren"], "sfm.no_model")
    assert (data[-1]["type"], data[-1]["status"]) == ("job", "failed")
    assert [e["event"] for e in events][-1] == "job"
    ids = [int(e["id"]) for e in events]
    assert ids == sorted(ids)


def test_events_resume_after_last_event_id(client: TestClient, workspace: Path) -> None:
    """Given a finished job and the id of an event already seen
    When the stream is opened again with Last-Event-ID
    Then only the later events are sent.
    """
    job_id = submit(client, new_scan(client, workspace))["id"]
    wait_for(client, job_id)
    events = sse(client, job_id)

    later = sse(client, job_id, last_id=int(events[2]["id"]))

    assert later == events[3:]


def test_events_stream_live_while_the_job_runs(client: TestClient, workspace: Path) -> None:
    """Given a job that is still running when the stream is opened
    When the stream is read
    Then it follows the job until the final event.
    """
    scan_id = new_scan(client, workspace, {"sfm": {"sleep": 0.5}})
    job_id = submit(client, scan_id)["id"]

    events = sse(client, job_id)

    assert events[-1]["data"]["status"] == "succeeded"
    assert events[-1]["data"]["percent"] == 100.0


# --- end to end with the real stage command --------------------------------------------------
@needs_ffmpeg
@pytest.mark.usefixtures("block_network")
def test_real_ingest_job_runs_offline(tmp_path: Path, workspace: Path) -> None:
    """Given the real stage command, a scan with a synthetic flight and a blocked network
    When an ingest job runs through the API
    Then it succeeds and the ingest manifest is available as result.
    """
    app = create_app(workspace, server_settings(), command=stage_command)
    with TestClient(app) as client:
        scan_id = client.post("/api/scans", json={"name": "E2E"}).json()["id"]
        video = write_video(tmp_path / "clip.mp4", panning(48))
        upload = client.post(
            f"/api/scans/{scan_id}/flights",
            params={"filename": "clip.mp4"},
            content=video.read_bytes(),
        )
        assert upload.status_code == 201, upload.text

        job = wait_for(client, submit(client, scan_id, stages=["ingest"])["id"])

        assert job["status"] == "succeeded", job
        assert client.get(f"/api/scans/{scan_id}/results/ingest").json()["status"] == "ok"
