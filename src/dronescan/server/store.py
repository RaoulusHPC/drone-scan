# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistent job store in SQLite (``<workspace>/jobs.sqlite3``, KAN-52).

Two tables: ``jobs`` (one row per job, current state) and ``job_events`` (append-only
progress log, the source of the SSE stream; ``seq`` is the SSE event id). A status change
and its event are written in one transaction, so a reader that sees a final status also
sees the final event. Every operation opens its own connection (thread-safe, WAL mode).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

DB_NAME = "jobs.sqlite3"
_BUSY_TIMEOUT_S = 30.0

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
FINAL: frozenset[JobStatus] = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
RESUMABLE: frozenset[JobStatus] = frozenset({"failed", "cancelled", "interrupted"})
EventType = Literal["job", "stage"]
StageStatus = Literal[
    "running", "ok", "skipped", "input_missing", "quality_insufficient", "oom", "error"
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL,
    created TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_scan ON jobs (scan_id);
CREATE TABLE IF NOT EXISTS job_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs (id),
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS job_events_job ON job_events (job_id, seq);
"""


def _now() -> datetime:
    return datetime.now(UTC)


class Job(BaseModel):
    """A pipeline job: stages of one scan, run one after another."""

    id: str
    scan_id: str
    stages: list[str]
    profile: str | None = Field(description="Profile of the stages; None: the server's profile.")
    force: bool
    status: JobStatus = "queued"
    stages_done: int = Field(default=0, description="Finished stages; resume starts after them.")
    current_stage: str | None = None
    progress: float = Field(default=0.0, ge=0, le=100, description="Percent of the stages done.")
    oom_retried: bool = False
    exit_code: int | None = None
    error: str | None = None
    error_code: str | None = None
    warnings: list[str] = []
    hints: list[str] = []
    created: datetime = Field(default_factory=_now)
    started: datetime | None = None
    finished: datetime | None = None


class JobEvent(BaseModel):
    """Progress event as sent over SSE (event name = ``type``, id = ``seq``)."""

    seq: int = 0
    job_id: str
    type: EventType
    status: JobStatus
    stage: str | None = None
    stage_status: StageStatus | None = None
    stage_index: int | None = Field(default=None, description="0-based index in the job.")
    stage_count: int
    percent: float
    warnings: list[str] = []
    hints: list[str] = []
    error: str | None = None
    error_code: str | None = None


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


class JobStore:
    """All reads and writes of jobs and events."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One connection per operation; the ``with`` block is one transaction."""
        with closing(sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_S)) as conn, conn:
            yield conn

    def add(self, job: Job, event: JobEvent | None = None) -> Job:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, scan_id, created, data) VALUES (?, ?, ?, ?)",
                (job.id, job.scan_id, job.created.isoformat(), job.model_dump_json()),
            )
            if event is not None:
                self._insert_event(conn, event)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else Job.model_validate_json(row[0])

    def jobs(self, scan_id: str | None = None) -> list[Job]:
        """Jobs, newest first."""
        query = "SELECT data FROM jobs"
        args: tuple[Any, ...] = ()
        if scan_id is not None:
            query, args = query + " WHERE scan_id = ?", (scan_id,)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY created DESC, rowid DESC", args).fetchall()
        return [Job.model_validate_json(row[0]) for row in rows]

    def save(self, job: Job, event: JobEvent | None = None) -> Job:
        """Store the new state of ``job`` (and its event) atomically."""
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET data = ? WHERE id = ?", (job.model_dump_json(), job.id))
            if event is not None:
                self._insert_event(conn, event)
        return job

    def claim_next(self) -> Job | None:
        """Oldest queued job, already switched to ``running`` (no event yet)."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for (data,) in conn.execute("SELECT data FROM jobs ORDER BY created, rowid"):
                job = Job.model_validate_json(data)
                if job.status == "queued":
                    job.status = "running"
                    job.started = job.started or _now()
                    conn.execute(
                        "UPDATE jobs SET data = ? WHERE id = ?", (job.model_dump_json(), job.id)
                    )
                    return job
        return None

    def events_after(self, job_id: str, seq: int) -> list[JobEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT seq, data FROM job_events WHERE job_id = ? AND seq > ? ORDER BY seq",
                (job_id, seq),
            ).fetchall()
        return [JobEvent.model_validate({**json.loads(data), "seq": s}) for s, data in rows]

    @staticmethod
    def _insert_event(conn: sqlite3.Connection, event: JobEvent) -> None:
        conn.execute(
            "INSERT INTO job_events (job_id, data) VALUES (?, ?)",
            (event.job_id, event.model_dump_json(exclude={"seq"})),
        )
