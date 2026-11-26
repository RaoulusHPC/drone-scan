# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scan directory and flight import (architecture §5.3).

``workspace/<scan_id>/scan.yaml`` lists the flights; each flight video is copied to
``flights/<flight_id>/video.<ext>``. ``scan.yaml`` is written atomically.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from dronescan.capture.camera import CameraProfile
from dronescan.capture.probe import ProbeError, VideoMeta, probe_video

SCAN_FILE = "scan.yaml"
LOCK_FILE = ".scan.lock"
_HASH_CHUNK = 1024 * 1024
_MAX_ID_TRIES = 1000


class ScanError(Exception):
    """Operation failed because of the input; the message names cause and remedy (German)."""


class FlightMeta(VideoMeta):
    id: str = Field(pattern=r"^f\d{2,}$")  # used as directory name by the stages
    source_name: str
    sha256: str
    added: datetime
    camera: CameraProfile | None = None  # copy of the profile; None: COLMAP estimates (KAN-29)


class ScanMeta(BaseModel):
    id: str
    name: str
    created: datetime
    flights: list[FlightMeta] = Field(default_factory=list)


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def _write_scan(scan_dir: Path, meta: ScanMeta) -> None:
    text = yaml.safe_dump(
        meta.model_dump(mode="json"), sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    fd, tmp = tempfile.mkstemp(dir=scan_dir, prefix=".scan-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, scan_dir / SCAN_FILE)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _scan_lock(scan_dir: Path) -> Iterator[None]:
    """Serialise read-modify-write of scan.yaml between concurrent imports."""
    with (scan_dir / LOCK_FILE).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def create_scan(workspace: Path, name: str) -> ScanMeta:
    """Create ``workspace/<id>/scan.yaml``; the id is a slug of ``name`` (unique)."""
    slug = slugify(name)
    if not slug:
        raise ScanError(
            "Der Scan-Name enthält keine Buchstaben oder Ziffern. Abhilfe: einen Namen wie "
            "'Haus Muster EG' verwenden."
        )
    workspace.mkdir(parents=True, exist_ok=True)
    for index in range(1, _MAX_ID_TRIES):
        scan_id = slug if index == 1 else f"{slug}-{index}"
        scan_dir = workspace / scan_id
        try:
            scan_dir.mkdir()
        except FileExistsError:
            continue
        meta = ScanMeta(id=scan_id, name=name.strip(), created=datetime.now(UTC))
        _write_scan(scan_dir, meta)
        return meta
    raise ScanError(f"Zu viele Scans mit dem Namen '{slug}'. Abhilfe: einen anderen Namen wählen.")


def resolve_scan_dir(workspace: Path, scan: str) -> Path:
    """Accept a scan id (below ``workspace``) or a path to the scan directory."""
    for candidate in (workspace / scan, Path(scan)):
        if (candidate / SCAN_FILE).is_file():
            return candidate
    raise ScanError(
        f"Scan '{scan}' nicht gefunden. Abhilfe: zuerst `drone-scan scan create <name>` "
        "ausführen oder --workspace prüfen."
    )


def load_scan(scan_dir: Path) -> ScanMeta:
    try:
        data = yaml.safe_load((scan_dir / SCAN_FILE).read_text(encoding="utf-8"))
        return ScanMeta.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ScanError(
            f"{SCAN_FILE} in '{scan_dir.name}' ist nicht lesbar. Abhilfe: Datei prüfen oder Scan "
            "neu anlegen."
        ) from exc


def flight_video(scan_dir: Path, flight: FlightMeta) -> Path:
    """Path of the imported video (``flights/<id>/video.<ext>``); may not exist."""
    flight_dir = scan_dir / "flights" / flight.id
    return next(iter(sorted(flight_dir.glob("video.*"))), flight_dir / "video.mp4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def add_flight(
    scan_dir: Path, video: Path, camera: CameraProfile | None = None, move: bool = False
) -> FlightMeta:
    """Validate ``video``, copy it into the scan and register it in ``scan.yaml``.

    ``camera`` is stored as a copy, so the scan stays reproducible if the profile changes.
    ``move`` moves instead of copying (server uploads: a temporary file on the same disk).
    """
    load_scan(scan_dir)  # fail early if scan.yaml is unreadable
    if not video.is_file():
        raise ScanError(f"Video '{video.name}' nicht gefunden. Abhilfe: Pfad prüfen.")
    try:
        probed = probe_video(video)
    except ProbeError as exc:
        raise ScanError(str(exc)) from exc

    scan_flights = scan_dir / "flights"
    with _scan_lock(scan_dir):
        # Reload under the lock so flights added by parallel imports are kept.
        meta = load_scan(scan_dir)
        created_flights_dir = not scan_flights.exists()
        flight_dir: Path | None = None
        try:
            number = len(meta.flights) + 1
            while flight_dir is None:
                flight_id = f"f{number:02d}"
                try:
                    (scan_flights / flight_id).mkdir(parents=True)
                    flight_dir = scan_flights / flight_id
                except FileExistsError:
                    number += 1
            target = flight_dir / f"video{video.suffix.lower() or '.mp4'}"
            if move:
                shutil.move(video, target)
            else:
                shutil.copyfile(video, target)
            flight = FlightMeta(
                **probed.model_dump(),
                id=flight_id,
                source_name=video.name,
                sha256=_sha256(target),
                added=datetime.now(UTC),
                camera=camera,
            )
            meta.flights.append(flight)
            _write_scan(scan_dir, meta)
        except OSError as exc:
            _cleanup(flight_dir, scan_flights, created_flights_dir)
            raise ScanError(
                f"'{video.name}' konnte nicht importiert werden ({exc.strerror or 'Dateifehler'}). "
                "Abhilfe: freien Speicherplatz und Leserechte prüfen."
            ) from exc
        except BaseException:
            _cleanup(flight_dir, scan_flights, created_flights_dir)
            raise
    return flight


def _cleanup(flight_dir: Path | None, flights_root: Path, remove_root: bool) -> None:
    if flight_dir is not None:
        shutil.rmtree(flight_dir, ignore_errors=True)
    if remove_root:
        with contextlib.suppress(OSError):  # another import may already use it
            flights_root.rmdir()
