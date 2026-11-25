# SPDX-License-Identifier: AGPL-3.0-or-later
"""Machine-wide GPU lock: GPU stages never run in parallel (CLAUDE.md rule 2, KAN-80).

The lock is an ``flock(2)`` on a file in a directory that host, worker and colmap
mount in common (``workspace/``). Properties:

* The kernel releases the lock when the holding process dies (even SIGKILL or OOM
  kill), so there is no stale-lock deadlock and no PID file.
* The lock file is never deleted: unlinking would let two processes lock two
  different inodes of the same path.
* Containers share the lock because ``flock`` locks live on the shared file in the
  kernel of the common host. It does not work across hosts or on NFS.
* The descriptor is close-on-exec (Python default, kept deliberately). Children
  such as COLMAP or ``docker run`` therefore do not inherit it. If the runner is
  killed while such a child keeps running, the lock is gone although the child
  still uses the GPU; the next job may then overlap with it. Stop leftover
  COLMAP/docker processes manually after killing a runner (see docs/fehlercodes.md).
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import structlog

from dronescan.common.config import Settings

LOCK_FILE_NAME = ".gpu.lock"
_LOCK_FILE_MODE = 0o666  # reduced by umask; containers run as another uid than the host user

log = structlog.get_logger(__name__)


class GpuBusyError(RuntimeError):
    """Another GPU job still holds the lock after the timeout (``str()`` = cause + remedy)."""

    def __init__(self, cause: str, remedy: str) -> None:
        super().__init__(f"{cause} Abhilfe: {remedy}")
        self.cause = cause
        self.remedy = remedy


def lock_path(settings: Settings, scan_dir: Path) -> Path:
    """Lock file: ``settings.gpu.lock_file`` or ``.gpu.lock`` next to the scan directory."""
    configured = settings.gpu.lock_file
    return configured if configured is not None else scan_dir.resolve().parent / LOCK_FILE_NAME


def _open_lock_file(path: Path) -> int:
    flags = os.O_CLOEXEC
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT | flags, _LOCK_FILE_MODE)
    except PermissionError:
        # Created by another uid without write access: flock needs only read access.
        return os.open(path, os.O_RDONLY | flags)
    except OSError as exc:
        raise GpuBusyError(
            f"Die GPU-Sperrdatei '{path.name}' kann nicht angelegt werden ({exc.strerror}).",
            "Verzeichnis der Sperrdatei prüfen (existiert, beschreibbar) oder "
            "DRONESCAN_GPU__LOCK_FILE auf einen gemeinsam gemounteten, beschreibbaren Pfad setzen.",
        ) from exc


@contextmanager
def gpu_lock(path: Path, timeout_s: float, poll_s: float) -> Iterator[None]:
    """Hold the exclusive GPU lock for the ``with`` block.

    Waits up to ``timeout_s`` seconds (polling every ``poll_s``), then raises `GpuBusyError`.
    """
    fd = _open_lock_file(path)
    try:
        deadline = time.monotonic() + timeout_s
        waited = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
            if not waited:
                log.info("gpu_lock_waiting", timeout_s=timeout_s)
                waited = True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GpuBusyError(
                    "Eine andere GPU-Stufe läuft noch und belegt die GPU "
                    f"(Wartezeit von {timeout_s:g} s überschritten).",
                    "Warten, bis der andere Lauf fertig ist (z. B. 'docker ps' bzw. 'nvidia-smi'), "
                    "und die Stufe erneut starten, oder die Wartezeit mit "
                    "DRONESCAN_GPU__LOCK_TIMEOUT_S erhöhen.",
                )
            time.sleep(min(poll_s, remaining))
        log.info("gpu_lock_acquired")
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            log.info("gpu_lock_released")
    finally:
        os.close(fd)


def stage_gpu_lock(settings: Settings, scan_dir: Path) -> AbstractContextManager[None]:
    """`gpu_lock` configured from ``settings`` for a stage working on ``scan_dir``."""
    return gpu_lock(
        lock_path(settings, scan_dir), settings.gpu.lock_timeout_s, settings.gpu.lock_poll_s
    )
