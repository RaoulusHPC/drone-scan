# SPDX-License-Identifier: AGPL-3.0-or-later
"""Machine-wide GPU lock: GPU stages never run in parallel (CLAUDE.md rule 2, KAN-80).

The lock is an ``flock(2)`` on ``gpu.lock`` in the per-user state directory
(``$DRONESCAN_GPU_LOCK_DIR``, else ``$XDG_STATE_HOME/drone-scan``, else
``~/.local/state/drone-scan``). That is one file per machine and user, independent of
worktrees and of where a scan lives. Containers mount the same host directory
(compose.yaml) and point ``DRONESCAN_GPU__LOCK_FILE`` at the mounted path. Properties:

* The kernel releases the lock when the holding process dies (even SIGKILL or OOM
  kill), so there is no stale-lock deadlock and no PID file.
* The lock file is never deleted: unlinking would let two processes lock two
  different inodes of the same path.
* ``flock`` works across containers because they share the host kernel and the file;
  it does not work across hosts or on NFS.
* The descriptor is close-on-exec (kept deliberately). Children such as COLMAP or
  ``docker run`` therefore do not inherit it. If the runner is killed while such a
  child keeps running, the lock is gone although the child still uses the GPU; the
  next job may then overlap with it. The COLMAP adapter (`dronescan.sfm.colmap`) ends its
  containers itself in that case; stop other leftover docker processes manually
  after killing a runner (see docs/fehlercodes.md).
* No silent fallback: if the lock file cannot be used, the stage fails with a message.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from pathlib import Path

import structlog

from dronescan.common.config import Settings

LOCK_DIR_ENV_VAR = "DRONESCAN_GPU_LOCK_DIR"
STATE_HOME_ENV_VAR = "XDG_STATE_HOME"
LOCK_FILE_NAME = "gpu.lock"
STATE_SUBDIR = "drone-scan"
_LOCK_FILE_MODE = 0o666  # reduced by umask; containers may run as another uid than the host user
_LOCK_DIR_MODE = 0o755

log = structlog.get_logger(__name__)
_sleep = time.sleep  # indirection so tests can observe or interrupt the waiting


class GpuLockError(RuntimeError):
    """The GPU lock cannot be used at all (``str()`` = cause + remedy). Not "GPU busy"."""

    def __init__(self, cause: str, remedy: str) -> None:
        super().__init__(f"{cause} Abhilfe: {remedy}")
        self.cause = cause
        self.remedy = remedy


class GpuBusyError(GpuLockError):
    """Another GPU job still holds the lock after the timeout."""


def _lock_dir() -> Path:
    override = os.environ.get(LOCK_DIR_ENV_VAR)
    if override:
        return Path(override)
    state_home = os.environ.get(STATE_HOME_ENV_VAR)
    if state_home and Path(state_home).is_absolute():  # XDG: relative values are ignored
        return Path(state_home) / STATE_SUBDIR
    try:
        return Path.home() / ".local" / "state" / STATE_SUBDIR
    except (RuntimeError, KeyError) as exc:
        raise GpuLockError(
            "Das Home-Verzeichnis für die GPU-Sperrdatei ist nicht ermittelbar.",
            f"{LOCK_DIR_ENV_VAR} auf ein beschreibbares Verzeichnis setzen "
            "(dasselbe, das compose.yaml in die Container mountet).",
        ) from exc


def lock_path(settings: Settings) -> Path:
    """``settings.gpu.lock_file`` or the default ``gpu.lock`` in the per-user state directory."""
    configured = settings.gpu.lock_file
    return configured if configured is not None else _lock_dir() / LOCK_FILE_NAME


def _cannot_use(path: Path, exc: OSError) -> GpuLockError:
    return GpuLockError(
        f"Die GPU-Sperrdatei '{path.name}' kann nicht verwendet werden ({exc.strerror}).",
        "Verzeichnis der Sperrdatei prüfen (existiert, beschreibbar, lesbar für alle "
        "Container-Nutzer) oder DRONESCAN_GPU__LOCK_FILE bzw. "
        f"{LOCK_DIR_ENV_VAR} auf einen anderen Pfad setzen.",
    )


def _open_lock_file(path: Path) -> int:
    flags = os.O_CLOEXEC
    try:
        path.parent.mkdir(mode=_LOCK_DIR_MODE, parents=True, exist_ok=True)
        try:
            return os.open(path, os.O_RDWR | os.O_CREAT | flags, _LOCK_FILE_MODE)
        except PermissionError:
            # Created by another uid without write access: flock needs only read access.
            return os.open(path, os.O_RDONLY | flags)
    except OSError as exc:
        raise _cannot_use(path, exc) from exc


@contextmanager
def gpu_lock(path: Path, timeout_s: float, poll_s: float) -> Iterator[bool]:
    """Hold the exclusive GPU lock for the ``with`` block; yields whether it had to wait.

    Waits up to ``timeout_s`` seconds (polling every ``poll_s``), then raises
    `GpuBusyError`. Other problems with the lock file raise `GpuLockError`.
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
            except OSError as exc:  # e.g. ENOLCK: the lock itself does not work
                raise _cannot_use(path, exc) from exc
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
            _sleep(min(poll_s, remaining))
        log.info("gpu_lock_acquired", waited=waited)
        try:
            yield waited
        finally:
            with suppress(OSError):  # closing the fd below releases the lock anyway
                fcntl.flock(fd, fcntl.LOCK_UN)
            log.info("gpu_lock_released")
    finally:
        os.close(fd)


def stage_gpu_lock(settings: Settings) -> AbstractContextManager[bool]:
    """`gpu_lock` configured from ``settings``."""
    return gpu_lock(lock_path(settings), settings.gpu.lock_timeout_s, settings.gpu.lock_poll_s)
