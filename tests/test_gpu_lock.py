# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU lock (KAN-80): GPU stages never run in parallel, crashes release the lock.

No test relies on timing races: other holders are separate processes that report
via a pipe when they hold the lock, and waits use generous timeouts.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import pytest

from dronescan.common import gpu
from dronescan.common.config import Settings, load_settings
from dronescan.common.gpu_lock import GpuBusyError, gpu_lock, lock_path
from dronescan.common.stage import ExitCode, Stage, StageContext, run_stage

GENEROUS_S = 60.0
POLL_S = 0.01

_HOLDER = """
import sys
from pathlib import Path
from dronescan.common.gpu_lock import gpu_lock
with gpu_lock(Path(sys.argv[1]), 30, 0.01):
    print("locked", flush=True)
    sys.stdin.read()  # hold until the parent closes stdin (or kills us)
"""


@contextmanager
def holder_process(path: Path) -> Iterator[subprocess.Popen[str]]:
    """A separate process that holds the lock until stdin is closed."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        # structlog also prints to stdout: skip log lines until the marker
        for line in proc.stdout:
            if line.strip() == "locked":
                break
        else:
            pytest.fail("holder exited without taking the lock")
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout):
            if stream:
                stream.close()


@pytest.fixture
def lock_file(tmp_path: Path) -> Path:
    directory = tmp_path / "workspace"
    directory.mkdir()
    return directory / ".gpu.lock"


def can_lock(path: Path) -> bool:
    try:
        with gpu_lock(path, 0, POLL_S):
            return True
    except GpuBusyError:
        return False


def test_lock_is_exclusive_across_processes(lock_file: Path) -> None:
    """Given another process holds the lock, when we try without waiting,
    then we get a German GpuBusyError with a remedy."""
    with (
        holder_process(lock_file),
        pytest.raises(GpuBusyError) as info,
        gpu_lock(lock_file, 0, POLL_S),
    ):
        pytest.fail("lock must not be granted")
    assert "Abhilfe:" in str(info.value)
    assert "GPU" in info.value.cause


def test_waiter_gets_lock_after_holder_releases(lock_file: Path) -> None:
    """Given a holder, when a second job waits and the holder then exits,
    then the waiter acquires the lock within the timeout."""
    acquired = threading.Event()

    def wait_for_lock() -> None:
        with gpu_lock(lock_file, GENEROUS_S, POLL_S):
            acquired.set()

    with holder_process(lock_file) as holder:
        waiter = threading.Thread(target=wait_for_lock)
        waiter.start()
        assert not acquired.wait(0.3)  # still blocked while the holder lives
        assert holder.stdin is not None
        holder.stdin.close()  # holder exits normally
        assert acquired.wait(GENEROUS_S)
        waiter.join(GENEROUS_S)
    assert not waiter.is_alive()


def test_timeout_raises_after_waiting(lock_file: Path) -> None:
    """Given a holder that never releases, when the timeout is short,
    then GpuBusyError names the waited time and the remedy."""
    with (
        holder_process(lock_file),
        pytest.raises(GpuBusyError) as info,
        gpu_lock(lock_file, 0.2, POLL_S),
    ):
        pytest.fail("lock must not be granted")
    assert "0.2 s" in info.value.cause
    assert "DRONESCAN_GPU__LOCK_TIMEOUT_S" in info.value.remedy


def test_released_after_normal_exit(lock_file: Path) -> None:
    """Given a finished with-block, then the lock is free again and the file stays."""
    with gpu_lock(lock_file, 0, POLL_S):
        assert not can_lock(lock_file)
    assert can_lock(lock_file)
    assert lock_file.exists()  # never deleted (inode race)


def test_released_after_exception(lock_file: Path) -> None:
    """Given the with-block raises, then the lock is released."""
    with pytest.raises(ValueError), gpu_lock(lock_file, 0, POLL_S):
        raise ValueError("boom")
    assert can_lock(lock_file)


def test_released_after_holder_is_sigkilled(lock_file: Path) -> None:
    """Given the holder process is killed with SIGKILL (no cleanup runs),
    then the kernel releases the lock and we can take it (no deadlock)."""
    with holder_process(lock_file) as holder:
        assert not can_lock(lock_file)
        holder.send_signal(signal.SIGKILL)
        holder.wait(GENEROUS_S)
        with gpu_lock(lock_file, GENEROUS_S, POLL_S):
            pass


def test_lock_fd_is_not_inherited_by_children(lock_file: Path) -> None:
    """Given a child process started while the lock is held and outliving the
    holder, then the child does not keep the lock (close-on-exec, documented)."""
    with gpu_lock(lock_file, 0, POLL_S):
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
        )
    try:
        assert can_lock(lock_file)
    finally:
        child.kill()
        child.wait()
        if child.stdin:
            child.stdin.close()


def test_missing_lock_dir_gives_german_remedy(tmp_path: Path) -> None:
    """Given the lock directory does not exist, then GpuBusyError names the remedy."""
    with (
        pytest.raises(GpuBusyError) as info,
        gpu_lock(tmp_path / "missing" / ".gpu.lock", 0, POLL_S),
    ):
        pytest.fail("lock must not be granted")
    assert "DRONESCAN_GPU__LOCK_FILE" in info.value.remedy


def test_read_only_lock_file_of_other_uid_still_locks(lock_file: Path) -> None:
    """Given a lock file without write permission (other uid), flock still works."""
    lock_file.touch(mode=0o444)
    if os.access(lock_file, os.W_OK):
        pytest.skip("root ignores file permissions")
    with gpu_lock(lock_file, 0, POLL_S):
        assert not can_lock(lock_file)


def test_default_lock_is_shared_between_scans(tmp_path: Path) -> None:
    """Given two scans in the same workspace, then they use the same lock file."""
    settings = load_settings()
    first = lock_path(settings, tmp_path / "ws" / "scan-a")
    second = lock_path(settings, tmp_path / "ws" / "scan-b")
    assert first == second == tmp_path.resolve() / "ws" / ".gpu.lock"


def test_lock_file_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given DRONESCAN_GPU__LOCK_FILE, then it wins over the workspace default."""
    monkeypatch.setenv("DRONESCAN_GPU__LOCK_FILE", str(tmp_path / "custom.lock"))
    assert lock_path(load_settings(), tmp_path / "scan") == tmp_path / "custom.lock"


# --- stage runner integration -------------------------------------------------


class ProbeStage(Stage):
    name: ClassVar[str] = "probe"
    version: ClassVar[str] = "1"
    uses_gpu: ClassVar[bool] = False
    held_during_run: ClassVar[list[bool]] = []

    def __init__(self, lock_file: Path) -> None:
        self.lock_file = lock_file

    def inputs(self, scan_dir: Path) -> list[Path]:
        return [scan_dir / "in.txt"]

    def run(self, ctx: StageContext) -> None:
        self.held_during_run.append(not can_lock(self.lock_file))
        (ctx.out_dir / "out.txt").write_text("x")


class GpuProbeStage(ProbeStage):
    name: ClassVar[str] = "gpu_probe"
    uses_gpu: ClassVar[bool] = True


class FailingGpuStage(GpuProbeStage):
    def run(self, ctx: StageContext) -> None:
        raise ValueError("boom")


class InterruptedGpuStage(GpuProbeStage):
    def run(self, ctx: StageContext) -> None:
        raise KeyboardInterrupt


@pytest.fixture
def probe_setup(tmp_path: Path, lock_file: Path) -> tuple[Path, Settings]:
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "in.txt").write_text("data")
    base = load_settings()
    gpu_cfg = base.gpu.model_copy(
        update={"lock_file": lock_file, "lock_timeout_s": 0.2, "lock_poll_s": POLL_S}
    )
    ProbeStage.held_during_run.clear()
    return scan, base.model_copy(update={"gpu": gpu_cfg})


@pytest.fixture(autouse=True)
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner calls these for GPU stages; the lock tests need no real torch."""
    monkeypatch.setattr(gpu, "apply_vram_limit", lambda settings, device=None: 1.0)
    monkeypatch.setattr(gpu, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(gpu, "peak_reserved_mib", lambda: 1.0)


def test_gpu_stage_holds_lock_while_running_and_releases(
    probe_setup: tuple[Path, Settings], lock_file: Path
) -> None:
    """Given a GPU stage, when it runs, then the lock is held during run()
    and free afterwards."""
    scan, settings = probe_setup
    result = run_stage(GpuProbeStage(lock_file), scan, settings)
    assert result.exit_code is ExitCode.OK
    assert ProbeStage.held_during_run == [True]
    assert can_lock(lock_file)


def test_non_gpu_stage_takes_no_lock(probe_setup: tuple[Path, Settings], lock_file: Path) -> None:
    """Given a non-GPU stage, when it runs while another process holds the lock,
    then it neither waits nor holds the lock itself."""
    scan, settings = probe_setup
    with holder_process(lock_file):
        result = run_stage(ProbeStage(lock_file), scan, settings)
    assert result.exit_code is ExitCode.OK
    assert not result.skipped
    assert ProbeStage.held_during_run == [True]  # held by the other process only


def test_non_gpu_stage_does_not_hold_lock(
    probe_setup: tuple[Path, Settings], lock_file: Path
) -> None:
    """Given a non-GPU stage without a foreign holder, then the lock is free during run()."""
    scan, settings = probe_setup
    run_stage(ProbeStage(lock_file), scan, settings)
    assert ProbeStage.held_during_run == [False]


def test_cached_gpu_stage_takes_no_lock(
    probe_setup: tuple[Path, Settings], lock_file: Path
) -> None:
    """Given a GPU stage with a valid cache, when another process holds the lock,
    then the stage is skipped without waiting."""
    scan, settings = probe_setup
    assert run_stage(GpuProbeStage(lock_file), scan, settings).exit_code is ExitCode.OK
    with holder_process(lock_file):
        result = run_stage(GpuProbeStage(lock_file), scan, settings)
    assert result.skipped
    assert result.exit_code is ExitCode.OK


def test_busy_gpu_stage_exits_1_with_remedy_and_does_not_run(
    probe_setup: tuple[Path, Settings], lock_file: Path
) -> None:
    """Given another job holds the GPU, when a GPU stage starts, then it exits 1 with a
    German message and remedy in the manifest and never runs."""
    scan, settings = probe_setup
    with holder_process(lock_file):
        result = run_stage(GpuProbeStage(lock_file), scan, settings)
    assert result.exit_code is ExitCode.ERROR
    assert result.manifest.status == "error"
    assert "GPU-Stufe" in (result.manifest.error or "")
    assert any(h.startswith("Abhilfe:") for h in result.manifest.hints)
    assert ProbeStage.held_during_run == []


def test_gpu_stage_releases_lock_after_stage_failure(
    probe_setup: tuple[Path, Settings], lock_file: Path
) -> None:
    """Given a GPU stage that raises, then the lock is free afterwards."""
    scan, settings = probe_setup
    assert run_stage(FailingGpuStage(lock_file), scan, settings).exit_code is ExitCode.ERROR
    assert can_lock(lock_file)


def test_gpu_stage_releases_lock_after_interrupt(
    probe_setup: tuple[Path, Settings], lock_file: Path
) -> None:
    """Given a GPU stage interrupted with Ctrl+C, then the lock is free afterwards."""
    scan, settings = probe_setup
    with pytest.raises(KeyboardInterrupt):
        run_stage(InterruptedGpuStage(lock_file), scan, settings)
    assert can_lock(lock_file)
