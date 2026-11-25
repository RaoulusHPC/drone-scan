# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU lock (KAN-80): GPU stages never run in parallel, crashes release the lock.

No test relies on timing races: other holders are separate processes that report
via a pipe when they hold the lock, and waits use generous timeouts.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
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

from dronescan.common import gpu_lock as gpu_lock_module
from dronescan.common.config import Settings, load_settings
from dronescan.common.gpu_lock import GpuBusyError, GpuLockError, gpu_lock, lock_path
from dronescan.common.stage import ExitCode, Manifest, Stage, StageContext, StageResult, run_stage

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


def test_missing_lock_dir_is_created_on_demand(tmp_path: Path) -> None:
    """Given the lock directory does not exist yet, then it is created."""
    path = tmp_path / "a" / "b" / "gpu.lock"
    with gpu_lock(path, 0, POLL_S):
        assert path.is_file()


def test_unusable_lock_dir_gives_lock_error_not_busy(tmp_path: Path) -> None:
    """Given a lock directory below a regular file, then GpuLockError (not GpuBusyError)
    names the remedy."""
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(GpuLockError) as info, gpu_lock(blocker / "gpu.lock", 0, POLL_S):
        pytest.fail("lock must not be granted")
    assert not isinstance(info.value, GpuBusyError)
    assert "DRONESCAN_GPU__LOCK_FILE" in info.value.remedy
    assert "belegt" not in info.value.cause


def test_vanishing_lock_file_in_read_only_fallback_is_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given the read-only fallback open fails with FileNotFoundError,
    then it becomes a German GpuLockError instead of a raw OSError."""
    real_open = os.open

    def fake_open(path: str | Path, flags: int, mode: int = 0o777) -> int:
        if flags & os.O_RDWR:
            raise PermissionError(13, "Permission denied")
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(os, "open", fake_open)
    with pytest.raises(GpuLockError) as info, gpu_lock(tmp_path / "gpu.lock", 0, POLL_S):
        pytest.fail("lock must not be granted")
    monkeypatch.setattr(os, "open", real_open)
    assert "Abhilfe:" in str(info.value)


def test_flock_failure_other_than_busy_is_mapped(
    lock_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given flock fails with ENOLCK, then GpuLockError (not GpuBusyError) is raised at once."""

    def broken_flock(fd: int, op: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", broken_flock)
    with pytest.raises(GpuLockError) as info, gpu_lock(lock_file, GENEROUS_S, POLL_S):
        pytest.fail("lock must not be granted")
    assert not isinstance(info.value, GpuBusyError)


def test_read_only_lock_file_of_other_uid_still_locks(lock_file: Path) -> None:
    """Given a lock file without write permission (other uid), flock still works."""
    lock_file.touch(mode=0o444)
    if os.access(lock_file, os.W_OK):
        pytest.skip("root ignores file permissions")
    with gpu_lock(lock_file, 0, POLL_S):
        assert not can_lock(lock_file)


def test_default_lock_is_one_file_per_machine(tmp_path: Path) -> None:
    """Given XDG_STATE_HOME, then the default lock is gpu.lock below drone-scan there,
    independent of any scan or worktree location."""
    assert lock_path(load_settings()) == tmp_path / "xdg-state" / "drone-scan" / "gpu.lock"


def test_lock_dir_env_overrides_state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given DRONESCAN_GPU_LOCK_DIR (also used by compose.yaml), then it wins."""
    monkeypatch.setenv("DRONESCAN_GPU_LOCK_DIR", str(tmp_path / "locks"))
    assert lock_path(load_settings()) == tmp_path / "locks" / "gpu.lock"


def test_default_falls_back_to_home_local_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given no XDG_STATE_HOME (or a relative one), then ~/.local/state/drone-scan is used."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_STATE_HOME")
    expected = tmp_path / ".local" / "state" / "drone-scan" / "gpu.lock"
    assert lock_path(load_settings()) == expected
    monkeypatch.setenv("XDG_STATE_HOME", "relative/dir")
    assert lock_path(load_settings()) == expected


def test_unusable_home_fails_without_silent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given no state dir and no determinable home, then GpuLockError names the remedy."""
    monkeypatch.delenv("XDG_STATE_HOME")

    def no_home() -> Path:
        raise RuntimeError("no home")

    monkeypatch.setattr(Path, "home", no_home)
    with pytest.raises(GpuLockError) as info:
        lock_path(load_settings())
    assert "DRONESCAN_GPU_LOCK_DIR" in info.value.remedy


def test_lock_file_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given DRONESCAN_GPU__LOCK_FILE (set to the mounted path in containers), then it wins."""
    monkeypatch.setenv("DRONESCAN_GPU__LOCK_FILE", str(tmp_path / "custom.lock"))
    assert lock_path(load_settings()) == tmp_path / "custom.lock"


def test_timeout_defaults_come_from_the_profile() -> None:
    """Given the built-in profiles, then timeout and poll interval are defined there."""
    for name in ("low_vram_6gb", "default_8gb"):
        gpu_cfg = load_settings(name).gpu
        assert gpu_cfg.lock_timeout_s == 1800
        assert gpu_cfg.lock_poll_s == 1


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
    applies_vram_limit: ClassVar[bool] = False  # no torch needed for the lock tests


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


# --- manifest, re-check after waiting, torch independence ----------------------


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def replace_timeout(settings: Settings, timeout_s: float) -> Settings:
    gpu_cfg = settings.gpu.model_copy(update={"lock_timeout_s": timeout_s})
    return settings.model_copy(update={"gpu": gpu_cfg})


def read_probe_manifest(scan: Path) -> Manifest:
    return Manifest.model_validate_json((scan / "gpu_probe" / "manifest.json").read_text())


@pytest.fixture
def waiting_seen(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Set as soon as some lock wait had to sleep, i.e. is really blocked on the lock."""
    event = threading.Event()
    real_sleep = gpu_lock_module._sleep

    def observing_sleep(seconds: float) -> None:
        event.set()
        real_sleep(seconds)

    monkeypatch.setattr(gpu_lock_module, "_sleep", observing_sleep)
    return event


def test_busy_timeout_keeps_existing_manifest_untouched(
    probe_setup: tuple[Path, Settings], lock_file: Path
) -> None:
    """Given a valid ok manifest, when --force hits a busy GPU and times out,
    then the error result is returned and nothing in out_dir changes."""
    scan, settings = probe_setup
    assert run_stage(GpuProbeStage(lock_file), scan, settings).exit_code is ExitCode.OK
    before = snapshot(scan / "gpu_probe")
    with holder_process(lock_file):
        result = run_stage(GpuProbeStage(lock_file), scan, settings, force=True)
    assert result.exit_code is ExitCode.ERROR
    assert any(h.startswith("Abhilfe:") for h in result.manifest.hints)
    assert snapshot(scan / "gpu_probe") == before


def test_interrupt_while_waiting_keeps_existing_manifest_untouched(
    probe_setup: tuple[Path, Settings], lock_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a valid ok manifest, when Ctrl+C arrives while waiting for the lock,
    then KeyboardInterrupt propagates and out_dir stays unchanged."""
    scan, settings = probe_setup
    assert run_stage(GpuProbeStage(lock_file), scan, settings).exit_code is ExitCode.OK
    before = snapshot(scan / "gpu_probe")

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(gpu_lock_module, "_sleep", interrupt)
    with holder_process(lock_file), pytest.raises(KeyboardInterrupt):
        run_stage(GpuProbeStage(lock_file), scan, settings, force=True)
    assert snapshot(scan / "gpu_probe") == before


def test_changed_inputs_after_waiting_are_rehashed(
    probe_setup: tuple[Path, Settings], lock_file: Path, waiting_seen: threading.Event
) -> None:
    """Given the input changes while the stage waits for the lock,
    then the manifest records the new hash, not the one from before the wait."""
    scan, settings = probe_setup
    patient = replace_timeout(settings, GENEROUS_S)
    with holder_process(lock_file) as holder:
        thread = threading.Thread(target=lambda: run_stage(GpuProbeStage(lock_file), scan, patient))
        thread.start()
        assert waiting_seen.wait(GENEROUS_S)
        (scan / "in.txt").write_text("changed while waiting")
        assert holder.stdin is not None
        holder.stdin.close()
        thread.join(GENEROUS_S)
    assert not thread.is_alive()
    manifest = read_probe_manifest(scan)
    assert manifest.inputs["in.txt"] == hashlib.sha256(b"changed while waiting").hexdigest()


class GatedStage(Stage):
    """GPU stage for real cross-process runs; optionally blocks inside run() until stdin closes."""

    name: ClassVar[str] = "gated"
    version: ClassVar[str] = "1"
    uses_gpu: ClassVar[bool] = True
    applies_vram_limit: ClassVar[bool] = False
    runs: ClassVar[list[str]] = []

    def inputs(self, scan_dir: Path) -> list[Path]:
        return [scan_dir / "in.txt"]

    def run(self, ctx: StageContext) -> None:
        self.runs.append(ctx.scan_dir.name)
        if os.environ.get("TEST_GATE") == "1":
            print("running", flush=True)
            sys.stdin.read()
        (ctx.out_dir / "out.txt").write_bytes(ctx.inputs[0].read_bytes())


def child_main() -> None:
    """Entry point of the child process: one real run_stage on argv[1]."""
    result = run_stage(GatedStage(), Path(sys.argv[1]), load_settings())
    sys.exit(int(result.exit_code))


@contextmanager
def gated_child(scan: Path, lock_file: Path) -> Iterator[subprocess.Popen[str]]:
    """A separate process running run_stage(GatedStage) that holds the GPU lock inside run()."""
    env = {**os.environ, "TEST_GATE": "1", "DRONESCAN_GPU__LOCK_FILE": str(lock_file)}
    command = [sys.executable, "-c", "from tests.test_gpu_lock import child_main; child_main()"]
    proc = subprocess.Popen(
        [*command, str(scan)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.strip() == "running":
                break
        else:
            pytest.fail("child exited before running the stage")
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout):
            if stream:
                stream.close()


def make_scan(path: Path, content: str = "data") -> Path:
    path.mkdir()
    (path / "in.txt").write_text(content)
    return path


def patient_settings(lock_file: Path) -> Settings:
    base = load_settings()
    gpu_cfg = base.gpu.model_copy(
        update={"lock_file": lock_file, "lock_timeout_s": GENEROUS_S, "lock_poll_s": POLL_S}
    )
    return base.model_copy(update={"gpu": gpu_cfg})


def run_in_thread(scan: Path, settings: Settings) -> tuple[threading.Thread, list[StageResult]]:
    results: list[StageResult] = []
    thread = threading.Thread(
        target=lambda: results.append(run_stage(GatedStage(), scan, settings))
    )
    thread.start()
    return thread, results


def test_waiting_stage_is_served_from_cache_after_identical_stage_finished(
    tmp_path: Path, lock_file: Path, waiting_seen: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given two real run_stage calls on the same scan in separate processes, when the second
    waits for the lock while the first runs, then the second is served from the cache and does
    not compute again."""
    scan = make_scan(tmp_path / "scan")
    GatedStage.runs.clear()
    monkeypatch.delenv("TEST_GATE", raising=False)
    with gated_child(scan, lock_file) as child:
        thread, results = run_in_thread(scan, patient_settings(lock_file))
        assert waiting_seen.wait(GENEROUS_S)
        assert child.stdin is not None
        child.stdin.close()
        thread.join(GENEROUS_S)
        assert child.wait(GENEROUS_S) == 0
    assert not thread.is_alive()
    (result,) = results
    assert result.skipped
    assert result.exit_code is ExitCode.OK
    assert GatedStage.runs == []


def test_waiting_stage_on_other_scan_runs_after_the_holder(
    tmp_path: Path, lock_file: Path, waiting_seen: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given two real run_stage calls on different scans in separate processes, when the second
    waits for the lock, then it runs after the first finished (own cache, no skip)."""
    first = make_scan(tmp_path / "scan-a", "a")
    second = make_scan(tmp_path / "scan-b", "b")
    GatedStage.runs.clear()
    monkeypatch.delenv("TEST_GATE", raising=False)
    with gated_child(first, lock_file) as child:
        thread, results = run_in_thread(second, patient_settings(lock_file))
        assert waiting_seen.wait(GENEROUS_S)
        assert GatedStage.runs == []  # blocked while the first process holds the GPU
        assert child.stdin is not None
        child.stdin.close()
        thread.join(GENEROUS_S)
        assert child.wait(GENEROUS_S) == 0
    (result,) = results
    assert not result.skipped
    assert result.exit_code is ExitCode.OK
    assert GatedStage.runs == ["scan-b"]
    assert (first / "gated" / "out.txt").read_text() == "a"
    assert (second / "gated" / "out.txt").read_text() == "b"


class TorchlessGpuStage(GpuProbeStage):
    name: ClassVar[str] = "torchless"


class TorchGpuStage(GpuProbeStage):
    name: ClassVar[str] = "torch_gpu"
    applies_vram_limit: ClassVar[bool] = True


def test_stage_without_vram_limit_takes_lock_but_makes_no_torch_calls(
    probe_setup: tuple[Path, Settings], lock_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given no torch installed and a GPU stage with applies_vram_limit = False (COLMAP),
    then it runs under the lock, without torch calls and without a VRAM peak."""
    scan, settings = probe_setup
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    result = run_stage(TorchlessGpuStage(lock_file), scan, settings)
    assert result.exit_code is ExitCode.OK
    assert result.manifest.peak_vram_mb is None
    assert ProbeStage.held_during_run == [True]


def test_stage_with_vram_limit_still_needs_torch(
    probe_setup: tuple[Path, Settings], lock_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given no torch installed, a GPU stage with the default applies_vram_limit = True
    fails with exit 1 and the lock is free afterwards."""
    scan, settings = probe_setup
    monkeypatch.setitem(sys.modules, "torch", None)
    result = run_stage(TorchGpuStage(lock_file), scan, settings)
    assert result.exit_code is ExitCode.ERROR
    assert can_lock(lock_file)
