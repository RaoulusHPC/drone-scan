# SPDX-License-Identifier: AGPL-3.0-or-later
"""Worker adapter: runs the trainer in the ``worker`` image, one `docker run` per attempt.

Same isolation as the COLMAP adapter (ADR-0003): no network (CLAUDE.md rule 3), the calling
user's uid, exactly one GPU, and a watchdog that kills the trainer when the runner dies, so
the GPU lock (KAN-80) never outlives the job. The trainer applies the PyTorch allocator limit
itself; the peak VRAM of its process *including* the CUDA context is sampled here with
nvidia-smi, matched by the host PIDs of the container (``docker top``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Self

import structlog

from dronescan.sfm.colmap import WATCHDOG, Mount

WORKER_IMAGE = "drone-scan/worker:0.1.0"  # same tag as in compose.yaml (checked by a test)
_SAMPLE_INTERVAL_S = 0.5
_TOOL_TIMEOUT_S = 10
DOCKER_FAILED = 125  # `docker run` itself failed, the trainer never started
KILLED = 137  # SIGKILL, usually the kernel's out-of-memory killer

log = structlog.get_logger(__name__)


class WorkerError(Exception):
    """Running the worker container failed; the message is user-facing (German), no paths."""


class WorkerUnavailableError(WorkerError):
    """Docker or the worker image is missing, or the container could not be started."""


class WorkerTimeoutError(WorkerError):
    """The trainer did not finish in time and was stopped."""


def docker_image_id(image: str) -> str:
    """Id of a local image or ``missing`` (also if docker does not respond)."""
    if shutil.which("docker") is None:
        return "missing"
    command = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=_TOOL_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return "missing"
    return proc.stdout.strip() if proc.returncode == 0 else "missing"


def _run_tool(command: list[str]) -> str:
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=_TOOL_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""  # a failed query under load must not end the sampling
    return proc.stdout if proc.returncode == 0 else ""


class ContainerVramSampler:
    """Polls the GPU memory of all processes of one container (host PIDs) on one GPU."""

    def __init__(self, device: int, container: str) -> None:
        self._query = ["nvidia-smi", "-i", str(device), "--format=csv,noheader,nounits"]
        self._query.append("--query-compute-apps=pid,used_memory")
        self._top = ["docker", "top", container, "-eo", "pid"]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak: float | None = None

    def _sample(self) -> None:
        pids = {line.strip() for line in _run_tool(self._top).splitlines()[1:] if line.strip()}
        if not pids:
            return  # container not started yet or already gone
        used = 0.0
        for line in _run_tool(self._query).splitlines():
            pid, _, memory = (part.strip() for part in line.partition(","))
            if pid in pids and memory.isdigit():
                used += float(memory)
        if used:
            self.peak = max(used, self.peak or 0.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(_SAMPLE_INTERVAL_S)

    def __enter__(self) -> Self:
        if shutil.which("nvidia-smi"):
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()


class Worker:
    """Runs Python modules of drone-scan in the worker image with fixed mounts."""

    def __init__(self, mounts: list[Mount], gpu_device: int) -> None:
        self._mounts = mounts
        self._gpu_device = gpu_device
        self.peak_vram_mib: float | None = None  # over all runs, also failed ones
        self.last_peak_vram_mib: float | None = None  # of the last run

    def run(self, module: str, args: list[str], *, log_file: Path, timeout_s: float) -> int:
        """Run ``python -m module args``; returns its exit code (output goes to ``log_file``)."""
        if shutil.which("docker") is None:
            raise WorkerUnavailableError("Docker ist nicht installiert.")
        if docker_image_id(WORKER_IMAGE) == "missing":
            raise WorkerUnavailableError(
                f"Das Docker-Image '{WORKER_IMAGE}' fehlt oder Docker ist nicht erreichbar."
            )
        container = f"dronescan-worker-{uuid.uuid4().hex[:12]}"
        arguments = [
            "docker", "run", "--rm", "--interactive", "--pull", "never", "--name", container,
            "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
            "--runtime", "nvidia", "--gpus", f"device={self._gpu_device}",
            "--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        ]  # fmt: skip
        for mount in self._mounts:
            arguments += ["--mount", mount.argument()]
        arguments += [WORKER_IMAGE, "sh", "-c", WATCHDOG, "sh", "python", "-m", module, *args]

        log_file.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        sampler = ContainerVramSampler(self._gpu_device, container)
        with log_file.open("w", encoding="utf-8") as output:
            proc = subprocess.Popen(
                arguments, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT
            )
            try:
                with sampler:
                    proc.wait(timeout_s)
            except BaseException as exc:
                # Killing only the docker client would leave the container on the GPU.
                with suppress(subprocess.TimeoutExpired):  # the watchdog ends it anyway
                    subprocess.run(
                        ["docker", "kill", container],
                        capture_output=True,
                        check=False,
                        timeout=_TOOL_TIMEOUT_S,
                    )
                proc.kill()
                proc.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    log.error("worker_timeout", module=module, timeout_s=timeout_s)
                    raise WorkerTimeoutError(
                        "Das Training wurde nach Ablauf der Zeitgrenze abgebrochen."
                    ) from exc
                raise
            finally:
                if proc.stdin is not None:
                    proc.stdin.close()
                self.last_peak_vram_mib = sampler.peak
                if sampler.peak is not None:
                    self.peak_vram_mib = max(sampler.peak, self.peak_vram_mib or 0.0)
        seconds = round(time.monotonic() - started, 1)
        log.info("worker_finished", module=module, returncode=proc.returncode, seconds=seconds)
        if proc.returncode == DOCKER_FAILED:
            raise WorkerUnavailableError("Der Worker-Container konnte nicht gestartet werden.")
        if proc.returncode == KILLED:
            raise MemoryError(f"worker ({module})")
        return proc.returncode
