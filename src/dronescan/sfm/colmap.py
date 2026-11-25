# SPDX-License-Identifier: AGPL-3.0-or-later
"""COLMAP adapter: one `docker run` per COLMAP command (separate process, ADR-0003).

The container has no network (CLAUDE.md rule 3), runs as the calling user and sees exactly one
GPU; CPU-only commands get none. COLMAP is not bound by PyTorch's allocator limit, so the peak
VRAM of its process (including the CUDA context) is measured with nvidia-smi.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import structlog

COLMAP_IMAGE = "drone-scan/colmap:4.2.1"  # same tag as in compose.yaml (checked by a test)
_SAMPLE_INTERVAL_S = 0.25
_DOCKER_FAILED = 125  # `docker run` itself failed, the command never started
_TOOL_TIMEOUT_S = 10  # a hanging helper must not block the stage while it holds the GPU lock
_KILLED = 137  # SIGKILL, usually the kernel's out-of-memory killer
_GPU_OOM = re.compile(
    r"not enough gpu memory|out[ _]of[ _]memory|cudaErrorMemoryAllocation", re.IGNORECASE
)
_HOST_OOM = "std::bad_alloc"
# Runs in the container: stdin is a pipe from the runner. If the runner dies for any reason
# (even SIGKILL), the pipe closes and COLMAP is killed. Otherwise it would keep the GPU busy
# after the kernel has released the runner's GPU lock (KAN-80).
WATCHDOG = (
    'exec 3<&0; "$@" & pid=$!; (cat <&3 >/dev/null; kill -KILL "$pid") 2>/dev/null & wait "$pid"'
)

log = structlog.get_logger(__name__)


class ColmapError(Exception):
    """COLMAP failed; the message is user-facing (German) and contains no paths."""

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


class ColmapUnavailableError(ColmapError):
    """Docker or the COLMAP image is missing, or the container could not be started."""


class ColmapTimeoutError(ColmapError):
    """The command did not finish in time and was stopped."""


class ColmapOutOfMemoryError(ColmapError):
    """The GPU has not enough free memory."""


@dataclass(frozen=True)
class Mount:
    source: Path
    target: str
    read_only: bool = False

    def argument(self) -> str:
        """Value for ``--mount`` (CSV, so any character is allowed in the path)."""
        source = f"source={self.source.resolve()}".replace('"', '""')
        fields = ["type=bind", f'"{source}"', f"target={self.target}"]
        return ",".join([*fields, "readonly"] if self.read_only else fields)


@dataclass(frozen=True)
class StepResult:
    seconds: float


class _VramSampler:
    """Polls the memory of the ``colmap`` process on one GPU while a command runs."""

    def __init__(self, device: int) -> None:
        self._command = ["nvidia-smi", "-i", str(device), "--format=csv,noheader,nounits"]
        self._command.append("--query-compute-apps=used_memory,process_name")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak: float | None = None

    def _sample(self) -> None:
        try:
            proc = subprocess.run(
                self._command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_TOOL_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired):
            return  # a failed query under load must not end the sampling
        for line in proc.stdout.splitlines():  # no output if the query failed
            memory, _, name = line.partition(",")
            if Path(name.strip()).name == "colmap" and memory.strip().isdigit():
                self.peak = max(float(memory), self.peak or 0.0)

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


def image_id() -> str:
    """Id of the local COLMAP image or ``missing`` (also if docker does not respond)."""
    if shutil.which("docker") is None:
        return "missing"
    command = ["docker", "image", "inspect", "--format", "{{.Id}}", COLMAP_IMAGE]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=_TOOL_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return "missing"
    return proc.stdout.strip() if proc.returncode == 0 else "missing"


class Colmap:
    """Runs COLMAP commands with fixed mounts; the output of each step goes to a log file."""

    def __init__(self, mounts: list[Mount], gpu_device: int, log_dir: Path) -> None:
        self._mounts = mounts
        self._gpu_device = gpu_device
        self._log_dir = log_dir
        self._steps = 0
        self.peak_vram_mib: float | None = None  # over all commands, also failed ones

    def run(
        self,
        command: str,
        options: Mapping[str, str | int | float],
        *,
        gpu: bool = False,
        timeout_s: float | None = None,
    ) -> StepResult:
        """Run one command; after ``timeout_s`` it is stopped and ColmapTimeoutError is raised."""
        if shutil.which("docker") is None:
            raise ColmapUnavailableError("Docker ist nicht installiert.")
        if image_id() == "missing":
            raise ColmapUnavailableError(
                f"Das Docker-Image '{COLMAP_IMAGE}' fehlt oder Docker ist nicht erreichbar."
            )
        container = f"dronescan-colmap-{uuid.uuid4().hex[:12]}"
        arguments = self._arguments(container, gpu)
        arguments += ["sh", "-c", WATCHDOG, "sh", "colmap", command, "--log_target", "stderr"]
        for name, value in options.items():
            arguments += [f"--{name}", str(value)]

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._steps += 1  # numbered: a retry must not overwrite the log of the first attempt
        log_file = self._log_dir / f"{self._steps:02d}_{command}.log"
        started = time.monotonic()
        sampler = _VramSampler(self._gpu_device)
        with log_file.open("w", encoding="utf-8") as output:
            proc = subprocess.Popen(
                arguments, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT
            )
            try:
                if gpu:
                    with sampler:
                        proc.wait(timeout_s)
                else:
                    proc.wait(timeout_s)
            except BaseException as exc:
                # Killing only the docker client would leave the container on the GPU.
                with suppress(subprocess.TimeoutExpired):  # the watchdog ends the container anyway
                    subprocess.run(
                        ["docker", "kill", container],
                        capture_output=True,
                        check=False,
                        timeout=_TOOL_TIMEOUT_S,
                    )
                proc.kill()
                proc.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    log.error("colmap_timeout", command=command, timeout_s=timeout_s)
                    raise ColmapTimeoutError(f"COLMAP antwortet nicht ({command}).") from exc
                raise
            finally:
                if proc.stdin is not None:
                    proc.stdin.close()
                if sampler.peak is not None:
                    self.peak_vram_mib = max(sampler.peak, self.peak_vram_mib or 0.0)
        result = StepResult(time.monotonic() - started)
        self._check(command, proc.returncode, log_file.read_text("utf-8", errors="replace"), gpu)
        log.info("colmap_finished", command=command, seconds=round(result.seconds, 1))
        return result

    def _arguments(self, container: str, gpu: bool) -> list[str]:
        arguments = [
            "docker",
            "run",
            "--rm",
            "--interactive",
            "--pull",
            "never",
            "--name",
            container,
        ]
        arguments += ["--network", "none", "--user", f"{os.getuid()}:{os.getgid()}"]
        if gpu:
            arguments += ["--runtime", "nvidia", "--gpus", f"device={self._gpu_device}"]
            arguments += ["--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID"]
        else:  # the image asks for all GPUs if nvidia is the default runtime of the host
            arguments += ["--env", "NVIDIA_VISIBLE_DEVICES=void"]
        for mount in self._mounts:
            arguments += ["--mount", mount.argument()]
        return [*arguments, COLMAP_IMAGE]

    def _check(self, command: str, returncode: int, output: str, gpu: bool) -> None:
        # COLMAP reports some GPU memory problems only in its log and still exits with 0.
        if gpu and _GPU_OOM.search(output):
            raise ColmapOutOfMemoryError("COLMAP: Der Grafikspeicher reicht nicht.", output)
        if returncode == 0:
            return
        log.error("colmap_failed", command=command, returncode=returncode)
        if returncode == _DOCKER_FAILED:
            raise ColmapUnavailableError(
                "Der COLMAP-Container konnte nicht gestartet werden.", output
            )
        if returncode == _KILLED or _HOST_OOM in output:
            raise MemoryError(f"COLMAP ({command})")
        raise ColmapError(f"COLMAP ist fehlgeschlagen ({command}, Code {returncode}).", output)
