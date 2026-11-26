# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stand-in for ``python -m dronescan <stage>`` in the server tests (KAN-52).

Usage: ``fake_stage.py <stage> <scan_dir> <profile> <force 0|1>``. The behaviour per stage
comes from ``<scan_dir>/fake.json``: ``{"<stage>": {"exit": 0, "sleep": 0.0, "warnings": [],
"hints": [], "error_code": null, "ignore_sigint": false, "no_manifest": false,
"exit_by_profile": {"<profile>": <exit>}}}``. Every run appends ``<stage> <profile> <force>``
to ``<scan_dir>/calls.log``; a second stage process running at the same time is recorded
in ``<scan_dir>/../overlap.log``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# No dronescan import: it would add about a second per stage process. The server validates
# the manifest with the real model (dronescan.common.stage.Manifest) when it reads it.
MANIFEST_NAME = "manifest.json"
STATUS = {0: "ok", 1: "error", 10: "input_missing", 20: "quality_insufficient", 30: "oom"}


def main() -> int:
    stage, scan_dir, profile, force = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    spec = json.loads((scan_dir / "fake.json").read_text()).get(stage, {})
    marker = scan_dir.parent / "stage.running"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        with (scan_dir.parent / "overlap.log").open("a") as handle:
            handle.write(f"{stage}\n")
    if spec.get("ignore_sigint"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    with (scan_dir / "calls.log").open("a") as handle:  # tests wait for this line
        handle.write(f"{stage} {profile} {force}\n")
    started = datetime.now(UTC)
    code = int(spec.get("exit_by_profile", {}).get(profile, spec.get("exit", 0)))
    error: str | None = None
    error_code = spec.get("error_code")
    try:
        deadline = time.monotonic() + float(spec.get("sleep", 0.0))
        while time.monotonic() < deadline:
            time.sleep(0.02)
    except KeyboardInterrupt:
        code, error, error_code = 1, "Abgebrochen", "common.interrupted"
    finally:
        marker.unlink(missing_ok=True)
    if spec.get("no_manifest"):
        return code
    if code and error is None:
        error = f"Fehler in {stage}"
    out = scan_dir / stage
    out.mkdir(exist_ok=True)
    manifest = {
        "stage": stage,
        "version": "1",
        "profile": profile,
        "inputs": {},
        "params": {},
        "tools": {},
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "peak_vram_mb": None,
        "status": STATUS[code],
        "exit_code": code,
        "error": error,
        "error_code": error_code if code else None,
        "warnings": spec.get("warnings", []),
        "hints": spec.get("hints", []),
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest))
    for name, content in spec.get("files", {}).items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(content)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
