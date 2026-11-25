# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every tracked source file carries an SPDX license header (KAN-77)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER = "# SPDX-License-Identifier: AGPL-3.0-or-later"
SLASH_HEADER = "// SPDX-License-Identifier: AGPL-3.0-or-later"
ROOTS = ("src", "tests", "tools", "viewer", "flight")
SUFFIXES = {".py", ".ts", ".tsx", ".js", ".sh"}
_SKIPPABLE = re.compile(r"^(#!|#.*coding[:=])")


def _expected(path: Path) -> str:
    return SLASH_HEADER if path.suffix in {".ts", ".tsx", ".js"} else HEADER


def has_header(path: Path) -> bool:
    """Return True if the header is the first line, after shebang/encoding lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines) and idx < 2 and _SKIPPABLE.match(lines[idx]):
        idx += 1
    return idx < len(lines) and lines[idx].strip() == _expected(path)


def tracked_sources(root: Path = REPO_ROOT) -> list[Path]:
    """List source files tracked by version control under the checked roots."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *ROOTS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = [root / name for name in out.split("\0") if name]
    return sorted(f for f in files if f.suffix in SUFFIXES)


def missing_headers(files: list[Path]) -> list[Path]:
    """Return the files without the SPDX header."""
    return [f for f in files if not has_header(f)]


def test_all_tracked_sources_have_spdx_header() -> None:
    """
    Given all tracked source files under src/, tests/, tools/, viewer/, flight/
    When their first lines are checked
    Then each carries the SPDX AGPL-3.0-or-later header.
    """
    files = tracked_sources()
    assert files, "no tracked source files found"
    missing = [str(f.relative_to(REPO_ROOT)) for f in missing_headers(files)]
    assert not missing, "Missing SPDX header in:\n" + "\n".join(missing)


def test_missing_header_is_detected(tmp_path: Path) -> None:
    """
    Given a file without header and files with header (plain and after a shebang)
    When the check runs
    Then only the headerless file is reported.
    """
    bad = tmp_path / "bad.py"
    bad.write_text("print('x')\n", encoding="utf-8")
    plain = tmp_path / "ok.py"
    plain.write_text(HEADER + "\n", encoding="utf-8")
    shebang = tmp_path / "run.sh"
    shebang.write_text("#!/bin/sh\n" + HEADER + "\n", encoding="utf-8")
    assert missing_headers([bad, plain, shebang]) == [bad]
