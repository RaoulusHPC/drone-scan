# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sample clips go through Git LFS and `make setup` handles git-lfs (KAN-79, AC 1 and AC 2)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def lfs_filter_for(path: str) -> str:
    result = subprocess.run(
        ["git", "check-attr", "filter", "--", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    # Output format: "<path>: filter: <value>"
    return result.stdout.rsplit(":", 1)[-1].strip()


@pytest.mark.parametrize(
    "path",
    [
        "data/samples/kurz.mp4",
        "data/samples/kurz.mov",
        "data/samples/kurz.mkv",
        "data/samples/rundgang/kurz.mp4",
        "data/samples/rundgang/kurz.mov",
        "data/samples/rundgang/kurz.mkv",
        "data/samples/KURZ.MP4",
        "data/samples/KURZ.MOV",
        "data/samples/KURZ.MKV",
        "data/samples/rundgang/KURZ.MP4",
        "data/samples/rundgang/KURZ.MOV",
        "data/samples/rundgang/KURZ.MKV",
        "data/samples/rundgang/Kurz.Mp4",
    ],
)
def test_sample_clips_are_routed_through_lfs(path: str) -> None:
    """Given a sample clip in data/samples/ or one of its subfolders,
    When Git resolves the filter attribute for the path,
    Then the path is handled by the lfs filter (no video file needs to exist).
    """
    assert lfs_filter_for(path) == "lfs"


def test_clips_outside_samples_are_not_routed_through_lfs() -> None:
    """Given a video file outside data/samples/,
    When Git resolves the filter attribute,
    Then no LFS filter applies, so private recordings are not silently pushed to LFS.
    """
    assert lfs_filter_for("workspace/scan-001/kurz.mp4") == "unspecified"
    assert lfs_filter_for("data/raw/wohnung.mp4") == "unspecified"


def test_patch_files_keep_their_attributes() -> None:
    """Given the existing .gitattributes rules,
    When the patch attributes are read,
    Then the *.patch line is still present (regression guard for the LFS change).
    """
    assert "*.patch -text -whitespace" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


def _setup_recipe() -> str:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^setup:.*?(?=^\S[^\n]*:)", makefile, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, "setup target not found in Makefile"
    return match.group(0)


def test_make_setup_installs_lfs_hooks_only_when_git_lfs_is_available() -> None:
    """Given a machine where git-lfs may be missing,
    When `make setup` runs,
    Then `git lfs install --local` runs only if `git-lfs` is on the PATH.
    """
    recipe = _setup_recipe()

    assert "command -v git-lfs" in recipe
    assert "git lfs install --local" in recipe
    assert recipe.index("command -v git-lfs") < recipe.index("git lfs install --local")


def test_make_setup_explains_missing_git_lfs_and_does_not_fail() -> None:
    """Given git-lfs is not installed,
    When `make setup` runs,
    Then a German message names cause and remedy (sudo apt install git-lfs), setup continues.
    """
    recipe = _setup_recipe()

    assert "sudo apt install git-lfs" in recipe
    assert "nicht installiert" in recipe
    assert "Setup wird fortgesetzt" in recipe


def test_make_setup_tolerates_lfs_hook_conflicts() -> None:
    """Given an existing Git hook (e.g. pre-push) that blocks `git lfs install`,
    When `git lfs install --local` fails,
    Then the failure is reported as a hint and does not abort `make setup`.
    """
    recipe = _setup_recipe()

    assert "git lfs install --local ||" in recipe
    assert "pre-push" in recipe
