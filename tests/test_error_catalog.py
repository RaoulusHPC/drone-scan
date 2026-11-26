# SPDX-License-Identifier: AGPL-3.0-or-later
"""Error catalog: one source for docs and code, no unknown or orphaned codes (KAN-56)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import dronescan
from dronescan.common import error_catalog as catalog
from dronescan.common.config import load_settings
from dronescan.common.stage import (
    ExitCode,
    InputMissingError,
    OutOfMemoryError,
    QualityError,
    StageError,
    run_stage,
)
from dronescan.ingest.stage import TOO_FEW_HINTS, IngestStage

SRC = Path(dronescan.__file__).parent
STAGE_FILES = [SRC / name / "stage.py" for name in ("ingest", "sfm", "recon")]
STAGE_ERRORS = {"StageError", "InputMissingError", "QualityError", "OutOfMemoryError"}
EXIT_OF = {"StageError": 1, "InputMissingError": 10, "QualityError": 20, "OutOfMemoryError": 30}
# Developer tooling and internal signals, never shown to users of the pipeline.
EXEMPT = {"SmokeError", "_OutsideScanError"}  # the latter is mapped by run_stage
VALID_EXIT_CODES = {int(code) for code in ExitCode} | {2}


def _error_classes() -> dict[str, ast.ClassDef]:
    """Exception classes defined in the user-facing packages (parsed, torch is not imported)."""
    classes: dict[str, ast.ClassDef] = {}
    for package in ("common", "capture", "ingest", "sfm", "recon"):
        for path in sorted((SRC / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes[node.name] = node
    builtin = {"Exception", "RuntimeError", "ValueError", "OSError", "LookupError"}
    errors: set[str] = set()
    changed = True
    while changed:  # transitive closure over the base classes
        changed = False
        for name, node in classes.items():
            bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
            if name not in errors and bases & (builtin | errors):
                errors.add(name)
                changed = True
    return {name: classes[name] for name in errors}


def _carries_code(name: str, classes: dict[str, ast.ClassDef]) -> bool:
    """True for StageError subclasses and classes whose __init__ takes ``code``."""
    node = classes[name]
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            names = {arg.arg for arg in item.args.args + item.args.kwonlyargs}
            if "code" in names:
                return True
    bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
    return name == "StageError" or any(b in classes and _carries_code(b, classes) for b in bases)


def test_catalog_entries_are_well_formed() -> None:
    """
    Given the catalog
    When it is loaded
    Then ids are "<scope>.<slug>", exit codes valid, and every entry has cause and hints.
    """
    entries = catalog.load_catalog()
    assert entries
    for code, entry in entries.items():
        assert re.fullmatch(r"[a-z]+\.[a-z0-9_]+", code), code
        assert entry.exit_code in VALID_EXIT_CODES and entry.exit_code != 0, code
        assert code.split(".")[0] == entry.scope, code
        assert entry.cause.strip() and entry.hints, code


def test_every_exit_code_has_a_catalog_entry() -> None:
    """
    Given the stage contract (1, 10, 20, 30) and the CLI usage error (2)
    When the catalog is checked
    Then each of them has at least one entry.
    """
    used = {entry.exit_code for entry in catalog.load_catalog().values()}
    assert used == VALID_EXIT_CODES - {0}


def test_stage_error_defaults_match_the_catalog() -> None:
    """
    Given the StageError classes
    When their default code is looked up
    Then it exists with the same exit code.
    """
    for cls in (StageError, InputMissingError, QualityError, OutOfMemoryError):
        assert catalog.get(cls.default_code).exit_code == int(cls.exit_code), cls


@pytest.mark.parametrize("path", STAGE_FILES, ids=lambda p: p.parent.name)
def test_stage_errors_name_a_catalog_code(path: Path) -> None:
    """
    Given the stages ingest, sfm and recon
    When they raise a StageError (or subclass)
    Then each raise site passes a literal ``code=`` from the catalog with the class's exit code.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in STAGE_ERRORS
    ]
    assert calls
    for call in calls:
        assert isinstance(call.func, ast.Name)
        where = f"{path.name}:{call.lineno}"
        value = next((k.value for k in call.keywords if k.arg == "code"), None)
        assert isinstance(value, ast.Constant) and isinstance(value.value, str), where
        assert catalog.get(value.value).exit_code == EXIT_OF[call.func.id], where


def test_every_error_class_is_in_the_catalog() -> None:
    """
    Given all exception classes in common, capture, ingest, sfm and recon
    When they are compared with the catalog
    Then each is a StageError (carries a code), carries a code itself or is listed in `exceptions`.
    """
    listed = {name for entry in catalog.load_catalog().values() for name in entry.exceptions}
    classes = _error_classes()
    for name in classes:
        if name in EXEMPT or _carries_code(name, classes):
            continue
        assert name in listed, f"{name} is not in error_catalog.yaml (exceptions)"


def test_no_orphaned_catalog_entries() -> None:
    """
    Given the catalog
    When it is compared with the code
    Then listed exception classes exist, and every code is used in src or mapped from a class.
    """
    classes = _error_classes()
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SRC.rglob("*.py")
        if path.name != "error_catalog.py"
    )
    for code, entry in catalog.load_catalog().items():
        for name in entry.exceptions:
            assert name in classes, f"{code}: unknown exception class {name}"
        assert f'"{code}"' in sources or entry.exceptions, f"{code} is orphaned"


def test_hints_in_code_come_from_the_catalog() -> None:
    """
    Given hint texts that stages build from other modules
    When they are compared with the catalog
    Then the texts are identical (the catalog stays the single source).
    """
    assert catalog.hints("ingest.too_few_frames") == TOO_FEW_HINTS


def test_docs_table_matches_the_catalog() -> None:
    """
    Given docs/fehlercodes.md
    When its generated block is compared with the catalog
    Then they are identical; otherwise run `python -m dronescan.common.error_catalog --write`.
    """
    text = catalog.DOCS_PATH.read_text(encoding="utf-8")
    start = text.index(catalog.BEGIN_MARKER) + len(catalog.BEGIN_MARKER)
    block = text[start : text.index(catalog.END_MARKER)].strip("\n")
    assert block == catalog.render_markdown()


def test_docs_mention_no_unknown_code() -> None:
    """
    Given docs/fehlercodes.md
    When searching for catalog-like codes
    Then every one exists in the catalog and every catalog id appears.
    """
    text = catalog.DOCS_PATH.read_text(encoding="utf-8")
    ids = set(catalog.load_catalog())
    assert all(f"`{code}`" in text for code in ids)
    scopes = "|".join({code.split(".")[0] for code in ids})
    mentioned = set(re.findall(rf"`((?:{scopes})\.[a-z0-9_]+)`", text))
    mentioned -= {"models.lock"}  # a file name, not a code
    assert mentioned <= ids, mentioned - ids


def test_failed_stage_stores_its_code_in_the_manifest(tmp_path: Path) -> None:
    """
    Given a scan path that does not exist
    When a stage runs
    Then the result has exit code 10 and the matching code from the catalog.
    """
    result = run_stage(IngestStage(), tmp_path / "missing", load_settings("low_vram_6gb"))
    assert result.exit_code is ExitCode.INPUT_MISSING
    assert result.manifest.error_code == "common.scan_dir_missing"
    assert catalog.get("common.scan_dir_missing").exit_code == 10


def test_capture_and_cli_codes_exist_in_the_catalog() -> None:
    """
    Given calibration.py and cli.py
    When they raise CalibrationError or print a code via _echo_code
    Then each literal code is in the catalog, and a CalibrationError exits with the catalog code.
    """
    names = {"CalibrationError": 1, "_echo_code": 0}
    found = 0
    paths = (
        SRC / "capture" / "calibration.py",
        SRC / "cli.py",
        SRC / "common" / "models_pull.py",
        SRC / "common" / "models_lock.py",
        SRC / "common" / "licenses.py",
    )
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id == "PullError":  # code is a keyword or the catalogued default
                keyword = next((k.value for k in node.keywords if k.arg == "code"), None)
                if keyword is not None:
                    assert isinstance(keyword, ast.Constant), f"{path.name}:{node.lineno}"
                    assert catalog.get(str(keyword.value)).exit_code == 1
                    found += 1
            if node.func.id in names and len(node.args) > names[node.func.id]:
                arg = node.args[names[node.func.id]]
                if not isinstance(arg, ast.Constant):  # _echo_code(exc.code): checked in __init__
                    assert node.func.id == "_echo_code", f"{path.name}:{node.lineno}"
                    continue
                catalog.get(str(arg.value))
                found += 1
    assert found >= 8


def test_pull_error_default_code_is_catalogued() -> None:
    """
    Given PullError without an explicit code
    When it is created
    Then it carries a catalog code with exit code 1 (`models pull` exits with 1).
    """
    from dronescan.common.models_pull import PullError

    assert catalog.get(PullError("x").code).exit_code == 1
