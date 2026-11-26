# SPDX-License-Identifier: AGPL-3.0-or-later
"""Catalog of all user-facing error codes (KAN-56, US-041).

``error_catalog.yaml`` is the single source of truth: the stages take their remedy texts
from it, and ``docs/fehlercodes.md`` contains the table rendered by `render_markdown`
(a test fails when the two differ). Regenerate the docs with
``python -m dronescan.common.error_catalog --write``.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

CATALOG_PATH = Path(__file__).with_name("error_catalog.yaml")
DOCS_PATH = Path(__file__).resolve().parents[3] / "docs" / "fehlercodes.md"
BEGIN_MARKER = "<!-- BEGIN error-catalog (generated: python -m dronescan.common.error_catalog) -->"
END_MARKER = "<!-- END error-catalog -->"


class ErrorCode(BaseModel):
    """One catalog entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    exit_code: int
    scope: str
    cause: str
    hints: list[str]
    exceptions: list[str] = []


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, ErrorCode]:
    """All codes by id; raises ValueError for duplicate ids."""
    data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    codes: dict[str, ErrorCode] = {}
    for raw in data["codes"]:
        entry = ErrorCode.model_validate(raw)
        if entry.id in codes:
            raise ValueError(f"duplicate error code {entry.id}")
        codes[entry.id] = entry
    return codes


def get(code: str) -> ErrorCode:
    """The entry for ``code``; KeyError names unknown codes (a programming error)."""
    try:
        return load_catalog()[code]
    except KeyError:
        raise KeyError(f"unknown error code '{code}' (add it to error_catalog.yaml)") from None


def hints(code: str) -> list[str]:
    """Fresh list of the remedy texts of ``code``."""
    return list(get(code).hints)


def hint(code: str) -> str:
    """The single remedy text of ``code`` (the entry must have exactly one)."""
    texts = get(code).hints
    if len(texts) != 1:
        raise ValueError(f"error code {code} has {len(texts)} hints, expected 1")
    return texts[0]


def cause(code: str) -> str:
    return get(code).cause


def render_markdown() -> str:
    """Table of all codes, grouped by exit code, as it appears in docs/fehlercodes.md."""
    lines = ["| Fehlercode | Exitcode | Ursache | Abhilfe |", "|---|---|---|---|"]
    for entry in sorted(load_catalog().values(), key=lambda e: (e.exit_code, e.id)):
        remedy = "<br>".join(_cell(text) for text in entry.hints)
        lines.append(f"| `{entry.id}` | {entry.exit_code} | {_cell(entry.cause)} | {remedy} |")
    return "\n".join(lines)


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _docs_block() -> str:
    return f"{BEGIN_MARKER}\n{render_markdown()}\n{END_MARKER}"


def update_docs(path: Path = DOCS_PATH) -> bool:
    """Rewrite the generated block in ``path``; True if the file changed."""
    text = path.read_text(encoding="utf-8")
    start, end = text.index(BEGIN_MARKER), text.index(END_MARKER) + len(END_MARKER)
    new = text[:start] + _docs_block() + text[end:]
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--write" in args:
        changed = update_docs()
        print("docs/fehlercodes.md aktualisiert." if changed else "docs/fehlercodes.md aktuell.")
    else:
        print(render_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
