# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model registry ``models.lock`` (ADR-0007, KAN-24): one entry per model weight file.

Read by the license gate (KAN-23) and by ``drone-scan models pull``/``list``. Everything in
this module is offline: stages resolve weights with :func:`resolve_model`, the only network
access is the explicit download in :mod:`dronescan.common.models_pull` (CLAUDE.md rule 3).
For models the license of the WEIGHTS decides; the code license is recorded as well.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Self
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

# Names, versions and file names become directories/files below the models directory:
# no separators, no leading dot (so never "." or ".."), no whitespace.
SAFE_NAME = r"^[A-Za-z0-9][A-Za-z0-9._+-]*$"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_HASH_CHUNK_BYTES = 1024 * 1024


def is_allowed_url(url: str) -> bool:
    """https only; plain http only to the local machine (test server, local mirror)."""
    parts = urlsplit(url)
    if not parts.hostname:
        return False
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS)


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=SAFE_NAME, description="Unique name, e.g. 'aliked-n16'")
    version: str = Field(pattern=SAFE_NAME, description="Version of the weights")
    source: str = Field(min_length=1, description="Model card / provenance (repository, paper)")
    url: str = Field(description="Direct download URL of the weight file (https)")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the weight file")
    weights_license: str = Field(min_length=1, description="SPDX id of the weights (decides)")
    code_license: str = Field(min_length=1, description="SPDX id of the model code")
    filename: str | None = Field(
        default=None, pattern=SAFE_NAME, description="Local file name (default: from the URL)"
    )
    size_bytes: PositiveInt | None = Field(
        default=None, description="Expected size; a larger download is aborted"
    )

    @field_validator("url")
    @classmethod
    def _https_only(cls, url: str) -> str:
        if not is_allowed_url(url):
            raise ValueError("url muss mit https:// beginnen (http:// nur für 127.0.0.1/localhost)")
        return url

    @model_validator(mode="after")
    def _file_name_is_safe(self) -> Self:
        if not re.fullmatch(SAFE_NAME, self.file_name):
            raise ValueError(
                "Dateiname aus der URL nicht ableitbar oder unzulässig (erlaubt: A-Z, a-z, 0-9, "
                "'._+-'), 'filename' angeben"
            )
        return self

    @property
    def file_name(self) -> str:
        """Local file name: ``filename`` or the last path segment of the URL."""
        return self.filename or unquote(PurePosixPath(urlsplit(self.url).path).name)


class ModelsLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[ModelEntry] = []

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [m.name for m in self.models]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Modellname mehrfach vergeben: {', '.join(duplicates)}")
        return self

    def get(self, name: str) -> ModelEntry | None:
        return next((m for m in self.models if m.name == name), None)


def load_models_lock(path: Path) -> ModelsLock:
    """Parse ``models.lock``; raises ValueError/ValidationError for malformed files."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return ModelsLock()
    if not isinstance(data, dict):
        raise ValueError("models.lock muss eine YAML-Zuordnung mit dem Schlüssel 'models' sein")
    return ModelsLock.model_validate(data)


def model_path(entry: ModelEntry, models_dir: Path) -> Path:
    """Where the weights of ``entry`` live: ``<models_dir>/<name>/<version>/<file>``."""
    return models_dir / entry.name / entry.version / entry.file_name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class ModelStatus(StrEnum):
    OK = "ok"
    MISSING = "fehlt"
    CORRUPT = "beschädigt"


def model_status(entry: ModelEntry, models_dir: Path) -> ModelStatus:
    path = model_path(entry, models_dir)
    if not path.is_file():
        return ModelStatus.MISSING
    return ModelStatus.OK if file_sha256(path) == entry.sha256 else ModelStatus.CORRUPT


class ModelNotAvailableError(LookupError):
    """Weights are not in models.lock, not downloaded or do not match their hash."""


def resolve_model(name: str, lock: ModelsLock, models_dir: Path, verify: bool = True) -> Path:
    """Path of verified local weights; never downloads (CLAUDE.md rule 3).

    Stages map :class:`ModelNotAvailableError` to exit 10 (input missing).
    """
    entry = lock.get(name)
    if entry is None:
        raise ModelNotAvailableError(
            f"Modell '{name}' steht nicht in models.lock. Abhilfe: Eintrag mit Lizenzprüfung "
            "ergänzen lassen (PO, Subagent license-auditor), dann 'drone-scan models pull'."
        )
    path = model_path(entry, models_dir)
    try:
        status = model_status(entry, models_dir) if verify else None
    except OSError as exc:
        raise ModelNotAvailableError(
            f"Modell '{name}' ({entry.version}) ist nicht lesbar ({exc.strerror or exc}). "
            "Abhilfe: Leserechte im Modellverzeichnis prüfen (DRONESCAN_MODELS_DIR)."
        ) from exc
    if not path.is_file() or status is ModelStatus.MISSING:
        raise ModelNotAvailableError(
            f"Modell '{name}' ({entry.version}) ist nicht heruntergeladen. Abhilfe: einmalig "
            f"'drone-scan models pull {name}' ausführen (einziger Netzwerkzugriff)."
        )
    if status is ModelStatus.CORRUPT:
        raise ModelNotAvailableError(
            f"Modell '{name}' ({entry.version}) ist beschädigt (SHA-256 weicht ab). Abhilfe: "
            f"'drone-scan models pull {name}' lädt es neu."
        )
    return path
