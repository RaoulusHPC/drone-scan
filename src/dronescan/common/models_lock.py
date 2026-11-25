# SPDX-License-Identifier: AGPL-3.0-or-later
"""Schema of ``models.lock`` (ADR-0007): one entry per model weight file.

Used by the license gate (KAN-23) and later by ``drone-scan models pull`` (KAN-24).
For models the license of the WEIGHTS decides.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the weight file")
    license: str = Field(min_length=1, description="SPDX id of the weights")
    source: str = Field(min_length=1, description="Model card / download URL")


class ModelsLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[ModelEntry] = []


def load_models_lock(path: Path) -> ModelsLock:
    """Parse ``models.lock``; raises ValueError/ValidationError for malformed files."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return ModelsLock()
    if not isinstance(data, dict):
        raise ValueError("models.lock muss eine YAML-Zuordnung mit dem Schlüssel 'models' sein")
    return ModelsLock.model_validate(data)
