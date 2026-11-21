"""Central configuration: YAML profiles plus environment overrides.

Priority (highest first):
1. environment variables ``DRONESCAN_*`` (nested sections via ``__``,
   e.g. ``DRONESCAN_SPLAT__MAX_GAUSSIANS``),
2. the selected YAML profile (``--profile`` argument > ``DRONESCAN_PROFILE`` >
   ``low_vram_6gb``),
3. nothing else: every value must come from a profile, so stages contain no
   magic numbers (CLAUDE.md).

VRAM values are GiB (1024**3 bytes).
"""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_PROFILE = "low_vram_6gb"
ENV_PREFIX = "DRONESCAN_"
PROFILE_ENV_VAR = f"{ENV_PREFIX}PROFILE"
MAX_VRAM_LIMIT_GB = 80.0  # sanity bound (largest data-centre GPUs), catches unit mistakes
_PROFILE_NAME = re.compile(r"^[a-z0-9_]+$")


class ProfileError(ValueError):
    """A configuration profile is missing or malformed."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GpuSettings(_Section):
    allow_fp16: bool = Field(
        description="fp16/AMP allowed; only used on sm_70+ (Pascal stays fp32, ADR-0010)."
    )


class SfmSettings(_Section):
    max_image_edge_px: PositiveInt = Field(description="Longest image edge for feature extraction.")


class SplatSettings(_Section):
    max_iterations: PositiveInt
    max_gaussians: PositiveInt


class SearchSettings(_Section):
    detector_batch_size: PositiveInt


class _EnvWithoutProfile(EnvSettingsSource):
    """Env source that ignores ``DRONESCAN_PROFILE``.

    The profile name only *selects* the YAML file (see `load_settings`); it must
    not override the name of the profile that was actually loaded.
    """

    def __call__(self) -> dict[str, Any]:
        data = super().__call__()
        data.pop("profile", None)
        return data


class Settings(BaseSettings):
    """Effective configuration of one pipeline run."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    profile: str = DEFAULT_PROFILE
    vram_limit_gb: float = Field(
        gt=0,
        le=MAX_VRAM_LIMIT_GB,
        description="PyTorch allocator limit per process in GiB (CUDA context comes on top).",
    )
    gpu: GpuSettings
    sfm: SfmSettings
    splat: SplatSettings
    search: SearchSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment beats the profile values, which are passed as init kwargs.
        # No .env or secrets files: configuration stays explicit and offline.
        return (_EnvWithoutProfile(settings_cls), init_settings)


def _builtin_profiles_dir() -> Path:
    return Path(str(resources.files("dronescan.common") / "profiles"))


def available_profiles(profiles_dir: Path | None = None) -> list[str]:
    directory = profiles_dir or _builtin_profiles_dir()
    return sorted(p.stem for p in directory.glob("*.yaml"))


def _read_profile(name: str, profiles_dir: Path) -> dict[str, Any]:
    if not _PROFILE_NAME.fullmatch(name):
        raise ProfileError(
            f"Ungültiger Profilname '{name}'. Abhilfe: nur Kleinbuchstaben, Ziffern und '_' "
            f"verwenden, z. B. '{DEFAULT_PROFILE}'."
        )
    path = profiles_dir / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(available_profiles(profiles_dir)) or "keine"
        raise ProfileError(
            f"Profil '{name}' nicht gefunden. Verfügbare Profile: {known}. "
            f"Abhilfe: --profile bzw. {PROFILE_ENV_VAR} auf eines davon setzen."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(
            f"Profil '{name}' ({path.name}) kann nicht gelesen werden: {exc}. "
            "Abhilfe: YAML-Syntax prüfen (Einrückung mit Leerzeichen, 'schlüssel: wert')."
        ) from exc
    if not isinstance(data, dict):
        raise ProfileError(
            f"Profil '{name}' ({path.name}) ist keine YAML-Zuordnung. "
            "Abhilfe: Schlüssel wie 'vram_limit_gb: 5.0' auf oberster Ebene angeben."
        )
    return data


def load_settings(profile: str | None = None, profiles_dir: Path | None = None) -> Settings:
    """Load the effective settings for ``profile`` (see module docstring for priority)."""
    name = profile if profile is not None else os.environ.get(PROFILE_ENV_VAR) or DEFAULT_PROFILE
    data = _read_profile(name, profiles_dir or _builtin_profiles_dir())
    data["profile"] = name
    return Settings(**data)


def describe_validation_error(exc: ValidationError, profile: str | None = None) -> str:
    """German, user-facing description of invalid settings with a concrete remedy."""
    lines = ["Ungültige Konfiguration:"]
    for error in exc.errors():
        path = [str(part) for part in error["loc"]]
        env_name = ENV_PREFIX + "__".join(p.upper() for p in path)
        lines.append(
            f"- {'.'.join(path)} = {error.get('input')!r}: {error['msg']}. "
            f"Abhilfe: Umgebungsvariable {env_name} korrigieren bzw. entfernen "
            f"oder den Wert im Profil '{profile or DEFAULT_PROFILE}' anpassen."
        )
    return "\n".join(lines)
