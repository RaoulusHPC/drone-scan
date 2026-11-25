# SPDX-License-Identifier: AGPL-3.0-or-later
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
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator
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
    lock_file: Path | None = Field(
        default=None,
        description="Machine-wide GPU lock file (KAN-80). Default: gpu.lock in "
        "$DRONESCAN_GPU_LOCK_DIR, else $XDG_STATE_HOME/drone-scan, else ~/.local/state/drone-scan. "
        "Containers set this to the mounted path.",
    )
    lock_timeout_s: float = Field(
        ge=0,
        description="Seconds a GPU stage waits for another GPU job before it gives up (0 = fail "
        "immediately).",
    )
    lock_poll_s: float = Field(gt=0, description="Seconds between attempts to take the GPU lock.")


class IngestSettings(_Section):
    analysis_fps: float = Field(
        gt=0, le=60, description="Frames per second checked for sharpness and similarity."
    )
    analysis_edge_px: int = Field(ge=64, description="Longest edge of the analysis frames.")
    keyframes_per_s: float = Field(gt=0, description="Target rate: one keyframe per time window.")
    min_rel_sharpness: float = Field(
        ge=0, le=1, description="Sharpness threshold as a fraction of the median of the video."
    )
    similarity_edge_px: int = Field(
        ge=8, description="Longest edge of the thumbnails compared by the duplicate filter."
    )
    max_similarity: float = Field(
        gt=0,
        description="Correlation with the previous keyframe from which a frame is a duplicate "
        "(above 1 disables the filter).",
    )
    max_image_edge_px: PositiveInt = Field(description="Longest edge of the extracted JPEGs.")
    jpeg_qscale: int = Field(ge=1, le=31, description="ffmpeg JPEG quality (2 = very high).")
    min_keyframes: PositiveInt = Field(
        description="Fewer keyframes in the whole scan cannot be reconstructed (exit 20)."
    )
    min_decoded_ratio: float = Field(
        ge=0, le=1, description="Warn if less of the video duration could be decoded."
    )

    @model_validator(mode="after")
    def _enough_analysis_frames(self) -> Self:
        if self.keyframes_per_s > self.analysis_fps:
            raise ValueError("keyframes_per_s must not exceed analysis_fps")
        return self


class SfmSettings(_Section):
    max_image_edge_px: PositiveInt = Field(description="Longest image edge for feature extraction.")
    max_num_features: PositiveInt = Field(description="SIFT features per image.")
    max_num_matches: PositiveInt = Field(
        description="Features per image the GPU matcher can hold (determines its VRAM)."
    )
    camera_model: str = Field(
        pattern=r"^[A-Z_]+$", description="COLMAP camera model, one camera per flight."
    )
    exhaustive_max_keyframes: PositiveInt = Field(
        description="Up to this many keyframes all pairs are matched, above it sequential "
        "matching with loop detection (PO decision 2026-11-25)."
    )
    sequential_overlap: PositiveInt = Field(description="Neighbours matched per keyframe.")
    loop_vocab_words: PositiveInt = Field(
        description="Words of the vocabulary tree built from the scan's own features."
    )
    loop_vocab_iterations: PositiveInt = Field(description="k-means iterations of the tree.")
    loop_vocab_rounds: PositiveInt = Field(description="k-means restarts of the tree.")
    loop_vocab_max_descriptors: PositiveInt = Field(
        description="Descriptors sampled for building the tree (CPU time, RAM)."
    )
    min_registration_ratio: float = Field(
        gt=0, le=1, description="Share of keyframes in the largest model; below it exit 20."
    )
    target_registration_ratio: float = Field(
        gt=0, le=1, description="Target Q-3; below it the stage warns."
    )
    oom_retry_image_edge_px: PositiveInt = Field(
        description="Longest image edge for the second attempt after running out of VRAM."
    )
    oom_retry_num_features: PositiveInt = Field(
        description="SIFT features per image for the second attempt."
    )
    oom_retry_num_matches: PositiveInt = Field(
        description="Match capacity for the second attempt (SIFT may exceed the feature limit)."
    )

    @model_validator(mode="after")
    def _consistent_limits(self) -> Self:
        if self.min_registration_ratio > self.target_registration_ratio:
            raise ValueError("min_registration_ratio must not exceed target_registration_ratio")
        if self.max_num_matches < self.max_num_features:
            raise ValueError("max_num_matches must not be smaller than max_num_features")
        if (
            self.oom_retry_image_edge_px >= self.max_image_edge_px
            or self.oom_retry_num_features >= self.max_num_features
            or self.oom_retry_num_matches >= self.max_num_matches
        ):
            raise ValueError("oom_retry_* values must be smaller than the normal ones")
        if self.oom_retry_num_matches < self.oom_retry_num_features:
            raise ValueError(
                "oom_retry_num_matches must not be smaller than oom_retry_num_features"
            )
        return self


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
    gpu_device: int = Field(
        ge=0,
        description="nvidia-smi index of the GPU for containers started by stages; same "
        "variable (DRONESCAN_GPU_DEVICE) as compose.yaml.",
    )
    gpu: GpuSettings
    ingest: IngestSettings
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
