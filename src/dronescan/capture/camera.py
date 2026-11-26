# SPDX-License-Identifier: AGPL-3.0-or-later
"""Camera profiles (KAN-29, US-014): COLMAP camera model plus optional intrinsics.

A profile is a YAML file: shipped ones live in ``capture/cameras/<name>.yaml``, any other
file (a calibration result, or ``sfm/cameras/<flight>.yaml`` estimated by COLMAP) is loaded
by its path. Intrinsics use COLMAP's parameter order and pixel convention (the image corner is
(0, 0), the centre of the first pixel is (0.5, 0.5)) and refer to ``width`` x ``height``;
`CameraProfile.params_for` scales them to the size of the keyframes.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator

from dronescan.common.camera_models import CameraModel

# Number of parameters and how many of them are pixel values (focal length(s), principal
# point); the remaining distortion coefficients refer to normalised coordinates and do not
# change with the image size.
_PARAMS: dict[str, tuple[int, int]] = {
    "SIMPLE_PINHOLE": (3, 3),
    "PINHOLE": (4, 4),
    "SIMPLE_RADIAL": (4, 3),
    "RADIAL": (5, 3),
    "OPENCV": (8, 4),
    "OPENCV_FISHEYE": (8, 4),
}
# fit_size() rounds the keyframe size to even pixels: that alone must not count as a
# different aspect ratio (2 px of 360 px are 0.6 %).
_ASPECT_TOLERANCE = 0.01
_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"


class CameraError(ValueError):
    """A camera profile is missing or malformed; the message names cause and remedy."""


class Intrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: PositiveInt = Field(description="Image width the parameters refer to (pixels).")
    height: PositiveInt = Field(description="Image height the parameters refer to (pixels).")
    params: list[float] = Field(description="COLMAP parameters of the camera model, in order.")


class CameraProfile(BaseModel):
    """How COLMAP models one camera; without intrinsics COLMAP estimates them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=_NAME_PATTERN)
    description: str = ""
    model: CameraModel
    intrinsics: Intrinsics | None = None
    source: Literal["manual", "calibration", "colmap"] = "manual"
    rms_error_px: float | None = Field(
        default=None, ge=0, description="Reprojection error of the calibration (pixels)."
    )

    @model_validator(mode="after")
    def _param_count(self) -> Self:
        if self.intrinsics is not None:
            expected = _PARAMS[self.model][0]
            if len(self.intrinsics.params) != expected:
                raise ValueError(
                    f"{self.model} needs {expected} params, got {len(self.intrinsics.params)}"
                )
        return self

    def params_for(self, width: int, height: int) -> list[float] | None:
        """Intrinsics scaled to ``width`` x ``height``; None without intrinsics or if the
        aspect ratio differs (another video mode: the values would not fit)."""
        if self.intrinsics is None:
            return None
        scale_x = width / self.intrinsics.width
        scale_y = height / self.intrinsics.height
        if abs(scale_x / scale_y - 1) > _ASPECT_TOLERANCE:
            return None
        params = self.intrinsics.params
        pixel_params = _PARAMS[self.model][1]
        if pixel_params == 3:  # f, cx, cy
            scales = [(scale_x + scale_y) / 2, scale_x, scale_y]
        else:  # fx, fy, cx, cy
            scales = [scale_x, scale_y, scale_x, scale_y]
        scaled = [value * scale for value, scale in zip(params[:pixel_params], scales, strict=True)]
        return scaled + params[pixel_params:]


def is_camera_name(name: str) -> bool:
    return re.fullmatch(_NAME_PATTERN, name) is not None


def param_count(model: str) -> int:
    return _PARAMS[model][0]


def principal_point_index(model: str) -> int:
    """Index of cx (cy follows): after f, or after fx and fy."""
    return _PARAMS[model][1] - 2


def _builtin_dir() -> Path:
    return Path(str(resources.files("dronescan.capture") / "cameras"))


def available_cameras() -> list[str]:
    """Names of the shipped camera profiles."""
    return sorted(path.stem for path in _builtin_dir().glob("*.yaml"))


def read_camera(path: Path) -> CameraProfile:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CameraProfile.model_validate(data)
    except OSError as exc:
        raise CameraError(
            f"Kameraprofil '{path.name}' kann nicht gelesen werden. Abhilfe: Pfad prüfen."
        ) from exc
    except (yaml.YAMLError, ValidationError) as exc:
        raise CameraError(
            f"Kameraprofil '{path.name}' ist ungültig ({_first_error(exc)}). Abhilfe: Datei mit "
            "einem mitgelieferten Profil vergleichen (`drone-scan camera list`) oder neu "
            "kalibrieren."
        ) from exc


def load_camera(spec: str) -> CameraProfile:
    """Load a shipped profile by name or any profile file by its path."""
    path = Path(spec)
    if path.suffix in {".yaml", ".yml"} or len(path.parts) > 1:
        if not path.is_file():
            raise CameraError(f"Kameraprofil '{path.name}' nicht gefunden. Abhilfe: Pfad prüfen.")
        return read_camera(path)
    builtin = _builtin_dir() / f"{spec}.yaml"
    if not builtin.is_file():
        known = ", ".join(available_cameras()) or "keine"
        raise CameraError(
            f"Kameraprofil '{spec}' nicht gefunden. Mitgelieferte Profile: {known}. Abhilfe: "
            "einen dieser Namen oder den Pfad zu einer Profildatei (*.yaml) angeben."
        )
    return read_camera(builtin)


def write_camera(profile: CameraProfile, path: Path) -> None:
    data = profile.model_dump(mode="json", exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None),
        encoding="utf-8",
    )


def _first_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        error = exc.errors()[0]
        location = ".".join(str(part) for part in error["loc"]) or "Datei"
        return f"{location}: {error['msg']}"
    return "kein gültiges YAML"
