# SPDX-License-Identifier: AGPL-3.0-or-later
"""Central configuration with YAML profiles and env overrides (KAN-20, AC 1 and AC 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dronescan.common.config import (
    DEFAULT_PROFILE,
    ProfileError,
    Settings,
    available_profiles,
    load_settings,
)


def write_profile(directory: Path, name: str, body: str) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(body, encoding="utf-8")
    return path


VALID_PROFILE = """\
vram_limit_gb: 3.0
gpu:
  allow_fp16: false
sfm:
  max_image_edge_px: 800
splat:
  max_iterations: 1000
  max_gaussians: 10000
search:
  detector_batch_size: 1
"""


def test_shipped_profiles_are_available() -> None:
    """Given the packaged profiles,
    When they are listed,
    Then low_vram_6gb (default) and default_8gb exist.
    """
    assert {"low_vram_6gb", "default_8gb"} <= set(available_profiles())
    assert DEFAULT_PROFILE == "low_vram_6gb"


def test_default_profile_respects_6gb_budget() -> None:
    """Given no profile argument and no environment,
    When the settings are loaded,
    Then the low_vram_6gb profile is used and leaves headroom for the CUDA context
    below the 5.5 GB peak budget (Q-2).
    """
    settings = load_settings()

    assert settings.profile == "low_vram_6gb"
    assert settings.vram_limit_gb <= 5.0


def test_default_8gb_profile_has_larger_limits() -> None:
    """Given the default_8gb profile,
    When it is loaded,
    Then its limits are at least those of low_vram_6gb.
    """
    low = load_settings("low_vram_6gb")
    high = load_settings("default_8gb")

    assert high.profile == "default_8gb"
    assert high.vram_limit_gb > low.vram_limit_gb
    assert high.vram_limit_gb <= 7.0  # display + CUDA context on an 8 GB card
    assert high.splat.max_gaussians >= low.splat.max_gaussians
    assert high.sfm.max_image_edge_px >= low.sfm.max_image_edge_px


def test_profile_selected_via_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given DRONESCAN_PROFILE=default_8gb,
    When the settings are loaded without argument,
    Then the default_8gb profile is used.
    """
    monkeypatch.setenv("DRONESCAN_PROFILE", "default_8gb")

    assert load_settings().profile == "default_8gb"


def test_profile_argument_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given DRONESCAN_PROFILE=default_8gb and an explicit profile argument,
    When the settings are loaded,
    Then the argument wins (CLI --profile beats the environment).
    """
    monkeypatch.setenv("DRONESCAN_PROFILE", "default_8gb")

    settings = load_settings("low_vram_6gb")

    assert settings.profile == "low_vram_6gb"
    assert settings.vram_limit_gb < load_settings("default_8gb").vram_limit_gb


def test_vram_limit_overridden_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given DRONESCAN_VRAM_LIMIT_GB=4.25,
    When any profile is loaded,
    Then the environment value replaces the profile value.
    """
    monkeypatch.setenv("DRONESCAN_VRAM_LIMIT_GB", "4.25")

    assert load_settings("default_8gb").vram_limit_gb == pytest.approx(4.25)


def test_nested_value_overridden_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given DRONESCAN_SPLAT__MAX_GAUSSIANS=12345,
    When the profile is loaded,
    Then only that nested value changes and the rest of the section is kept.
    """
    baseline = load_settings()
    monkeypatch.setenv("DRONESCAN_SPLAT__MAX_GAUSSIANS", "12345")

    settings = load_settings()

    assert settings.splat.max_gaussians == 12345
    assert settings.splat.max_iterations == baseline.splat.max_iterations


@pytest.mark.parametrize("value", ["-1", "0", "abc"])
def test_invalid_vram_limit_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Given an invalid DRONESCAN_VRAM_LIMIT_GB (negative, zero, not a number),
    When the settings are loaded,
    Then a validation error names the field.
    """
    monkeypatch.setenv("DRONESCAN_VRAM_LIMIT_GB", value)

    with pytest.raises(ValidationError, match="vram_limit_gb"):
        load_settings()


def test_unknown_profile_explains_remedy() -> None:
    """Given a profile name that does not exist,
    When the settings are loaded,
    Then a ProfileError names the available profiles as remedy.
    """
    with pytest.raises(ProfileError) as excinfo:
        load_settings("huge_24gb")

    message = str(excinfo.value)
    assert "huge_24gb" in message
    assert "low_vram_6gb" in message


@pytest.mark.parametrize("name", ["../etc/passwd", "Low-VRAM", ""])
def test_invalid_profile_name_is_rejected(name: str) -> None:
    """Given a profile name with path separators or invalid characters,
    When the settings are loaded,
    Then a ProfileError is raised before any file is read.
    """
    with pytest.raises(ProfileError):
        load_settings(name)


def test_unknown_key_in_profile_is_rejected(tmp_path: Path) -> None:
    """Given a profile file with a misspelled key,
    When it is loaded,
    Then validation fails instead of silently ignoring the typo.
    """
    write_profile(tmp_path, "typo", VALID_PROFILE + "vram_limt_gb: 2.0\n")

    with pytest.raises(ValidationError, match="vram_limt_gb"):
        load_settings("typo", profiles_dir=tmp_path)


def test_yaml_syntax_error_is_explained(tmp_path: Path) -> None:
    """Given a profile file with broken YAML syntax,
    When it is loaded,
    Then a ProfileError names the profile and the remedy instead of a traceback.
    """
    write_profile(tmp_path, "syntax", "vram_limit_gb: [5.0\n")

    with pytest.raises(ProfileError, match=r"(?s)syntax.*Abhilfe"):
        load_settings("syntax", profiles_dir=tmp_path)


def test_profile_must_be_a_mapping(tmp_path: Path) -> None:
    """Given a profile file that is not a YAML mapping,
    When it is loaded,
    Then a ProfileError explains the expected format.
    """
    write_profile(tmp_path, "broken", "- just\n- a list\n")

    with pytest.raises(ProfileError, match="broken"):
        load_settings("broken", profiles_dir=tmp_path)


def test_custom_profile_directory(tmp_path: Path) -> None:
    """Given a valid profile in a custom directory,
    When it is loaded,
    Then all sections come from that file.
    """
    write_profile(tmp_path, "tiny", VALID_PROFILE)

    settings = load_settings("tiny", profiles_dir=tmp_path)

    assert isinstance(settings, Settings)
    assert settings.profile == "tiny"
    assert settings.vram_limit_gb == pytest.approx(3.0)
    assert settings.search.detector_batch_size == 1
