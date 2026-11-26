# SPDX-License-Identifier: AGPL-3.0-or-later
"""Camera profiles and COLMAP intrinsics estimation (KAN-29, US-014).

The sfm tests use the fake docker of tests/test_sfm.py; videos are synthetic (ffmpeg).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from dronescan.capture.camera import (
    CameraError,
    CameraProfile,
    Intrinsics,
    available_cameras,
    load_camera,
    write_camera,
)
from dronescan.capture.scan import ScanMeta, load_scan
from dronescan.cli import app
from dronescan.common.config import load_settings
from dronescan.common.stage import ExitCode
from dronescan.ingest.stage import FlightSummary
from tests.synthetic_room import views
from tests.test_scan_import import _create as create
from tests.test_scan_import import require_ffmpeg, video, ws  # noqa: F401  (fixtures)
from tests.test_sfm import (  # noqa: F401  (fixtures)
    MAX_PROCESS_MIB,
    SMALL,
    FakeDocker,
    docker,
    make_scan,
    needs_colmap,
    room_scan,
    sfm,
)

runner = CliRunner()

CALIBRATED = CameraProfile(
    name="testcam",
    model="OPENCV",
    intrinsics=Intrinsics(
        width=1280, height=960, params=[1000, 1002, 640.5, 480.5, -0.1, 0.02, 0.001, -0.002]
    ),
    source="calibration",
    rms_error_px=0.3,
)


def set_camera(scan_dir: Path, camera: CameraProfile | None, flight: int = 0) -> None:
    meta = load_scan(scan_dir)
    flights = list(meta.flights)
    flights[flight] = flights[flight].model_copy(update={"camera": camera})
    meta = ScanMeta(id=meta.id, name=meta.name, created=meta.created, flights=flights)
    (scan_dir / "scan.yaml").write_text(yaml.safe_dump(meta.model_dump(mode="json")))


def write_summary(scan_dir: Path, flight: str, width: int = 640, height: int = 480) -> None:
    """ingest/<flight>/manifest.json with the keyframe size."""
    summary = FlightSummary(
        flight_id=flight,
        video=f"flights/{flight}/video.mp4",
        video_sha256="0" * 64,
        params={},
        tools={},
        frame_width=width,
        frame_height=height,
        analysed_frames=10,
        decoded_s=10,
        sharpness_min=1,
        sharpness_median=1,
        sharpness_max=1,
        sharpness_threshold=1,
        counts={},
    )
    (scan_dir / "ingest" / flight / "manifest.json").write_text(summary.model_dump_json())


# ---------- AC 1: profile mechanism ----------


def test_smartphone_profile_is_shipped_without_guessed_values() -> None:
    """Given the shipped camera profiles,
    When the smartphone profile is loaded,
    Then it names a COLMAP camera model but no intrinsics (COLMAP estimates them; PO decision
    2026-11-25: no data-sheet values, no drone profiles yet).
    """
    assert available_cameras() == ["smartphone"]
    profile = load_camera("smartphone")
    assert profile.model == "SIMPLE_RADIAL"
    assert profile.intrinsics is None
    assert profile.description


@pytest.mark.parametrize(
    ("model", "count"),
    [("PINHOLE", 4), ("SIMPLE_RADIAL", 4), ("OPENCV", 8), ("OPENCV_FISHEYE", 8)],
)
def test_profile_file_round_trip(tmp_path: Path, model: str, count: int) -> None:
    """Given a profile with intrinsics for one of the camera models,
    When it is written and loaded by its path,
    Then it is unchanged.
    """
    params = [float(value) for value in range(1, count + 1)]
    profile = CameraProfile(
        name="cam",
        model=model,  # type: ignore[arg-type]
        intrinsics=Intrinsics(width=100, height=50, params=params),
    )
    write_camera(profile, tmp_path / "cam.yaml")

    assert load_camera(str(tmp_path / "cam.yaml")) == profile


def test_wrong_parameter_count_is_explained(tmp_path: Path) -> None:
    """Given a profile file whose parameters do not fit the model,
    When it is loaded,
    Then a CameraError names the problem and a remedy.
    """
    path = tmp_path / "broken.yaml"
    path.write_text(
        "name: broken\nmodel: OPENCV\nintrinsics: {width: 10, height: 10, params: [1, 2, 3]}\n"
    )
    with pytest.raises(CameraError, match="OPENCV needs 8 params.*Abhilfe"):
        load_camera(str(path))


def test_unknown_model_and_name_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fov.yaml"
    path.write_text("name: fov\nmodel: FOV\n")
    with pytest.raises(CameraError, match="ungültig"):
        load_camera(str(path))
    with pytest.raises(CameraError, match="Mitgelieferte Profile: smartphone"):
        load_camera("dji_neo_2")
    with pytest.raises(CameraError, match="nicht gefunden"):
        load_camera(str(tmp_path / "missing.yaml"))


def test_intrinsics_are_scaled_to_the_keyframe_size() -> None:
    """Given a profile calibrated at 1280x960,
    When the keyframes are 640x480,
    Then focal length and principal point are halved and distortion stays.
    """
    params = CALIBRATED.params_for(640, 480)

    assert params == pytest.approx([500, 501, 320.25, 240.25, -0.1, 0.02, 0.001, -0.002])


def test_single_focal_models_scale_f_cx_cy() -> None:
    profile = CameraProfile(
        name="simple",
        model="SIMPLE_RADIAL",
        intrinsics=Intrinsics(width=1600, height=900, params=[1200, 800, 450, 0.05]),
    )
    assert profile.params_for(1600, 900) == pytest.approx([1200, 800, 450, 0.05])
    # fit_size() rounds to even pixels: 1066x600 is still the same aspect ratio
    assert profile.params_for(1066, 600) == pytest.approx([799.75, 533.0, 300.0, 0.05], rel=1e-3)


def test_other_aspect_ratio_gives_no_intrinsics() -> None:
    """Given a profile calibrated in 4:3,
    When the keyframes are 16:9 (another video mode),
    Then the values are not used.
    """
    assert CALIBRATED.params_for(1280, 720) is None
    assert CameraProfile(name="x", model="PINHOLE").params_for(10, 10) is None


def test_camera_list_shows_the_shipped_profiles() -> None:
    result = runner.invoke(app, ["camera", "list"])

    assert result.exit_code == 0
    assert "smartphone: SIMPLE_RADIAL, Intrinsik schätzt COLMAP" in result.output


def test_flight_add_stores_a_copy_of_the_profile(ws: Path, video: Path, tmp_path: Path) -> None:  # noqa: F811
    """Given a calibrated profile file,
    When a flight is added with --camera <path>,
    Then scan.yaml holds a copy of the profile, so the scan stays reproducible.
    """
    write_camera(CALIBRATED, tmp_path / "testcam.yaml")
    scan_id = create(ws)

    with_file = runner.invoke(
        app,
        ["flight", "add", scan_id, str(video), "--workspace", str(ws)]
        + ["--camera", str(tmp_path / "testcam.yaml")],
    )
    (tmp_path / "testcam.yaml").unlink()
    by_name = runner.invoke(
        app,
        ["flight", "add", scan_id, str(video), "--workspace", str(ws), "--camera", "smartphone"],
    )
    without = runner.invoke(app, ["flight", "add", scan_id, str(video), "--workspace", str(ws)])

    assert with_file.exit_code == by_name.exit_code == without.exit_code == 0, with_file.output
    assert "Kamera testcam" in with_file.output
    flights = load_scan(ws / scan_id).flights
    assert flights[0].camera == CALIBRATED
    assert flights[1].camera == load_camera("smartphone")
    assert flights[2].camera is None


def test_flight_add_with_unknown_camera_exits_2(ws: Path, video: Path) -> None:  # noqa: F811
    scan_id = create(ws)

    result = runner.invoke(
        app, ["flight", "add", scan_id, str(video), "--workspace", str(ws), "--camera", "tello"]
    )

    assert result.exit_code == 2
    assert "smartphone" in result.output
    assert load_scan(ws / scan_id).flights == []


# ---------- AC 1 + 2: sfm uses the profile, estimates and stores the intrinsics ----------


def test_without_profile_colmap_estimates_with_the_default_model(
    tmp_path: Path,
    docker: FakeDocker,  # noqa: F811
) -> None:
    """Given a flight without camera profile,
    When sfm runs,
    Then COLMAP gets the model from the VRAM profile (sfm.camera_model) and no start values,
    and the estimated intrinsics are stored in the scan as a reusable camera profile.
    """
    scan = make_scan(tmp_path / "scan", 10)
    docker.models(10)

    result = sfm(scan)

    assert result.exit_code is ExitCode.OK
    options = docker.options("feature_extractor")
    assert options["--ImageReader.camera_model"] == "SIMPLE_RADIAL"
    assert "--ImageReader.camera_params" not in options
    estimated = load_camera(str(scan / "sfm" / "cameras" / "f01.yaml"))
    assert estimated.source == "colmap"
    assert estimated.model == "SIMPLE_RADIAL"
    assert estimated.intrinsics == Intrinsics(width=640, height=480, params=[500, 320, 240, 0])
    cameras = result.manifest.metrics["cameras"]
    assert cameras["f01"]["profile"] is None
    assert cameras["f01"]["estimated"]["params"] == [500, 320, 240, 0]


def test_profile_model_and_scaled_start_values_reach_colmap(
    tmp_path: Path,
    docker: FakeDocker,  # noqa: F811
) -> None:
    """Given a flight with a calibrated OPENCV profile (1280x960) and keyframes of 640x480,
    When sfm runs,
    Then the feature extractor gets the model and the scaled intrinsics as start values.
    """
    scan = make_scan(tmp_path / "scan", 10)
    write_summary(scan, "f01")
    set_camera(scan, CALIBRATED)
    docker.models(10)

    result = sfm(scan)

    assert result.exit_code is ExitCode.OK
    options = docker.options("feature_extractor")
    assert options["--ImageReader.camera_model"] == "OPENCV"
    params = [float(value) for value in options["--ImageReader.camera_params"].split(",")]
    assert params == pytest.approx([500, 501, 320.25, 240.25, -0.1, 0.02, 0.001, -0.002])
    assert result.manifest.metrics["cameras"]["f01"]["profile"] == "testcam"
    assert result.manifest.warnings == []


def test_flights_with_different_cameras_are_extracted_separately(
    tmp_path: Path,
    docker: FakeDocker,  # noqa: F811
) -> None:
    """Given two flights, one with a fisheye profile and one without,
    When sfm runs,
    Then each camera setup gets its own feature extraction with its own image list, matching
    and mapping still run once over all flights, and both estimates are stored.
    """
    scan = make_scan(tmp_path / "scan", 6, 4)
    set_camera(scan, CameraProfile(name="fisheye", model="OPENCV_FISHEYE"), flight=1)
    docker.models(10)

    result = sfm(scan)

    assert result.exit_code is ExitCode.OK
    assert docker.commands().count("feature_extractor") == 2
    assert docker.commands().count("mapper") == 1
    first, second = docker.options("feature_extractor"), docker.options("feature_extractor", 1)
    assert first["--ImageReader.camera_model"] == "SIMPLE_RADIAL"
    assert second["--ImageReader.camera_model"] == "OPENCV_FISHEYE"
    work = scan / "sfm" / "colmap"
    lists = [
        (work / options["--image_list_path"].removeprefix("/sfm/")).read_text().split()
        for options in (first, second)
    ]
    assert lists == [
        [f"f01/frames/{n:06d}.jpg" for n in range(1, 7)],
        [f"f02/frames/{n:06d}.jpg" for n in range(1, 5)],
    ]
    assert sorted(p.name for p in (scan / "sfm" / "cameras").iterdir()) == ["f01.yaml", "f02.yaml"]


def test_profile_for_another_aspect_ratio_warns_and_estimates(
    tmp_path: Path,
    docker: FakeDocker,  # noqa: F811
) -> None:
    """Given a profile calibrated in 4:3 and 16:9 keyframes,
    When sfm runs,
    Then the start values are dropped with a warning and a hint, COLMAP estimates.
    """
    scan = make_scan(tmp_path / "scan", 10)
    write_summary(scan, "f01", 640, 360)
    set_camera(scan, CALIBRATED)
    docker.models(10)

    result = sfm(scan)

    assert result.exit_code is ExitCode.OK
    options = docker.options("feature_extractor")
    assert options["--ImageReader.camera_model"] == "OPENCV"
    assert "--ImageReader.camera_params" not in options
    assert "Seitenverhältnis (1280x960)" in result.manifest.warnings[0]
    assert any("neu kalibrieren" in hint for hint in result.manifest.hints)


def test_changed_profile_invalidates_the_cache(
    tmp_path: Path,
    docker: FakeDocker,  # noqa: F811
) -> None:
    """Given a finished sfm run,
    When the camera profile of the flight changes,
    Then sfm runs again instead of using the cache.
    """
    scan = make_scan(tmp_path / "scan", 10)
    docker.models(10)
    assert sfm(scan).exit_code is ExitCode.OK
    assert sfm(scan).skipped

    set_camera(scan, CameraProfile(name="fisheye", model="OPENCV_FISHEYE"))
    result = sfm(scan)

    assert not result.skipped
    assert docker.options("feature_extractor", 1)["--ImageReader.camera_model"] == "OPENCV_FISHEYE"


def test_oom_in_a_later_camera_group_re_extracts_all_groups(
    tmp_path: Path,
    docker: FakeDocker,  # noqa: F811
) -> None:
    """Given two flights with different camera setups,
    When the GPU runs out of memory in the second feature extraction,
    Then the database is rebuilt: the retry extracts both groups again with smaller values.
    """
    scan = make_scan(tmp_path / "scan", 6, 4)
    set_camera(scan, CameraProfile(name="fisheye", model="OPENCV_FISHEYE"), flight=1)
    docker.fail("feature_extractor", "ERROR: Not enough GPU memory", code=1, skip=1)
    docker.models(10)

    result = sfm(scan)

    assert result.exit_code is ExitCode.OK, result.manifest.error
    assert result.manifest.metrics["oom_retry"] is True
    calls = [docker.options("feature_extractor", n) for n in range(4)]
    assert [c["--ImageReader.camera_model"] for c in calls] == [
        "SIMPLE_RADIAL",
        "OPENCV_FISHEYE",
        "SIMPLE_RADIAL",
        "OPENCV_FISHEYE",
    ]
    edges = [int(c["--FeatureExtraction.max_image_size"]) for c in calls]
    assert edges[2] == edges[3] < edges[0] == edges[1]
    database = (scan / "sfm" / "colmap" / "database.db").read_text().split()
    assert database == ["/sfm/image_list_0.txt", "/sfm/image_list_1.txt"]


def test_missing_keyframe_size_exits_10(tmp_path: Path, docker: FakeDocker) -> None:  # noqa: F811
    scan = make_scan(tmp_path / "scan", 10)
    set_camera(scan, CALIBRATED)

    result = sfm(scan)

    assert result.exit_code is ExitCode.INPUT_MISSING
    assert "Flug f01" in (result.manifest.error or "")


def test_estimated_profile_can_be_reused_for_a_flight(
    tmp_path: Path,
    docker: FakeDocker,  # noqa: F811
) -> None:
    """Given the intrinsics COLMAP estimated for a flight,
    When the stored file is given as camera profile,
    Then it loads as a profile with these values.
    """
    scan = make_scan(tmp_path / "scan", 10)
    docker.models(10)
    sfm(scan)

    profile = load_camera(str(scan / "sfm" / "cameras" / "f01.yaml"))

    assert profile.params_for(640, 480) == [500, 320, 240, 0]


# ---------- real COLMAP (GTX 1060) ----------


@pytest.mark.gpu
@needs_colmap
@pytest.mark.parametrize("camera", [None, "pinhole"])
def test_real_colmap_estimates_the_true_focal_length(tmp_path: Path, camera: str | None) -> None:
    """Given 30 views of the synthetic room (f = 0.8 * width, principal point in the centre),
    once without profile and once with a PINHOLE profile calibrated at twice the size,
    When the stage runs with COLMAP on the GTX 1060,
    Then COLMAP accepts the model and scaled start values, registers all views, and the stored
    estimate matches the true focal length within 1 %.
    """
    width, height = SMALL
    scan = room_scan(tmp_path / "scan", views(30))
    if camera is not None:
        write_summary(scan, "f01", width, height)
        intrinsics = Intrinsics(
            width=2 * width, height=2 * height, params=[1.6 * width, 1.6 * width, width, height]
        )
        set_camera(scan, CameraProfile(name=camera, model="PINHOLE", intrinsics=intrinsics))

    result = sfm(scan, load_settings("low_vram_6gb"))

    assert result.exit_code is ExitCode.OK, result.manifest.error
    assert result.manifest.metrics["registration_ratio"] == 1.0
    assert result.manifest.peak_vram_mb is not None
    assert result.manifest.peak_vram_mb <= MAX_PROCESS_MIB
    estimated = load_camera(str(scan / "sfm" / "cameras" / "f01.yaml"))
    assert estimated.model == ("PINHOLE" if camera else "SIMPLE_RADIAL")
    assert estimated.intrinsics is not None
    assert estimated.intrinsics.params[0] == pytest.approx(0.8 * width, rel=0.01)
