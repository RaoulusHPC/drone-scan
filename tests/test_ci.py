"""CI pipeline and SITL test gating (KAN-17)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MAX_CI_MINUTES = 10  # AC 3


def load_workflow() -> dict[Any, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def all_steps(workflow: dict[Any, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def test_ci_runs_on_every_push_and_pull_request() -> None:
    """Given the GitHub Actions workflow,
    When its triggers are read,
    Then it runs on every push and every pull request (AC 1).
    """
    # PyYAML parses the bare key `on` as boolean True.
    triggers = load_workflow()[True]

    assert "push" in triggers
    assert "pull_request" in triggers


def test_ci_runs_make_check() -> None:
    """Given the workflow steps,
    When they are executed,
    Then `make check` runs, so a failing test or type error fails the job (AC 1/2).
    """
    commands = [step.get("run", "") for step in all_steps(load_workflow())]

    assert any(re.search(r"\bmake check\b", cmd) for cmd in commands)
    assert any("uv sync --locked" in cmd for cmd in commands)


def test_ci_job_has_time_limit() -> None:
    """Given every CI job,
    When it runs,
    Then GitHub cancels it after at most 10 minutes (AC 3).
    """
    for job in load_workflow()["jobs"].values():
        assert 0 < job["timeout-minutes"] <= MAX_CI_MINUTES


def test_actions_are_pinned_to_commit_sha() -> None:
    """Given third-party actions in the workflow,
    When they are referenced,
    Then they are pinned to a full commit SHA (supply-chain safety).
    """
    uses = [step["uses"] for step in all_steps(load_workflow()) if "uses" in step]

    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


SITL_TEST = """
import pytest

@pytest.mark.sitl
def test_needs_simulator():
    assert True
"""


@pytest.fixture
def sitl_project(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini("[pytest]\nmarkers =\n    sitl: sitl\n    gpu: gpu\n")
    pytester.makepyfile(test_sitl_demo=SITL_TEST)
    return pytester


def test_sitl_tests_are_skipped_without_simulator(
    sitl_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a test marked `sitl` and no DRONESCAN_SITL=1,
    When the test suite runs (e.g. `make check` in CI),
    Then the test is skipped with a reason instead of failing.
    """
    monkeypatch.delenv("DRONESCAN_SITL", raising=False)

    result = sitl_project.runpytest("-rs")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*DRONESCAN_SITL=1*"])


def test_sitl_tests_run_with_simulator(
    sitl_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a test marked `sitl` and DRONESCAN_SITL=1 (set by `make test-sitl`),
    When the test suite runs,
    Then the test is executed.
    """
    monkeypatch.setenv("DRONESCAN_SITL", "1")

    sitl_project.runpytest().assert_outcomes(passed=1)


def test_sitl_tests_keep_dronescan_env(
    sitl_project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given DRONESCAN_SITL=1,
    When a `sitl` test runs,
    Then the env isolation fixture keeps the DRONESCAN_* variables visible to it.
    """
    sitl_project.makepyfile(
        test_sitl_env="""
import os
import pytest

@pytest.mark.sitl
def test_sees_env():
    assert os.environ["DRONESCAN_SITL"] == "1"
"""
    )
    monkeypatch.setenv("DRONESCAN_SITL", "1")

    sitl_project.runpytest().assert_outcomes(passed=2)
