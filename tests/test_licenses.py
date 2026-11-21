"""License gate (KAN-23): allow-/denylist, models.lock, copyleft imports."""

from __future__ import annotations

import json
import shutil
from email.message import Message
from pathlib import Path

import pytest

from dronescan.common.licenses import (
    Component,
    Finding,
    Policy,
    Scope,
    check_component,
    load_policy,
    main,
    model_components,
    npm_components,
    resolve_python_license,
    run_gate,
    runtime_packages,
    scan_imports,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy() -> Policy:
    return load_policy(ROOT / "licenses")


def comp(license_: str | None, scope: Scope = "runtime", name: str = "pkg") -> Component:
    return Component(kind="python", name=name, version="1.0", license=license_, scope=scope)


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def test_policy_files_name_the_forbidden_projects(policy: Policy) -> None:
    """Given licenses/denylist.yaml,
    When it is loaded,
    Then Inria 3DGS, DUSt3R, MASt3R and SuperPoint weights are denied (AC 1).
    """
    assert {"diff-gaussian-rasterization", "gaussian-splatting", "dust3r", "mast3r"} <= set(
        policy.denied_packages
    )
    denied_model_patterns = " ".join(p.pattern for p, _ in policy.denied_models)
    assert "superpoint" in denied_model_patterns
    assert {"MIT", "Apache-2.0", "PSF-2.0"} <= policy.allowed


@pytest.mark.parametrize(
    "expression",
    ["MIT", "Apache-2.0 OR BSD-2-Clause", "MIT OR GPL-3.0-only", "(MIT AND BSD-3-Clause)"],
)
def test_permissive_expressions_pass(policy: Policy, expression: str) -> None:
    """Given a permissive SPDX expression (OR: one allowed alternative is enough),
    When a runtime dependency is checked,
    Then there is no error.
    """
    assert errors(check_component(comp(expression), policy)) == []


@pytest.mark.parametrize("license_", ["CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "Non-Commercial Use Only"])
def test_nc_license_fails(policy: Policy, license_: str) -> None:
    """Given a dependency with a non-commercial license,
    When it is checked (runtime or dev),
    Then the gate fails (Q-8).
    """
    assert errors(check_component(comp(license_), policy))
    assert errors(check_component(comp(license_, scope="dev"), policy))


def test_copyleft_runtime_dependency_fails(policy: Policy) -> None:
    """Given a GPL package in the runtime dependencies,
    When it is checked,
    Then the gate fails: copyleft only as separate process/container (AC 3).
    """
    found = errors(check_component(comp("GPL-3.0-only"), policy))
    assert found and "Copyleft" in found[0].message


def test_copyleft_dev_tool_is_only_a_warning(policy: Policy) -> None:
    """Given a GPL tool used only in development (separate process, never shipped),
    When it is checked,
    Then it is reported as warning, not as error.
    """
    findings = check_component(comp("GPL-2.0-or-later", scope="dev"), policy)
    assert findings and errors(findings) == []


@pytest.mark.parametrize("license_", [None, "", "Proprietary", "UNKNOWN"])
def test_unclear_license_fails(policy: Policy, license_: str | None) -> None:
    """Given a dependency without a recognisable SPDX license,
    When it is checked,
    Then the gate fails (unclear licenses are forbidden, ADR-0007).
    """
    assert errors(check_component(comp(license_), policy))


def test_denylisted_package_fails_even_with_permissive_metadata(policy: Policy) -> None:
    """Given the Inria rasterizer with misleading MIT metadata,
    When it is checked,
    Then the denylist wins.
    """
    found = errors(check_component(comp("MIT", name="diff_gaussian_rasterization"), policy))
    assert found and "Inria" in found[0].message


def make_metadata(**fields: str | list[str]) -> Message:
    msg = Message()
    for key, value in fields.items():
        for item in value if isinstance(value, list) else [value]:
            msg[key.replace("_", "-")] = item
    return msg


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"Name": "a", "License_Expression": "MIT OR Apache-2.0"}, "MIT OR Apache-2.0"),
        ({"Name": "b", "License": "BSD-3-Clause"}, "BSD-3-Clause"),
        ({"Name": "c", "Classifier": ["License :: OSI Approved :: MIT License"]}, "MIT"),
        ({"Name": "d", "License": "ISC License"}, "ISC"),
        ({"Name": "nodeenv", "License": "BSD"}, "BSD-3-Clause"),  # override
        ({"Name": "e", "License": "BSD"}, None),  # ambiguous without override
    ],
)
def test_python_license_resolution(
    policy: Policy, fields: dict[str, str | list[str]], expected: str | None
) -> None:
    """Given package metadata in its various forms,
    When the license is resolved,
    Then the override, License-Expression, License field or classifier is used in that order.
    """
    assert resolve_python_license(make_metadata(**fields), policy) == expected


def test_runtime_scope_comes_from_uv_lock(tmp_path: Path) -> None:
    """Given a uv.lock with runtime and dev dependencies,
    When the runtime closure is computed,
    Then transitive runtime deps are included and dev-only deps are not.
    """
    (tmp_path / "uv.lock").write_text(
        """
version = 1
[[package]]
name = "drone-scan"
dependencies = [{ name = "typer" }]
[package.dev-dependencies]
dev = [{ name = "pytest" }]
[[package]]
name = "typer"
dependencies = [{ name = "rich" }]
[[package]]
name = "rich"
[[package]]
name = "pytest"
dependencies = [{ name = "pluggy" }]
[[package]]
name = "pluggy"
""",
        encoding="utf-8",
    )

    assert runtime_packages(tmp_path / "uv.lock", "drone-scan") == {"typer", "rich"}


def write_models_lock(path: Path, entries: list[dict[str, str]]) -> Path:
    path.write_text(json.dumps({"models": entries}), encoding="utf-8")  # JSON is valid YAML
    return path


MODEL = {
    "name": "aliked-n16",
    "version": "1.0",
    "sha256": "a" * 64,
    "license": "BSD-3-Clause",
    "source": "https://github.com/Shiaoming/ALIKED",
}


def test_models_lock_with_nc_weights_fails(tmp_path: Path, policy: Policy) -> None:
    """Given models.lock with a model whose weights are CC-BY-NC,
    When the gate checks it,
    Then the build fails (Q-8).
    """
    lock = write_models_lock(tmp_path / "models.lock", [{**MODEL, "license": "CC-BY-NC-4.0"}])

    components, findings = model_components(lock)
    all_findings = findings + [f for c in components for f in check_component(c, policy)]

    assert errors(all_findings)


def test_denylisted_model_name_fails(tmp_path: Path, policy: Policy) -> None:
    """Given SuperPoint weights in models.lock (even with a permissive license field),
    When the gate checks it,
    Then the denylist rejects it and recommends ALIKED + LightGlue.
    """
    lock = write_models_lock(tmp_path / "models.lock", [{**MODEL, "name": "SuperPoint_v1"}])

    components, _ = model_components(lock)
    found = errors(check_component(components[0], policy))

    assert found and "ALIKED" in found[0].message


def test_cc_by_is_allowed_for_models_only(tmp_path: Path, policy: Policy) -> None:
    """Given CC-BY-4.0 (allowed for data/weights),
    When used by a model vs. a Python package,
    Then the model passes and the package fails.
    """
    lock = write_models_lock(tmp_path / "models.lock", [{**MODEL, "license": "CC-BY-4.0"}])
    components, findings = model_components(lock)

    assert findings == [] and errors(check_component(components[0], policy)) == []
    assert errors(check_component(comp("CC-BY-4.0"), policy))


def test_incomplete_models_lock_entry_fails(tmp_path: Path) -> None:
    """Given a models.lock entry without sha256 or source,
    When the gate reads it,
    Then it reports the missing fields.
    """
    entry = {k: v for k, v in MODEL.items() if k not in {"sha256", "source"}}
    _, findings = model_components(write_models_lock(tmp_path / "models.lock", [entry]))

    assert errors(findings)
    assert "sha256" in findings[0].message


def test_npm_licenses_are_read_from_node_modules(tmp_path: Path, policy: Policy) -> None:
    """Given a viewer with node_modules (incl. scoped packages),
    When the gate collects npm packages,
    Then their licenses are checked like Python ones.
    """
    modules = tmp_path / "viewer" / "node_modules"
    for name, lic in [("three", "MIT"), ("@scope/bad", "CC-BY-NC-4.0")]:
        pkg = modules / name
        pkg.mkdir(parents=True)
        (pkg / "package.json").write_text(
            json.dumps({"name": name, "version": "1", "license": lic})
        )

    components = npm_components(tmp_path / "viewer")
    bad = [c for c in components if errors(check_component(c, policy))]

    assert {c.name for c in components} == {"three", "@scope/bad"}
    assert [c.name for c in bad] == ["@scope/bad"]


def test_container_only_import_in_src_is_reported(tmp_path: Path, policy: Policy) -> None:
    """Given src/ code that imports a copyleft tool reserved for containers,
    When the imports are scanned,
    Then the violation names file and module (AC 3).
    """
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "mesh.py").write_text("import os\nfrom pymeshlab import MeshSet\n", encoding="utf-8")

    found = errors(scan_imports(tmp_path / "src", policy, extra_modules=set()))

    assert found and "pymeshlab" in found[0].message and "mesh.py" in found[0].message


def test_import_of_copyleft_distribution_is_reported(tmp_path: Path, policy: Policy) -> None:
    """Given src/ importing a module whose distribution is copyleft,
    When the imports are scanned,
    Then it is reported.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("import gplthing.sub\n", encoding="utf-8")

    assert errors(scan_imports(src, policy, extra_modules={"gplthing"}))


def test_repository_passes_the_gate() -> None:
    """Given the current repository (uv.lock, installed env, models.lock, src/),
    When the license gate runs,
    Then there are no errors.
    """
    assert errors(run_gate(ROOT)) == []


def test_main_exit_code_reflects_violations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a repository copy with NC weights in models.lock,
    When `python -m dronescan.common.licenses` runs,
    Then it exits 1 and prints a German report; the real repo exits 0.
    """
    assert main([str(ROOT)]) == 0

    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(ROOT / "licenses", repo / "licenses")
    shutil.copy(ROOT / "uv.lock", repo / "uv.lock")
    (repo / "src").mkdir()
    write_models_lock(repo / "models.lock", [{**MODEL, "license": "CC-BY-NC-4.0"}])

    assert main([str(repo)]) == 1
    assert "Lizenz-Gate" in capsys.readouterr().out
