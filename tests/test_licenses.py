# SPDX-License-Identifier: AGPL-3.0-or-later
"""License gate (KAN-23, ADR-0012): allow-/denylist, NC marking, models.lock, copyleft imports."""

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
    python_components,
    resolve_python_license,
    run_gate,
    runtime_packages,
    scan_imports,
    scan_repository,
)

ROOT = Path(__file__).resolve().parents[1]
MIT_CLASSIFIER = "License :: OSI Approved :: MIT License"
APACHE_CLASSIFIER = "License :: OSI Approved :: Apache Software License"


@pytest.fixture
def policy() -> Policy:
    return load_policy(ROOT / "licenses")


def comp(license_: str | None, scope: Scope = "runtime", name: str = "pkg") -> Component:
    return Component(kind="python", name=name, version="1.0", license=license_, scope=scope)


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def nc(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "nc"]


def test_policy_files_mark_the_known_nc_projects(policy: Policy) -> None:
    """Given licenses/denylist.yaml,
    When it is loaded,
    Then Inria 3DGS, DUSt3R, MASt3R, SiftGPU and SuperPoint weights are known as NC (ADR-0012).
    """
    assert {
        "diff-gaussian-rasterization",
        "gaussian-splatting",
        "dust3r",
        "mast3r",
    } <= set(policy.nc_packages)
    assert "colmap/siftgpu" in {name for name, _ in policy.nc_image_components}
    nc_model_patterns = " ".join(p.pattern for p, _ in policy.nc_models)
    assert "superpoint" in nc_model_patterns
    assert {"MIT", "Apache-2.0", "PSF-2.0", "BSL-1.0", "Zlib", "0BSD", "CC0-1.0", "curl"} <= (
        policy.allowed
    )


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
def test_nc_license_is_marked_not_failed(policy: Policy, license_: str) -> None:
    """Given a dependency with a non-commercial license,
    When it is checked (runtime or dev),
    Then the gate does not fail but marks it as NC (ADR-0012).
    """
    for scope in ("runtime", "dev"):
        findings = check_component(comp(license_, scope=scope), policy)
        assert errors(findings) == []
        assert len(nc(findings)) == 1 and "ADR-0012" in findings[0].message


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


@pytest.mark.parametrize("scope", ["runtime", "dev"])
def test_nc_does_not_hide_copyleft(policy: Policy, scope: Scope) -> None:
    """Given an SPDX expression that combines copyleft AND an NC license,
    When it is checked,
    Then it is marked NC and the copyleft part is still judged as before
    (runtime: error, dev: warning); with OR the NC alternative may be chosen.
    """
    findings = check_component(comp("AGPL-3.0-only AND CC-BY-NC-4.0", scope=scope), policy)
    assert nc(findings)
    if scope == "runtime":
        assert errors(findings) and "Copyleft" in errors(findings)[0].message
    else:
        assert errors(findings) == [] and any(f.severity == "warning" for f in findings)

    either = check_component(comp("GPL-3.0-only OR CC-BY-NC-4.0", scope=scope), policy)
    assert nc(either) and errors(either) == []


def test_image_components_are_reported_as_nc() -> None:
    """Given NC components compiled into the Docker images (e.g. SiftGPU in COLMAP),
    When the gate runs on the repository,
    Then each one appears in the report as NC (ADR-0012), without failing the gate.
    """
    findings = run_gate(ROOT)
    assert "image:colmap/siftgpu" in {f.subject for f in nc(findings)}
    assert errors(findings) == []


def test_known_nc_package_is_marked_even_with_permissive_metadata(policy: Policy) -> None:
    """Given the Inria rasterizer with misleading MIT metadata,
    When it is checked,
    Then it is marked as NC from the denylist (no error).
    """
    findings = check_component(comp("MIT", name="diff_gaussian_rasterization"), policy)
    assert errors(findings) == []
    assert nc(findings) and "Inria" in findings[0].message


def test_hard_denylist_still_fails(tmp_path: Path) -> None:
    """Given a package and a model in the hard-ban sections `packages`/`models`,
    When they are checked,
    Then the gate fails regardless of their license.
    """
    licenses = tmp_path / "licenses"
    shutil.copytree(ROOT / "licenses", licenses)
    deny = (licenses / "denylist.yaml").read_text(encoding="utf-8")
    deny = deny.replace(
        "packages: []", "packages:\n  - name: evil-pkg\n    reason: gesperrt"
    ).replace("models: []", "models:\n  - pattern: 'evil'\n    reason: gesperrt")
    (licenses / "denylist.yaml").write_text(deny, encoding="utf-8")
    strict = load_policy(licenses)
    model = Component(
        kind="model", name="evil-net", version="1", license="MIT", scope="runtime", source=""
    )

    assert errors(check_component(comp("MIT", name="evil_pkg"), strict))
    assert errors(check_component(model, strict))


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
        ({"Name": "nodeenv", "Version": "1.12.0", "License": "BSD"}, "BSD-3-Clause"),
        ({"Name": "nodeenv", "Version": "2.0.0", "License": "BSD"}, None),  # override pinned
        (
            {"Name": "f", "License_Expression": "MIT", "Classifier": [APACHE_CLASSIFIER]},
            "MIT",
        ),
        ({"Name": "e", "License": "BSD"}, None),  # ambiguous without override
        (  # short NC field wins over a permissive classifier (NC must stay visible)
            {"Name": "g", "License": "Non-Commercial Use Only", "Classifier": [MIT_CLASSIFIER]},
            "Non-Commercial Use Only",
        ),
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


def test_models_lock_with_nc_weights_is_marked(tmp_path: Path, policy: Policy) -> None:
    """Given models.lock with a model whose weights are CC-BY-NC,
    When the gate checks it,
    Then the build passes and the model is marked as NC (ADR-0012).
    """
    lock = write_models_lock(tmp_path / "models.lock", [{**MODEL, "license": "CC-BY-NC-4.0"}])

    components, findings = model_components(lock)
    all_findings = findings + [f for c in components for f in check_component(c, policy)]

    assert errors(all_findings) == []
    assert len(nc(all_findings)) == 1


def test_known_nc_model_name_is_marked(tmp_path: Path, policy: Policy) -> None:
    """Given SuperPoint weights in models.lock (even with a permissive license field),
    When the gate checks it,
    Then it is marked as NC with the reason from the denylist.
    """
    lock = write_models_lock(tmp_path / "models.lock", [{**MODEL, "name": "SuperPoint_v1"}])

    components, _ = model_components(lock)
    findings = check_component(components[0], policy)

    assert errors(findings) == []
    assert nc(findings) and "SuperPoint" in findings[0].message


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
    marked = [c for c in components if nc(check_component(c, policy))]

    assert {c.name for c in components} == {"three", "@scope/bad"}
    assert [c.name for c in marked] == ["@scope/bad"]
    assert not any(errors(check_component(c, policy)) for c in components)


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
    """Note: depends on the locally installed environment (`uv sync --locked`).

    Given the current repository (uv.lock, installed env, models.lock, src/),
    When the license gate runs,
    Then there are no errors.
    """
    assert errors(run_gate(ROOT)) == []


def test_main_exit_code_reflects_violations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given repository copies with NC weights resp. weights with an unclear license,
    When `python -m dronescan.common.licenses` runs,
    Then NC passes (exit 0) but is reported; the unclear license exits 1; the real repo exits 0.
    """
    assert main([str(ROOT)]) == 0

    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(ROOT / "licenses", repo / "licenses")
    shutil.copy(ROOT / "uv.lock", repo / "uv.lock")
    (repo / "src").mkdir()
    write_models_lock(repo / "models.lock", [{**MODEL, "license": "CC-BY-NC-4.0"}])
    capsys.readouterr()

    assert main([str(repo)]) == 0
    out = capsys.readouterr().out
    assert "NC      model:aliked" in out and "2 NC-Komponenten" in out  # + SiftGPU image

    write_models_lock(repo / "models.lock", [{**MODEL, "license": "Proprietary"}])
    assert main([str(repo)]) == 1
    assert "Lizenz-Gate fehlgeschlagen" in capsys.readouterr().out


@pytest.mark.parametrize(
    "expression",
    ["(MIT OR Apache-2.0) AND GPL-3.0-only", "MIT AND (GPL-3.0-only OR AGPL-3.0-only)"],
)
def test_brackets_with_and_cannot_hide_copyleft(policy: Policy, expression: str) -> None:
    """Given an expression where every alternative needs a copyleft license,
    When a runtime dependency is checked,
    Then the gate fails (brackets are parsed, not dropped).
    """
    assert errors(check_component(comp(expression), policy))


@pytest.mark.parametrize(
    "expression", ["LGPL-3.0-or-later", "GPL-2.0-or-later WITH Classpath-exception-2.0 OR MIT"]
)
def test_lgpl_and_with_exceptions(policy: Policy, expression: str) -> None:
    """Given LGPL (dynamic use allowed) or a WITH exception with a permissive alternative,
    When a runtime dependency is checked,
    Then it passes.
    """
    assert errors(check_component(comp(expression), policy)) == []


@pytest.mark.parametrize("expression", ["(MIT", "MIT OR", "AND MIT"])
def test_malformed_expression_is_unclear(policy: Policy, expression: str) -> None:
    """Given a malformed SPDX expression,
    When it is checked,
    Then it is reported as unclear (fail closed).
    """
    assert errors(check_component(comp(expression), policy))


def test_long_permissive_license_text_is_no_nc_false_positive(policy: Policy) -> None:
    """Given a full license text mentioning 'commercial or non-commercial' plus an MIT classifier,
    When the license is resolved,
    Then the classifier is used and the package passes.
    """
    text = (
        "Permission is hereby granted ... for any purpose, commercial or non-commercial, "
        "and by any means. THE SOFTWARE IS PROVIDED AS IS ..." * 2
    )
    meta = make_metadata(Name="g", License=text, Classifier=[MIT_CLASSIFIER])

    resolved = resolve_python_license(meta, policy)

    assert resolved == "MIT"
    assert errors(check_component(comp(resolved), policy)) == []


def test_nc_pattern_does_not_match_inside_words(policy: Policy) -> None:
    """Given a short license id that merely contains the letters 'nc' (e.g. 'NCSA'),
    When it is checked,
    Then it is not mistaken for a non-commercial license (but still unknown → allowlist).
    """
    found = errors(check_component(comp("NCSA"), policy))
    assert found and "non-commercial" not in found[0].message


def test_missing_uv_lock_fails_closed(tmp_path: Path, policy: Policy) -> None:
    """Given no uv.lock,
    When Python components are collected,
    Then an error is reported instead of treating everything as dev.
    """
    components, findings = python_components(policy, tmp_path / "uv.lock", dists=[])

    assert components == [] and errors(findings)


def test_locked_but_not_installed_is_warned(tmp_path: Path, policy: Policy) -> None:
    """Given a locked runtime package that is not installed,
    When the gate runs,
    Then it warns that the package was not checked.
    """
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "drone-scan"\ndependencies = [{ name = "typer" }]\n'
        '[[package]]\nname = "typer"\n',
        encoding="utf-8",
    )

    _, findings = python_components(policy, lock, dists=[])

    assert [f.subject for f in findings if f.severity == "warning"] == ["python:typer"]


def test_extras_count_as_runtime(tmp_path: Path) -> None:
    """Given a dependency that only comes in through an extra,
    When the runtime closure is computed,
    Then it counts as runtime (copyleft there must fail, not just warn).
    """
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "drone-scan"\n'
        '[package.optional-dependencies]\nserver = [{ name = "uvicorn" }]\n'
        '[[package]]\nname = "uvicorn"\n',
        encoding="utf-8",
    )

    assert runtime_packages(lock) == {"uvicorn"}


@pytest.mark.parametrize(
    "content",
    [
        "- just a list",
        "models:\n  - name: x",
        "models:\n  - " + json.dumps({**MODEL, "sha256": "abc"}),
    ],
)
def test_malformed_models_lock_is_reported(tmp_path: Path, content: str) -> None:
    """Given a malformed models.lock (no mapping, missing fields, bad sha256),
    When the gate reads it,
    Then it reports an error with a remedy instead of crashing.
    """
    path = tmp_path / "models.lock"
    path.write_text(content, encoding="utf-8")

    _, findings = model_components(path)

    assert errors(findings) and "Abhilfe" in findings[0].message


def test_npm_pnpm_indirect_and_legacy_licenses(tmp_path: Path, policy: Policy) -> None:
    """Given pnpm's layout (indirect deps under node_modules/.pnpm) and a legacy 'licenses' array,
    When npm components are collected,
    Then indirect packages and legacy licenses are included.
    """
    modules = tmp_path / "viewer" / "node_modules"
    indirect = modules / ".pnpm" / "dep@1.0.0" / "node_modules" / "dep"
    indirect.mkdir(parents=True)
    nc_pkg = {"name": "dep", "version": "1.0.0", "license": "CC-BY-NC-4.0"}
    (indirect / "package.json").write_text(json.dumps(nc_pkg))
    legacy = modules / "old"
    legacy.mkdir()
    legacy_pkg = {"name": "old", "version": "0.1", "licenses": [{"type": "MIT"}]}
    (legacy / "package.json").write_text(json.dumps(legacy_pkg))

    components = {c.name: c for c in npm_components(tmp_path / "viewer")}

    assert components["old"].license == "MIT"
    assert nc(check_component(components["dep"], policy))


def test_viewer_without_node_modules_fails(tmp_path: Path) -> None:
    """Given viewer/package.json but no installed node_modules,
    When the gate runs,
    Then it fails instead of silently skipping npm.
    """
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "licenses", repo / "licenses")
    shutil.copy(ROOT / "uv.lock", repo / "uv.lock")
    shutil.copy(ROOT / "models.lock", repo / "models.lock")
    (repo / "viewer").mkdir()
    (repo / "viewer" / "package.json").write_text("{}", encoding="utf-8")

    assert any(f.subject == "npm" for f in errors(run_gate(repo)))


def test_dynamic_import_of_container_only_module_is_reported(
    tmp_path: Path, policy: Policy
) -> None:
    """Given importlib.import_module('ultralytics') in src/,
    When imports are scanned,
    Then it is reported like a normal import.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "det.py").write_text("import importlib\nm = importlib.import_module('ultralytics')\n")

    assert errors(scan_imports(src, policy, extra_modules=set()))


def test_known_nc_model_matched_by_source(tmp_path: Path, policy: Policy) -> None:
    """Given a model whose name is harmless but whose source is the SuperPoint repo,
    When the gate checks it,
    Then the NC list matches on the source.
    """
    lock = write_models_lock(
        tmp_path / "models.lock",
        [
            {
                **MODEL,
                "name": "keypoints",
                "source": "https://github.com/magicleap/SuperPointPretrainedNetwork",
            }
        ],
    )
    components, _ = model_components(lock)

    assert nc(check_component(components[0], policy))


@pytest.mark.parametrize(
    "path",
    [
        "third_party/SiftGPU/SiftGPU.cpp",
        "vendor/diff_gaussian_rasterization/__init__.py",
        "models/dust3r.py",
        "weights/aliked-n16.pth",
        "data/model.ONNX",
    ],
)
def test_nc_code_or_weights_in_public_repo_fail(policy: Policy, path: str) -> None:
    """Given a tracked file that vendors a known NC project or commits model weights,
    When the repository is scanned,
    Then the gate fails: NC code/weights must not land in the public repo (ADR-0012).
    """
    found = errors(scan_repository(ROOT, policy, tracked=["src/ok.py", path]))
    assert [f.subject for f in found] == [f"repo:{path}"]


def test_nc_submodule_fails(tmp_path: Path, policy: Policy) -> None:
    """Given a .gitmodules entry pointing to an NC project,
    When the repository is scanned,
    Then the gate fails.
    """
    (tmp_path / ".gitmodules").write_text(
        '[submodule "x"]\n\tpath = ext/x\n\turl = https://github.com/naver/dust3r.git\n'
    )
    assert errors(scan_repository(tmp_path, policy, tracked=[]))


def test_dockerfile_building_nc_upstream_is_fine(tmp_path: Path, policy: Policy) -> None:
    """Given the real repository (COLMAP with SiftGPU is only built locally from upstream),
    When it is scanned,
    Then there are no findings; outside a git work tree the scan is skipped.
    """
    assert scan_repository(ROOT, policy) == []
    assert scan_repository(tmp_path, policy) == []
