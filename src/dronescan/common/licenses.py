"""License gate (KAN-23, ADR-0007): checks Python and npm dependencies, models.lock and
imports in src/ against licenses/allowlist.yaml and licenses/denylist.yaml.

Run offline with ``python -m dronescan.common.licenses [repo_root]`` (``make license-gate``,
part of ``make check`` and CI). Exit 1 on any violation.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from email.message import Message
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import yaml

Kind = Literal["python", "npm", "model"]
Scope = Literal["runtime", "dev"]
Severity = Literal["error", "warning"]

ROOT_PROJECT = "drone-scan"
MODEL_FIELDS = ("name", "version", "sha256", "license", "source")

# Trove classifiers and free-text License fields that map unambiguously to SPDX.
_ALIASES: dict[str, str] = {
    "mit license": "MIT",
    "mit": "MIT",
    "apache software license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "isc license": "ISC",
    "isc license (iscl)": "ISC",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "python software foundation license": "PSF-2.0",
}
_TOKEN = re.compile(r"\(|\)|\bOR\b|\bAND\b|\bWITH\b|[^\s()]+")


def normalize_name(name: str) -> str:
    """PEP 503 normalisation (also fine for npm names)."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Policy:
    allowed: frozenset[str]
    data_allowed: frozenset[str]
    copyleft: frozenset[str]
    patterns: tuple[re.Pattern[str], ...]
    denied_packages: Mapping[str, str]
    denied_models: tuple[tuple[re.Pattern[str], str], ...]
    container_only: frozenset[str]
    overrides: Mapping[str, str]


@dataclass(frozen=True)
class Component:
    kind: Kind
    name: str
    version: str
    license: str | None
    scope: Scope


@dataclass(frozen=True)
class Finding:
    severity: Severity
    subject: str
    message: str


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: YAML-Zuordnung erwartet")
    return data


def load_policy(licenses_dir: Path) -> Policy:
    allow = _read_yaml(licenses_dir / "allowlist.yaml")
    deny = _read_yaml(licenses_dir / "denylist.yaml")
    return Policy(
        allowed=frozenset(allow.get("licenses", [])),
        data_allowed=frozenset(allow.get("data_licenses", [])),
        copyleft=frozenset(deny.get("copyleft", [])),
        patterns=tuple(re.compile(p, re.IGNORECASE) for p in deny.get("license_patterns", [])),
        denied_packages={normalize_name(e["name"]): e["reason"] for e in deny.get("packages", [])},
        denied_models=tuple(
            (re.compile(e["pattern"], re.IGNORECASE), e["reason"]) for e in deny.get("models", [])
        ),
        container_only=frozenset(deny.get("container_only_modules", [])),
        overrides={
            normalize_name(name): entry["license"]
            for name, entry in (allow.get("overrides") or {}).items()
        },
    )


def _alternatives(expression: str) -> list[list[str]]:
    """Split an SPDX expression into OR-alternatives of AND-ed license ids.

    Parentheses are flattened; this is sufficient for the expressions found in
    package metadata ("A OR B", "A AND B", "(A OR B)").
    """
    tokens = [t for t in _TOKEN.findall(expression) if t not in "()"]
    alternatives: list[list[str]] = [[]]
    skip_next = False
    for token in tokens:
        if skip_next:  # exception id after WITH
            skip_next = False
        elif token == "OR":
            alternatives.append([])
        elif token == "AND":
            continue
        elif token == "WITH":
            skip_next = True
        else:
            alternatives[-1].append(token)
    return [alt for alt in alternatives if alt]


def _label(c: Component) -> str:
    return f"{c.kind}:{c.name}=={c.version}"


def check_component(c: Component, policy: Policy) -> list[Finding]:
    """Evaluate one dependency or model against the policy."""
    subject = _label(c)
    name = normalize_name(c.name)
    if c.kind != "model" and name in policy.denied_packages:
        return [Finding("error", subject, f"Verboten (Denylist): {policy.denied_packages[name]}.")]
    if c.kind == "model":
        for pattern, reason in policy.denied_models:
            if pattern.search(c.name):
                return [Finding("error", subject, f"Verbotenes Modell (Denylist): {reason}.")]
    license_ = (c.license or "").strip()
    if any(p.search(license_) for p in policy.patterns):
        return [
            Finding(
                "error",
                subject,
                f"Nutzungsbeschränkte Lizenz '{license_}' (non-commercial/research-only) ist "
                "verboten. Abhilfe: Alternative mit MIT/BSD/Apache-2.0 wählen.",
            )
        ]
    alternatives = _alternatives(license_) if license_ else []
    if not alternatives:
        return [
            Finding(
                "error",
                subject,
                "Lizenz unklar. Abhilfe: Lizenzdatei prüfen und in licenses/allowlist.yaml "
                "unter 'overrides' mit Quelle eintragen (PO-Freigabe).",
            )
        ]
    allowed = policy.allowed | (policy.data_allowed if c.kind == "model" else frozenset())
    if any(all(lic in allowed for lic in alt) for alt in alternatives):
        return []
    if any(all(lic in allowed | policy.copyleft for lic in alt) for alt in alternatives):
        if c.scope == "dev" and c.kind == "python":
            return [
                Finding(
                    "warning",
                    subject,
                    f"Copyleft-Lizenz '{license_}' nur als Entwicklungswerkzeug (separater "
                    "Prozess) – nicht ausliefern.",
                )
            ]
        return [
            Finding(
                "error",
                subject,
                f"Copyleft-Lizenz '{license_}' ist nur als separater Prozess bzw. Container "
                "erlaubt, nicht als Laufzeit-Abhängigkeit (ADR-0007). Abhilfe: permissive "
                "Alternative wählen oder als eigenes Werkzeug/Container anbinden.",
            )
        ]
    return [
        Finding(
            "error",
            subject,
            f"Lizenz '{license_}' steht nicht in licenses/allowlist.yaml. Abhilfe: Lizenz "
            "prüfen lassen (Subagent license-auditor) und vom PO freigeben lassen.",
        )
    ]


def resolve_python_license(meta: metadata.PackageMetadata | Message, policy: Policy) -> str | None:
    """Override > License-Expression > License field (SPDX/alias) > classifiers."""
    name = normalize_name(meta.get("Name", ""))
    if name in policy.overrides:
        return policy.overrides[name]
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    field = (meta.get("License") or "").strip()
    known = policy.allowed | policy.copyleft | policy.data_allowed
    if field in known:
        return field
    if field.lower() in _ALIASES:
        return _ALIASES[field.lower()]
    if any(p.search(field) for p in policy.patterns):
        return field  # keep the restricting text so the gate reports it
    classifiers = {
        _ALIASES.get(c.split("::")[-1].strip().lower())
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    } - {None}
    if classifiers:
        return " OR ".join(sorted(c for c in classifiers if c))
    return None


def runtime_packages(lock_path: Path, root: str = ROOT_PROJECT) -> set[str]:
    """Normalised names of all transitive runtime dependencies of ``root`` in uv.lock."""
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    deps: dict[str, list[str]] = {
        normalize_name(p["name"]): [normalize_name(d["name"]) for d in p.get("dependencies", [])]
        for p in lock.get("package", [])
    }
    seen: set[str] = set()
    stack = list(deps.get(normalize_name(root), []))
    while stack:
        name = stack.pop()
        if name not in seen:
            seen.add(name)
            stack.extend(deps.get(name, []))
    return seen


def python_components(
    policy: Policy, lock_path: Path, dists: Iterable[metadata.Distribution] | None = None
) -> list[Component]:
    runtime = runtime_packages(lock_path) if lock_path.is_file() else set()
    components = []
    for dist in dists if dists is not None else metadata.distributions():
        meta = dist.metadata
        name = normalize_name(meta["Name"])
        if name == ROOT_PROJECT:
            continue
        scope: Scope = "runtime" if name in runtime else "dev"
        components.append(
            Component("python", name, meta["Version"], resolve_python_license(meta, policy), scope)
        )
    return components


def npm_components(viewer_dir: Path) -> list[Component]:
    """All packages in viewer/node_modules (treated as shipped, i.e. runtime)."""
    modules = viewer_dir / "node_modules"
    if not modules.is_dir():
        return []
    manifests = list(modules.glob("*/package.json")) + list(modules.glob("@*/*/package.json"))
    components = []
    for path in sorted(manifests):
        data = json.loads(path.read_text(encoding="utf-8"))
        lic = data.get("license")
        if isinstance(lic, dict):
            lic = lic.get("type")
        name = str(data.get("name") or path.parent.name)
        components.append(
            Component("npm", name, str(data.get("version", "?")), lic or None, "runtime")
        )
    return components


def model_components(models_lock: Path) -> tuple[list[Component], list[Finding]]:
    if not models_lock.is_file():
        return [], [Finding("error", "models.lock", "models.lock fehlt.")]
    data = yaml.safe_load(models_lock.read_text(encoding="utf-8")) or {}
    components: list[Component] = []
    findings: list[Finding] = []
    for index, entry in enumerate(data.get("models") or []):
        missing = [f for f in MODEL_FIELDS if not entry.get(f)]
        subject = f"model:{entry.get('name', f'#{index}')}"
        if missing:
            findings.append(
                Finding("error", subject, f"Eintrag unvollständig, es fehlt: {', '.join(missing)}.")
            )
            continue
        components.append(
            Component("model", entry["name"], str(entry["version"]), entry["license"], "runtime")
        )
    return components, findings


def scan_imports(src_dir: Path, policy: Policy, extra_modules: set[str]) -> list[Finding]:
    """Report imports in src/ of container-only or copyleft modules (AC 3)."""
    forbidden = policy.container_only | extra_modules
    findings = []
    for path in sorted(src_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in forbidden:
                    findings.append(
                        Finding(
                            "error",
                            f"{path.relative_to(src_dir).as_posix()}:{node.lineno}",
                            f"Import von '{top}' (Copyleft) aus {path.name} ist verboten. "
                            "Abhilfe: Werkzeug als separaten Prozess/Container aufrufen.",
                        )
                    )
    return findings


def _copyleft_modules(components: list[Component], policy: Policy) -> set[str]:
    """Top-level modules provided by installed distributions with copyleft-only licenses."""
    copyleft = {
        c.name
        for c in components
        if c.kind == "python"
        and c.license
        and all(any(lic in policy.copyleft for lic in alt) for alt in _alternatives(c.license))
    }
    mapping = metadata.packages_distributions()
    return {
        mod for mod, dists in mapping.items() if any(normalize_name(d) in copyleft for d in dists)
    }


def run_gate(root: Path) -> list[Finding]:
    policy = load_policy(root / "licenses")
    python = python_components(policy, root / "uv.lock")
    models, findings = model_components(root / "models.lock")
    components = python + npm_components(root / "viewer") + models
    for component in components:
        findings.extend(check_component(component, policy))
    if (root / "src").is_dir():
        findings.extend(scan_imports(root / "src", policy, _copyleft_modules(python, policy)))
    return findings


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else Path.cwd()
    findings = run_gate(root)
    errors = [f for f in findings if f.severity == "error"]
    for f in findings:
        marker = "FEHLER " if f.severity == "error" else "Warnung"
        print(f"{marker} {f.subject}: {f.message}")
    status = "fehlgeschlagen" if errors else "bestanden"
    print(f"Lizenz-Gate {status}: {len(errors)} Fehler, {len(findings) - len(errors)} Warnungen.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
