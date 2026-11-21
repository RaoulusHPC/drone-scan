"""License gate (KAN-23, ADR-0007, ADR-0012): checks Python and npm dependencies, models.lock
and imports in src/ against licenses/allowlist.yaml and licenses/denylist.yaml.

Run offline with ``python -m dronescan.common.licenses [repo_root]`` (``make license-gate``,
part of ``make check`` and CI). Exit 1 on any violation. Non-commercial/research-only
components are allowed (ADR-0012) but reported with severity ``nc`` so they can be found
completely before any redistribution or commercial use.

The Python check reads the *installed* environment (CI: ``uv sync --locked``) and uses
``uv.lock`` for the runtime/dev scope; locked packages that are not installed are reported
as warnings, so they are not silently skipped.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from email.message import Message
from importlib import metadata
from itertools import product
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from dronescan.common.models_lock import load_models_lock

Kind = Literal["python", "npm", "model"]
Scope = Literal["runtime", "dev"]
Severity = Literal["error", "warning", "nc"]

ROOT_PROJECT = "drone-scan"
_MAX_ID_LENGTH = 80  # longer License fields are free text, not an SPDX id/expression

# Trove classifiers and short License fields that map unambiguously to SPDX.
# "BSD License" is deliberately missing: it is ambiguous (2- or 3-clause) and needs an override.
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
_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def normalize_name(name: str) -> str:
    """PEP 503 normalisation (also fine for npm names)."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Policy:
    allowed: frozenset[str]
    data_allowed: frozenset[str]
    copyleft: frozenset[str]
    nc_patterns: tuple[re.Pattern[str], ...]
    nc_packages: Mapping[str, str]
    nc_models: tuple[tuple[re.Pattern[str], str], ...]
    nc_image_components: tuple[tuple[str, str], ...]  # (name, reason), not visible to the gate
    denied_packages: Mapping[str, str]
    denied_models: tuple[tuple[re.Pattern[str], str], ...]
    container_only: frozenset[str]
    overrides: Mapping[str, tuple[str, str | None]]  # name -> (license, pinned version)


class Component(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Kind
    name: str
    version: str
    license: str | None
    scope: Scope
    source: str = ""


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    severity: Severity
    subject: str
    message: str


def _error(subject: str, message: str) -> Finding:
    return Finding(severity="error", subject=subject, message=message)


def _warning(subject: str, message: str) -> Finding:
    return Finding(severity="warning", subject=subject, message=message)


def _nc(subject: str, message: str) -> Finding:
    return Finding(severity="nc", subject=subject, message=message)


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
        nc_patterns=tuple(
            re.compile(p, re.IGNORECASE) for p in deny.get("nc_license_patterns") or []
        ),
        nc_packages={normalize_name(e["name"]): e["reason"] for e in deny.get("nc_packages") or []},
        nc_models=tuple(
            (re.compile(e["pattern"], re.IGNORECASE), e["reason"])
            for e in deny.get("nc_models") or []
        ),
        nc_image_components=tuple(
            (e["name"], e["reason"]) for e in deny.get("nc_image_components") or []
        ),
        denied_packages={
            normalize_name(e["name"]): e["reason"] for e in deny.get("packages") or []
        },
        denied_models=tuple(
            (re.compile(e["pattern"], re.IGNORECASE), e["reason"]) for e in deny.get("models") or []
        ),
        container_only=frozenset(deny.get("container_only_modules", [])),
        overrides={
            normalize_name(name): (entry["license"], entry.get("version"))
            for name, entry in (allow.get("overrides") or {}).items()
        },
    )


class _Parser:
    """SPDX expression → disjunctive normal form (list of AND-ed license id sets).

    Grammar: expr := term (OR term)* ; term := atom (AND atom)* ;
    atom := '(' expr ')' | ID [WITH ID]. Raises ValueError for malformed input.
    """

    def __init__(self, expression: str) -> None:
        self.tokens = _TOKEN.findall(expression)
        self.pos = 0

    def parse(self) -> list[frozenset[str]]:
        result = self._expr()
        if self.pos != len(self.tokens):
            raise ValueError("trailing tokens")
        return result

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end")
        self.pos += 1
        return token

    def _expr(self) -> list[frozenset[str]]:
        result = self._term()
        while self._peek() == "OR":
            self._take()
            result = result + self._term()
        return result

    def _term(self) -> list[frozenset[str]]:
        result = self._atom()
        while self._peek() == "AND":
            self._take()
            right = self._atom()
            result = [a | b for a, b in product(result, right)]
        return result

    def _atom(self) -> list[frozenset[str]]:
        token = self._take()
        if token == "(":
            inner = self._expr()
            if self._take() != ")":
                raise ValueError("missing ')'")
            return inner
        if token in {")", "OR", "AND", "WITH"}:
            raise ValueError(f"unexpected {token}")
        if self._peek() == "WITH":  # license exception, e.g. GPL-2.0-only WITH Classpath
            self._take()
            self._take()
        return [frozenset({token})]


def alternatives(expression: str) -> list[frozenset[str]] | None:
    """Alternatives of an SPDX expression, or None if it cannot be parsed."""
    try:
        return _Parser(expression).parse()
    except ValueError:
        return None


def _label(c: Component) -> str:
    return f"{c.kind}:{c.name}=={c.version}"


_NC_HINT = "Erlaubt (ADR-0012), aber vor Weitergabe oder kommerzieller Nutzung neu prüfen."


def _nc_findings(c: Component, policy: Policy) -> list[Finding]:
    """Known non-commercial/research-only packages and models (ADR-0012: allowed, but marked)."""
    subject = _label(c)
    if c.kind == "model":
        return [
            _nc(subject, f"{reason}. {_NC_HINT}")
            for pattern, reason in policy.nc_models
            if pattern.search(c.name) or pattern.search(c.source)
        ][:1]
    name = normalize_name(c.name)
    if name in policy.nc_packages:
        return [_nc(subject, f"{policy.nc_packages[name]}. {_NC_HINT}")]
    return []


def check_component(c: Component, policy: Policy) -> list[Finding]:
    """Evaluate one dependency or model against the policy."""
    subject = _label(c)
    name = normalize_name(c.name)
    if c.kind != "model" and name in policy.denied_packages:
        return [
            _error(
                subject,
                f"Verboten (Denylist): {policy.denied_packages[name]}. "
                "Abhilfe: Abhängigkeit entfernen und eine zulässige Alternative wählen.",
            )
        ]
    if c.kind == "model":
        for pattern, reason in policy.denied_models:
            if pattern.search(c.name) or pattern.search(c.source):
                return [
                    _error(
                        subject,
                        f"Verbotenes Modell (Denylist): {reason}. "
                        "Abhilfe: Eintrag aus models.lock entfernen.",
                    )
                ]
    marked = _nc_findings(c, policy)
    license_ = (c.license or "").strip()
    if not any(p.search(license_) for p in policy.nc_patterns):
        return marked + _check_license(c, license_, policy)
    marked = marked or [
        _nc(
            subject,
            f"Nutzungsbeschränkte Lizenz '{license_}' (non-commercial/research-only). {_NC_HINT}",
        )
    ]
    alts = alternatives(license_)
    if not alts:  # free text such as "Non-Commercial Use Only": NC is all we can tell
        return marked
    # SPDX expression: NC ids count as allowed, the rest (e.g. copyleft) is still checked.
    nc_ids = frozenset(
        atom for alt in alts for atom in alt if any(p.search(atom) for p in policy.nc_patterns)
    )
    return marked + _check_license(c, license_, policy, extra_allowed=nc_ids)


def _check_license(
    c: Component, license_: str, policy: Policy, extra_allowed: frozenset[str] = frozenset()
) -> list[Finding]:
    subject = _label(c)
    alts = alternatives(license_) if license_ else None
    if not alts:
        return [
            _error(
                subject,
                f"Lizenz unklar ('{license_ or 'keine Angabe'}'). Abhilfe: Lizenzdatei prüfen "
                "und in licenses/allowlist.yaml unter 'overrides' mit Quelle eintragen, bei "
                "NC-Lizenzen mit deren Kennung, z. B. 'LicenseRef-NC' (PO-Freigabe).",
            )
        ]
    allowed = (
        policy.allowed | extra_allowed | (policy.data_allowed if c.kind == "model" else frozenset())
    )
    if any(alt <= allowed for alt in alts):
        return []
    if any(alt <= allowed | policy.copyleft for alt in alts):
        if c.scope == "dev" and c.kind == "python":
            return [
                _warning(
                    subject,
                    f"Copyleft-Lizenz '{license_}' ist nur als Entwicklungswerkzeug (separater "
                    "Prozess) zulässig. Abhilfe: nie aus src/ importieren und nicht ausliefern; "
                    "besser eine permissive Alternative wählen.",
                )
            ]
        return [
            _error(
                subject,
                f"Copyleft-Lizenz '{license_}' ist nur als separater Prozess bzw. Container "
                "erlaubt, nicht als Laufzeit-Abhängigkeit (ADR-0007). Abhilfe: permissive "
                "Alternative wählen oder als eigenes Werkzeug/Container anbinden.",
            )
        ]
    return [
        _error(
            subject,
            f"Lizenz '{license_}' steht nicht in licenses/allowlist.yaml. Abhilfe: Lizenz "
            "prüfen lassen (Subagent license-auditor) und vom PO freigeben lassen.",
        )
    ]


def resolve_python_license(meta: metadata.PackageMetadata | Message, policy: Policy) -> str | None:
    """Override > License-Expression > short License field > classifiers.

    Long License fields are free text (often the full license) and are only used when
    there is nothing better; NC patterns are applied to short, id-like fields only to
    avoid false positives such as "commercial or non-commercial use is permitted".
    """
    name = normalize_name(meta.get("Name") or "")
    if name in policy.overrides:
        license_, version = policy.overrides[name]
        if version is None or version == meta.get("Version"):
            return license_
        # Override is pinned to another version: fall through, a re-check is needed.
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    field = (meta.get("License") or "").strip()
    short = 0 < len(field) <= _MAX_ID_LENGTH and "\n" not in field
    if short:
        if alternatives(field) and all(
            alt <= policy.allowed | policy.copyleft | policy.data_allowed
            for alt in alternatives(field) or []
        ):
            return field
        if field.lower() in _ALIASES:
            return _ALIASES[field.lower()]
        if any(p.search(field) for p in policy.nc_patterns):
            return field  # keep the restricting text so the gate reports it
    classifiers = sorted(
        {
            _ALIASES[key]
            for c in meta.get_all("Classifier") or []
            if c.startswith("License ::") and (key := c.split("::")[-1].strip().lower()) in _ALIASES
        }
    )
    if classifiers:
        return " OR ".join(classifiers)
    return None


def _lock_graph(lock: dict[str, Any]) -> dict[str, list[tuple[str, bool]]]:
    """name -> [(dependency, has_marker)] incl. optional (extra) dependencies."""
    graph: dict[str, list[tuple[str, bool]]] = {}
    for package in lock.get("package", []):
        edges = [
            (normalize_name(d["name"]), "marker" in d) for d in package.get("dependencies", [])
        ]
        for extra in (package.get("optional-dependencies") or {}).values():
            edges += [(normalize_name(d["name"]), "marker" in d) for d in extra]
        graph[normalize_name(package["name"])] = edges
    return graph


def _closure(
    graph: dict[str, list[tuple[str, bool]]], start: list[str], unmarked: bool
) -> set[str]:
    seen: set[str] = set()
    stack = list(start)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(dep for dep, marked in graph.get(name, []) if not (unmarked and marked))
    return seen


def runtime_packages(lock_path: Path, root: str = ROOT_PROJECT) -> set[str]:
    """Normalised names of all transitive runtime dependencies (incl. extras) of ``root``."""
    graph = _lock_graph(tomllib.loads(lock_path.read_text(encoding="utf-8")))
    return _closure(graph, [dep for dep, _ in graph.get(normalize_name(root), [])], False)


def _expected_installed(lock_path: Path, root: str = ROOT_PROJECT) -> set[str]:
    """Locked runtime + dev packages without platform markers (should be installed in CI)."""
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    graph = _lock_graph(lock)
    root_entry: dict[str, Any] = next(
        (p for p in lock.get("package", []) if normalize_name(p["name"]) == normalize_name(root)),
        {},
    )
    start = [dep for dep, marked in graph.get(normalize_name(root), []) if not marked]
    for group in (root_entry.get("dev-dependencies") or {}).values():
        start += [normalize_name(d["name"]) for d in group if "marker" not in d]
    return _closure(graph, start, True)


def python_components(
    policy: Policy, lock_path: Path, dists: Iterable[metadata.Distribution] | None = None
) -> tuple[list[Component], list[Finding]]:
    if not lock_path.is_file():
        return [], [
            _error(
                "uv.lock",
                "uv.lock fehlt, ohne Lockfile ist der Laufzeit-Scope unbekannt. "
                "Abhilfe: 'uv lock' ausführen und uv.lock einchecken.",
            )
        ]
    runtime = runtime_packages(lock_path)
    components: list[Component] = []
    findings: list[Finding] = []
    installed: set[str] = set()
    for dist in dists if dists is not None else metadata.distributions():
        meta = dist.metadata
        raw_name = meta.get("Name")
        if not raw_name:
            findings.append(
                _warning(
                    "python:?",
                    "Installiertes Paket ohne Namen in den Metadaten übersprungen. "
                    "Abhilfe: Umgebung mit 'uv sync --locked' neu aufbauen.",
                )
            )
            continue
        name = normalize_name(raw_name)
        installed.add(name)
        if name == ROOT_PROJECT:
            continue
        scope: Scope = "runtime" if name in runtime else "dev"
        components.append(
            Component(
                kind="python",
                name=name,
                version=meta.get("Version") or "?",
                license=resolve_python_license(meta, policy),
                scope=scope,
            )
        )
    for name in sorted(_expected_installed(lock_path) - installed):
        findings.append(
            _warning(
                f"python:{name}",
                "In uv.lock, aber nicht installiert – Lizenz nicht geprüft. "
                "Abhilfe: 'uv sync --locked --all-extras --all-groups' vor dem Gate ausführen.",
            )
        )
    return components, findings


def _npm_license(data: dict[str, Any]) -> str | None:
    lic = data.get("license")
    if isinstance(lic, dict):
        lic = lic.get("type")
    if not lic and isinstance(data.get("licenses"), list):  # legacy field
        types = [
            str(e.get("type")) for e in data["licenses"] if isinstance(e, dict) and e.get("type")
        ]
        lic = " OR ".join(types) if types else None
    return str(lic) if lic else None


def npm_components(viewer_dir: Path) -> list[Component]:
    """All packages in viewer/node_modules incl. pnpm's indirect dependencies (runtime)."""
    modules = viewer_dir / "node_modules"
    if not modules.is_dir():
        return []
    patterns = [
        "*/package.json",
        "@*/*/package.json",
        ".pnpm/*/node_modules/*/package.json",
        ".pnpm/*/node_modules/@*/*/package.json",
    ]
    seen: dict[tuple[str, str], Component] = {}
    for pattern in patterns:
        for path in sorted(modules.glob(pattern)):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data.get("name"):
                continue
            key = (str(data["name"]), str(data.get("version", "?")))
            seen.setdefault(
                key,
                Component(
                    kind="npm",
                    name=key[0],
                    version=key[1],
                    license=_npm_license(data),
                    scope="runtime",
                ),
            )
    return list(seen.values())


def _npm_findings(viewer_dir: Path) -> list[Finding]:
    if (viewer_dir / "package.json").is_file() and not (viewer_dir / "node_modules").is_dir():
        return [
            _error(
                "npm",
                "viewer/package.json vorhanden, aber keine node_modules – npm-Lizenzen "
                "ungeprüft. Abhilfe: 'pnpm install --frozen-lockfile' im Ordner viewer/ "
                "vor dem Gate ausführen.",
            )
        ]
    return []


def model_components(models_lock: Path) -> tuple[list[Component], list[Finding]]:
    if not models_lock.is_file():
        return [], [
            _error(
                "models.lock",
                "models.lock fehlt. Abhilfe: Datei mit 'models: []' anlegen "
                "(Modelle kommen per 'drone-scan models pull').",
            )
        ]
    try:
        lock = load_models_lock(models_lock)
    except (ValueError, ValidationError, yaml.YAMLError) as exc:
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return [], [
            _error(
                "models.lock",
                f"Eintrag ungültig oder unvollständig ({detail}). Abhilfe: jedes Modell braucht "
                "name, version, sha256 (64 Hex-Zeichen), license und source.",
            )
        ]
    return [
        Component(
            kind="model",
            name=m.name,
            version=m.version,
            license=m.license,
            scope="runtime",
            source=m.source,
        )
        for m in lock.models
    ], []


def _imported_modules(tree: ast.AST) -> Iterable[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, node.lineno
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            )
        ):
            yield node.args[0].value, node.lineno


def scan_imports(src_dir: Path, policy: Policy, extra_modules: set[str]) -> list[Finding]:
    """Report imports in src/ of container-only or copyleft modules (AC 3)."""
    forbidden = policy.container_only | extra_modules
    findings = []
    for path in sorted(src_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for module, lineno in _imported_modules(tree):
            top = module.split(".")[0]
            if top in forbidden:
                findings.append(
                    _error(
                        f"{path.relative_to(src_dir).as_posix()}:{lineno}",
                        f"Import von '{top}' (Copyleft) aus {path.name} ist verboten. "
                        "Abhilfe: Werkzeug als separaten Prozess/Container aufrufen.",
                    )
                )
    return findings


# Model weight formats that must never be committed (CLAUDE.md rule 5, ADR-0012: public repo).
_WEIGHT_SUFFIXES = frozenset(
    {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".tflite", ".engine"}
)


def _tracked_files(root: Path) -> list[str] | None:
    """Files tracked by git (incl. submodule paths), or None outside a git work tree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.splitlines() if result.returncode == 0 else None


def scan_repository(root: Path, policy: Policy, tracked: list[str] | None = None) -> list[Finding]:
    """No NC code or weights in the public repository (ADR-0012): tracked paths and submodule
    URLs must not be named after a known NC project, and no model weights may be committed."""
    files = _tracked_files(root) if tracked is None else tracked
    if files is None:
        return []
    nc_names = {name: reason for name, reason in policy.nc_packages.items()} | {
        normalize_name(name.rsplit("/", 1)[-1]): reason
        for name, reason in policy.nc_image_components
    }
    findings = []
    for path in files:
        parts = {normalize_name(Path(part).stem) for part in Path(path).parts}
        parts |= {normalize_name(part) for part in Path(path).parts}
        hits = sorted(parts & nc_names.keys())
        if hits:
            findings.append(
                _error(
                    f"repo:{path}",
                    f"NC-Komponente '{hits[0]}' im öffentlichen Repository ({nc_names[hits[0]]}). "
                    "Abhilfe: nicht vendoren/kopieren; nur lokal aus der Upstream-Quelle bauen "
                    "(Dockerfile) bzw. per 'drone-scan models pull' laden (ADR-0012).",
                )
            )
        elif Path(path).suffix.lower() in _WEIGHT_SUFFIXES:
            findings.append(
                _error(
                    f"repo:{path}",
                    "Modellgewichte gehören nicht ins Repository. Abhilfe: Datei aus Git "
                    "entfernen, in models.lock eintragen und per 'drone-scan models pull' laden.",
                )
            )
    gitmodules = root / ".gitmodules"
    if gitmodules.is_file():
        for line in gitmodules.read_text(encoding="utf-8").splitlines():
            key, _, url = line.partition("=")
            if key.strip() != "url":
                continue
            repo_name = normalize_name(
                url.strip().rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
            )
            if repo_name in nc_names:
                findings.append(
                    _error(
                        "repo:.gitmodules",
                        f"Submodule mit NC-Komponente '{repo_name}' ({nc_names[repo_name]}). "
                        "Abhilfe: Submodule entfernen, nur lokal bauen (ADR-0012).",
                    )
                )
    return findings


def _copyleft_modules(components: list[Component], policy: Policy) -> set[str]:
    """Top-level modules provided by installed distributions without a non-copyleft option."""
    copyleft = set()
    for c in components:
        alts = alternatives(c.license or "") if c.kind == "python" else None
        if (
            alts
            and not any(alt <= policy.allowed for alt in alts)
            and any(alt & policy.copyleft for alt in alts)
        ):
            copyleft.add(c.name)
    mapping = metadata.packages_distributions()
    return {
        mod for mod, dists in mapping.items() if any(normalize_name(d) in copyleft for d in dists)
    }


def run_gate(root: Path) -> list[Finding]:
    policy = load_policy(root / "licenses")
    python, findings = python_components(policy, root / "uv.lock")
    models, model_findings = model_components(root / "models.lock")
    findings += model_findings + _npm_findings(root / "viewer")
    for component in python + npm_components(root / "viewer") + models:
        findings.extend(check_component(component, policy))
    if (root / "src").is_dir():
        findings.extend(scan_imports(root / "src", policy, _copyleft_modules(python, policy)))
    findings.extend(
        _nc(f"image:{name}", f"{reason}. {_NC_HINT}") for name, reason in policy.nc_image_components
    )
    findings.extend(scan_repository(root, policy))
    return findings


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else Path.cwd()
    findings = run_gate(root)
    markers = {"error": "FEHLER ", "warning": "Warnung", "nc": "NC     "}
    counts = {severity: 0 for severity in markers}
    for f in findings:
        counts[f.severity] += 1
        print(f"{markers[f.severity]} {f.subject}: {f.message}")
    status = "fehlgeschlagen" if counts["error"] else "bestanden"
    print(
        f"Lizenz-Gate {status}: {counts['error']} Fehler, {counts['warning']} Warnungen, "
        f"{counts['nc']} NC-Komponenten (vor Weitergabe neu prüfen, ADR-0012)."
    )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
