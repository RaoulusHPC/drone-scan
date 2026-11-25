#!/usr/bin/env python3
"""Validate backlog/backlog.yaml and export it.

Outputs:
  docs/user-stories.md        human-readable backlog (German)
  backlog/jira-import.csv     CSV for Jira's external-system importer (fallback if MCP is unavailable)

Usage: python3 tools/backlog_export.py [--jira-map backlog/jira-map.json]
The optional jira map (local id -> Jira key, e.g. {"US-001": "KAN-13"}) adds Jira keys to the markdown.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRIO_JIRA = {"Must": "Highest", "Should": "High", "Could": "Medium", "Won't": "Low"}
MS_NAMES = {
    "m0": "M0 Fundament & Spikes",
    "m1": "M1 MVP „Video → Begehung“",
    "m2": "M2 Gegenstandssuche",
    "m3": "M3 Drohnen-Erfassung",
    "m4": "M4 Präsentationsreife",
    "m5": "M5 Assistierter Flug",
}
MS_ORDER = list(MS_NAMES)


def load() -> dict:
    return yaml.safe_load((ROOT / "backlog/backlog.yaml").read_text(encoding="utf-8"))


def validate(d: dict) -> list[str]:
    errors: list[str] = []
    stories = {s["id"]: s for s in d["stories"]}
    epics = {e["id"] for e in d["epics"]}
    personas = {p["id"] for p in d["personas"]}
    if len(stories) != len(d["stories"]):
        errors.append("duplicate story ids")
    for s in d["stories"]:
        sid = s["id"]
        if s["epic"] not in epics:
            errors.append(f"{sid}: unknown epic {s['epic']}")
        if s["as"] not in personas:
            errors.append(f"{sid}: unknown persona {s['as']}")
        if s["sp"] not in (1, 2, 3, 5, 8):
            errors.append(f"{sid}: story points must be Fibonacci <= 8")
        if s["prio"] not in PRIO_JIRA:
            errors.append(f"{sid}: bad prio {s['prio']}")
        if s["ms"] not in MS_NAMES:
            errors.append(f"{sid}: bad milestone {s['ms']}")
        if len(s["ac"]) < 2:
            errors.append(f"{sid}: needs >= 2 acceptance criteria")
        for dep in s["deps"]:
            if dep not in stories:
                errors.append(f"{sid}: unknown dependency {dep}")
            elif MS_ORDER.index(stories[dep]["ms"]) > MS_ORDER.index(s["ms"]):
                errors.append(f"{sid} ({s['ms']}) depends on later milestone {dep} ({stories[dep]['ms']})")
    # cycle check
    state: dict[str, int] = {}

    def visit(n: str, path: list[str]) -> None:
        if state.get(n) == 1:
            errors.append("cycle: " + " -> ".join(path + [n]))
            return
        if state.get(n) == 2:
            return
        state[n] = 1
        for m in stories[n]["deps"]:
            if m in stories:
                visit(m, path + [n])
        state[n] = 2

    for sid in stories:
        visit(sid, [])
    return errors


def persona_name(d: dict, pid: str) -> str:
    return next(p["name"] for p in d["personas"] if p["id"] == pid)


def story_text(d: dict, s: dict) -> str:
    return f"Als **{persona_name(d, s['as'])}** möchte ich {s['want']}, damit {s['so']}."


def export_markdown(d: dict, jira: dict[str, str]) -> str:
    key = lambda sid: f"{sid} / {jira[sid]}" if sid in jira else sid  # noqa: E731
    by_epic: dict[str, list[dict]] = defaultdict(list)
    for s in d["stories"]:
        by_epic[s["epic"]].append(s)
    total = sum(s["sp"] for s in d["stories"])
    lines = [
        "# User Stories – drone-scan",
        "",
        "> Generiert aus `backlog/backlog.yaml` mit `tools/backlog_export.py` – **nicht von Hand bearbeiten**.",
        "> Führendes System für Status und Sprints ist **Jira (Projekt „DRN“ (Schlüssel `KAN`))**. Priorität: MoSCoW → Jira (Must=Highest, Should=High, Could=Medium).",
        "",
        f"**Umfang:** {len(d['epics'])} Epics · {len(d['stories'])} Stories/Tasks · {total} Story Points",
        "",
        "## Personas",
        "",
        "| ID | Persona | Beschreibung |",
        "|---|---|---|",
    ]
    lines += [f"| {p['id']} | {p['name']} | {p['desc']} |" for p in d["personas"]]
    lines += ["", "## Übersicht nach Meilenstein", "", "| Meilenstein | Stories | SP | Must | Should | Could |", "|---|---|---|---|---|---|"]
    for ms in MS_ORDER:
        ss = [s for s in d["stories"] if s["ms"] == ms]
        cnt = lambda p: sum(1 for s in ss if s["prio"] == p)  # noqa: E731
        lines.append(f"| {MS_NAMES[ms]} | {len(ss)} | {sum(s['sp'] for s in ss)} | {cnt('Must')} | {cnt('Should')} | {cnt('Could')} |")
    lines += ["", "## Übersicht nach Epic", "", "| Epic | Titel | Stories | SP |", "|---|---|---|---|"]
    for e in d["epics"]:
        ss = by_epic[e["id"]]
        ek = f"{e['id']} / {jira[e['id']]}" if e["id"] in jira else e["id"]
        lines.append(f"| {ek} | {e['title']} | {len(ss)} | {sum(s['sp'] for s in ss)} |")
    for e in d["epics"]:
        ek = f"{e['id']} / {jira[e['id']]}" if e["id"] in jira else e["id"]
        lines += ["", f"## {ek} – {e['title']}", ""]
        for s in by_epic[e["id"]]:
            deps = ", ".join(key(x) for x in s["deps"]) or "–"
            lines += [
                f"### {key(s['id'])}: {s['title']}",
                "",
                f"`{s['type']}` · **{s['prio']}** · {s['sp']} SP · {MS_NAMES[s['ms']]} · Labels: {', '.join(s['labels'])} · Abhängig von: {deps}",
                "",
                story_text(d, s),
                "",
                "**Akzeptanzkriterien**",
                "",
            ]
            lines += [f"- [ ] {a}" for a in s["ac"]]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def jira_description(d: dict, s: dict) -> str:
    ac = "\n".join(f"- {a}" for a in s["ac"])
    deps = ", ".join(s["deps"]) or "–"
    return (
        f"{story_text(d, s)}\n\n## Akzeptanzkriterien\n{ac}\n\n"
        f"## Kontext\n- Meilenstein: {MS_NAMES[s['ms']]}\n- Lokale ID: {s['id']} (backlog/backlog.yaml)\n"
        f"- Abhängig von: {deps}\n- Architektur: docs/architektur.md"
    )


def export_csv(d: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Issue ID", "Parent ID", "Issue Type", "Summary", "Description", "Priority", "Story Points", "Labels", "Labels", "Labels", "Labels"])
        for e in d["epics"]:
            w.writerow([e["id"], "", "Epic", f"{e['id']} {e['title']}", f"Epic {e['id']} – Meilenstein {MS_NAMES[e['ms']]}", "High", "", e["ms"], "epic", "", ""])
        for s in d["stories"]:
            labels = ([s["ms"]] + s["labels"] + ["", "", "", ""])[:4]
            w.writerow([s["id"], s["epic"], s["type"], s["title"], jira_description(d, s), PRIO_JIRA[s["prio"]], s["sp"], *labels])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jira-map", type=Path, default=ROOT / "backlog/jira-map.json")
    args = ap.parse_args()
    d = load()
    errors = validate(d)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    jira = json.loads(args.jira_map.read_text()) if args.jira_map.exists() else {}
    (ROOT / "docs/user-stories.md").write_text(export_markdown(d, jira), encoding="utf-8")
    export_csv(d, ROOT / "backlog/jira-import.csv")
    print(f"ok: {len(d['epics'])} epics, {len(d['stories'])} stories, {sum(s['sp'] for s in d['stories'])} SP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
