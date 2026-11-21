# ADR-0009: Technologiestack

- **Status:** angenommen
- **Datum:** 2026-11-20

## Kontext
Ein Einzelgründer entwickelt mit Claude Code; der Stack soll verbreitet, gut dokumentiert und containerisierbar sein.

## Entscheidung
Python 3.12 + uv, FastAPI, Pydantic, Typer, pytest, ruff, mypy; TypeScript + Vite, three.js-basierter Splat-Renderer, Playwright; Docker Compose mit NVIDIA Container Toolkit; Jira (Projekt „DRN“ (Schlüssel KAN)) für Backlog und Sprints.

## Konsequenzen
+ Gute Unterstützung durch Claude Code, große Ökosysteme
- Zwei Sprachen (Python/TS) im Repo
