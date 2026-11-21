# drone-scan – identical commands for the PO and Claude Code (CLAUDE.md, "Stack & Befehle").
# Recipes must be indented with tabs.

UV ?= uv
VRAM_LIMIT_GB ?= 5.5

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check test-gpu smoke bench viewer-dev

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create Python 3.12 env (uv.lock), install pre-commit hooks
	$(UV) sync --locked
	$(UV) run pre-commit install
	@if [ -f viewer/package.json ]; then cd viewer && pnpm install; else echo "viewer/ fehlt noch – pnpm install übersprungen"; fi
	@if [ -f docker-compose.yml ] || [ -f compose.yaml ]; then docker compose build; else echo "Docker-Compose-Datei fehlt noch (KAN-18) – docker build übersprungen"; fi

lint: ## ruff lint + format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format: ## Apply ruff fixes and formatting
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck: ## mypy --strict
	$(UV) run mypy

test: ## CPU tests (no gpu, no bench)
	$(UV) run pytest -m "not gpu and not bench"

# The license gate (CLAUDE.md) is added to `check` with KAN-23 (US-008).
check: lint typecheck test ## Run before every PR: lint, types, CPU tests

test-gpu: ## GPU tests with VRAM limit
	DRONESCAN_VRAM_LIMIT_GB=$(VRAM_LIMIT_GB) $(UV) run pytest -m gpu

smoke: ## Pipeline on data/samples/kurz.mp4
	@echo "make smoke: noch nicht verfügbar – kommt mit den Pipeline-Stufen (M1)." >&2; exit 1

bench: ## Benchmarks on reference rooms
	@echo "make bench: noch nicht verfügbar – kommt mit den Referenzräumen (M1)." >&2; exit 1

viewer-dev: ## Vite dev server for viewer/
	@echo "make viewer-dev: noch nicht verfügbar – kommt mit dem Viewer (M1)." >&2; exit 1
