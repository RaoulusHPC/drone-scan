# drone-scan – identical commands for the PO and Claude Code (CLAUDE.md, "Stack & Befehle").
# Recipes must be indented with tabs.

UV ?= uv
VRAM_LIMIT_GB ?= 5.0
# GPU tests run on the GTX 1060 only; the RTX 2070 SUPER (index 1) needs the PO's consent (CLAUDE.md).
GPU_DEVICES ?= 0

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test license-gate check test-gpu test-sitl docker-check smoke bench viewer-dev

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create Python 3.12 env (uv.lock), install pre-commit hooks
	$(UV) sync --locked
	$(UV) run pre-commit install
	@if command -v git-lfs >/dev/null 2>&1; then \
		git lfs install --local || echo "Hinweis: 'git lfs install --local' ist fehlgeschlagen (vermutlich existiert bereits ein Git-Hook, z. B. pre-push). Setup wird fortgesetzt; Beispielclips in data/samples/ ggf. manuell prüfen." >&2; \
	else \
		echo "Hinweis: git-lfs ist nicht installiert – Beispielclips in data/samples/ werden ohne LFS gespeichert. Abhilfe: sudo apt install git-lfs, danach make setup erneut ausführen. Setup wird fortgesetzt." >&2; \
	fi
	@if [ -f viewer/package.json ]; then cd viewer && pnpm install; else echo "viewer/ fehlt noch – pnpm install übersprungen"; fi
	@mkdir -p workspace data
	@mkdir -p "$${DRONESCAN_GPU_LOCK_DIR:-$${XDG_STATE_HOME:-$$HOME/.local/state}/drone-scan}"  # GPU lock dir, mounted into the containers (KAN-80)
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

license-gate: ## Check dependency, npm and model licenses (licenses/*.yaml, models.lock)
	$(UV) run python -m dronescan.common.licenses .

check: lint typecheck test license-gate ## Run before every PR: lint, types, CPU tests, licenses

test-gpu: ## GPU tests on the GTX 1060 with VRAM limit (GiB)
	@# pytest exit code 5 = no tests collected (no GPU tests yet) -> not a failure
	CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(GPU_DEVICES) \
		DRONESCAN_VRAM_LIMIT_GB=$(VRAM_LIMIT_GB) $(UV) run pytest -m gpu || [ $$? -eq 5 ]

test-sitl: ## Flight tests against ArduPilot SITL (needs the SITL container, M5)
	@# pytest exit code 5 = no tests collected (no SITL tests yet) -> not a failure
	DRONESCAN_SITL=1 $(UV) run pytest -m sitl || [ $$? -eq 5 ]

docker-check: ## GPU visible in the worker container? (expects e.g. (6, 1) on the GTX 1060)
	@# create the bind-mount directories as the current user (Docker would create them as root)
	@mkdir -p workspace data
	@mkdir -p "$${DRONESCAN_GPU_LOCK_DIR:-$${XDG_STATE_HOME:-$$HOME/.local/state}/drone-scan}"  # GPU lock dir, mounted into the containers (KAN-80)
	docker compose run --rm worker python -W always -c \
		"import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"

smoke: ## Pipeline on data/samples/kurz.mp4
	@echo "make smoke: noch nicht verfügbar – kommt mit den Pipeline-Stufen (M1)." >&2; exit 1

bench: ## Benchmarks on reference rooms
	@echo "make bench: noch nicht verfügbar – kommt mit den Referenzräumen (M1)." >&2; exit 1

viewer-dev: ## Vite dev server for viewer/
	@echo "make viewer-dev: noch nicht verfügbar – kommt mit dem Viewer (M1)." >&2; exit 1
