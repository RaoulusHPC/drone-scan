# syntax=docker/dockerfile:1
# GPU worker: drone-scan + PyTorch (cu126) + gsplat, built for sm_61 (GTX 1060) and sm_75 (RTX 2060).
# Versions pinned per ADR-0010. Build: `docker compose build worker`.
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04
ARG UV_VERSION=0.13.9

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------- build stage: compile gsplat with nvcc ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG TORCH_VERSION=2.14.1
ARG GSPLAT_VERSION=v1.5.3
ARG GSPLAT_COMMIT=937e29912570c372bed6747a5c9bf85fed877bae
ARG TORCH_CUDA_ARCH_LIST="6.1;7.5"
ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_HTTP_TIMEOUT=300
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

# Order matters for the layer cache: PyTorch (~2.5 GB) and the gsplat nvcc build do not depend
# on src/, so they come first; edits under src/ only re-run the last `uv sync`.
RUN uv venv /opt/venv --python /usr/bin/python3.12

# PyTorch < 2.15 with CUDA 12.6: the last wheels that still support Pascal (ADR-0010).
RUN uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu126 "torch==${TORCH_VERSION}+cu126"

# gsplat compiled ahead of time for both architectures (no JIT on the user's machine).
# Pinned patch (ADR-0010): C++20 for torch 2.14 headers and a Pascal (sm_61) fallback for
# cg::labeled_partition, which needs sm_70+. `git apply --check` fails the build if it no
# longer applies cleanly (e.g. after a gsplat update).
COPY docker/patches/gsplat-v1.5.3-pascal.patch /tmp/gsplat-pascal.patch
RUN uv pip install --python /opt/venv/bin/python setuptools wheel ninja \
    && git clone --branch "${GSPLAT_VERSION}" --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat \
    && test "$(git -C /tmp/gsplat rev-parse HEAD)" = "${GSPLAT_COMMIT}" \
    && cd /tmp/gsplat \
    && git apply --check /tmp/gsplat-pascal.patch \
    && git apply /tmp/gsplat-pascal.patch \
    && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
        uv pip install --python /opt/venv/bin/python --no-build-isolation . \
    && mkdir -p /opt/notices/gsplat \
    && cp LICENSE /opt/notices/gsplat/LICENSE \
    && cp gsplat/cuda/csrc/third_party/glm/copying.txt /opt/notices/gsplat/glm.copying.txt \
    && cp /tmp/gsplat-pascal.patch /opt/notices/gsplat/MODIFICATIONS.patch \
    && rm -rf /tmp/gsplat /tmp/gsplat-pascal.patch

# AC: gsplat must contain device code for both architectures (fails the build otherwise).
RUN so="$(find /opt/venv -path '*gsplat*' -name 'csrc*.so' | head -n 1)" \
    && test -n "$so" \
    && cuobjdump --list-elf "$so" > /tmp/gsplat-elf.txt \
    && grep -q 'sm_61' /tmp/gsplat-elf.txt \
    && grep -q 'sm_75' /tmp/gsplat-elf.txt \
    && rm /tmp/gsplat-elf.txt

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# --inexact keeps PyTorch and gsplat, which are not part of uv.lock.
RUN uv sync --locked --no-dev --no-editable --inexact --python /usr/bin/python3.12

# ---------- runtime stage ----------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# ffmpeg (Ubuntu 24.04: 6.1, provides ffprobe) is used by `flight add` and `ingest` (KAN-81).
# It runs only as an external process, never linked; its license texts stay in
# /usr/share/doc/ffmpeg (CLAUDE.md rule 1, image local only).
COPY --from=build /opt/venv /opt/venv
COPY LICENSE /usr/share/doc/drone-scan/LICENSE
COPY --from=build /opt/notices/gsplat/ /usr/share/doc/gsplat/
WORKDIR /workspace
# uid 1000 ("ubuntu" in the base image) matches the usual desktop user, so files in
# the mounted workspace/ stay owned by the user.
USER ubuntu
CMD ["drone-scan", "--help"]

# ---------- test stage: worker + dev group (pytest) for `make test-gpu` (KAN-41) ----------
# Built only on request (compose profile "test"). The tests and the code under test come from
# the repository mounted read-only at /repo (PYTHONPATH=/repo/src), so source edits need no
# rebuild; this image only adds the locked dev dependencies to the worker environment.
FROM runtime AS test
USER root
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_HTTP_TIMEOUT=300 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /tmp/lock
COPY pyproject.toml uv.lock README.md ./
# --only-group dev: no project install; --inexact keeps PyTorch, gsplat and drone-scan.
RUN uv sync --locked --inexact --only-group dev --python /usr/bin/python3.12 \
    && rm -rf /tmp/lock /root/.cache/uv
WORKDIR /repo
USER ubuntu
CMD ["pytest", "-m", "gpu and torch", "-p", "no:cacheprovider"]
