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

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --python /usr/bin/python3.12

# PyTorch < 2.15 with CUDA 12.6: the last wheels that still support Pascal (ADR-0010).
RUN uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu126 "torch==${TORCH_VERSION}+cu126"

# gsplat compiled ahead of time for both architectures (no JIT on the user's machine).
RUN uv pip install --python /opt/venv/bin/python setuptools wheel ninja \
    && git clone --branch "${GSPLAT_VERSION}" --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/nerfstudio-project/gsplat.git /tmp/gsplat \
    && test "$(git -C /tmp/gsplat rev-parse HEAD)" = "${GSPLAT_COMMIT}" \
    && cd /tmp/gsplat \
    && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
        uv pip install --python /opt/venv/bin/python --no-build-isolation . \
    && mkdir -p /opt/notices/gsplat \
    && cp LICENSE /opt/notices/gsplat/LICENSE \
    && cp gsplat/cuda/csrc/third_party/glm/copying.txt /opt/notices/gsplat/glm.copying.txt \
    && rm -rf /tmp/gsplat

# ---------- runtime stage ----------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
COPY LICENSE /usr/share/doc/drone-scan/LICENSE
COPY --from=build /opt/notices/gsplat/ /usr/share/doc/gsplat/
WORKDIR /workspace
# uid 1000 ("ubuntu" in the base image) matches the usual desktop user, so files in
# the mounted workspace/ stay owned by the user.
USER ubuntu
CMD ["drone-scan", "--help"]
