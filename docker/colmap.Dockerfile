# syntax=docker/dockerfile:1
# COLMAP 4.2.1 (BSD-3-Clause) built for sm_61 (GTX 1060) and sm_75 (RTX 2060).
# Runs as a separate process/container (stage `sfm`); headless: no GUI/Qt, no CGAL (GPL parts),
# OpenBLAS instead of Intel MKL (proprietary). Offline: no runtime downloads, no fetched ONNX
# runtime (CLAUDE.md rule 3); no AGPL LSD detector. Build: `docker compose build colmap`.
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04

# ---------- build stage ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG COLMAP_VERSION=4.2.1
ARG COLMAP_COMMIT=bd1fcf654d2dd8fefa1466999c190a246f83f4b9
ARG CUDA_ARCHITECTURES="61;75"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libopenimageio-dev libmetis-dev libgoogle-glog-dev \
        libsqlite3-dev libceres-dev libcurl4-openssl-dev libssl-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/include/opencv4  # OpenImageIO's CMake config expects this directory
RUN git clone --branch "${COLMAP_VERSION}" --depth 1 https://github.com/colmap/colmap.git /colmap \
    && test "$(git -C /colmap rev-parse HEAD)" = "${COLMAP_COMMIT}" \
    && cmake -S /colmap -B /colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DBLA_VENDOR=OpenBLAS \
        -DGUI_ENABLED=OFF \
        -DOPENGL_ENABLED=OFF \
        -DCGAL_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        -DDOWNLOAD_ENABLED=OFF \
        -DONNX_ENABLED=OFF \
        -DFETCH_ONNX=OFF \
        -DLSD_ENABLED=OFF \
    && cmake --build /colmap/build --target install
# License texts of COLMAP and its bundled/fetched third-party code (ADR-0007).
RUN mkdir -p /colmap-notices/thirdparty \
    && cp /colmap/COPYING.txt /colmap-notices/ \
    && for f in /colmap/src/thirdparty/*/LICENSE /colmap/build/_deps/*-src/LICENSE; do \
         if [ -f "$f" ]; then \
           name="$(basename "$(dirname "$f")")"; cp "$f" "/colmap-notices/thirdparty/${name%-src}.LICENSE"; \
         fi; \
       done

# ---------- runtime stage ----------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libboost-program-options1.83.0 libgomp1 libmetis5 libceres4t64 \
        libopenimageio2.4t64 libgoogle-glog0v6t64 libsqlite3-0 libcurl4t64 libssl3t64 \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /colmap-install/ /usr/local/
COPY --from=build /colmap-notices/ /usr/share/doc/colmap/
WORKDIR /workspace
USER ubuntu
CMD ["colmap", "help"]
