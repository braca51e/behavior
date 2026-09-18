# GR00T N1.7 (wensi-ai/Isaac-GR00T challenge fork) — train + serve image.
#
# Host needs ONLY Docker + NVIDIA Container Toolkit. No conda, no uv on the host.
#
# Build (BuildKit recommended — caches large NVIDIA wheels across retries):
#   scripts/docker/build_groot.sh
#
# NOTE: nvidia/Cosmos-Reason2-2B is gated — pass HF_TOKEN and accept the gate
# on Hugging Face before first train.

# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # Large wheels from pypi.nvidia.com often timeout at the default 30s.
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=4 \
    GIT_LFS_SKIP_SMUDGE=1 \
    GROOT_DIR=/opt/Isaac-GR00T \
    DATASET_PATH=/data/demos \
    REPO_ID=behavior-1k/2026-challenge-demos \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs \
        build-essential pkg-config \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        # torchcodec (GR00T video demos) needs FFmpeg shared libs at import time
        ffmpeg \
        libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt
# Challenge fork (has scripts/b1k/*). Pin shallow clone for reproducible builds.
RUN git clone --depth 1 https://github.com/wensi-ai/Isaac-GR00T.git Isaac-GR00T

WORKDIR /opt/Isaac-GR00T
# Retry: torch+cu128 pulls multi-GB nvidia-* wheels; flaky CDNs are common.
# BuildKit cache keeps successful wheel downloads across failed attempts.
COPY install_deps.sh /tmp/install_deps.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    chmod +x /tmp/install_deps.sh && /tmp/install_deps.sh && rm -f /tmp/install_deps.sh

ENV VIRTUAL_ENV=/opt/Isaac-GR00T/.venv \
    PATH="/opt/Isaac-GR00T/.venv/bin:${PATH}" \
    PYTHONPATH=/opt/Isaac-GR00T \
    # Triton JIT cache (writable in container)
    TRITON_CACHE_DIR=/tmp/triton_cache

WORKDIR /opt/Isaac-GR00T
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Runtime mounts (host paths via train_*.sh): demos, checkpoints, HF model cache.
VOLUME ["/data/demos", "/checkpoints", "/root/.cache/huggingface"]

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["help"]
