# GR00T N1.7 (wensi-ai/Isaac-GR00T challenge fork) — train + serve image.
#
# Host needs ONLY Docker + NVIDIA Container Toolkit. No conda, no uv on the host.
#
# Build:
#   docker build -t b1k-groot -f docker/groot/Dockerfile docker/groot
# Or:
#   scripts/docker/build_groot.sh
#
# NOTE: nvidia/Cosmos-Reason2-2B is gated — pass HF_TOKEN and accept the gate
# on Hugging Face before first train.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    GIT_LFS_SKIP_SMUDGE=1 \
    GROOT_DIR=/opt/Isaac-GR00T \
    DATASET_PATH=/data/demos \
    REPO_ID=behavior-1k/2026-challenge-demos

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs \
        python3.10 python3.10-venv python3-pip \
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
RUN uv sync --frozen --python 3.10 \
    && uv pip install --python .venv/bin/python websockets huggingface_hub

ENV VIRTUAL_ENV=/opt/Isaac-GR00T/.venv \
    PATH="/opt/Isaac-GR00T/.venv/bin:${PATH}" \
    PYTHONPATH=/opt/Isaac-GR00T

WORKDIR /opt/Isaac-GR00T
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

VOLUME ["/data/demos", "/checkpoints", "/root/.cache"]

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["help"]
