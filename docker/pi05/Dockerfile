# π0.5 (OpenPI 'behavior' fork) — train + serve image.
#
# Host needs ONLY Docker + NVIDIA Container Toolkit. No conda, no uv on the host.
# uv runs *inside* the image at build time; runtime PATH points at the baked .venv.
#
# Build:
#   docker build -t b1k-pi05 -f docker/pi05/Dockerfile docker/pi05
# Or:
#   scripts/docker/build_pi05.sh

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    GIT_LFS_SKIP_SMUDGE=1 \
    # Runtime defaults (override with -e)
    REPO_ID=behavior-1k/2026-challenge-demos \
    DATASET_PATH=/data/demos \
    OPENPI_DIR=/opt/openpi \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Install uv (build-time only from the host's POV — never required on the host).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt
RUN git clone --depth 1 -b behavior https://github.com/wensi-ai/openpi.git openpi \
    && cd openpi \
    && git submodule update --init --recursive

WORKDIR /opt/openpi
# Resolve + install the challenge training env into /.venv (project-local).
RUN uv sync \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . \
    && uv pip install websockets huggingface_hub

# Prefer the baked venv for every command — user never types `uv run`.
ENV VIRTUAL_ENV=/opt/openpi/.venv \
    PATH="/opt/openpi/.venv/bin:${PATH}" \
    PYTHONPATH=/opt/openpi/src:/opt/openpi

WORKDIR /opt/openpi
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Mounts expected at runtime:
#   -v $DATA_ROOT:/data/demos
#   -v $CKPT_ROOT:/checkpoints
VOLUME ["/data/demos", "/checkpoints", "/root/.cache"]

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["help"]
