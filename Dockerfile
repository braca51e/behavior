# b1k 2026 BEHAVIOR Challenge policy server (Docker serving option).
#
# The organizers run OmniGibson OUTSIDE the container and connect over the
# WebSocket policy client.  The image serves our policy on port 8000 (overridable
# via --port).  Build with a 24 GB GPU available; the real pi0.5 backend loads
# the checkpoint from /workspace/data/checkpoints/pi05_b1k.
#
#   docker build -t b1k-policy .
#   docker run --gpus all -p 8000:8000 -v $(pwd)/data:/workspace/data b1k-policy \
#       python -m b1k.server --config configs/server.yaml --port 8000 --task 0
#
# Cold-start target < 300 s (design week 4, M4).

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Serving deps only (no simulator, no training stack).  torch is pulled in for
# the real "vla" backend; the "echo"/"noop" backends never import it at runtime.
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install numpy msgpack websockets PyYAML torch

# Ship the repo (serving code + configs + docs/plans + data stubs).
COPY src ./src
COPY configs ./configs
COPY docs ./docs
COPY pyproject.toml ./

ENV PYTHONPATH=/workspace/src

# Data (checkpoints, demo actions, detector bundles, stats) is mounted at
# runtime so the image stays small; the fixture demo actions ship inside.
RUN mkdir -p data/demos data/outputs

EXPOSE 8000

# Health check (organizer connectivity + our own readiness).
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

# Default: serve task 0; the evaluator reconnects/resets per rollout.  For real
# multi-task serving, the harness fans out N containers (or the 50-port fleet)
# each pinned to a task via --task.
CMD ["python", "-m", "b1k.server", "--config", "configs/server.yaml", "--port", "8000"]
