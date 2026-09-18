#!/usr/bin/env bash
# Build the GR00T N1.7 training/serve image (no host uv/conda).
#
# Uses BuildKit so uv's download cache survives retries when pypi.nvidia.com
# times out on large CUDA wheels (torch / cudnn / cusparse / tensorrt).
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG="${TAG:-b1k-groot}"
export DOCKER_BUILDKIT=1
echo "+ DOCKER_BUILDKIT=1 docker build -t $TAG -f docker/groot/Dockerfile docker/groot"
docker build -t "$TAG" -f docker/groot/Dockerfile docker/groot
echo "built $TAG"
