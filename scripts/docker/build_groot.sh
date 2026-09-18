#!/usr/bin/env bash
# Build the GR00T N1.7 training/serve image (no host uv/conda).
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG="${TAG:-b1k-groot}"
echo "+ docker build -t $TAG -f docker/groot/Dockerfile docker/groot"
docker build -t "$TAG" -f docker/groot/Dockerfile docker/groot
echo "built $TAG"
