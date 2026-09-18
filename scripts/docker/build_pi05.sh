#!/usr/bin/env bash
# Build the π0.5 OpenPI training/serve image (no host uv/conda).
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG="${TAG:-b1k-pi05}"
echo "+ docker build -t $TAG -f docker/pi05/Dockerfile docker/pi05"
docker build -t "$TAG" -f docker/pi05/Dockerfile docker/pi05
echo "built $TAG"
