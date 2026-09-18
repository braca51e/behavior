#!/usr/bin/env bash
# Build the OmniGibson / BEHAVIOR-1K v3.9.2 eval image yourself (official Dockerfile).
#
# Needs: lots of disk (~50–100 GB free while building), Docker + BuildKit,
#        NVIDIA Container Toolkit. Accepts NVIDIA EULA via setup.sh flags.
#
# Faster alternative (no build): docker pull stanfordvl/behavior:3.9.2
#
# Usage:
#   scripts/docker/build_behavior_eval.sh
#   TAG=b1k-eval scripts/docker/build_behavior_eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${TAG:-b1k-eval}"
B1K_REF="${B1K_REF:-v3.9.2}"
SRC_DIR="${B1K_SRC:-$PWD/third_party/BEHAVIOR-1K}"

echo "+ BEHAVIOR-1K source: $SRC_DIR ($B1K_REF)"
if [[ ! -f "$SRC_DIR/docker/Dockerfile" ]]; then
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --depth 1 -b "$B1K_REF" \
    https://github.com/StanfordVL/BEHAVIOR-1K.git "$SRC_DIR"
fi

export DOCKER_BUILDKIT=1
echo "+ docker build -t $TAG -f docker/Dockerfile (from BEHAVIOR-1K root)"
echo "  This is LONG and LARGE (CUDA toolkit + Isaac/OmniGibson). Expect tens of GB."
docker build -t "$TAG" -f docker/Dockerfile "$SRC_DIR"
echo "built $TAG"
echo "Use:  EVAL_IMAGE=$TAG scripts/docker/eval_headless.sh"
