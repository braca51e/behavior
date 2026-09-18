#!/usr/bin/env bash
# Train / norm-stats / serve π0.5 via the b1k-pi05 image.
#
#   DATA_ROOT=data/demos CKPT_ROOT=data/checkpoints/pi05 \
#     scripts/docker/train_pi05.sh norm-stats
#   DATA_ROOT=data/demos CKPT_ROOT=data/checkpoints/pi05 EXP_NAME=radio \
#     scripts/docker/train_pi05.sh train
#   PATH_TO_CKPT=... scripts/docker/train_pi05.sh serve
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${TAG:-b1k-pi05}"
DATA_ROOT="${DATA_ROOT:-$PWD/data/demos}"
CKPT_ROOT="${CKPT_ROOT:-$PWD/data/checkpoints/pi05}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CMD="${1:-help}"
shift || true

mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$HF_CACHE"

extra=()
[[ -n "${HF_TOKEN:-}" ]] && extra+=(-e "HF_TOKEN=$HF_TOKEN")
[[ -n "${EXP_NAME:-}" ]] && extra+=(-e "EXP_NAME=$EXP_NAME")
[[ -n "${BATCH_SIZE:-}" ]] && extra+=(-e "BATCH_SIZE=$BATCH_SIZE")
[[ -n "${PATH_TO_CKPT:-}" ]] && extra+=(-e "PATH_TO_CKPT=$PATH_TO_CKPT")
[[ -n "${TASK_NAME:-}" ]] && extra+=(-e "TASK_NAME=$TASK_NAME")
[[ -n "${PORT:-}" ]] && extra+=(-e "PORT=$PORT")

ports=()
[[ "$CMD" == "serve" ]] && ports+=(-p "${PORT:-8000}:8000")

exec docker run --rm -it --gpus all \
  -v "$DATA_ROOT:/data/demos" \
  -v "$CKPT_ROOT:/checkpoints" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e "DATASET_PATH=/data/demos" \
  -e "REPO_ID=${REPO_ID:-behavior-1k/2026-challenge-demos}" \
  "${extra[@]}" \
  "${ports[@]}" \
  "$TAG" "$CMD" "$@"
