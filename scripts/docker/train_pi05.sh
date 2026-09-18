#!/usr/bin/env bash
# Train / norm-stats / serve π0.5 via the b1k-pi05 image.
#
# Demos, checkpoints, and HF models persist on the host under data/ (see
# scripts/docker/_persist_paths.sh). Re-runs reuse the cache — no re-download.
#
#   scripts/docker/train_pi05.sh norm-stats
#   EXP_NAME=radio scripts/docker/train_pi05.sh train
#   PATH_TO_CKPT=... scripts/docker/train_pi05.sh serve
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${TAG:-b1k-pi05}"
CKPT_DEFAULT=pi05
# shellcheck source=scripts/docker/_persist_paths.sh
source "$(dirname "$0")/_persist_paths.sh"

CMD="${1:-help}"
shift || true

extra=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  extra+=(-e "HF_TOKEN=$HF_TOKEN" -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN")
fi
[[ -n "${EXP_NAME:-}" ]] && extra+=(-e "EXP_NAME=$EXP_NAME")
[[ -n "${BATCH_SIZE:-}" ]] && extra+=(-e "BATCH_SIZE=$BATCH_SIZE")
[[ -n "${PATH_TO_CKPT:-}" ]] && extra+=(-e "PATH_TO_CKPT=$PATH_TO_CKPT")
[[ -n "${TASK_NAME:-}" ]] && extra+=(-e "TASK_NAME=$TASK_NAME")
[[ -n "${PORT:-}" ]] && extra+=(-e "PORT=$PORT")

ports=()
[[ "$CMD" == "serve" ]] && ports+=(-p "${PORT:-8000}:8000")

SHM_SIZE="${SHM_SIZE:-16g}"

exec docker run --rm -it --gpus all \
  --shm-size="$SHM_SIZE" \
  -v "$DATA_ROOT:/data/demos" \
  -v "$CKPT_ROOT:/checkpoints" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e "DATASET_PATH=/data/demos" \
  -e "REPO_ID=${REPO_ID:-behavior-1k/2026-challenge-demos}" \
  -e "HF_HOME=/root/.cache/huggingface" \
  -e "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface" \
  -e "TRANSFORMERS_CACHE=/root/.cache/huggingface" \
  -e "HF_HUB_CACHE=/root/.cache/huggingface" \
  -e "OPENPI_DATA_HOME=/checkpoints" \
  "${extra[@]}" \
  "${ports[@]}" \
  "$TAG" "$CMD" "$@"
