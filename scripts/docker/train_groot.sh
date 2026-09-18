#!/usr/bin/env bash
# Train / serve GR00T N1.7 via the b1k-groot image.
#
#   export HF_TOKEN=hf_xxx   # required (gated Cosmos backbone)
#   DATA_ROOT=data/demos CKPT_ROOT=data/checkpoints/groot \
#     scripts/docker/train_groot.sh deploy-modality
#   scripts/docker/train_groot.sh train
#   PATH_TO_CKPT=... scripts/docker/train_groot.sh serve
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${TAG:-b1k-groot}"
DATA_ROOT="${DATA_ROOT:-$PWD/data/demos}"
CKPT_ROOT="${CKPT_ROOT:-$PWD/data/checkpoints/groot}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CMD="${1:-help}"
shift || true

mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$HF_CACHE"

if [[ "$CMD" == "train" && -z "${HF_TOKEN:-}" ]]; then
  echo "FAIL: export HF_TOKEN (gated nvidia/Cosmos-Reason2-2B + GR00T-N1.7-3B)" >&2
  exit 1
fi

extra=()
[[ -n "${HF_TOKEN:-}" ]] && extra+=(-e "HF_TOKEN=$HF_TOKEN")
[[ -n "${EXP_NAME:-}" ]] && extra+=(-e "EXP_NAME=$EXP_NAME")
[[ -n "${TASK_NAME:-}" ]] && extra+=(-e "TASK_NAME=$TASK_NAME")
[[ -n "${NUM_GPUS:-}" ]] && extra+=(-e "NUM_GPUS=$NUM_GPUS")
[[ -n "${GLOBAL_BATCH_SIZE:-}" ]] && extra+=(-e "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE")
[[ -n "${MAX_STEPS:-}" ]] && extra+=(-e "MAX_STEPS=$MAX_STEPS")
[[ -n "${PATH_TO_CKPT:-}" ]] && extra+=(-e "PATH_TO_CKPT=$PATH_TO_CKPT")
[[ -n "${PORT:-}" ]] && extra+=(-e "PORT=$PORT")

ports=()
[[ "$CMD" == "serve" ]] && ports+=(-p "${PORT:-8000}:8000")

exec docker run --rm -it --gpus all \
  -v "$DATA_ROOT:/data/demos" \
  -v "$CKPT_ROOT:/checkpoints" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e "DATASET_PATH=/data/demos" \
  "${extra[@]}" \
  "${ports[@]}" \
  "$TAG" "$CMD" "$@"
