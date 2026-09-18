#!/usr/bin/env bash
# Train / serve GR00T N1.7 via the b1k-groot image.
#
# Demos, checkpoints, and HF models persist on the host under data/ (see
# scripts/docker/_persist_paths.sh). Re-runs reuse the cache — no re-download.
#
#   export HF_TOKEN=hf_xxx   # required (gated Cosmos backbone)
#   scripts/docker/train_groot.sh deploy-modality
#   scripts/docker/train_groot.sh train
#   PATH_TO_CKPT=... scripts/docker/train_groot.sh serve
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${TAG:-b1k-groot}"
CKPT_DEFAULT=groot
# shellcheck source=scripts/docker/_persist_paths.sh
source "$(dirname "$0")/_persist_paths.sh"

CMD="${1:-help}"
shift || true

if [[ "$CMD" == "train" || "$CMD" == "deploy-modality" ]]; then
  if [[ ! -f "$DATA_ROOT/meta/info.json" ]]; then
    echo "FAIL: missing $DATA_ROOT/meta/info.json (no demos on host)" >&2
    echo "  Run:  scripts/docker/download_demos.sh 0" >&2
    echo "  Then: scripts/docker/train_groot.sh train" >&2
    exit 1
  fi
fi

# train needs meta/modality.json — write it automatically if missing.
if [[ "$CMD" == "train" && ! -f "$DATA_ROOT/meta/modality.json" ]]; then
  echo "+ missing meta/modality.json — running deploy-modality first"
  docker run --rm --gpus all \
    -v "$DATA_ROOT:/data/demos" \
    -e "DATASET_PATH=/data/demos" \
    "$TAG" deploy-modality
  if [[ ! -f "$DATA_ROOT/meta/modality.json" ]]; then
    echo "FAIL: deploy-modality did not create $DATA_ROOT/meta/modality.json" >&2
    exit 1
  fi
fi

if [[ "$CMD" == "train" && -z "${HF_TOKEN:-}" ]]; then
  echo "FAIL: export HF_TOKEN (gated nvidia/Cosmos-Reason2-2B + GR00T-N1.7-3B)" >&2
  exit 1
fi

# HF_TOKEN wins over cached hf auth login — a revoked/typo'd token always 401s.
if [[ "$CMD" == "train" ]]; then
  echo "+ preflight: hf auth whoami (HF_TOKEN must be a live Read token)"
  if ! docker run --rm -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -e HF_HOME=/root/.cache/huggingface \
      -v "$HF_CACHE:/root/.cache/huggingface" \
      "$TAG" shell -c 'hf auth whoami'; then
    echo "FAIL: HF_TOKEN is invalid or expired." >&2
    echo "  1) Revoke old tokens at https://huggingface.co/settings/tokens" >&2
    echo "  2) Create a new Read token (do not paste it into chat)" >&2
    echo "  3) export HF_TOKEN=hf_...   # fresh value, no quotes/spaces" >&2
    echo "  4) Confirm: docker run --rm -e HF_TOKEN $TAG shell -c 'hf auth whoami'" >&2
    exit 1
  fi
fi

extra=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  extra+=(-e "HF_TOKEN=$HF_TOKEN" -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN")
fi
[[ -n "${EXP_NAME:-}" ]] && extra+=(-e "EXP_NAME=$EXP_NAME")
[[ -n "${TASK_NAME:-}" ]] && extra+=(-e "TASK_NAME=$TASK_NAME")
[[ -n "${NUM_GPUS:-}" ]] && extra+=(-e "NUM_GPUS=$NUM_GPUS")
[[ -n "${GLOBAL_BATCH_SIZE:-}" ]] && extra+=(-e "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE")
[[ -n "${MAX_STEPS:-}" ]] && extra+=(-e "MAX_STEPS=$MAX_STEPS")
[[ -n "${PATH_TO_CKPT:-}" ]] && extra+=(-e "PATH_TO_CKPT=$PATH_TO_CKPT")
[[ -n "${PORT:-}" ]] && extra+=(-e "PORT=$PORT")
# W&B: default offline inside entrypoint; set WANDB_MODE=online + WANDB_API_KEY to sync live.
[[ -n "${WANDB_MODE:-}" ]] && extra+=(-e "WANDB_MODE=$WANDB_MODE")
[[ -n "${WANDB_API_KEY:-}" ]] && extra+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
[[ -n "${WANDB_PROJECT:-}" ]] && extra+=(-e "WANDB_PROJECT=$WANDB_PROJECT")
[[ -n "${WANDB_ENTITY:-}" ]] && extra+=(-e "WANDB_ENTITY=$WANDB_ENTITY")
[[ -n "${WANDB_RUN_GROUP:-}" ]] && extra+=(-e "WANDB_RUN_GROUP=$WANDB_RUN_GROUP")

ports=()
[[ "$CMD" == "serve" ]] && ports+=(-p "${PORT:-8000}:8000")

# PyTorch DataLoader + video shard cache needs far more than Docker's 64MB /dev/shm default.
SHM_SIZE="${SHM_SIZE:-16g}"

exec docker run --rm -it --gpus all \
  --shm-size="$SHM_SIZE" \
  -v "$DATA_ROOT:/data/demos" \
  -v "$CKPT_ROOT:/checkpoints" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e "DATASET_PATH=/data/demos" \
  -e "HF_HOME=/root/.cache/huggingface" \
  -e "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface" \
  -e "TRANSFORMERS_CACHE=/root/.cache/huggingface" \
  -e "HF_HUB_CACHE=/root/.cache/huggingface" \
  "${extra[@]}" \
  "${ports[@]}" \
  "$TAG" "$CMD" "$@"
