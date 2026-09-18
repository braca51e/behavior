#!/usr/bin/env bash
# Entrypoint for b1k-groot. Official GR00T N1.7 baseline commands without host uv/conda.
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-/data/demos}"
TASK_NAME="${TASK_NAME:-turning_on_radio}"
EXP_NAME="${EXP_NAME:-b1k-$TASK_NAME}"
PATH_TO_CKPT="${PATH_TO_CKPT:-}"
PORT="${PORT:-8000}"
NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"   # official 2048 is for 8× big GPUs
MAX_STEPS="${MAX_STEPS:-150000}"

cd /opt/Isaac-GR00T

usage() {
  cat <<'EOF'
b1k-groot commands (no host uv/conda):

  help
  deploy-modality   Write meta/modality.json into every task under DATASET_PATH
  train             Fine-tune GR00T N1.7 (single GPU default)
  serve             Websocket serve on :8000 (needs PATH_TO_CKPT)
  shell             Interactive bash in the baked venv

Env:
  DATASET_PATH        demos root (default /data/demos)
  TASK_NAME           challenge task (default turning_on_radio)
  EXP_NAME            default b1k-$TASK_NAME
  PATH_TO_CKPT        checkpoint dir for serve
  NUM_GPUS            default 1
  GLOBAL_BATCH_SIZE   default 128 (official doc: 2048 on 8 GPUs)
  MAX_STEPS           default 150000
  HF_TOKEN            required for gated Cosmos-Reason2-2B + GR00T-N1.7-3B
  PORT                serve port (default 8000)
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  help|-h|--help)
    usage
    ;;
  deploy-modality)
    exec python scripts/b1k/deploy_modality.py "$DATASET_PATH" "$@"
    ;;
  train)
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "WARN: HF_TOKEN unset — gated backbone download will likely fail." >&2
    fi
    mkdir -p /checkpoints
    # Single-GPU path; multi-GPU uses torchrun when NUM_GPUS>1.
    if [[ "$NUM_GPUS" -le 1 ]]; then
      exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        WANDB_MODE="${WANDB_MODE:-offline}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
        python scripts/b1k/train_b1k.py \
          --experiment-name "$EXP_NAME" \
          --base-model-path nvidia/GR00T-N1.7-3B \
          --dataset-path "$DATASET_PATH" \
          --embodiment-tag NEW_EMBODIMENT \
          --modality-config-path examples/b1k/r1pro.py \
          --num-gpus 1 \
          --global-batch-size "$GLOBAL_BATCH_SIZE" \
          --output-dir /checkpoints \
          --save-steps 1500 --save-total-limit 5 --max-steps "$MAX_STEPS" \
          --dataloader-num-workers 4 --decode-only-used-frames \
          "$@"
    else
      exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        WANDB_MODE="${WANDB_MODE:-offline}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
        torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-29500}" \
          scripts/b1k/train_b1k.py \
          --experiment-name "$EXP_NAME" \
          --base-model-path nvidia/GR00T-N1.7-3B \
          --dataset-path "$DATASET_PATH" \
          --embodiment-tag NEW_EMBODIMENT \
          --modality-config-path examples/b1k/r1pro.py \
          --num-gpus "$NUM_GPUS" \
          --global-batch-size "$GLOBAL_BATCH_SIZE" \
          --output-dir /checkpoints \
          --save-steps 1500 --save-total-limit 5 --max-steps "$MAX_STEPS" \
          --dataloader-num-workers 8 --decode-only-used-frames \
          "$@"
    fi
    ;;
  serve)
    if [[ -z "$PATH_TO_CKPT" ]]; then
      echo "FAIL: set PATH_TO_CKPT to a checkpoint directory" >&2
      exit 1
    fi
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python scripts/b1k/serve_b1k.py \
        --model-path "$PATH_TO_CKPT" \
        --modality-config-path examples/b1k/r1pro.py \
        --embodiment-tag NEW_EMBODIMENT \
        --host 0.0.0.0 --port "$PORT" \
        "$@"
    ;;
  shell|bash)
    exec bash "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac
