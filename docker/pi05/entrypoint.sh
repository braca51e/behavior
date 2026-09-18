#!/usr/bin/env bash
# Entrypoint for b1k-pi05. Dispatches official OpenPI baseline commands without uv/conda.
set -euo pipefail

REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos}"
DATASET_PATH="${DATASET_PATH:-/data/demos}"
EXP_NAME="${EXP_NAME:-pi05_b1k}"
BATCH_SIZE="${BATCH_SIZE:-8}"          # 64 needs multi-GPU / 40GB+; 8 fits ~16–24GB
ACTION_HORIZON="${ACTION_HORIZON:-16}"
PORT="${PORT:-8000}"
PATH_TO_CKPT="${PATH_TO_CKPT:-}"
TASK_NAME="${TASK_NAME:-turning_on_radio}"

cd /opt/openpi

usage() {
  cat <<'EOF'
b1k-pi05 commands (no host uv/conda):

  help
  norm-stats          Compute π0.5 norm stats → /checkpoints/assets/...
  train               Fine-tune pi05_b1k (single GPU by default)
  serve               Websocket serve on :8000 (needs PATH_TO_CKPT)
  shell               Interactive bash in the baked venv
  <any args...>       Pass-through to python in the venv

Env:
  DATASET_PATH   demos root (default /data/demos)
  REPO_ID        HF repo id (default behavior-1k/2026-challenge-demos)
  EXP_NAME       experiment name (default pi05_b1k)
  BATCH_SIZE     default 8 (official doc uses 64 on big GPUs)
  PATH_TO_CKPT   checkpoint step dir for serve
  TASK_NAME      serve task (default turning_on_radio)
  PORT           serve port (default 8000)
  HF_TOKEN       optional; helps base-weight / rate limits
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  help|-h|--help)
    usage
    ;;
  norm-stats)
    exec python scripts/compute_norm_stats.py pi05_b1k \
      --data.repo_id="$REPO_ID" \
      --data.base_config.dataset_root="$DATASET_PATH" \
      "$@"
    ;;
  train)
    mkdir -p /checkpoints
    # Point OpenPI outputs at the mounted volume when possible.
    export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/checkpoints}"
    exec env XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" \
      python scripts/b1k/train_b1k.py pi05_b1k \
        --exp_name="$EXP_NAME" \
        --overwrite \
        --batch_size="$BATCH_SIZE" \
        --data.repo_id="$REPO_ID" \
        --data.base_config.dataset_root="$DATASET_PATH" \
        "$@"
    ;;
  serve)
    if [[ -z "$PATH_TO_CKPT" ]]; then
      echo "FAIL: set PATH_TO_CKPT to a checkpoint step directory" >&2
      exit 1
    fi
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" \
      python scripts/b1k/serve_b1k.py \
        --robot b1k/R1Pro \
        --task "b1k/$TASK_NAME" \
        --repo-id "$REPO_ID" \
        --policy.config pi05_b1k \
        --policy.dir "$PATH_TO_CKPT" \
        --control_mode receding_horizon \
        --action_horizon "$ACTION_HORIZON" \
        --port "$PORT" \
        "$@"
    ;;
  shell|bash)
    exec bash "$@"
    ;;
  *)
    # Pass-through: `docker run ... python -c ...` style via entrypoint override,
    # or `docker run b1k-pi05 python scripts/...`
    exec "$cmd" "$@"
    ;;
esac
