#!/usr/bin/env bash
# Shared host paths for Docker train/serve wrappers.
# All downloads land on the host so containers never re-fetch unnecessarily.
#
#   data/demos/                  challenge LeRobot demos (download_demos.sh)
#   data/checkpoints/{pi05,groot} training outputs
#   data/cache/huggingface/      HF models (Cosmos, GR00T, tokenizers, …)
#
# Override with DATA_ROOT / CKPT_ROOT / HF_CACHE if needed.
# shellcheck disable=SC2034

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/demos}"
# Caller sets CKPT_DEFAULT before sourcing (pi05 vs groot).
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/data/checkpoints/${CKPT_DEFAULT:-groot}}"
HF_CACHE="${HF_CACHE:-$REPO_ROOT/data/cache/huggingface}"

mkdir -p "$DATA_ROOT" "$CKPT_ROOT" "$HF_CACHE"

echo "+ persist demos       $DATA_ROOT  →  /data/demos"
echo "+ persist checkpoints $CKPT_ROOT  →  /checkpoints"
echo "+ persist HF models   $HF_CACHE   →  /root/.cache/huggingface"
