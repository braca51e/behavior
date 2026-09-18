#!/usr/bin/env bash
# Headless OmniGibson eval in Docker (CLI-only / no host conda).
#
# Two containers:
#   [b1k-groot serve :8000]  ←──WS──  [EVAL_IMAGE = stanfordvl/behavior:3.9.2 or self-built b1k-eval]
#
# Prefer:
#   docker pull stanfordvl/behavior:3.9.2
# Or build yourself (slow/huge):
#   scripts/docker/build_behavior_eval.sh   # → image b1k-eval
#
# Usage:
#   # Terminal A: PATH_TO_CKPT=... scripts/docker/train_groot.sh serve
#   # Terminal B:
#   scripts/docker/eval_headless.sh
#
# Assets (first run downloads into OG_DATA if empty):
#   export OG_DATA=$PWD/data/omnigibson_data
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${TASK:-turning_on_radio}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
INSTANCE="${INSTANCE:-0}"
MAX_STEPS="${MAX_STEPS:-2000}"
OUT_DIR="${OUT_DIR:-$PWD/outputs/groot_docker_eval}"
# Official challenge-aligned image (or TAG from build_behavior_eval.sh).
EVAL_IMAGE="${EVAL_IMAGE:-stanfordvl/behavior:3.9.2}"
OG_DATA="${OG_DATA:-$PWD/data/omnigibson_data}"
SHM_SIZE="${SHM_SIZE:-16g}"

mkdir -p "$OUT_DIR" "$OG_DATA"

if ! curl -sf -o /dev/null "http://${HOST}:${PORT}/healthz"; then
  echo "FAIL: no policy on http://${HOST}:${PORT}/healthz" >&2
  echo "  Start:  PATH_TO_CKPT=... scripts/docker/train_groot.sh serve" >&2
  exit 1
fi

echo "+ policy OK on :${PORT}"
echo "+ EVAL_IMAGE=$EVAL_IMAGE"
echo "+ OG_DATA=$OG_DATA  (OmniGibson /data mount)"
echo "+ OUT_DIR=$OUT_DIR"

# stanfordvl/behavior entrypoint: conda activate behavior && exec "$@"
docker run --rm --gpus all --network=host --privileged \
  --shm-size="$SHM_SIZE" \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e OMNIGIBSON_HEADLESS=1 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNIGIBSON_DATA_PATH=/data \
  -v "$OG_DATA:/data" \
  -v "$OUT_DIR:/out" \
  "$EVAL_IMAGE" \
  python -m omnigibson.eval.eval \
    --task-name "$TASK" \
    --host "$HOST" --port "$PORT" \
    --instance-indices "$INSTANCE" \
    --num-rollouts 1 \
    --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
    --output-dir /out \
    --max-steps "$MAX_STEPS" \
    --headless \
    --write-video

echo "done → $OUT_DIR/videos/"
ls -lh "$OUT_DIR/videos" 2>/dev/null || true
