#!/usr/bin/env bash
# One Docker eval entrypoint for challenge OmniGibson ↔ policy on :8000.
#
# Default: HEADLESS=1 + write MP4 (CLI / no display).
# Optional:  HEADLESS=0  for a LIVE Isaac window (needs DISPLAY).
#
# Policy must already be up (separate container — Isaac is not in b1k-groot):
#   PATH_TO_CKPT=... scripts/docker/train_groot.sh serve
#
# Eval image (pick one):
#   docker pull stanfordvl/behavior:3.9.2          # recommended
#   scripts/docker/build_behavior_eval.sh          # self-build → b1k-eval
#
# Usage:
#   scripts/docker/eval.sh
#   HEADLESS=0 scripts/docker/eval.sh              # LIVE GUI
#   TASK=turning_on_radio MAX_STEPS=2000 scripts/docker/eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${TASK:-turning_on_radio}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
INSTANCE="${INSTANCE:-0}"
MAX_STEPS="${MAX_STEPS:-2000}"
OUT_DIR="${OUT_DIR:-$PWD/outputs/groot_docker_eval}"
EVAL_IMAGE="${EVAL_IMAGE:-stanfordvl/behavior:3.9.2}"
OG_DATA="${OG_DATA:-$PWD/data/omnigibson_data}"
SHM_SIZE="${SHM_SIZE:-16g}"
HEADLESS="${HEADLESS:-1}"
WRITE_VIDEO="${WRITE_VIDEO:-1}"

mkdir -p "$OUT_DIR" "$OG_DATA"

if ! curl -sf -o /dev/null "http://${HOST}:${PORT}/healthz"; then
  echo "FAIL: no policy on http://${HOST}:${PORT}/healthz" >&2
  echo "  Start:  PATH_TO_CKPT=... scripts/docker/train_groot.sh serve" >&2
  exit 1
fi

eval_args=(
  python -m omnigibson.eval.eval
  --task-name "$TASK"
  --host "$HOST" --port "$PORT"
  --instance-indices "$INSTANCE"
  --num-rollouts 1
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper
  --output-dir /out
  --max-steps "$MAX_STEPS"
)

if [[ "$HEADLESS" == "1" ]]; then
  eval_args+=(--headless)
  og_headless=1
  display_env=()
  echo "+ mode: HEADLESS (default)  WRITE_VIDEO=$WRITE_VIDEO"
else
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "FAIL: HEADLESS=0 needs DISPLAY set" >&2
    exit 1
  fi
  eval_args+=(--no-headless)
  og_headless=0
  display_env=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
  echo "+ mode: LIVE GUI  DISPLAY=$DISPLAY  WRITE_VIDEO=$WRITE_VIDEO"
fi
[[ "$WRITE_VIDEO" == "1" ]] && eval_args+=(--write-video)

echo "+ EVAL_IMAGE=$EVAL_IMAGE"
echo "+ OUT_DIR=$OUT_DIR"

docker run --rm --gpus all --network=host --privileged \
  --shm-size="$SHM_SIZE" \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e "OMNIGIBSON_HEADLESS=$og_headless" \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNIGIBSON_DATA_PATH=/data \
  "${display_env[@]}" \
  -v "$OG_DATA:/data" \
  -v "$OUT_DIR:/out" \
  "$EVAL_IMAGE" \
  "${eval_args[@]}"

echo "done → $OUT_DIR/"
ls -lh "$OUT_DIR/videos" 2>/dev/null || true
