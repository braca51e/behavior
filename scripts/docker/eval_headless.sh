#!/usr/bin/env bash
# Headless OmniGibson eval in Docker (CLI-only / no host conda).
#
# Architecture (two containers — Isaac Sim is NOT inside b1k-groot):
#
#   [b1k-groot serve :8000]  ←──WS──  [EVAL_IMAGE runs omnigibson.eval.eval]
#                                      writes MP4 + JSON to host OUT_DIR
#
# Prerequisites:
#   1) Policy already serving:  PATH_TO_CKPT=... scripts/docker/train_groot.sh serve
#   2) An OmniGibson/Isaac image that has the 2026 challenge evaluator, e.g.:
#        export EVAL_IMAGE=stanfordvl/omnigibson:isaac_4_5
#      (challenge v3.9.2 may need a BEHAVIOR-1K checkout mounted — see below)
#   3) Dataset/assets available to that image (OMNIGIBSON_DATA_PATH or B1K mount)
#
# Usage:
#   scripts/docker/eval_headless.sh
#   TASK=turning_on_radio OUT_DIR=outputs/groot_provided_eval scripts/docker/eval_headless.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK="${TASK:-turning_on_radio}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
INSTANCE="${INSTANCE:-0}"
MAX_STEPS="${MAX_STEPS:-2000}"
OUT_DIR="${OUT_DIR:-$PWD/outputs/groot_docker_eval}"
EVAL_IMAGE="${EVAL_IMAGE:-stanfordvl/omnigibson:isaac_4_5}"
# Optional: mount a BEHAVIOR-1K v3.9.2 tree so `python -m omnigibson.eval.eval` matches challenge.
B1K_ROOT="${B1K_ROOT:-}"
OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-}"

mkdir -p "$OUT_DIR"

if ! curl -sf -o /dev/null "http://${HOST}:${PORT}/healthz"; then
  echo "FAIL: no policy on http://${HOST}:${PORT}/healthz" >&2
  echo "  Start serve first (Docker):" >&2
  echo "    PATH_TO_CKPT=... scripts/docker/train_groot.sh serve" >&2
  exit 1
fi

echo "+ policy OK on :${PORT}"
echo "+ eval image: $EVAL_IMAGE"
echo "+ videos/json → $OUT_DIR"

mounts=(-v "$OUT_DIR:/out")
envs=(
  -e OMNI_KIT_ACCEPT_EULA=YES
  -e OMNIGIBSON_HEADLESS=1
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
)
[[ -n "$OMNIGIBSON_DATA_PATH" ]] && {
  mounts+=(-v "$OMNIGIBSON_DATA_PATH:/og_data")
  envs+=(-e OMNIGIBSON_DATA_PATH=/og_data)
}
[[ -n "$B1K_ROOT" ]] && mounts+=(-v "$B1K_ROOT:/b1k")

# Prefer challenge evaluator from mounted BEHAVIOR-1K; else image default module.
if [[ -n "$B1K_ROOT" ]]; then
  py_cmd=(
    bash -lc
    "cd /b1k && python -m omnigibson.eval.eval \
      --task-name $TASK \
      --host $HOST --port $PORT \
      --instance-indices $INSTANCE \
      --num-rollouts 1 \
      --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
      --output-dir /out \
      --max-steps $MAX_STEPS \
      --headless \
      --write-video"
  )
else
  py_cmd=(
    bash -lc
    "python -m omnigibson.eval.eval \
      --task-name $TASK \
      --host $HOST --port $PORT \
      --instance-indices $INSTANCE \
      --num-rollouts 1 \
      --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
      --output-dir /out \
      --max-steps $MAX_STEPS \
      --headless \
      --write-video"
  )
fi

echo "+ docker run $EVAL_IMAGE (headless + write-video)"
docker run --rm --gpus all --network=host \
  --shm-size="${SHM_SIZE:-16g}" \
  "${envs[@]}" \
  "${mounts[@]}" \
  "$EVAL_IMAGE" \
  "${py_cmd[@]}"

echo "done → $OUT_DIR/videos/  (scp the mp4 to your laptop to watch)"
ls -lh "$OUT_DIR/videos" 2>/dev/null || true
