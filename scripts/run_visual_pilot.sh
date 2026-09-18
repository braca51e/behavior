#!/usr/bin/env bash
# Small visual pilot: this repo's echo policy + OmniGibson evaluator.
# Prerequisites: conda env behavior392 (BEHAVIOR-1K v3.9.2) fully installed.
#
# Usage:
#   LIVE=1 scripts/run_visual_pilot.sh              # LIVE Isaac window, NO video
#   LIVE=1 MAX_STEPS=9000 scripts/run_visual_pilot.sh
#   scripts/run_visual_pilot.sh                     # headless + write MP4 (no autoplay)
#   WRITE_VIDEO=1 PLAY_VIDEO=1 scripts/run_visual_pilot.sh  # headless + record + play
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

TASK="${TASK:-turning_on_radio}"
PORT="${PORT:-8000}"
INSTANCE="${INSTANCE:-0}"
MAX_STEPS="${MAX_STEPS:-1200}"
OUT="${OUT:-$REPO/outputs/visual_pilot}"
LIVE="${LIVE:-0}"
CONDA_ENV="${CONDA_ENV:-behavior392}"
# LIVE defaults: no recording, no ffplay. Headless defaults: write video, no autoplay.
if [[ "$LIVE" == "1" ]]; then
  WRITE_VIDEO="${WRITE_VIDEO:-0}"
  PLAY_VIDEO="${PLAY_VIDEO:-0}"
else
  WRITE_VIDEO="${WRITE_VIDEO:-1}"
  PLAY_VIDEO="${PLAY_VIDEO:-0}"
fi

mkdir -p "$OUT"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

if [[ "$LIVE" == "1" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "FAIL: LIVE=1 needs a display (DISPLAY is empty). Run from a desktop session."
    exit 1
  fi
  echo "LIVE mode: Isaac Sim GUI on DISPLAY=$DISPLAY (no video unless WRITE_VIDEO=1)"
fi

# 1) policy server (system/repo python — echo backend, CPU, tiny)
pkill -f "b1k.server.*--port ${PORT}" 2>/dev/null || true
sleep 1
PYTHONPATH="$REPO/src" PYTHONNOUSERSITE=1 \
  python3 -m b1k.server --config "$REPO/configs/server.yaml" --port "$PORT" --task "$TASK" \
  >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$OUT/server.pid"
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/healthz" || true)
  [[ "$code" == "200" ]] && break
  sleep 0.5
done
[[ "$code" == "200" ]] || { echo "FAIL: policy server healthz"; tail -50 "$OUT/server.log"; exit 1; }
echo "policy server OK on :$PORT (pid $SERVER_PID, backend=echo)"

# 2) official evaluator (needs GPU + OmniGibson)
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES
unset EXP_PATH CARB_APP_PATH ISAAC_PATH

CMD=(
  python -m omnigibson.eval.eval
  --task-name "$TASK"
  --host 127.0.0.1 --port "$PORT"
  --instance-indices "$INSTANCE"
  --num-rollouts 1
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper
  --output-dir "$OUT"
  --max-steps "$MAX_STEPS"
)
[[ "$WRITE_VIDEO" == "1" ]] && CMD+=(--write-video)
# OmniGibson eval defaults --headless=True; omitting the flag still runs headless.
# LIVE needs the explicit --no-headless (BooleanOptionalAction).
if [[ "$LIVE" == "1" ]]; then
  CMD+=(--no-headless)
else
  CMD+=(--headless)
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

echo "metrics:"
ls -1 "$OUT"/json/"${TASK}"_*.json 2>/dev/null || true

if [[ "$WRITE_VIDEO" == "1" ]]; then
  VID=$(ls -1t "$OUT"/videos/"${TASK}"*.mp4 2>/dev/null | head -1 || true)
  if [[ -n "${VID:-}" ]]; then
    echo "video -> $VID"
    if [[ "$PLAY_VIDEO" == "1" ]]; then
      if command -v ffplay >/dev/null; then
        ffplay -autoexit -loglevel error "$VID" || true
      elif command -v vlc >/dev/null; then
        vlc --play-and-exit "$VID" || true
      else
        echo "open the MP4 with any video player"
      fi
    fi
  else
    echo "WARN: no MP4 found under $OUT/videos"
  fi
fi
