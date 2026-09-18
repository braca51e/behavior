#!/usr/bin/env bash
# Download one (or more) challenge demo chunks into DATA_ROOT using the host `hf`
# CLI — or fall back to this repo's download module. No conda/uv required.
#
#   scripts/docker/download_demos.sh 0
#   scripts/docker/download_demos.sh 0 1 2
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_ROOT="${DATA_ROOT:-$PWD/data/demos}"
mkdir -p "$DATA_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <task_id> [task_id...]" >&2
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  for tid in "$@"; do
    CHUNK=$(printf "chunk-%03d" "$tid")
    echo "+ hf download chunk $CHUNK -> $DATA_ROOT"
    hf download behavior-1k/2026-challenge-demos \
      --repo-type dataset \
      --local-dir "$DATA_ROOT" \
      --include "data/$CHUNK/**" \
      --include "meta/episodes/$CHUNK/**" \
      --include "videos/*/$CHUNK/**" \
      --include "meta/info.json" \
      --include "meta/stats.json" \
      --include "meta/tasks.parquet" \
      --include "meta/tasks.jsonl"
  done
else
  PYTHONPATH="$PWD/src" python3 -m src.training.download --tasks "$@" --root "$DATA_ROOT"
fi
echo "demos -> $DATA_ROOT"
