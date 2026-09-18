#!/usr/bin/env bash
# Download one (or more) challenge demo chunks into DATA_ROOT.
# Prefers the host `hf` CLI; otherwise uses Python huggingface_hub (no CLI).
#
#   scripts/docker/download_demos.sh 0
#   scripts/docker/download_demos.sh 0 1 2
#
# Optional: export HF_TOKEN=... if the dataset requires auth.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_ROOT="${DATA_ROOT:-$PWD/data/demos}"
mkdir -p "$DATA_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <task_id> [task_id...]" >&2
  exit 1
fi

# Always go through the Python module — it uses `hf` when on PATH, else
# huggingface_hub.snapshot_download (pip install huggingface_hub).
if ! PYTHONPATH="$PWD/src" python3 -m src.training.download --tasks "$@" --root "$DATA_ROOT"; then
  echo "FAIL: demo download failed." >&2
  echo "  pip install -U 'huggingface_hub[cli]'   # then re-run" >&2
  echo "  # or use Docker:" >&2
  echo "  docker run --rm -e HF_TOKEN -v \"$DATA_ROOT:/data/demos\" \\" >&2
  echo "    b1k-groot shell -c 'hf download behavior-1k/2026-challenge-demos --repo-type dataset --local-dir /data/demos --include \"data/chunk-000/**\" --include \"meta/episodes/chunk-000/**\" --include \"videos/*/chunk-000/**\" --include meta/info.json --include meta/stats.json --include meta/tasks.parquet --include meta/tasks.jsonl'" >&2
  exit 1
fi
