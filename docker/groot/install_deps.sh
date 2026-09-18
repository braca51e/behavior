#!/bin/sh
# Install Isaac-GR00T deps with retries (pypi.nvidia.com timeouts are common).
set -eu

ok=0
attempt=1
while [ "$attempt" -le 6 ]; do
  echo "uv sync attempt ${attempt}/6"
  if uv sync --frozen --python 3.10; then
    ok=1
    break
  fi
  echo "uv sync failed (attempt ${attempt}); sleeping before retry..."
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done

if [ "$ok" -ne 1 ]; then
  echo "uv sync failed after 6 attempts" >&2
  exit 1
fi

uv pip install --python .venv/bin/python websockets huggingface_hub
