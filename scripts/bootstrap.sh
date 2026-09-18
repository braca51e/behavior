#!/usr/bin/env bash
# Bootstrap a full training/eval box (design.md week 1).  This machine is a
# *serving/verification* box (CPU, no simulator), so by default bootstrap only
# verifies the serving deps and prints the training/eval commands without
# running the heavy steps.  Pass --full to actually clone BEHAVIOR-1K v3.9.2
# and start the (multi-TB) demo download.
#
#   scripts/bootstrap.sh             # check serving deps (safe, fast)
#   scripts/bootstrap.sh --full      # clone v3.9.2 + conda envs + start download
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

echo "== serving dependencies =="
python3 -c "import numpy, msgpack, websockets, yaml; print('serving deps OK (numpy %s, msgpack %s, websockets %s)' % (numpy.__version__, msgpack.__version__, websockets.__version__))"
python3 -m pytest tests/ -q
echo "unit tests passed"

if [ "${1:-}" != "--full" ]; then
    cat <<'NOTE'

Serving box is ready (this is the CPU verification path).

Full training/eval bootstrap (run on a GPU box with conda):
  git clone -b v3.9.2 https://github.com/StanfordVL/BEHAVIOR-1K.git /path/to/B1K
  cd /path/to/B1K && ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
  conda activate behavior
  # demo download (per-task chunks; 3.27 TB total — start early):
  PYTHONPATH=$REPO/src python3 -m src.training.download --tasks 0 1 2 --root data/demos
  # then train (week 2):
  PYTHONPATH=$REPO/src python3 -m src.training.vla_finetune --out-dir data/checkpoints/pi05_b1k
  PYTHONPATH=$REPO/src python3 -m src.training.detector_train --out-dir data/detectors
  # and serve with the real model:
  #   configs/server.yaml: policy.backend: vla, policy.checkpoint_dir: data/checkpoints/pi05_b1k
NOTE
    exit 0
fi

echo "== full bootstrap (GPU box) =="
B1K="${BEHAVIOR_1K:-/path/to/B1K}"
git clone -b v3.9.2 https://github.com/StanfordVL/BEHAVIOR-1K.git "$B1K"
( cd "$B1K" && ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval )
PYTHONPATH="$REPO/src" python3 -m src.training.download --root data/demos
echo "bootstrap complete; activate the 'behavior' conda env and run self-eval --mode sim."
