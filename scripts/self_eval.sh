#!/usr/bin/env bash
# Reproducible self-eval (design.md contract: scripts/self_eval.sh ->
# outputs/<date>/report.md).
#
# Steps:
#   1. Run the local unit test suite (must pass) — verifies protocol, planner,
#      progress, odometry, controller, scoring, and the no-privileged-import
#      boundary.  CPU-only, no simulator.
#   2. Generate the fixtures if missing (deterministic, seeded).
#   3. Run the self-eval (fixture mode by default) -> outputs/<date>/report.md.
#
# Usage:
#   scripts/self_eval.sh                 # fixture mode (CPU)
#   MODE=sim TASKS="0 1 3" scripts/self_eval.sh   # real sim (needs OmniGibson+GPU)
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${MODE:-fixture}"
TASKS="${TASKS:-}"
OUT="${OUT:-}"

echo "== [1/3] unit tests =="
python3 -m pytest tests/ -q

echo "== [2/3] fixtures =="
python3 tests/fixtures/make_fixtures.py

echo "== [3/3] self-eval (mode=$MODE) =="
ARGS=( --mode "$MODE" )
if [ -n "$TASKS" ]; then ARGS+=( --tasks $TASKS ); fi
if [ -n "$OUT" ]; then ARGS+=( --out "$OUT" ); fi
python3 -m evalharness.self_eval "${ARGS[@]}"

REPORT=$(ls -1d outputs/*/report.md 2>/dev/null | sort | tail -1)
echo
echo "DONE. Report: ${REPORT:-outputs/<date>/report.md}"
