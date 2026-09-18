#!/usr/bin/env bash
# Package the submission zip per the challenge contract (spec section 3 / 7.1).
#
# Produces submission/ with:
#   <track>.<testset>.<team>.<affiliation>.<date>/
#     json/    <task>_<instance>_<rollout>.json   (the evaluator's outputs — do NOT edit)
#     server/  our policy server code (b1k package + configs + r1pro.yaml)
#     wrapper/ the eval wrapper proof (obs restrictions honored)
#     Dockerfile, self-eval report
#   + a portal-upload manifest for the <=1000 rollout MP4 videos.
#
# The JSON + videos are produced by the REAL evaluator (self_eval --mode sim) on
# a GPU box; this script assembles the rest.  Set TEAM/DATE/TESTSET env vars.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

TRACK="${TRACK:-standard}"
TESTSET="${TESTSET:-public}"
TEAM="${TEAM:-b1k}"
AFFIL="${AFFIL:-nous}"
DATE="${DATE:-$(date +%Y%m%d)}"
SUB="submission/${TRACK}.${TESTSET}.${TEAM}.${AFFIL}.${DATE}"
JSON_SRC="${JSON_SRC:-outputs/${DATE}/b1k_eval/json}"

mkdir -p "$SUB/json" "$SUB/server" "$SUB/wrapper"

# 1. Evaluator metrics (must be the real, unedited per-rollout JSONs).
if [ -d "$JSON_SRC" ]; then
    cp -r "$JSON_SRC"/. "$SUB/json/"
    echo "copied $(ls "$SUB/json" | wc -l) metrics JSONs"
else
    echo "WARN: no metrics at $JSON_SRC — json/ left empty (run self_eval --mode sim first)"
fi

# 2. Server code (the WebSocket policy server — .py) + robot config + wrapper.
cp -r src/b1k "$SUB/server/"
cp configs/r1pro.yaml "$SUB/server/r1pro.yaml"
cp configs/server.yaml "$SUB/server/server.yaml"
cp Dockerfile "$SUB/server/Dockerfile"
# Wrapper proof: the obs-restriction audit (which keys the server consumes).
cp scripts/obs_wrapper_audit.py "$SUB/wrapper/" 2>/dev/null || true

# 3. Self-eval report (reproducibility evidence).
REPORT=$(ls -1d outputs/*/report.md 2>/dev/null | sort | tail -1 || true)
if [ -n "${REPORT:-}" ]; then
    cp "$REPORT" "$SUB/self_eval_report.md"
fi

# 4. Video portal manifest (one line per rollout MP4: path + url placeholder).
{
  echo "# video portal manifest: <local mp4> <portal url>"
  if [ -d "${JSON_SRC%/json}/videos" ]; then
    find "${JSON_SRC%/json}/videos" -name '*.mp4' | sort
  fi
} > "$SUB/videos_manifest.txt"

echo
echo "submission staged at: $SUB"
echo "next: zip it and upload per the challenge submission guidelines."
echo "  cd submission && zip -r ${TRACK}.${TESTSET}.${TEAM}.${AFFIL}.${DATE}.zip ${TRACK}.${TESTSET}.${TEAM}.${AFFIL}.${DATE}"
