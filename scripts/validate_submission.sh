#!/usr/bin/env bash
# Local validation entrypoint for the b1k 2026 BEHAVIOR solution.
#
# Two usages:
#   1) Full harness (default) — runs the unit tests, fixtures, the live-server
#      check, edge cases, sample-data round-trip, the scoring-parity gate, and
#      writes validation/validation_report.md + .json.  CPU-only, no simulator.
#        scripts/validate_submission.sh
#        scripts/validate_submission.sh --quick
#   2) Score an existing sample / submission folder (format + exact official Q):
#        scripts/validate_submission.sh validation/sample_data/standard.public.b1k.nous.20260910
#        scripts/validate_submission.sh /path/to/standard.public.b1k.nous.20260912
#
# The harness is the local stand-in for the organizers' evaluator + score_utils
# scoring: it verifies the solution is functional and submission-ready without
# relying on external challenge testing.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}:${REPO}"

# If a submission-folder path was given, score it directly and exit.
if [ "$#" -ge 1 ]; then
    SUB="$1"
    python3 - "$SUB" <<'PY'
import sys, json
from pathlib import Path
import validation.validator as v
sub = Path(sys.argv[1])
if not sub.exists():
    print(f"FAIL: {sub} does not exist", file=sys.stderr)
    sys.exit(2)
res = v.validate_submission(sub)
print(f"folder: {res.folder}  (track={res.track} testset={res.testset})")
print(f"rollouts scored: {res.n_rollouts}")
print(f"overall Q:       {res.overall_q:.6f}")
print(f"task_sr:         {res.overall_task_sr:.6f}")
print(f"time_score:      {res.overall_time_score:.6f}")
if res.warnings:
    print(f"warnings ({len(res.warnings)}):")
    for w in res.warnings[:12]:
        print("   -", w)
if res.errors:
    print(f"ERRORS ({len(res.errors)}):")
    for e in res.errors[:20]:
        print("   !", e)
print("VERDICT:", "PASS" if res.passed else "FAIL")
sys.exit(0 if res.passed else 1)
PY
    exit $?
fi

# Otherwise: run the full harness.
exec python3 validation/run_validation.py "$@"
