# Validation report — b1k / 2026 BEHAVIOR challenge

- generated: **2026-09-14 14:34:56** (mode: `full`, 18.26s)
- verdict: **PASS — functional and submission-ready**

| # | gate | status | detail |
|---|---|---|---|
| 1 | unit-tests | PASS | 116 tests passed |
| 2 | fixtures | PASS | recorded obs/action + metrics fixtures present |
| 3 | live-server | PASS | live B1KServer serves finite, in-bounds 23-dim actions on the recorded obs |
| 4 | edge-cases | PASS | 11/11 edge cases handled safely |
| 5 | sample-data | PASS | sample submissions exist and match frozen expected scores to 1e-12 |
| 6 | scoring | PASS | local scoring reproduces the exact official score_utils math |
| 7 | report | PASS | wrote validation_report.md + validation_report.json (verdict PASS) |

## unit-tests

- pytest exit 0
- ........................................................................ [ 62%]
- ............................................                             [100%]
- 116 passed in 7.33s

## fixtures

- obs_payload.bin 15.2 MB; 5 metrics JSONs; action_23d.bin yes

## live-server

- task 0 (turning_on_radio): 30 steps, 0 bad actions
- task 1 (picking_up_trash): 30 steps, 0 bad actions
- task 3 (cleaning_up_plates_and_food): 30 steps, 0 bad actions

## edge-cases

- ok  empty obs -> finite 23-dim
- ok  proprio-only obs -> finite 23-dim
- ok  non-dict binary rejected (raises)
- ok  wrong-dim action rejected
- ok  scalar decode rejected
- ok  base velocity clamped to r1pro output limits
- ok  validator accepts exactly the legal file
- ok  out-of-bounds instance flagged
- ok  unknown-task filename flagged
- ok  single legal rollout scores and passes
- ok  malformed folder name rejected

## sample-data

- standard.public.b1k.nous.20260912: rollouts=2000 Q=0.457500 task_sr=0.1400 time=1.199153 drift=0.00e+00 MATCH
- standard.public.b1k.nous.20260910: rollouts=15 Q=0.004375 task_sr=0.0025 time=0.010826 drift=0.00e+00 MATCH

## scoring

- hand-computed case: Q=0.416667 (want 0.416667), task_sr=0.333333 (want 0.333333), time=0.500000 (want 0.5), taskA q=0.750 (want 0.750), taskB sr=0.500 (want 0.500)
- time_score == official 3 - 2/nt across 6 values
- raw/score_utils not importable (needs omnigibson) -> skipped live parity (hand-computed parity covers it)

> This harness verifies the serving pipeline, output format, edge cases, and the scoring math locally (CPU, no simulator). It is a stand-in for the organizers' evaluator; a real leaderboard Q still requires `--mode sim` on a GPU box with OmniGibson v3.9.2.
