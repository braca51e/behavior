# b1k — 2026 BEHAVIOR Challenge solution (serving side)

A clean, runnable repository implementing the winning-solution design in
`docs/design.md` for the [2026 Stanford BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html).

The challenge: solve 100 full-length household tasks in the BEHAVIOR-1K /
OmniGibson simulator with an R1Pro robot, using **only onboard RGB + depth +
proprioception** at eval. The ranked metric is **Q** = mean over 100 tasks of
the BDDL goal-predicate satisfaction fraction (partial credit).

This repo contains the **serving system** (the WebSocket policy server + the
"System 2" brain: planner, progress watchdog, odometry, predicate detectors,
VLA wrapper + controller), the **quarantined training pipeline** (download,
norm stats, skill-plan mining, VLA/detector/adapter training), and the
**eval harness** (Q aggregation with the exact leaderboard math + report).
Everything above the big model is deterministic and **CPU-testable without a
GPU, simulator, or the 3.27 TB demo set** — that is the design's
`server → controller → {planner, perception, policy}` dependency rule.

> Status: **functional MVP / serving-contract-complete.** The VLA backend
> defaults to `echo` (replays recorded demo actions) so the full serving
> pipeline is verifiable end-to-end here. Switch to the real fine-tuned pi0.5
> (`policy.backend: vla`) on a GPU box for real Q — see `docs/design.md` for
> the 5-week plan to get there.
>
> **New here? Read `docs/solution.md`** — the full guide: challenge summary,
> why this solution is intended to win, architecture, assumptions, install +
> self-test with expected outputs, GPU path, and the submission walkthrough.
> **Running local simulations on a GPU box? Read
> `docs/gpu-simulation.md`** — prerequisites, environment setup, data/config
> preparation, run + test + visualize commands, troubleshooting, and expected
> outputs.
> **Want a single task on GPU with a video? Read
> `docs/visual-pilot-step-by-step.md`** — full step-by-step from driver check
> through `behavior392` install, `echo` policy serve, OmniGibson eval, and
> watching the MP4 (`scripts/run_visual_pilot.sh`).
> **Train π0.5 / GR00T without host conda/uv? Read
> `docs/docker-training.md`** — Docker images that bake the official baseline
> stacks (`docker/pi05`, `docker/groot`) and how to train + serve + eval.
> `docs/challenge-spec.md` is the authoritative requirements spec;
> `docs/design.md` is the strategy rationale.

---

## Self-test on a fresh checkout (CPU-only — do this first)

No GPU, simulator, or dataset download required:

```bash
# 1) install prerequisites
python3 -m pip install -r requirements.txt

# 2) unit tests — EXPECT: "115 passed"
python3 -m pytest tests/ -q

# 3) full validation harness — EXPECT: "== 7 passed, 0 failed, 0 skipped =="
scripts/validate_submission.sh            # or: python3 validation/run_validation.py

# 4) end-to-end WebSocket check — EXPECT: "WS END-TO-END TEST PASSED"
python3 tests/e2e_websocket.py
```

What the 7 harness gates check (see `validation/README.md`): the 115 unit
tests, the recorded fixtures, a live `B1KServer` serving finite in-bounds
23-dim actions on the recorded obs, 11 edge cases, sample submissions
re-scoring to frozen expected outputs (1e-12), and exact parity with the
official `score_utils` scoring math. The verdict is written to
`validation/validation_report.md`. This verifies the **pipeline, format, and
math** — a real leaderboard Q still requires `--mode sim` on a
BEHAVIOR-1K v3.9.2 + GPU box (below).

---

## Layout

```
behavior_challenge/
├── src/
│   ├── b1k/                 # SERVING (imported at eval time; no privileged deps)
│   │   ├── server.py        #   WebSocket policy server  (python -m b1k.server)
│   │   ├── protocol.py      #   obs parsing + msgpack encode/decode (hard contract)
│   │   ├── controller.py    #   action-chunk cache, receding horizon, safety clamps
│   │   ├── config.py        #   dataclass runtime config (configs/server.yaml)
│   │   ├── embodiment.py    #   R1Pro constants: 23-dim action, 61-dim proprio, limits
│   │   ├── planner/         #   skill_tax (31 skills) · miner · task_planner · progress
│   │   ├── perception/      #   odometry · rooms (room prior) · detectors/ (labels·pred_models·state)
│   │   └── policy/          #   vla (skill-conditioned; echo/vla/noop backends) · norm · chunking
│   ├── training/            # TRAINING-ONLY (privileged; NEVER imported by b1k — CI-checked)
│   │   └── download · lerobot_io · stats · skill_annotations · vla_finetune · adapters · detector_train
│   └── evalharness/         # run_eval · parallel · aggregate (Q math) · report · self_eval
├── configs/
│   ├── r1pro.yaml           # verbatim robot config (do not modify)
│   ├── server.yaml          # serving config (MVP: policy.backend = echo)
│   └── training.yaml        # training config
├── docs/                    # challenge-spec · design · solution (guide) · task ids/descriptions · plans/
├── tests/                   # 115 unit tests + fixtures + e2e WebSocket check
├── validation/              # local validation harness: sample submissions + expected
│                            #   outputs + format/edge-case/scoring gates (CPU-only)
├── scripts/                 # self_eval.sh · validate_submission.sh · bootstrap.sh
│                            #   · make_submission.sh · obs_wrapper_audit.py
├── data/                    # (gitignored large) bddl_goals.json · human_stats.jsonl · demos/<t>/actions.npy
├── outputs/                 # (gitignored) self-eval reports + videos
├── raw/                     # captured challenge evidence (evaluator v3.9.2 source, r1pro.yaml, misc csv)
├── Dockerfile               # serving image (organizer runs OmniGibson outside, connects via WS)
├── pyproject.toml           # src-layout package (editable install)
└── requirements.txt
```

**Import rule (enforced by `tests/test_imports.py` + CI):** nothing under `src/b1k`
imports from `src/training` or `omnigibson`. No privileged path can leak into
serving — by construction.

---

## Quick start (this machine — CPU, no GPU/simulator/dataset)

```bash
# 1. (optional) editable install so the documented entrypoint resolves bare
python3 -m pip install -e . --no-build-isolation --no-deps

# 2. Regenerate fixtures + run the full local check (tests + self-eval report)
scripts/self_eval.sh

# 3. Run the policy server (MVP entrypoint from the design)
PYTHONPATH=src python3 -m b1k.server --config configs/server.yaml --port 8000 --task 0
#    → serves WS on :8000, answers GET /healthz with 200.

# 4. Prove the obs wrapper is onboard-only (challenge inspection requirement)
PYTHONPATH=src python3 scripts/obs_wrapper_audit.py
```

`scripts/self_eval.sh` produces `outputs/<date>/report.md`:
- the **serving-pipeline verification** (live `B1KServer` per fixture task on the
  recorded obs fixture → every step a finite 23-dim action; clearly labeled, *not*
  a leaderboard Q), and
- the **scoring math** run on the 5 fixture metrics JSONs with the exact
  `score_utils` formula (Q, task_sr, time_score).

Real Q requires `--mode sim` on an OmniGibson v3.9.2 + GPU box (see below).

---

## The serving contract (what the evaluator sees)

Mirrors `omnigibson.eval.policies.WebsocketPolicy` (v3.9.2):

| Direction | Wire format |
|---|---|
| readiness | `GET /healthz` → HTTP 200 |
| evaluator → server | one **binary** frame per step: msgpack `dict` of the *flattened* obs (keys joined by `::`; head RGB 720², wrists 480², depths, 61-dim proprio) |
| server → evaluator | one **binary** frame per step: msgpack `{"action": [23 floats]}` (R1Pro `action_dim`) |

Action layout (23-dim): `[0:3]` base velocity, `[3:7]` trunk, `[7:14]` left arm,
`[14]` left gripper, `[15:22]` right arm, `[22]` right gripper. Base velocity is
clamped to the `r1pro.yaml` output limits (±0.75 x-y, ±1.0 yaw); gripper commands
pass a deadband; non-finite actions are suppressed to park. See
`src/b1k/embodiment.py` for the full slice tables (copied verbatim from the
challenge's canonical sources in `raw/`).

Per-step pipeline (`src/b1k/server.py::handle_frame`):
`Frame → PredicateTracker (3-frame hysteresis) → Progress watchdog
(CONTINUE/ADVANCE/RETRY/REPLAN/SEARCH/FINISH) → Odometry → ActionController
(receding horizon K=16, chunk blending) → 23-dim action`. `FINISH` parks the
robot the moment all BDDL goal predicates are satisfied (the efficiency
tie-breaker engine).

---

## Real training / evaluation (GPU box)

The heavy path is fully scaffolded in `src/training` and `src/evalharness`; it
needs conda + the BEHAVIOR-1K v3.9.2 clone + the demo dataset. `scripts/bootstrap.sh`
prints the exact sequence:

```bash
# clone + envs (BEHAVIOR-1K v3.9.2)
git clone -b v3.9.2 https://github.com/StanfordVL/BEHAVIOR-1K.git /path/to/B1K
cd /path/to/B1K && ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
conda activate behavior

# 1) per-task demo download (3.27 TB total; start early)
PYTHONPATH=$PWD/src python3 -m src.training.download --tasks 0 1 2 --root data/demos
# 2) recompute norm stats (post Aug-2026 velocity fix)
PYTHONPATH=$PWD/src python3 -m src.training.stats --data-root data/demos --out data/stats.json
# 3) mine skill plans (demos → docs/plans/<task_id>.json)
PYTHONPATH=$PWD/src python3 -m src.training.skill_annotations --data-root data/demos --tasks 0
PYTHONPATH=$PWD/src python3 -m src.training.vla_finetune --out-dir data/checkpoints/pi05_b1k
PYTHONPATH=$PWD/src python3 -m src.training.detector_train --out-dir data/detectors
# 4) serve the REAL model
#    configs/server.yaml: policy.backend: vla, policy.checkpoint_dir: data/checkpoints/pi05_b1k
PYTHONPATH=$PWD/src python3 -m b1k.server --config configs/server.yaml --port 8000 --task 0
# 5) drive the real evaluator over our server
PYTHONPATH=$PWD/src python3 -m evalharness.run_eval --task turning_on_radio --port 8000
# 6) real-Q self-eval (50-port fan-out + report)
MODE=sim TASKS="0 1 3" scripts/self_eval.sh
# 7) package the submission zip
TEAM=b1k scripts/make_submission.sh
```

Docker serving (organizer-recommended): `docker build -t b1k-policy .` then run
with `--gpus all -p 8000:8000 -v $(pwd)/data:/workspace/data b1k-policy`.

---

## Tests

```bash
python3 -m pytest tests/ -q          # 115 tests, CPU-only
python3 tests/e2e_websocket.py       # real `python -m b1k.server` over a live WS
```

Coverage: protocol msgpack round-trip + the 15 MB recorded obs fixture
(`tests/fixtures/obs_payload.bin`), 31-skill taxonomy, skill-plan miner (on the
synthetic annotation fixture), progress watchdog state machine (ADVANCE/RETRY/
REPLAN/SEARCH/FINISH + budgets), odometry vs synthetic ground truth, controller
(clamps/deadband/receding/blend/park), Q/time_score aggregation (exact
`score_utils` formula on the 5 metrics fixtures), and the no-privileged-import
boundary. Fixtures are regenerated deterministically by
`tests/fixtures/make_fixtures.py`.

## Local validation harness (submission-ready check)

`validation/` is a CPU-only harness that verifies the solution is **functional
and submission-ready without relying on external challenge testing** — the
local stand-in for the organizers' evaluator + `score_utils` scoring. It checks
output format (submission folder name, `<task>_<instance>_<rollout>.json`
naming, per-rollout metrics schema), edge cases (empty/proprio-only obs,
wrong-dim/non-finite actions, out-of-bounds instances, unknown task names,
malformed folders), and the **exact official scoring math** (Q / task_sr /
time_score) against frozen sample datasets. See `validation/README.md`.

```bash
scripts/validate_submission.sh                    # full harness -> validation/validation_report.md
scripts/validate_submission.sh <submission-folder> # score one folder (format + official Q)
python3 -m pytest tests/test_validation.py -q     # CI gate
```

It ships two seeded sample submissions (2,000 + 15 metrics JSONs) with frozen
`expected_*.json` aggregate scores; the harness re-scores them and fails if the
math drifts by more than 1e-12. Like the self-eval fixture mode, the harness
verifies the *pipeline and the math* — it does not claim a leaderboard Q, which
still requires `--mode sim` on an OmniGibson v3.9.2 + GPU box.

## Limitations (honest)

- The `echo`/`noop` VLA backends make the **serving contract** verifiable here,
  but are not a policy that scores Q — the real pi0.5 `vla` backend is the
  integration seam (OpenPI 'behavior' fork), to be wired at training time.
- BDDL goal predicates for the fixture tasks are hand-provided
  (`data/bddl_goals.json`); at eval the real BDDL is identical train/eval, so the
  `Progress` early-stop + partial-credit logic is what earns Q.
- Detector backends default to deterministic heuristics; trained classifiers
  (`data/detectors/*.pt`) plug in via `policy.detector_bundle_dir` once
  `detector_train` has run.
- Cross-room search uses velocity-integration odometry + a coarse occupancy
  grid (VO model optional). The week-1 risk item (episode-end semantics for a
  parked robot) is addressed by the `FINISH`→park path and the no-op fixture.

See `docs/design.md` (strategy, tradeoffs, risks, 5-week plan) and
`docs/challenge-spec.md` (authoritative requirements) for the full reasoning.
