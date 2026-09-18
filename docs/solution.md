# b1k — 2026 BEHAVIOR Challenge: Solution & Submission Guide

**Companion docs:** [`challenge-spec.md`](challenge-spec.md) (authoritative requirements) ·
[`design.md`](design.md) (strategy rationale, tradeoffs, risks, 5-week plan) ·
[`../README.md`](../README.md) (repo tour) · [`validation/README.md`](../validation/README.md) (harness detail) ·
[`gpu-simulation.md`](gpu-simulation.md) (step-by-step GPU-box local simulation: run, test, visualize, troubleshooting)

This guide is written for a **new user who clones the repo and wants to**:
1. understand what the challenge is and how this solution attacks it,
2. install and run everything locally (CPU box — no GPU, simulator, or 3.27 TB dataset needed),
3. validate the solution locally (self-test with exact expected outputs),
4. take it to a GPU box for real Q and package the official submission.

---

## 1. The challenge in one page

The **2026 BEHAVIOR Challenge** (Stanford, 2nd edition) is a single-track
embodied-AI benchmark: a policy must solve **100 full-length household tasks** in
the house-scale BEHAVIOR-1K / OmniGibson simulator with the default **R1Pro**
robot, using **only onboard observations** at eval — RGB (head 720×720, wrists
480×480), depth, and 61-dim proprioception. Episodes average ~6 minutes
(27 skills per trajectory); the robot's **global pose is not observable** at
eval, so cross-room search must be self-localized.

**Ranking metric — Q (partial credit):**

```
Q = mean over 100 tasks of ( mean over scored instances of
                              #goal BDDL predicates satisfied / #total goal predicates )
```

A half-done task scores 0.5. Ties are broken by human-normalized efficiency
metrics: `time_score = 3 − 2/normalized_time` (fast → >1, at 1.5× human time → 0)
and normalized base/EEF travel distance. Each task has 40 pre-sampled test
instances (ids 301–340); self-eval/leaderboard reporting uses the first 10
public, 1 rollout each, scored by the organizers' `omnigibson.eval` driver
(v3.9.2) + `score_utils` math.

**Hard constraints:** eval obs are onboard-only (organizers inspect the wrapper
code); the policy must fit a **single 24 GB GPU** (Docker serving, recommended)
or self-host **≥50 ports**; output per-rollout JSONs and videos **must not be
edited**; the deadline is **2026-10-16**. Full detail: `challenge-spec.md`
§1–§12.

**What you can train on:** 20,000 human teleop demos (3.27 TB, LeRobot v3.0,
MIT-licensed, public on HF), per-episode language annotations, BDDL task
definitions (identical train/eval), and task metadata (rooms). Privileged sim
info is allowed **only at training time** (e.g., to build predicate-detector
labels).

---

## 2. Why this solution is intended to win

The design (full reasoning, tradeoffs, and rejected alternatives in
`design.md` §1, §6) is a **hierarchical "System 2 / System 1" agent** built on
the two provided baselines, chosen to exploit the three consequences of the
scoring math:

1. **Partial credit dominates** → a policy that reliably earns *some* predicates
   on *all 100* task families beats one that maxes out 30 and 0-scores 70. So
   the policy is a **single shared, skill-conditioned VLA** (π0.5, OpenPI
   `behavior` fork) fine-tuned **once** on all 20k demos, with per-task LoRA
   adapters reserved for the weakest quartile. Skill-conditioning
   (31-skill taxonomy mined from the demos' language annotations) amortizes
   100 fine-tunes into ~1 training run and shares manipulation skills across
   the dozens of tasks that reuse them.
2. **Robustness is the differentiator** → ~6-minute episodes with goal
   forgetting, stuck loops, and cross-room search. Above the VLA sits a
   deterministic "System 2" brain: a **rule-based planner** (mined per-task
   skill-sequence plans, no LLM at eval) that decomposes the BDDL goal into
   short subgoals, **onboard BDDL predicate detectors** (small vision
   classifiers trained with privileged teacher labels from the demos) feeding a
   **progress state machine** with RETRY / REPLAN / SEARCH / FINISH watchdogs
   (p90 budget caps), and **velocity-integration odometry + occupancy grid +
   room prior** (from legal task metadata) for navigation without global pose.
3. **Efficiency tie-breakers are a free second prize** → the `FINISH`
   watchdog parks the robot the moment all goal predicates are satisfied, so
   `time_score` is computed on the much shorter episode; the occupancy grid
   biases direct routes to cut normalized distance. Both come almost free from
   the progress tracker.

**Everything above the VLA is small, deterministic, and CPU-testable** —
enforced by the import boundary (`src/b1k` never imports `src/training` or
`omnigibson`; CI-checked). That is why this repo can be fully validated on a
laptop while the heavy model work happens on a GPU box.

**Status today:** functional MVP / serving-contract-complete. The VLA backend
defaults to `echo` (replays recorded demo actions, CPU-only) so the *serving
pipeline* is verifiable end-to-end here; the `vla` backend is the documented
integration seam for the fine-tuned π0.5. The self-test below proves the
pipeline, format, edge cases, and exact official scoring math; real Q requires
`--mode sim` on a GPU box (§6).

---

## 3. Architecture

```
                    ┌────────────────────────────────────────────┐
  evaluator         │  src/b1k — SERVING (Docker; 24 GB)         │
  (omnigibson      │                                            │
   v3.9.2)   ──WS──▶  server.py   msgpack protocol, /healthz,   │
                    │                  per-WS reset             │
                    │    │                                      │
                    │    ▼                                      │
                    │  progress.py  CONTINUE/ADVANCE/RETRY/     │
                    │               REPLAN/SEARCH/FINISH        │
                    │    │         (predicate-driven watchdog)  │
                    │    ├──▶ task_planner.py  mined skill plan,│
                    │    │                subgoal queue         │
                    │    ├──▶ odometry.py    pose + occupancy   │
                    │    │                (search frontiers)    │
                    │    ├──▶ rooms.py       RoomPrior (legal   │
                    │    │                task metadata)        │
                    │    └──▶ controller.py  receding horizon   │
                    │                  K=16, chunk blending,    │
                    │                  safety clamps, park      │
                    │                     │                     │
                    │                     ▼                     │
                    │  vla.py  skill-conditioned VLA wrapper    │
                    │           backend: echo | vla | noop      │
                    └────────────────────────────────────────────┘
  ─── never imported at serve time (CI boundary) ────────────────
                    ┌────────────────────────────────────────────┐
                    │  src/training — download · lerobot_io ·    │
                    │  stats · skill_annotations · vla_finetune  │
                    │  · adapters · detector_train               │
                    │  src/evalharness — run_eval · parallel ·   │
                    │  aggregate (exact score_utils math) ·      │
                    │  report · self_eval                        │
                    │  validation — local 7-gate harness         │
                    └────────────────────────────────────────────┘
```

Per eval step (inside `src/b1k`): `msgpack obs → Frame (head/wrist RGB+depth,
61-dim proprio) → PredicateTracker (3-frame hysteresis) → Progress watchdog →
Odometry → ActionController (VLA re-query every 16 steps) → 23-float action`.
Action layout (R1Pro): `[0:3]` base velocity, `[3:7]` trunk, `[7:14]` left arm,
`[14]` left gripper, `[15:22]` right arm, `[22]` right gripper — clamped to the
`r1pro.yaml` output limits, non-finite values suppressed to park.

Key artifacts baked into the repo (no retraining needed to serve):
`docs/plans/<task_id>.json` (mined per-task skill plans),
`data/bddl_goals.json` (goal predicates for the fixture tasks),
`data/demos/<task>/actions.npy` (recorded demo actions for the `echo` backend).

---

## 4. Assumptions

Documented here so a fresh reader can find them; these are the load-bearing
assumptions of the design:

1. **Scoring is per `score_utils.py` v3.9.2, authoritative over the site docs.**
   Every average (Q, task_sr, time_score, distances) divides by the *full*
   instance count, so missing instances/tasks count as 0; overall Q is the mean
   across all 100 tasks. (The site says "10 held out" but the shipped scorer
   defines 20 public + 20 hidden — we code against the scorer and confirm at
   office hours; the self-score slice, instances 0–9, is unaffected.)
2. **Episode-end semantics:** when all predicates are satisfied we park
   (zero base, hold pose, closed grippers) and keep serving no-ops; the
   evaluator either ends the episode early (buying `time_score`) or runs to
   `max_steps` (1.5× human mean). Week-1 verification uses the `noop` backend
   fixture; the fallback is `--max-steps` = 1.45× human mean per task.
3. **BDDL goal text is identical train/eval**, so predicate detectors trained
   on demo ground truth (privileged at train time) are legitimate; at eval the
   detectors consume onboard RGB+depth only.
4. **`observation.state[0:3]` is robot-local base velocity** (July 2026
   convention) — odometry integrates it; the August 2026 velocity-field fix
   means norm stats **must be recomputed** on the re-synced dataset.
5. **π0.5 fits 24 GB** in bf16 with 720²+2×480² inputs at 32-step chunks
   (≈6–7 GB weights, ≈4–6 GB activations, <2 GB detectors/VO) — matches the
   baseline fork's proven `serve_b1k.py` path.
6. **Task metadata is legal at eval** (BDDL, NL descriptions, room lists) —
   used by the planner and `RoomPrior`, not as privileged *state*.
7. **Simulator nondeterminism** is expected; no rollout cherry-picking, 1
   rollout per instance on the fixed index sets.

---

## 5. Installation & self-test (fresh checkout, CPU-only)

**Prereqs:** Python ≥ 3.10, `git`, `curl`. No GPU, no simulator, no dataset
download. The only third-party packages are the small serving deps + pytest.

```bash
git clone <repo-url> behavior_challenge && cd behavior_challenge

# 1) install prerequisites (all CPU, a few seconds)
python3 -m pip install -r requirements.txt          # numpy, msgpack, websockets, PyYAML, pytest

# 2) (optional) editable install so the bare entrypoint / b1k-server alias resolves
python3 -m pip install -e . --no-build-isolation    # adds nothing beyond deps above

# 3) unit tests — EXPECT: "116 passed"
python3 -m pytest tests/ -q

# 4) full validation harness — EXPECT: "== 7 passed, 0 failed, 0 skipped =="
scripts/validate_submission.sh                      # or: python3 validation/run_validation.py
#   --quick variant (skips the live-server gate) — EXPECT: "6 passed, 0 failed, 0 skipped"
scripts/validate_submission.sh --quick

# 5) end-to-end WebSocket check — EXPECT: "WS END-TO-END TEST PASSED"
python3 tests/e2e_websocket.py

# 6) reproducible self-eval (fixture mode) — EXPECT: outputs/<date>/report.md
scripts/self_eval.sh

# 7) smoke-test the serving entrypoint in a second terminal window
PYTHONPATH=src python3 -m b1k.server --config configs/server.yaml --port 8000 --task 0
#    → serves WS on :8000; from another shell:
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/healthz   # EXPECT: 200
#    (equivalently, tests/e2e_websocket.py in step 5 drives a real server process)
```

What the harness gates prove (details: `validation/README.md`):

| # | gate | verifies |
|---|------|----------|
| 1 | unit-tests | 116 tests: protocol round-trip on the 15 MB recorded obs fixture, 31-skill taxonomy, planner miner, progress watchdog, odometry, controller clamps/receding-horizon/park, Q/time_score aggregation, no-privileged-import boundary |
| 2 | fixtures | recorded obs + 23-float action + 5 metrics JSONs + annotations present (regenerated deterministically if missing) |
| 3 | live-server | real `B1KServer` serves 30 steps per fixture task on the recorded obs — every action finite, 23-dim, within `r1pro.yaml` base limits |
| 4 | edge-cases | 11/11: empty/proprio-only obs, wrong-dim & non-finite actions, out-of-bounds instances, unknown task names, malformed folder names — no crashes |
| 5 | sample-data | two seeded sample submissions (2,000 + 15 JSONs) re-score to frozen `expected_*.json` to 1e-12 |
| 6 | scoring | local validator reproduces the **exact official** `score_utils` math (hand-computed case + `time_score = 3 − 2/nt` across 6 values; live parity auto-skips without omnigibson) |
| 7 | report | `validation/validation_report.md` verdict **PASS** |

> **Honest boundary:** the harness verifies the *pipeline, format, and math* —
> not a leaderboard Q. A real Q requires `--mode sim` against the actual
> evaluator on a GPU box (§6). The sample scores (e.g. full sample Q=0.4575)
> come from synthetic seeded metrics and must never be quoted as results.

You can also score any submission folder directly (the CPU stand-in for the
organizer evaluator):

```bash
scripts/validate_submission.sh validation/sample_data/standard.public.b1k.nous.20260910
# EXPECT: VERDICT: PASS + overall Q / task_sr / time_score
```

---

## 6. Real training & evaluation (GPU box)

Everything below needs conda + BEHAVIOR-1K v3.9.2 + an NVIDIA GPU. **For the
complete GPU-box procedure — prerequisites, environment setup, data/config
preparation, running the simulation, tests, visualization, troubleshooting, and
expected outputs — follow [`gpu-simulation.md`](gpu-simulation.md).** The exact
sequence is printed by `scripts/bootstrap.sh` (default mode is safe on any box;
`--full` is **not** — it clones BEHAVIOR-1K, builds the env, and starts the
multi-TB demo download; run it only on the GPU box).

```bash
# 0) base + envs
scripts/bootstrap.sh --full        # clones v3.9.2, runs ./setup.sh, conda 'behavior' env
conda activate behavior

# 1) demos (3.27 TB total — start early; per-task chunks)
PYTHONPATH=$PWD/src python3 -m src.training.download --tasks 0 1 2 --root data/demos
# 2) recompute norm stats (MUST be post-Aug-2026 velocity fix)
PYTHONPATH=$PWD/src python3 -m src.training.stats --data-root data/demos --out data/stats.json
# 3) skill annotations → mine per-task plans → train
PYTHONPATH=$PWD/src python3 -m src.training.skill_annotations --data-root data/demos --tasks 0 1 3
#        (writes data/annotations.jsonl — the miner's input)
PYTHONPATH=$PWD/src python3 -m b1k.planner.mine_plans_cli --annotations data/annotations.jsonl --tasks 0 1 3
#        (writes docs/plans/<task_id>.json; planner falls back to move_to-only if absent)
PYTHONPATH=$PWD/src python3 -m src.training.vla_finetune --out-dir data/checkpoints/pi05_b1k
PYTHONPATH=$PWD/src python3 -m src.training.detector_train --out-dir data/detectors

# 4) serve the REAL model: configs/server.yaml → policy.backend: vla,
#    policy.checkpoint_dir: data/checkpoints/pi05_b1k
PYTHONPATH=src python3 -m b1k.server --config configs/server.yaml --port 8000 --task 0

# 5) drive the real evaluator against our server (per task; writes
#    outputs/<date>/b1k_eval/json/*.json + videos/*.mp4)
PYTHONPATH=src python3 -m evalharness.run_eval --task turning_on_radio --port 8000

# 6) real-Q self-eval: 50-port fan-out + aggregated report
#    (all 100 tasks:  TASKS="$(seq 0 99 | tr '\n' ' ')" scripts/self_eval.sh)
MODE=sim TASKS="0 1 3" scripts/self_eval.sh
#    EXPECT: outputs/<date>/report.md with per-task Q / task_sr / time_score
```

**Milestone path to a winning Q** (`design.md` §7): M1 (week 1) valid
submission artifact with the *baseline* π0.5 checkpoint — proves the whole
pipeline before any training; M2 (week 2) shared VLA + brain, Q≈0.40–0.50;
M3 (week 3) robustness/navigation + adapter batch 1, Q≈0.50–0.55; M4 (week 4)
efficiency pass, 24 GB soak on final-eval-class hardware, Docker build;
M5 (week 5) freeze Oct 12, final self-eval, package, submit before **Oct 16**.
Target Q ≥ 0.55 with clean efficiency margins.

---

## 7. How to submit

The challenge submission is a **served policy + unedited evaluator outputs**,
not a script. Contract (spec §3, §7.1):

**Serving (choose one):**
- **Docker (recommended):** `docker build -t b1k-policy .` then
  `docker run --gpus all -p 8000:8000 -v $(pwd)/data:/workspace/data b1k-policy
  python -m b1k.server --config configs/server.yaml --port 8000 --task 0`.
  Organizers run OmniGibson *outside* the container; cold-start target < 300 s.
- **IP-based:** self-host and expose **≥50 ports** (`evalharness/parallel.py`
  fans out N server workers over `8000..8049`).

**Output package** (folder + files must match the contract exactly — the local
validator enforces the same rules as `score_utils`):

```
<track>.<testset>.<team>.<affiliation>.<date>/
  json/   <task_name>_<instance_id>_<rollout_id>.json     # instance ids 301–340
  + portal link to ALL rollout MP4 videos (≤ 1,000)
  + policy server code (.py) + exact r1pro.yaml + wrapper code (onboard-only proof)
  + eval.camera_sensor_names mapping (head / left_wrist / right_wrist)
```

**Steps:**

```bash
# on the GPU box, after MODE=sim self-eval produced the real metrics + videos
# in outputs/<DATE>/b1k_eval/{json,videos}:
TEAM=b1k AFFIL=nous DATE=$(date +%Y%m%d) scripts/make_submission.sh
# → stages submission/standard.public.b1k.<team>.<affil>.<date>/ with json/,
#   server/ (b1k + r1pro.yaml + server.yaml + Dockerfile), wrapper/
#   (obs_wrapper_audit.py proof), self-eval report, videos_manifest.txt
# → verify the staged folder with the local scorer (must PASS):
scripts/validate_submission.sh submission/standard.public.b1k.nous.$DATE

# zip per the guidelines
cd submission && zip -r standard.public.b1k.nous.$DATE.zip standard.public.b1k.nous.$DATE
```

Then: complete the **participant registration form**, upload the zip + video
portal per the challenge Submission Guidelines, keep the JSONs/videos
**unedited** (organizer requirement), and note that submissions are
confidential unless you opt into disclosure — open-sourcing the repo (configs,
training scripts, this documentation) is separately eligible for the **$1,000
Outstanding Open-Source prize** and is planned for week 5. If any external
API were used (none are — the solution is fully local at eval), credentials +
quota + serving config would have to be provided, at your own cost.

---

## 8. Limitations (honest)

- **No real Q on this box.** The `echo`/`noop` VLA backends verify the serving
  contract; the fine-tuned π0.5 `vla` backend is the integration seam
  (`src/b1k/policy/vla.py`), wired at training time. All local scores are
  fixture/synthetic and are labeled as such in every report.
- **BDDL predicates for fixture tasks are hand-provided**
  (`data/bddl_goals.json`); real-eval BDDL is identical train/eval, but the
  `Progress` early-stop + partial-credit path has not yet been exercised on
  live sim episodes.
- **Detectors are deterministic heuristics** until `detector_train` runs;
  trained classifiers plug in via `policy.detector_bundle_dir`.
- **Navigation** is velocity-integration odometry + coarse occupancy grid; the
  optional depth-VO drift correction (`vo_model_path`) is not shipped.
  Cross-room search is the top design risk (`design.md` §6) and gets its own
  week-3 integration test.
- **Live `score_utils` parity check auto-skips** without omnigibson importable;
  the hand-computed + formula parity gates still enforce the exact math.
- **Dataset logistics** (3.27 TB download, 100×200-episode self-eval) assume
  multi-GPU/cluster time per `design.md` §7 — a single 4090 makes full
  self-eval a multi-day job.
- **Unresolved at submit time:** confirm the 20/20 public/hidden split and
  episode-end semantics at Monday office hours (both are spec-flagged).

## 9. Next steps

1. **GPU box, week 1:** `scripts/bootstrap.sh --full`; confirm WebSocket
   msgpack key strings + episode-end semantics with one baseline-π0.5 fixture
   rollout; start the 3.27 TB chunk download; reach **M1** (valid submission
   artifact on 10 tasks × 10 instances).
2. **Week 2:** skill-conditioned π0.5 finetune + detector training (F1 ≥ 0.85
   per family) + odometry blend gain → **M2** full self-eval, Q≈0.40–0.50.
3. **Week 3:** watchdog tuning, search integration tests, `RoomPrior` wiring,
   adapter batch 1 → **M3** Q≈0.50–0.55 with FINISH engaged.
4. **Week 4:** adapter batch 2, direct-route bias, 24 GB soak on
   final-eval-class hardware, Docker cold-start < 300 s, registration →
   **M4** regression-gated self-eval on the exact submission image.
5. **Week 5:** freeze Oct 12; final self-eval; `make_submission.sh` + portal
   upload; open-source release; submit before **2026-10-16**.

## 10. Document map

| Question | Doc |
|---|---|
| "What exactly must the submission satisfy?" | `docs/challenge-spec.md` (requirements, §11 acceptance checklist) |
| "Why this architecture? What did we reject? What are the risks?" | `docs/design.md` (strategy, tradeoffs, milestones M1–M5) |
| "How do I validate this repo locally?" | this file §5 + `validation/README.md` |
| "How do I run it?" | `README.md` (quick start, serving contract, GPU path) |
| "How do I run, test, and visualize local simulations on a GPU box?" | `docs/gpu-simulation.md` (this file §6 gives the condensed sequence) |
| "What are the 100 tasks?" | `docs/task-ids-100.txt`, `docs/task-descriptions-100.jsonl` |
| "Where is the evidence for the spec claims?" | `raw/` (v3.9.2 evaluator sources, r1pro.yaml, challenge pages) |
