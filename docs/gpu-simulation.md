# Running local simulations on a GPU box (b1k)

**Companion docs:** [`solution.md`](solution.md) (full solution & submission guide —
read §1–§5 first) · [`visual-pilot-step-by-step.md`](visual-pilot-step-by-step.md)
(end-to-end one-task visual pilot with `echo` + MP4) ·
[`docker-training.md`](docker-training.md) (π0.5 / GR00T train+serve Docker,
no host conda/uv) ·
[`gpu-commands-verified.md`](gpu-commands-verified.md)
(2026-09-14 verified-command notes with captured outputs) ·
[`../README.md`](../README.md) (repo tour) ·
[`challenge-spec.md`](challenge-spec.md) (authoritative requirements)

This guide gets you from a fresh GPU machine to **real simulation runs**: driving
the official BEHAVIOR-1K v3.9.2 OmniGibson evaluator against our WebSocket policy
server, scoring Q, and watching the results. It covers prerequisites,
environment setup, data/config preparation, running the simulation, running
tests, visualizing results, troubleshooting, and validating expected outputs.
Commands are imperative; every step states what you should see.

## 1. Overview

"Local simulation" in this repo means running the **official challenge
evaluator** on your own GPU box — not on the organizers' infrastructure:

```
sim mode  =  omnigibson.eval.eval (BEHAVIOR-1K v3.9.2, owns the simulator,
             needs the NVIDIA GPU)  ──WS──▶  our policy server (python -m b1k.server)
```

Two ways to serve during a sim run, both documented below:

- **One server per task** (pilot): start `python -m b1k.server --port 8000
  --task <name>` yourself, then drive one task with
  `python -m evalharness.run_eval --task <name> --port 8000`.
- **50-port fleet** (full self-eval): `MODE=sim scripts/self_eval.sh` spawns up
  to 50 task-pinned server workers (`evalharness/parallel.py`), waits for
  `/healthz`, runs the evaluator per task, aggregates, and writes the report.

GPU usage: the **simulator** (OmniGibson) needs the GPU; **training**
(`vla_finetune`, `detector_train`) needs a CUDA torch build; **VLA inference**
(`policy.backend: vla`) uses `policy.device` — `auto` resolves to `cuda` when
the torch build has CUDA, else `cpu`. The `echo`/`noop` backends need no GPU at
all. Serving, unit tests, e2e WebSocket checks, and the fixture-mode self-eval
are CPU-verifiable — run the tests below even before any simulator work.

**Status boundary (honest):** the fine-tuned π0.5 checkpoint and the demo
dataset are produced by this guide's training/data steps (week 1–2 of
`solution.md` §9). For the first pipeline check (milestone M1) you can serve
the **baseline** π0.5 checkpoint instead of training. Everything this guide
produces before that point is a *pipeline verification*, not a leaderboard Q.

---

## 2. Prerequisites

### 2.1 Hardware

| Requirement | Why |
|---|---|
| Single NVIDIA datacenter GPU, **≥ 24 GB VRAM** | The policy (π0.5 in bf16: ≈6–7 GB weights, ≈4–6 GB activations, < 2 GB detectors) must coexist with the OmniGibson sim on one GPU — the challenge's hard constraint (`solution.md` §1, §4.5) |
| ≥ 500 GB free disk for a pilot; **3.27 TB** for all 100 task chunks | Demo dataset `behavior-1k/2026-challenge-demos` (LeRobot v3.0), plus code + checkpoints |
| Headless-compatible (no display needed) | Evaluator runs with `--headless` (the default in `run_eval.py`) |
| Final-Q hardware = eval-class dGPU (e.g. RTX 4090) | The challenge design targets one 24 GB dGPU; unified-memory boards (Jetson Orin) do **not** meet the VRAM budget — use the eval hardware for final Q runs |

### 2.2 Software

- Ubuntu 20.04+/22.04 (or the distro your BEHAVIOR-1K v3.9.2 `setup.sh` supports)
- NVIDIA driver with CUDA 12 support (driver ≥ 525 for CUDA 12.1; ≥ 535 for
  12.4+). Reference: driver 540.4.0 / CUDA 12.6.
- conda (Miniconda3 or later)
- Python ≥ 3.10, `git`, `curl`, `zip`
- `ffplay` or any video player, for watching rollouts (optional but part of
  the visual loop)

### 2.3 Verify the box

```bash
nvidia-smi
# EXPECT: your dGPU + a "CUDA Version" ≥ 12.x matching the torch build you'll use

python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# EXPECT: a CUDA build, e.g. "2.3.1+cu121 12.1 True"
#        (torch.version.cuda=None or is_available()=False means a CPU-only build — see §8, row 2)
```

The driver's CUDA version must be **≥** the CUDA version of your torch build.

---

## 3. Environment setup

```bash
# 1) get the repo
git clone <repo-url> behavior_challenge && cd behavior_challenge

# 2) install serving deps (numpy, msgpack, websockets, PyYAML, pytest)
python3 -m pip install -r requirements.txt

# 3) (recommended) editable install so `python -m b1k.server` resolves bare
python3 -m pip install -e . --no-build-isolation --no-deps

# 4) baseline gate on the GPU box — EXPECT: "116 passed"
python3 -m pytest tests/ -q
#        If this fails, do NOT proceed to simulator work (see §7, §8).
```

```bash
# 5) BEHAVIOR-1K v3.9.2 + the 'behavior' conda env (OmniGibson, BDDL, Joylo,
#    dataset, eval) — long; the env must end with a CUDA torch build
git clone -b v3.9.2 https://github.com/StanfordVL/BEHAVIOR-1K.git /path/to/B1K
cd /path/to/B1K && ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
conda activate behavior

# 6) confirm CUDA torch inside the env — EXPECT: <ver> <cuda> True
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
#    If the env's torch is CPU-only, install the CUDA build that matches your
#    driver (e.g. 12.x):
python3 -m pip install --upgrade "torch>=2.1" --index-url https://download.pytorch.org/whl/cu121

# 7) Hugging Face CLI for the demo download (data step)
python3 -m pip install -U "huggingface_hub[cli]"
hf --help > /dev/null                # EXPECT: usage text (huggingface-cli is deprecated)
```

One-shot alternative: `scripts/bootstrap.sh --full` clones BEHAVIOR-1K v3.9.2,
runs `./setup.sh`, and starts the **full** demo download. Set
`BEHAVIOR_1K=/path/to/B1K` to control the clone path. Note its blast radius:
`--full` is for the GPU box only — it will consume hundreds of GB to
multi-TB of disk. The default `scripts/bootstrap.sh` (no flag) only verifies
serving deps + runs the unit tests and is safe anywhere.

```bash
# 8) sanity: serving deps OK + unit tests pass + prints the GPU-box sequence — EXPECT: exit 0
scripts/bootstrap.sh
```

---

## 4. Data & config preparation

### 4.1 Download the demos (3.27 TB total — start early)

The dataset is `behavior-1k/2026-challenge-demos`, sharded per task into
`chunk-000` … `chunk-099` (chunk = task id). Each task chunk pulls
`data/<chunk>/**`, `meta/episodes/<chunk>/**`, `videos/*/<chunk>/**`, plus the
shared `meta/*` files.

```bash
# preview the exact huggingface-cli argv per task — EXPECT: one "+ huggingface-cli …"
# line per task, touching data/, meta/episodes/, videos/
PYTHONPATH=$PWD/src python3 -m src.training.download --tasks 0 1 2 --root data/demos --dry-run

# pilot download (3 tasks, ~100 GB)
PYTHONPATH=$PWD/src python3 -m src.training.download --tasks 0 1 2 --root data/demos
#        EXPECT: huggingface-cli progress per chunk; ends 0

# full 100-task download (3.27 TB; omit --tasks — defaults to all 100)
PYTHONPATH=$PWD/src python3 -m src.training.download --root data/demos
#        EXPECT: ~100 sequential chunk downloads; hours-to-days of network time

# authenticate only if the repo or rate limits require it:
export HF_TOKEN=<your-token>

ls data/demos
# EXPECT: chunk-000/ … (data/, meta/, videos/ layout under data/demos/)
```

### 4.2 Compute norm stats

Norm stats must be the **post-Aug-2026 velocity-fix** ones — do not reuse
pre-fix stats.

```bash
PYTHONPATH=$PWD/src python3 -m src.training.stats --data-root data/demos --out data/stats.json
#        EXPECT: "stats from tasks […]" then data/stats.json written
#        (defaults to auto-detecting present data/chunk-NNN dirs; pass
#         --tasks 0 1 to restrict explicitly)
```

### 4.3 Train the model (pilot: one task; week 2: full)

```bash
# extract skill annotations from the demos (flat JSONL, one line per episode
# segment — the miner's input format; pilot set = fixture tasks 0, 1, 3)
PYTHONPATH=$PWD/src python3 -m src.training.skill_annotations --data-root data/demos --tasks 0 1 3
#        EXPECT: data/annotations.jsonl written ("wrote N annotation lines")

# mine per-task skill plans → docs/plans/<task_id>.json
PYTHONPATH=$PWD/src python3 -m b1k.planner.mine_plans_cli --annotations data/annotations.jsonl --tasks 0 1 3
#        EXPECT: "plans -> 0.json, 1.json, 3.json" in docs/plans/ (task names
#        come from docs/task-descriptions-100.jsonl; the planner falls back
#        to a move_to-only plan when a task's file is absent)

# fine-tune the skill-conditioned VLA → the checkpoint the server serves
PYTHONPATH=$PWD/src python3 -m src.training.vla_finetune --out-dir data/checkpoints/pi05_b1k
#        EXPECT: checkpoint files under data/checkpoints/pi05_b1k/

# train the onboard predicate detectors → plugged in via policy.detector_bundle_dir
PYTHONPATH=$PWD/src python3 -m src.training.detector_train --out-dir data/detectors
```

M1 alternative (`solution.md` §9): point `policy.checkpoint_dir` at the
**baseline** π0.5 checkpoint to prove the whole pipeline before any training.

### 4.4 Configure the server for the real model

Edit `configs/server.yaml` (the only keys that change for a sim run):

```yaml
policy:
  backend: vla                          # was: echo (MVP)
  checkpoint_dir: data/checkpoints/pi05_b1k   # was: null
  device: cuda                          # was: auto — auto also resolves to cuda
                                        # on a CUDA torch build; set it explicitly
                                        # so a silent CPU fallback is impossible
  # norm_stats_path: data/stats.json    (already correct)
  # demos_root: data/demos             (already correct)
  # detector_bundle_dir: data/detectors (set after §4.3 detectors are trained)
```

Keep `configs/r1pro.yaml` **verbatim** — the submission contract ships the
exact robot config, and the local validator checks action bounds against it.

---

## 5. Run the simulation

Do the CPU gate first, then pilot one task, then the full self-eval.

```bash
# 0) serving-pipeline gate (CPU-only, no simulator) — EXPECT: report + "DONE."
scripts/self_eval.sh
#        → tasks 0/1/3: 30/30 steps finite, 23-dim → OK
#        → fixture scoring: Q=0.6000 task_sr=0.4000 time_score=0.9833 rollouts=5
#        → outputs/<date>/report.md, labeled "not a leaderboard Q"
```

```bash
# 1) pilot: serve ONE task, drive the official evaluator (10 instances × 1 rollout)
PYTHONPATH=$PWD/src python3 -m b1k.server --config configs/server.yaml --port 8000 --task turning_on_radio
#        EXPECT: server up; from another shell:
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/healthz   # EXPECT: 200

# check the exact evaluator argv it will run — EXPECT: the omnigibson.eval.eval
# line printed below, nothing else
PYTHONPATH=$PWD/src python3 -m evalharness.run_eval --task turning_on_radio --port 8000 --dry-run
#   python -m omnigibson.eval.eval --task-name turning_on_radio \
#     --host 127.0.0.1 --port 8000 --instance-indices 0 1 2 3 4 5 6 7 8 9 \
#     --num-rollouts 1 --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
#     --output-dir outputs/b1k_eval --write-video --headless

# run it (one call = one task; the evaluator is a subprocess that owns the sim)
PYTHONPATH=$PWD/src python3 -m evalharness.run_eval --task turning_on_radio --port 8000
#        EXPECT: "metrics -> outputs/b1k_eval/json" (non-zero exit otherwise)
```

```bash
# 2) real-Q self-eval: 50-port fan-out (one task-pinned server per port),
#    healthz wait per port, evaluator per task, aggregate, report
MODE=sim TASKS="0 1 3" scripts/self_eval.sh
#        EXPECT: "+ python -m b1k.server …" per port, then per task
#        "  task <name> (port P) -> <json dir>", then
#        "[sim] report -> outputs/<date>/report.md  Q=0.xxxx rollouts=30"

# all 100 tasks (the full self-eval — multi-day on one 4090; run in the background)
MODE=sim TASKS="$(seq 0 99 | tr '\n' ' ')" scripts/self_eval.sh
```

`self_eval.sh` first re-runs the unit test suite (it must pass) and sets
`PYTHONPATH` for you. Environment knobs: `MODE` (fixture|sim), `TASKS`
(space-separated task ids), `OUT` (output dir; default `outputs/<date>`).

```bash
# 3) cleanup after any fleet run (workers are plain subprocesses)
pgrep -af "b1k.server" || echo "no leftover servers"
pkill -f "b1k.server"          # kill leftover workers if any
```

---

## 6. Run tests

Run these **on the GPU box** too — they gate a healthy environment before you
burn sim hours, and after any code/config change. All are CPU-verified
(they do not need the simulator); only the `MODE=sim` self-eval and the
evaluator need OmniGibson + GPU.

```bash
# 1) unit tests — EXPECT: "116 passed"
python3 -m pytest tests/ -q

# 2) end-to-end WebSocket serving contract — boots a real b1k.server on a free
#    port and drives the recorded obs over live WS — EXPECT: "WS END-TO-END TEST PASSED"
python3 tests/e2e_websocket.py

# 3) local submission validation harness (7 gates: unit tests, fixtures,
#    live server, edge cases, sample data, exact official scoring math,
#    report) — EXPECT: "7 passed, 0 failed, 0 skipped" + verdict PASS in
#    validation/validation_report.md
scripts/validate_submission.sh

# 4) proof the obs wrapper is onboard-only (challenge inspection requirement)
#    — EXPECT: "PASS: no privileged obs keys; server uses RGB + depth + proprio only."
PYTHONPATH=$PWD/src python3 scripts/obs_wrapper_audit.py

# 5) entrypoint smoke — proves bare `python -m b1k.server` works (no PYTHONPATH)
#    and /healthz returns 200 — EXPECT: "healthz OK on <port>"
python3 scripts/verify_entrypoint.py
```

`MODE=sim scripts/self_eval.sh` re-runs gate 1 automatically as its step [1/3]
— a failing suite blocks the sim run from starting.

---

## 7. Visualize results

There is no in-repo viewer by design: "visualize" = open the artifacts a sim
run produces.

```bash
REPORT=$(ls -1d outputs/*/report.md | sort | tail -1); echo $REPORT

# 1) the report — overall Q / task_sr / time_score, per-task table (worst first),
#    and a "Regressions vs previous run" section when a previous summary exists
cat $REPORT
cat $(dirname $REPORT)/summary.json

# 2) per-rollout metrics (one JSON per task × instance × rollout)
ls outputs/<date>/b1k_eval/json/
cat outputs/<date>/b1k_eval/json/turning_on_radio_0_0.json
#        EXPECT keys: task, instance_id, rollout_id, steps, success,
#        agent_distance, normalized_agent_distance, q_score.final,
#        time.{simulator_steps, simulator_time, normalized_time}

# 3) rollout videos (sim only; one MP4 per rollout — names follow the
#    evaluator's video writer; use the task prefix to find your episodes)
ls outputs/<date>/b1k_eval/videos/
ffplay outputs/<date>/b1k_eval/videos/turning_on_radio_*.mp4
#        EXPECT: head-camera view of the episode; the pilot run's JSONs in
#        outputs/b1k_eval/ (no date dir) land the same way

# 4) obs-layout dump — first frame of each connection (verify_obs_layout: true)
ls outputs/logs/

# 5) validation verdict (from the §6 harness)
cat validation/validation_report.md
```

**Regression tracking:** before a new run, `cp` the previous `outputs/<date>/
/summary.json` where your workflow keeps history — the report compares
per-task Q against it and lists drops > 0.005 (the week-4 "no model swap
without a per-task regression check" gate, `solution.md` §9).

**Submission note:** the portal upload is capped at 1,000 videos and
`scripts/make_submission.sh` generates `videos_manifest.txt` for the full set
(package step: `solution.md` §7).

---

## 8. Troubleshooting

| # | Symptom | Likely cause | Check / fix |
|---|---|---|---|
| 1 | `FAIL: omnigibson is not installed — sim mode requires the v3.9.2 evaluator` (self_eval `MODE=sim`, `run_eval` non-dry, or evaluator launch) | Not in the `behavior` env, or env was never built | `conda activate behavior` · `python -c "import omnigibson"` · if missing: `scripts/bootstrap.sh --full` on the GPU box (clone + `./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval`) |
| 2 | `torch.cuda.is_available() = False` with `policy.device: cuda` (VLA backend raises) | CPU-only torch build | `nvidia-smi` (driver present?) · `python -c "import torch; print(torch.version.cuda)"` · install a CUDA 12.x torch build matching the driver (§3 step 6) |
| 3 | `CUDA out of memory` during VLA inference or sim | π0.5 + OmniGibson don't fit the VRAM budget | `nvidia-smi` (current + max VRAM) · keep the π0.5 budget from `solution.md` §4.5: bf16, 720² + 2×480² inputs, 32-step chunks · don't stack extra workers on the one GPU |
| 4 | `FAIL: server on port <P> did not become healthy` after ~180 s | VLA checkpoint load is slow on first connect; worker crashed; wrong config | `pgrep -af "b1k.server"` (is the worker alive?) · read the worker's stderr · `curl http://127.0.0.1:<P>/healthz` · re-run with `echo` backend to isolate serving vs. model |
| 5 | Sim far slower than ~13.5 FPS (eval reference, RTX 4090) | Rendering/video overhead; wrong GPU picked | Evaluator already runs `--headless` · for timing runs, call `evalharness.run_eval.eval_cmd(..., write_video=False)` programmatically (the `run_eval` CLI has no video flag) · confirm `nvidia-smi` shows the 24 GB dGPU, not a small iGPU |
| 6 | `evaluator exited <n>` or `expected 10 JSONs for <task>, found <k<10>` | Evaluator crash (scene load, sim error) or WS protocol mismatch between evaluator v3.9.2 and our server | Read the evaluator's stderr (run_eval prints the full argv on failure) · `run_eval --dry-run` to re-check the argv · run the §6 e2e WS test to confirm the serving contract · compare obs payload keys against `outputs/logs/` |
| 7 | Download stalls / incomplete chunks | HF rate limits; network | Re-run the same `download` command (resumes per chunk) · `export HF_TOKEN=…` if authenticated · `ls data/demos/chunk-*` to see what landed |
| 8 | Q ≈ 0 on every task with all JSONs present | Checkpoint not actually trained (M1 confusion), or detectors/BDDL wiring missing | Serve `policy.backend: echo` on the pilot task to confirm the pipeline scores something · verify `checkpoint_dir` points at real weights · watch one rollout video (step 3) to see the robot's behavior |
| 9 | Metrics vary between runs of the same config | Expected — simulator nondeterminism; no rollout cherry-picking (1 rollout per instance, fixed indices) | Compare runs via the report's ΔQ column; the regression gate only flags drops > 0.005 |

**Box-level notes:** headless boxes need no display (the evaluator default is
`--headless`). For final Q runs, use the 24 GB dGPU eval-class hardware —
unified-memory boards (Jetson Orin) do not meet the VRAM budget and their
CPU-only system torch must be replaced as in §3 step 6.

---

## 9. Validation of expected outputs

Gate each phase before moving on. A phase that fails its EXPECT line is not
done — fix it (often via §8) before continuing.

| Phase | Command | EXPECT (pass threshold) |
|---|---|---|
| Box check | `nvidia-smi` + §2.3 torch one-liner | dGPU visible; `torch.cuda.is_available() = True` on a CUDA 12.x build |
| Env gate | `python3 -m pytest tests/ -q` | `116 passed`, exit 0 |
| Serving gate (fixture) | `scripts/self_eval.sh` | tasks 0/1/3 → `30/30 steps finite=… -> OK`; fixture scoring `Q=0.6000 task_sr=0.4000 time_score=0.9833 rollouts=5`; `outputs/<date>/report.md` written, labeled *not a leaderboard Q* |
| Pilot sim | §5 step 1 (`run_eval` one task) | exit 0; `outputs/b1k_eval/json/turning_on_radio_*.json` — **10 files** (10 instances × 1 rollout); `outputs/b1k_eval/videos/` — 10 MP4s; each JSON carries `q_score.final`, `time.normalized_time`, `steps` |
| Self-eval sim | `MODE=sim TASKS="0 1 3" scripts/self_eval.sh` | exit 0; `outputs/<date>/report.md` + `summary.json`; printed `rollouts=30` (10 instances × 3 tasks); per-task Q present in the report table |
| Submission package | `TEAM=b1k AFFIL=nous DATE=$(date +%Y%m%d) scripts/make_submission.sh` then `scripts/validate_submission.sh submission/standard.public.b1k.nous.$DATE` | staged folder matches the contract (`json/` naming `<task>_<instance>_<rollout>.json`, instance ids 301–340 at final-eval time); `VERDICT: PASS` |

**Interpretation rules:**

- `num_rollouts` must equal `#instances × #tasks × #rollouts` you asked for.
  Fewer → a task failed mid-run (check the per-task `WARN:` lines in the
  self-eval output, then §8.6).
- Fixture-mode Q (`0.6000`) and the sample-submission scores come from
  recorded/synthetic data and are **never** leaderboard Q. Do not quote them
  as results (`solution.md` §5 honest boundary).
- Real-Q reference points for sanity (not acceptance criteria): baseline
  pipeline (M1) should score *something > 0* on the pilot task; milestone
  targets are Q≈0.40–0.50 (M2) and Q ≥ 0.55 (M5, `solution.md` §6, §9).
- `time_score` is a tie-breaker, not a target: it is `3 − 2/normalized_time`
  per rollout, averaged — negative values are allowed (slower than 1.5× human).

---

## 10. Where things go (artifact map)

| Artifact | Path | Written by |
|---|---|---|
| Self-eval report + machine summary | `outputs/<date>/report.md`, `outputs/<date>/summary.json` | `evalharness.report` (fixture & sim modes) |
| Per-rollout metrics JSONs | `outputs/<date>/b1k_eval/json/<task>_<instance>_<rollout>.json` (sim via self-eval) · `outputs/b1k_eval/json/…` (pilot `run_eval`) | OmniGibson evaluator |
| Rollout videos (MP4) | same dirs under `videos/` | OmniGibson evaluator (`--write-video`) |
| Combined metrics copy | `outputs/<date>/json/` | `self_eval` (sim mode) |
| Obs-layout dump | `outputs/logs/` | server (`verify_obs_layout: true`) |
| Demo dataset | `data/demos/chunk-NNN/…` | `src.training.download` |
| Norm stats | `data/stats.json` | `src.training.stats` |
| VLA checkpoint | `data/checkpoints/pi05_b1k/` | `src.training.vla_finetune` |
| Detectors | `data/detectors/` | `src.training.detector_train` |
| Skill annotations (miner input) | `data/annotations.jsonl` | `src.training.skill_annotations` |
| Skill plans | `docs/plans/<task_id>.json` | `b1k.planner.miner.mine_plans` (CLI snippet, §4.3) |
| Validation verdict | `validation/validation_report.md` | `scripts/validate_submission.sh` |
| Submission package | `submission/standard.public.<team>.<affil>.<date>/` | `scripts/make_submission.sh` |

**Next step after a green pilot:** package the submission — `solution.md` §7
(`make_submission.sh`, zip, portal upload, registration form; deadline
**2026-10-16**).
