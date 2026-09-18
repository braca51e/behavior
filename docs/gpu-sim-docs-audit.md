# Audit: documentation coverage for GPU local simulation — run / test / visualize

**Task:** t_a0fd9417 (decomposed from t_782b37f6)
**Date:** 2026-09-14
**Method:** full-repo text search (`gpu|cuda|nvidia|device|conda|omnigibson|bootstrap|docker` in `*.md`; `visualiz|mp4|video|plot|ffmpeg|watch|portal` in docs/scripts/src) + direct reads of the GPU-path code (`src/evalharness/run_eval.py`, `self_eval.py`, `parallel.py`, `report.py`, `scripts/bootstrap.sh`, `scripts/self_eval.sh`).

**Docs inspected:** `README.md`, `docs/solution.md`, `docs/design.md`, `docs/challenge-spec.md`, `validation/README.md`, `scripts/bootstrap.sh`, `scripts/self_eval.sh`. (The `docs/icra-iros-2027-*` and `docs/feasibility-assessment-2027.md` files are research-scoping docs and do not document how to run this repo's simulation; they are excluded from the coverage table below.)

**Scope note:** the repo's "local simulation" = driving the BEHAVIOR-1K v3.9.2 OmniGibson evaluator against our WebSocket policy server (`--mode sim`). The CPU self-test (fixture mode) is documented well; this audit targets the **GPU path** specifically.

---

## Coverage matrix (GPU local simulation)

| Aspect | Status | Where documented |
|---|---|---|
| Prerequisites | **Present (partial)** | `docs/solution.md:246` ("needs conda + BEHAVIOR-1K v3.9.2 + an NVIDIA GPU"); `docs/challenge-spec.md:290-296` (benchmark machine: RTX 4090 / 24 GB final-eval GPUs; 13.52 FPS); `docs/solution.md:40-42` (24 GB VRAM budget, single-GPU constraint). |
| Environment setup | **Present** | `docs/solution.md:250-253` (`scripts/bootstrap.sh --full`, `conda activate behavior`); `scripts/bootstrap.sh:39-44` (clone v3.9.2, `./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval`); `docs/challenge-spec.md:232-237` (same install sequence); `README.md:164-168`. |
| Data / config requirements | **Ambiguous** | `docs/solution.md:255-266` (download 3.27 TB demos, recompute norm stats, train checkpoints, set `policy.backend: vla` + `checkpoint_dir` in `configs/server.yaml`). Missing: exact download flags for the full task set, HF credentials/token, and how the 24 GB VRAM budget maps to config. See gaps G4. |
| Run command | **Present** | `docs/solution.md:264-270` (`python -m b1k.server` to serve; `python -m evalharness.run_eval --task <name> --port 8000` to drive one task); `docs/solution.md:272-275` and `README.md:184` (`MODE=sim TASKS="0 1 3" scripts/self_eval.sh`); `src/evalharness/run_eval.py:44-62` (the exact `omnigibson.eval.eval` argv). |
| Test command | **Missing (GPU-specific)** | `docs/solution.md:250-275` (§6 GPU path) never says to re-run the CPU test suite on the GPU box; §5's `python3 -m pytest tests/ -q` is explicitly labeled CPU-only. No documented GPU-side smoke test beyond the `MODE=sim` self-eval. See gap G1. |
| Visualization command | **Missing** | No viewer command or tooling anywhere. `docs/solution.md:269` notes videos land in `outputs/<date>/b1k_eval/...videos/*.mp4`; `docs/solution.md:306-307` and `docs/challenge-spec.md:111` say videos go to a *portal link* for submission. No `ffplay`/`matplotlib`/report-plot step is documented or present in code. See gap G2. |
| Expected outputs | **Ambiguous** | `docs/solution.md:275` ("EXPECT: `outputs/<date>/report.md` with per-task Q / task_sr / time_score") for `MODE=sim`. Code confirms `report.md` + `summary.json` (`src/evalharness/report.py:53-105`) and `b1k_eval/json/*.json` + `videos/*.mp4` (`run_eval.py:15-18`). No *reference numbers* or pass/fail threshold for a real sim run, and the report is not labeled PASS/FAIL the way the CPU validation harness is (`validation/README.md`). See gap G3. |
| Troubleshooting | **Missing** | No troubleshooting section in any GPU doc. Only inline hints: `self_eval.py:147-153` (fails if `omnigibson` not importable → run `bootstrap.sh`), `run_eval.py:84-95` (evaluator non-zero exit / missing JSONs raise). No guidance for OOM, driver/CUDA mismatch, `healthz` timeout, or slow sim. See gap G5. |

---

## Gap list (concise, with references)

- **G1 — No documented GPU-side test command.** The CPU test suite (`python3 -m pytest tests/ -q`, `docs/solution.md:196`, `README.md:43`) is correct to run everywhere, but §6 ("Real training & evaluation (GPU box)", `docs/solution.md:244-284`) does not instruct the reader to run it (or `scripts/validate_submission.sh`) on the GPU box before/at the start of a sim run. A new contributor following only §6 has no "test" step to gate a healthy environment.
  - *Fix:* add a "Tests on the GPU box" subsection to `docs/solution.md` §6 (re-run `pytest`, `scripts/self_eval.sh` fixture mode as an env sanity check, plus a one-task `--dry-run` of `run_eval`).

- **G2 — No visualization command or tooling (biggest gap).** Sim mode produces `videos/*.mp4` (`docs/solution.md:269`; `run_eval.py:15-18`) and a `report.md`, but there is no documented way to *watch* a rollout or *view* results. The repo contains no plotting/viewer code (`grep matplotlib|ffplay|ffmpeg` in `src/`, `scripts/` returns nothing relevant). "How do I see the robot's episode?" is unanswered.
  - *Fix:* document at minimum (a) how to play an MP4 (`ffplay outputs/<date>/b1k_eval/videos/<task>_*.mp4`), and (b) how to read `report.md` / `summary.json`. Optionally add a tiny script to plot per-task Q trends across runs.

- **G3 — Expected outputs for sim mode are vague and have no pass threshold.** `docs/solution.md:275` says to EXPECT `report.md` with per-task metrics but gives no reference value, no "what good looks like", and no PASS/FAIL verdict for a real run (unlike the CPU validation harness which writes `validation/validation_report.md` with `PASS`, `validation/README.md`). A reader cannot tell if a `MODE=sim` run succeeded.
  - *Fix:* state the concrete artifacts (`report.md`, `summary.json`, `b1k_eval/json/*.json`, `videos/*.mp4`), what a successful run prints (`Q=… rollouts=…`, `self_eval.py:183`), and a sanity threshold (e.g. `num_rollouts` equals instances×tasks).

- **G4 — Data/config requirements are incomplete for a first-time GPU run.** `docs/solution.md:255-258` lists `download --tasks 0 1 2` (a 3-task sample) but does not document: the full 100-task download command/flag, whether an HF token is required, or which `configs/` keys must change for the real model (beyond `policy.backend`/`checkpoint_dir` at line 264-265). The 3.27 TB figure and VRAM budget are stated, but not tied to concrete config.
  - *Fix:* expand §6 data prep with the full-task download command, HF-auth note, and an explicit `configs/server.yaml` diff for sim mode.

- **G5 — No troubleshooting section.** Only two inline failure hints exist: `omnigibson` import failure (`self_eval.py:147-153`) and evaluator non-zero/missing-JSON (`run_eval.py:84-95`). Nothing covers the common GPU-box failures (CUDA OOM, driver/CUDA version mismatch, `/healthz` not becoming healthy, sim running far slower than 13.5 FPS, scene-load stalls).
  - *Fix:* add a "Troubleshooting (GPU box)" section to `docs/solution.md` mapping symptom → likely cause → command to check (`nvidia-smi`, server `healthz` curl, evaluator stderr).

- **Minor / consistency notes**
  - `docs/solution.md:252` says `scripts/bootstrap.sh --full` is "safe on any box", but `scripts/bootstrap.sh:41-43` will actually `git clone` and start the multi-TB download, so it is *not* safe to run casually on a CPU box — the doc and the script disagree on blast radius. Clarify the `--full` preconditions.
  - The evaluator flag `--env` is built in code (`run_eval.py:42`) but not surfaced in any doc or the `run_eval` CLI — ambiguous which Python env the evaluator subprocess uses.
  - `README.md:189-190` and `docs/solution.md:295` both give the Docker `--gpus all` serving command, which is the correct GPU note for the submission path; keep them in sync if either changes.

---

## Summary for downstream doc draft (t_95c33605)

The GPU **run** path is well documented (`docs/solution.md` §6). The **test**, **visualize**, **expected-output**, and **troubleshooting** aspects are under-documented or missing for the GPU path. Priorities: (1) add a GPU-side test subsection [G1], (2) add a visualization/viewer section [G2], (3) make sim expected outputs concrete with a pass threshold [G3], (4) complete data/config prep [G4], (5) add a troubleshooting section [G5]. All fixes belong in `docs/solution.md` §6 (or a new `docs/gpu-simulation.md`) in the existing imperative-command style.
