# Verified GPU local-simulation commands (b1k)

Verified on this box (luis-desktop, aarch64) on 2026-09-14 by actually running
every command in §1–§3. GPU-dependent commands in §4 are derived from code
(entrypoints + argparse) and marked with the exact prerequisites they need;
they fail cleanly and were verified to the point of the first missing dependency.

Handoff notes for the docs task: §1–§3 output is captured verbatim; §4 is the
command list a GPU box must run. Environment facts that any new doc must state:
the serving stack is CPU-verified end-to-end, but this machine's `torch` is a
CPU-only build and the simulator/training deps are absent (see §6 Blockers).

## 0. Environment snapshot (verified)

| Fact | Value |
|---|---|
| Python | 3.10.12 |
| GPU (nvidia-smi) | NVIDIA Orin (nvgpu), driver 540.4.0, CUDA 12.6 — present |
| torch | 2.0.1, `torch.version.cuda = None`, `cuda.is_available() = False` (CPU-only build) |
| `configs/server.yaml` `policy.device: auto` | resolves to `cpu` here (`b1k.config.resolve_device`) |
| Serving deps installed | numpy 1.26.4, msgpack 1.1.2, websockets 15.0.1, PyYAML 6.0.3, pytest 9.0.2 |
| Train extras installed | pandas 2.3.3, pyarrow 25.0.1, scipy 1.8.0, huggingface_hub |
| Missing modules | `omnigibson` (BEHAVIOR-1K v3.9.2), `openpi` ('behavior' fork) |
| `b1k` importable | yes, bare (`python -m b1k.server` works with no PYTHONPATH — editable install) |
| Fixture data present | `data/demos/{0,1}/actions.npy`, `data/bddl_goals.json`, `data/human_stats.jsonl`, `docs/plans/{0,1,3}.json`, `tests/fixtures/*` |

## 1. Prerequisites (all boxes)

```bash
python3 -m pip install -r requirements.txt          # numpy, msgpack, websockets, PyYAML, pytest
python3 -m pip install -e . --no-build-isolation --no-deps   # editable install so `python -m b1k.server` resolves bare
```

GPU box additionally (BEHAVIOR-1K v3.9.2 conda env):

```bash
git clone -b v3.9.2 https://github.com/StanfordVL/BEHAVIOR-1K.git /path/to/B1K
cd /path/to/B1K && ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
conda activate behavior
# torch must be a CUDA build (driver here is 540.4.0 / CUDA 12.6; the box's
# system torch 2.0.1 is CPU-only). Also: huggingface-cli (huggingface_hub).
```

No env vars are required for any command below except the optional ones listed
per script: `MODE`/`TASKS`/`OUT` (self_eval.sh), `TEAM`/`TRACK`/`TESTSET`/
`AFFIL`/`DATE`/`JSON_SRC` (make_submission.sh), `BEHAVIOR_1K` (bootstrap.sh
--full). There are no other CUDA-specific env vars in the code — GPU use is
driven by `policy.device: auto|cuda` in `configs/server.yaml`.

## 2. Run (serving) — verified

```bash
# Live policy server (MVP echo backend; CPU-verifiable). Verified: binds, /healthz 200.
PYTHONPATH=src python3 -m b1k.server --config configs/server.yaml --port 8000 --task 0
#   (PYTHONPATH unneeded once the editable install exists — scripts/verify_entrypoint.py
#    proves the bare `python -m b1k.server` entrypoint.)
```

- CLI args (src/b1k/server.py::main): `--config` (default configs/server.yaml),
  `--port`, `--host`, `--task` (name or id), `--log-level`.
- Readiness: `GET http://127.0.0.1:8000/healthz` → 200.
- Wire contract: binary msgpack obs in → binary msgpack `{"action": [23 floats]}`
  out; each WS (re)connect resets the episode (`reset_on_connect: true`).
- Real-model variant (GPU, after training): `configs/server.yaml`
  `policy.backend: vla`, `policy.checkpoint_dir: data/checkpoints/pi05_b1k`,
  `policy.device: cuda` — same command otherwise.

## 3. Test — verified (all executed, exit 0)

| Command | Captured result |
|---|---|
| `python3 -m pytest tests/ -q` | `115 passed in 7.34s` |
| `python3 tests/e2e_websocket.py` | `WS END-TO-END TEST PASSED` (boots a real `b1k.server` on a free port, drives recorded obs over live WS) |
| `scripts/validate_submission.sh` | `[PASS]` ×7 gates: unit-tests (115), fixtures, live-server, edge-cases 11/11, sample-data (frozen scores 1e-12), scoring (official score_utils math), report → `== 7 passed, 0 failed, 0 skipped (18.4s) ==`; verdict PASS in `validation/validation_report.md` |
| `scripts/self_eval.sh` (default fixture mode) | tasks 0/1/3: 30/30 steps finite 23-dim → OK; fixture scoring `Q=0.6000 task_sr=0.4000 time_score=0.9833 rollouts=5`; report → `outputs/20260914/report.md` |
| `PYTHONPATH=src python3 scripts/obs_wrapper_audit.py` | `PASS: no privileged obs keys; server uses RGB + depth + proprio only.` |
| `python3 scripts/verify_entrypoint.py` | `healthz OK on <port> (documented entrypoint works, no PYTHONPATH)` |
| `scripts/bootstrap.sh` (default, safe mode) | serving deps OK + unit tests pass + prints GPU-box sequence; exit 0 |

## 4. GPU-box path (derived from code; first missing dep verified)

Each step below was executed here up to the point where its missing dependency
surfaces; the captured failure is the expected one and is actionable.

```bash
# (1) per-task demo download (huggingface-cli; 3.27 TB total, per-task chunks)
PYTHONPATH=$PWD/src python3 -m src.training.download --tasks 0 1 2 --root data/demos
#   --dry-run verified: prints the exact huggingface-cli argv per chunk (data/chunk-000/**,
#   meta/episodes/chunk-000/**, videos/*/chunk-000/**, meta/*). FIX applied 2026-09-14:
#   --dry-run previously printed nothing (src/training/download.py).

# (2) norm stats (post-Aug-2026 velocity fix); needs downloaded chunks + pandas/pyarrow
PYTHONPATH=$PWD/src python3 -m src.training.stats --data-root data/demos --out data/stats.json

# (3) skill plans + training (needs openpi 'behavior' fork + CUDA torch + datasets)
PYTHONPATH=$PWD/src python3 -m src.training.skill_annotations --data-root data/demos --tasks 0
PYTHONPATH=$PWD/src python3 -m src.training.vla_finetune --out-dir data/checkpoints/pi05_b1k
PYTHONPATH=$PWD/src python3 -m src.training.detector_train --out-dir data/detectors

# (4) serve the REAL model (see §2 real-model variant)
PYTHONPATH=src python3 -m b1k.server --config configs/server.yaml --port 8000 --task 0

# (5) drive the official evaluator (needs omnigibson v3.9.2; one task per call)
PYTHONPATH=$PWD/src python3 -m evalharness.run_eval --task turning_on_radio --port 8000
#   --dry-run verified: prints exactly
#   python -m omnigibson.eval.eval --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
#     --instance-indices 0 1 2 3 4 5 6 7 8 9 --num-rollouts 1 \
#     --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
#     --output-dir outputs/b1k_eval --write-video --headless

# (6) real-Q self-eval: 50-port fan-out (evalharness/parallel.py) + aggregate + report
MODE=sim TASKS="0 1 3" scripts/self_eval.sh
#   verified failure on this box (expected blocker): "FAIL: omnigibson is not installed —
#   sim mode requires the v3.9.2 evaluator (run scripts/bootstrap.sh on a GPU box)."

# (7) package submission zip
TEAM=b1k scripts/make_submission.sh
```

Fleet/parallel utility (verified arg parsing; spawns `python -m b1k.server`
subprocesses one per port): `python3 -m src.evalharness.parallel --ports 50
--config configs/server.yaml` (use `--no-start` to just print commands).

## 5. Visualization — expected artifacts (no extra command needed)

There is no in-repo viewer by design; "visualize" = open these outputs:

- Rollout videos (sim only, requires `--write-video`, default on in run_eval):
  `outputs/<date>/b1k_eval/videos/*.mp4` (per-rollout MP4s; the submission
  contract also caps portal uploads at 1,000 videos and
  `scripts/make_submission.sh` generates `videos_manifest.txt`).
- Per-task/per-rollout metrics JSONs: `outputs/<date>/b1k_eval/json/
  <task>_<instance>_<rollout>.json`.
- Aggregated Markdown report with the leaderboard math (Q / task_sr /
  time_score per task, worst-first table, regression ΔQ vs previous run):
  `outputs/<date>/report.md` (+ `summary.json`), written by
  `evalharness/report.py::render_report` in both fixture and sim modes.
- Obs-layout dump (first frame of each connection, `verify_obs_layout: true`):
  `outputs/logs/`.
- Validation verdict: `validation/validation_report.md`.

Fixture mode produces `report.md` without videos (no simulator) — it verifies
the pipeline and scoring math, explicitly labeled "not a leaderboard Q".

## 6. Blockers / gaps found (2026-09-14)

1. **torch is CPU-only on this box** (`torch.version.cuda=None`). `device: auto`
   correctly falls back to CPU, but the `vla` backend / any CUDA inference needs
   a CUDA torch build on the GPU box (driver 540.4.0 = CUDA 12.6; consider
   torch ≥2.3 cu12x wheels — the system torch 2.0.1 predates 12.x).
2. **`omnigibson` absent** → `--mode sim`, `run_eval` (non-dry), and the real
   evaluator cannot run here; verified clean FAIL with actionable message.
3. **`openpi` 'behavior' fork absent** → `src.training.vla_finetune` /
   `detector_train` / the `vla` backend inference seam cannot run; their error
   messages point at bootstrap.
4. **Demo dataset not downloaded** (only fixture `actions.npy` for tasks 0/1);
   stats/plans/training need real chunks from HF `behavior-1k/2026-challenge-demos`.
5. **Bug fixed**: `src/training/download.py --dry-run` printed nothing; now
   prints the per-chunk `huggingface-cli` argv (tests re-run green).
6. **Hardware note**: this box's GPU is a Jetson Orin (unified memory); the
   challenge design targets a single 24 GB dGPU (e.g. 4090) for full 100-task
   runs — use the eval hardware for final Q runs.
7. **Docs gap (for the writer task)**: README/solution.md cover the GPU path in
   outline but do not state the torch-CUDA requirement, the exact expected
   outputs per command, or the visualization artifact locations (§2–§5 here);
   troubleshooting for the two verified failure modes is not documented.
