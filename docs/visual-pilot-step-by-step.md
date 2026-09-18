# Visual pilot — full step-by-step (GPU box)

**Goal:** run one BEHAVIOR Challenge task (`turning_on_radio`) on your NVIDIA
GPU, drive it with **this repo’s small open-source `echo` policy**, write an
MP4, and watch the robot execute.

**Companion docs:** [`gpu-simulation.md`](gpu-simulation.md) (full GPU-box
guide) · [`solution.md`](solution.md) (solution & submission) ·
[`../README.md`](../README.md) (repo tour)

**Verified on:** luis-desktop, 2026-09-14 — NVIDIA RTX 5000 Ada (16 GB),
driver 550.163.01 / CUDA 12.4, conda env `behavior392`, BEHAVIOR-1K `v3.9.2`.

---

## 0. What you will end up with

Two processes talking over WebSocket:

```
OmniGibson evaluator (GPU, conda env behavior392)
        │  binary msgpack obs each step
        ▼
python -m b1k.server  (CPU OK; this repo; echo backend)
        │  binary msgpack {"action": float32[23]}
        ▼
robot moves in sim → MP4 + metrics JSON
```

Expected pilot artifacts (after a successful run):

| Artifact | Path |
|---|---|
| Rollout video | `outputs/visual_pilot/videos/turning_on_radio_301_0.mp4` |
| Metrics JSON | `outputs/visual_pilot/json/turning_on_radio_301_0.json` |
| Policy server log | `outputs/visual_pilot/server.log` |
| Evaluator log | `outputs/visual_pilot_run.log` (if you redirect) |

**Honest expectation for `echo`:** the robot moves (replays recorded demo
actions), but `q_score` will usually be `0.0` and `success=false`. That is
correct — `echo` proves the **pipeline + visuals**, not leaderboard Q. Real Q
needs `policy.backend: vla` + a trained/baseline π0.5 checkpoint.

---

## 1. Prerequisites

### 1.1 Hardware / OS

| Need | Why |
|---|---|
| NVIDIA dGPU, preferably ≥ 16 GB VRAM | OmniGibson + Isaac Sim 5.1; 24 GB is the challenge’s eval budget |
| ≥ ~80 GB free disk for a pilot | Isaac Sim wheels + BEHAVIOR assets + challenge task instances |
| Ubuntu 22.04-class Linux | Matches BEHAVIOR-1K `setup.sh` |
| Display optional | Headless writes MP4; `LIVE=1` needs a display (`$DISPLAY`) |

### 1.2 Software you must already have

```bash
# NVIDIA driver
nvidia-smi
# EXPECT: your GPU name, Driver Version, CUDA Version ≥ 12.x

# conda
ls ~/miniconda3/etc/profile.d/conda.sh   # or ~/anaconda3/...
# EXPECT: file exists

# tools
which git curl python3 ffplay   # ffplay optional but used to auto-play MP4s
```

### 1.3 Important Python trap on this machine

If you have a **user-site** torch (e.g. `~/.local/lib/python3.10/site-packages`
with a CUDA-13 build), it can shadow the conda env’s CUDA-compatible torch and
make `torch.cuda.is_available()` return `False`.

**Always** set this inside the sim env:

```bash
export PYTHONNOUSERSITE=1
```

The visual-pilot script sets it for you.

---

## 2. Install this repo (policy server / serving stack)

From a terminal:

```bash
cd /home/luis/Documents/Behavior_Challenge   # or your clone path
export REPO="$(pwd)"

# 1) serving deps (CPU)
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-build-isolation --no-deps

# 2) CPU gate — do this before touching the simulator
python3 -m pytest tests/ -q
# EXPECT: all tests passed (≈115+)

python3 tests/e2e_websocket.py
# EXPECT: "WS END-TO-END TEST PASSED"
#         (boots a real b1k.server, drives the recorded obs fixture over WS)
```

What the e2e test proves: `/healthz` → 200, WebSocket obs→action contract,
JSON reset control, finite 23-dim actions.

---

## 3. Install BEHAVIOR-1K v3.9.2 (simulator + evaluator)

### 3.1 Clone the exact challenge tag

```bash
export PATH_TO_BEHAVIOR_1K=/home/luis/Documents/BEHAVIOR-1K

git clone --depth 1 -b v3.9.2 \
  https://github.com/StanfordVL/BEHAVIOR-1K.git \
  "$PATH_TO_BEHAVIOR_1K"

ls "$PATH_TO_BEHAVIOR_1K/OmniGibson/omnigibson/eval/eval.py"
# EXPECT: file exists
```

Do **not** use older tags (`v3.9.0`, OmniGibson 3.7.x). The challenge evaluator
and WS contract fixes ship in **v3.9.2**.

### 3.2 Create the conda env (preferred: official `setup.sh`)

Official one-shot (long; downloads Isaac Sim 5.1 + datasets):

```bash
cd "$PATH_TO_BEHAVIOR_1K"

# Match your driver. On this box nvidia-smi reported CUDA 12.4 → use 12.4.
# (setup.sh default is 12.8, which may not have matching torch wheels.)
./setup.sh --new-env behavior392 \
  --omnigibson --bddl --joylo --dataset --eval \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos \
  --cuda-version 12.4
```

**If `setup.sh` fails on `torch==2.7.0` / `torchcodec==0.5` for cu124**, continue
manually with the sequence in §3.3 (that is what was verified on this box).

### 3.3 Manual continue (verified workaround when setup.sh torch pin fails)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior392          # created by the failed/partial setup.sh
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES
unset EXP_PATH CARB_APP_PATH ISAAC_PATH

cd "$PATH_TO_BEHAVIOR_1K"
WORKDIR="$PATH_TO_BEHAVIOR_1K"
ARCH="$(uname -m)"

# --- PyTorch that matches driver CUDA 12.4 ---
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install "torchcodec==0.2.1" \
  --index-url https://download.pytorch.org/whl/cu124 || true
python -m pip install "numpy<2"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# EXPECT: ... True   and a GPU name if you also print torch.cuda.get_device_name(0)

# --- BDDL + OmniGibson (eval extras) ---
python -m pip install -e "$WORKDIR/bddl3"
python -m pip install -e "$WORKDIR/OmniGibson[eval]" --no-build-isolation
# NOTE: OmniGibson[eval] may upgrade torch (e.g. to 2.7+cu126). That is OK if
#       torch.cuda.is_available() stays True on your driver.

# --- Isaac Sim 5.1.0 (large download; many wheels from pypi.nvidia.com) ---
python -c "import isaacsim" 2>/dev/null && echo "isaacsim already installed" || {
  # Use the same wheel list as setup.sh (cp311 manylinux). See setup.sh
  # install_isaac_packages() if you need the exact package names.
  python -m pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com \
  || echo "If this fails on GLIBC, use setup.sh's curl+rename manylinux_2_31 path"
}

# Fix known Isaac ↔ websockets / packaging conflicts (from setup.sh)
ISAAC_PATH="$(python -c "import isaacsim, os; print(os.environ.get('ISAAC_PATH',''))" || true)"
if [ -n "${ISAAC_PATH:-}" ] && [ -d "$ISAAC_PATH/extscache" ]; then
  find "$ISAAC_PATH/extscache" -type d -name "websockets" -path "*/pip_prebundle/*" \
    -exec rm -rf {} + 2>/dev/null || true
fi
python -m pip install --force-reinstall cffi==1.17.1
python -m pip install --force-reinstall "websockets>=15.0.1"

# --- JoyLo (required by --eval) ---
python -m pip install -e "$WORKDIR/joylo"

# --- Datasets / assets (large; resume-safe if interrupted) ---
python -c "import omnigibson; print('og OK', omnigibson.__file__)"
python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
python -c "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"
```

Hugging Face may warn about unauthenticated rate limits; set `HF_TOKEN` if
downloads stall:

```bash
export HF_TOKEN=hf_xxx   # optional
```

### 3.4 Verify the sim env

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior392
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES

python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# EXPECT: cuda True

python -c "import omnigibson, isaacsim; print('og+isaac OK')"
# EXPECT: og+isaac OK  (first Isaac import can take a few seconds)
```

---

## 4. One-command visual pilot (recommended)

With the repo and `behavior392` ready:

```bash
cd /home/luis/Documents/Behavior_Challenge

# Headless sim → write MP4 → open it with ffplay/vlc
scripts/run_visual_pilot.sh
```

### Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `TASK` | `turning_on_radio` | Challenge task name |
| `INSTANCE` | `0` | Public-test instance index (maps to id 301 for task 0) |
| `MAX_STEPS` | `1200` | Cap episode length (use `200`–`400` for a fast first look) |
| `PORT` | `8000` | WebSocket / healthz port |
| `OUT` | `outputs/visual_pilot` | Metrics + videos directory |
| `LIVE` | `0` | `1` = open Isaac Sim GUI (no `--headless`) |
| `CONDA_ENV` | `behavior392` | Env that has OmniGibson v3.9.2 |

Examples:

```bash
# Fast short episode
MAX_STEPS=300 scripts/run_visual_pilot.sh

# Live window (needs DISPLAY)
LIVE=1 MAX_STEPS=400 scripts/run_visual_pilot.sh

# Different output dir
OUT=outputs/visual_pilot_$(date +%Y%m%d_%H%M) scripts/run_visual_pilot.sh
```

### What the script does (in order)

1. Starts `python -m b1k.server` with `configs/server.yaml` (`policy.backend: echo`).
2. Waits until `GET http://127.0.0.1:$PORT/healthz` returns **200**.
3. `conda activate behavior392` and runs:

   ```bash
   python -m omnigibson.eval.eval \
     --task-name turning_on_radio \
     --host 127.0.0.1 --port 8000 \
     --instance-indices 0 \
     --num-rollouts 1 \
     --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
     --output-dir outputs/visual_pilot \
     --write-video \
     --max-steps <MAX_STEPS> \
     --headless          # omitted when LIVE=1
   ```

4. Prints the metrics JSON path and plays the MP4.

### EXPECT during a healthy run

```
policy server OK on :8000 (pid …, backend=echo)
+ python -m omnigibson.eval.eval …
----- Starting OmniGibson. This will take 10-30 seconds... -----
… Isaac / Kit startup spam …
Connected to server!
Result: instance=301 rollout=0 steps=… success=False q_score=0.0 -> …/json/….json | video -> …/videos/….mp4
video -> outputs/visual_pilot/videos/turning_on_radio_301_0.mp4
```

First Isaac cold-start: **~2 minutes**. A 200-step capped episode after that:
roughly **~30–90 seconds** on this laptop GPU (varies with load).

---

## 5. Manual two-terminal run (same thing, more control)

Use this when debugging.

### Terminal A — policy server (repo Python is fine)

```bash
cd /home/luis/Documents/Behavior_Challenge
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1

python3 -m b1k.server \
  --config configs/server.yaml \
  --port 8000 \
  --task turning_on_radio

# EXPECT:
#   b1k server up on ws://127.0.0.1:8000 (task=turning_on_radio, backend=echo)
```

In another shell, check readiness:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/healthz
# EXPECT: 200
```

### Terminal B — evaluator (must be `behavior392`)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior392
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES
unset EXP_PATH CARB_APP_PATH ISAAC_PATH

cd /home/luis/Documents/Behavior_Challenge
mkdir -p outputs/visual_pilot

python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 \
  --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir outputs/visual_pilot \
  --write-video \
  --max-steps 300 \
  --headless
```

### Watch the video

```bash
ffplay outputs/visual_pilot/videos/turning_on_radio_301_0.mp4
# or:
vlc outputs/visual_pilot/videos/turning_on_radio_301_0.mp4
```

### Read the metrics

```bash
cat outputs/visual_pilot/json/turning_on_radio_301_0.json
```

Example successful **pipeline** result (not a good Q):

```json
{
  "task": "turning_on_radio",
  "instance_id": 301,
  "rollout_id": 0,
  "steps": 201,
  "success": false,
  "q_score": { "final": 0.0 },
  "time": {
    "simulator_steps": 201,
    "simulator_time": 6.7,
    "normalized_time": 10.69
  }
}
```

---

## 6. WebSocket contract (must match the official evaluator)

The official client is `omnigibson.eval.utils.network_utils.WebsocketClientPolicy`
(v3.9.2). This repo’s `src/b1k/server.py` must speak the same dialect:

| Step | Evaluator does | Our server must |
|---|---|---|
| 1 | `GET /healthz` | Return HTTP **200** |
| 2 | Open `ws://host:port` | Accept upgrade |
| 3 | `unpackb(conn.recv())` | **Immediately send** a msgpack metadata dict |
| 4 | Optionally `pack({"reset": True})` | Reset episode; **no reply** |
| 5 | Each step: send msgpack flattened obs (numpy ndarray extension) | Reply msgpack `{"action": float32[23] ndarray}` |

Also required for real OmniGibson frames:

- RGB cameras often arrive as **RGBA `(H,W,4)`** — strip alpha to `(H,W,3)`.
- Depth keys look like `…::depth_linear`.
- Proprio is a **61-dim** vector under `robot_r1::proprio`.

If metadata is missing, the evaluator deadlocks on connect. If `action` is a
Python `list` instead of a numpy array, you get:

```text
TypeError: expected np.ndarray (got list)
```

Those fixes live in `src/b1k/server.py` and `src/b1k/protocol.py` (already in
this checkout). Re-confirm with:

```bash
python3 tests/e2e_websocket.py
# EXPECT: WS END-TO-END TEST PASSED
#         and a printed "server metadata: {...}"
```

---

## 7. Config that the pilot uses

`configs/server.yaml` (MVP defaults):

```yaml
policy:
  backend: echo            # small open-source in-repo policy
  checkpoint_dir: null
  device: auto             # echo ignores GPU
  demos_root: data/demos   # recorded actions for echo
```

To switch later to a real model (not needed for the visual pilot):

```yaml
policy:
  backend: vla
  checkpoint_dir: data/checkpoints/pi05_b1k
  device: cuda
```

Keep `configs/r1pro.yaml` **verbatim** (submission / action-limit contract).

---

## 8. Optional next step — official π0.5 baseline (large)

Organizers ship a fine-tuned π0.5 checkpoint for `turning_on_radio` (~17 GB):

- Drive link (π0.5 column):  
  https://drive.google.com/file/d/1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2/view?usp=sharing  
- Walkthrough: https://behavior.stanford.edu/challenge/baselines.html  

High-level path (separate from this repo’s `echo` server):

```bash
# 1) clone OpenPI behavior fork + uv sync (see baselines page)
# 2) download + unzip the checkpoint → $PATH_TO_CKPT
# 3) serve:
#    uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/turning_on_radio \
#      --policy.config pi05_b1k --policy.dir $PATH_TO_CKPT --port 8000 ...
# 4) run the same omnigibson.eval.eval command as in §5 Terminal B
```

Skip this until the echo visual pilot is green.

---

## 9. Troubleshooting (failures seen on this box)

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` works but `torch.cuda.is_available()=False` | User-site torch (e.g. cu130) shadows conda | `export PYTHONNOUSERSITE=1` inside `behavior392` |
| `ModuleNotFoundError: omnigibson` | Wrong env, or editable path points at a deleted tree | `conda activate behavior392`; reinstall `-e $PATH_TO_BEHAVIOR_1K/OmniGibson[eval]` |
| `setup.sh` fails on `torch==2.7.0` for cu124 | Wheel not published for that CUDA tag | Use §3.3 (`torch==2.6.0+cu124`) then continue |
| Eval hangs after “Health check passed” with no “Connected” | Server never sent WS metadata | Update `src/b1k/server.py` (already fixed here); re-run e2e |
| `TypeError: expected np.ndarray (got list)` | Action encoded as a Python list | `encode_action` must pack a `float32` ndarray (fixed in `protocol.py`) |
| Video is ~48 bytes then grows | Normal — writer creates the file early | Wait until eval prints `Result: … video -> …` |
| `pkill -f 'python3 -m b1k.server'` kills your shell helper | Pattern matches the command line itself | Kill by PID file (`outputs/visual_pilot/server.pid`) or use `scripts/run_visual_pilot.sh` |
| HF “Permission denied” under `~/.cache/huggingface/xet/logs` | Log dir perms | `mkdir -p ~/.cache/huggingface/xet/logs && chmod -R u+rwX ~/.cache/huggingface/xet` |
| CUDA OOM | 16 GB is tight with video + full res | Lower `--max-steps`, keep one eval process, close other GPU apps |
| `q_score=0.0` with echo | Expected | Switch to `vla` + checkpoint for real Q |

---

## 10. Cleanup

```bash
# Stop leftover server / evaluator by PID if you started them manually
kill "$(cat outputs/visual_pilot/server.pid)" 2>/dev/null || true
kill "$(cat outputs/visual_pilot/eval.pid)" 2>/dev/null || true

# Or:
pkill -f 'b1k.server.*--port 8000' || true
# Avoid overly broad pkill patterns that match your current shell command.
```

---

## 11. Checklist (print and tick)

- [ ] `nvidia-smi` shows the dGPU  
- [ ] `python3 -m pytest tests/ -q` green in the repo  
- [ ] `python3 tests/e2e_websocket.py` → `WS END-TO-END TEST PASSED`  
- [ ] BEHAVIOR-1K `v3.9.2` cloned  
- [ ] `conda activate behavior392` + `PYTHONNOUSERSITE=1`  
- [ ] `import torch; torch.cuda.is_available()` → `True`  
- [ ] `import omnigibson, isaacsim` works  
- [ ] Robot + BEHAVIOR + 2026 challenge assets downloaded  
- [ ] `scripts/run_visual_pilot.sh` prints `policy server OK`  
- [ ] Evaluator prints `Connected to server!`  
- [ ] `outputs/visual_pilot/videos/turning_on_radio_*.mp4` is hundreds of KB+  
- [ ] `ffplay` / VLC shows the head-camera rollout  

When every box is checked, the visual pipeline on this GPU box is proven. Next
milestones (real Q) are in [`solution.md`](solution.md) §6–§9 and
[`gpu-simulation.md`](gpu-simulation.md) §4–§5.
