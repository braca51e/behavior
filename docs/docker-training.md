# Train π0.5 / GR00T with Docker (no host conda / uv)

**Official algorithms:** [BEHAVIOR Challenge Baselines — Train](https://behavior.stanford.edu/challenge/baselines.html#train)

**What this doc is:** a copy-paste path that works on a GPU box (e.g. Brev).  
You only need Docker + NVIDIA Container Toolkit + a GPU. Training runs **inside**
`b1k-pi05` / `b1k-groot`. OmniGibson LIVE eval stays on the host (`behavior392`).

---

## TL;DR — GR00T (most common path)

```bash
cd ~/behavior   # or your clone
git pull

# ---- once per machine ----
scripts/docker/build_groot.sh          # long; retries NVIDIA wheel downloads
pip install -U 'huggingface_hub[cli]'  # for demo download without `hf` CLI

# Accept gates in the browser (same HF account as the token):
#   https://huggingface.co/nvidia/Cosmos-Reason2-2B
#   https://huggingface.co/nvidia/GR00T-N1.7-3B
export HF_TOKEN=hf_...                 # NEW Read token — never paste into chat
# Optional: reuse an existing HF cache instead of data/cache/huggingface
export HF_CACHE=$HOME/.cache/huggingface

# ---- once per demo tree ----
scripts/docker/download_demos.sh 0     # task 0 = turning_on_radio → data/demos/
ls data/demos/meta/info.json           # MUST exist

# ---- train (auto-runs deploy-modality if meta/modality.json is missing) ----
scripts/docker/train_groot.sh train
# Checkpoints → data/checkpoints/groot/b1k-turning_on_radio/checkpoint-<step>/
```

**Different task:** download that task’s chunk, then  
`TASK_NAME=<task_name> scripts/docker/train_groot.sh train`  
(default `TASK_NAME` is `turning_on_radio`).

---

## TL;DR — π0.5

```bash
scripts/docker/build_pi05.sh
pip install -U 'huggingface_hub[cli]'
scripts/docker/download_demos.sh 0
ls data/demos/meta/info.json

export EXP_NAME=turning_on_radio_pilot
scripts/docker/train_pi05.sh norm-stats
BATCH_SIZE=8 scripts/docker/train_pi05.sh train
# Checkpoints under data/checkpoints/pi05/ …
```

---

## What stays on disk (do not re-download)

Containers use `--rm`. **All durable data is on the host.** Wrappers print the
three mounts every run:

| Host (defaults) | Container | Contents |
|---|---|---|
| `data/demos/` | `/data/demos` | LeRobot demos (`meta/info.json`, `data/chunk-000/`, videos) |
| `data/checkpoints/groot/` or `…/pi05/` | `/checkpoints` | Training checkpoints + OpenPI assets |
| `data/cache/huggingface/` | `/root/.cache/huggingface` | Cosmos / GR00T / tokenizers |

Override any path:

```bash
export DATA_ROOT=$PWD/data/demos
export CKPT_ROOT=$PWD/data/checkpoints/groot
export HF_CACHE=$HOME/.cache/huggingface   # share with other projects
```

After the first successful model download, later `train` runs should **not**
re-fetch multi‑GB weights (they may still touch small config files).

---

## Prerequisites (once)

1. **GPU + Docker**

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

2. **Build the image you need** (host never runs `uv` / `conda` for these stacks)

```bash
scripts/docker/build_groot.sh   # → image b1k-groot
# and/or
scripts/docker/build_pi05.sh    # → image b1k-pi05
```

GR00T build can take a long time (multi‑GB CUDA wheels from `pypi.nvidia.com`).
It retries automatically; re-run if the CDN times out.

3. **Demo download helper on the host**

```bash
pip install -U 'huggingface_hub[cli]'
```

`scripts/docker/download_demos.sh` uses the `hf` CLI if present, otherwise the
Python `huggingface_hub` API. No host conda required.

---

## Step-by-step: GR00T N1.7

### A. Hugging Face access

| Step | Detail |
|---|---|
| Gates | Open both model pages and accept access **with the same account** as your token |
| Token | Create a **Read** token at https://huggingface.co/settings/tokens |
| Export | `export HF_TOKEN=hf_...` in the shell that runs train (**do not commit or paste tokens**) |
| Verify | `train_groot.sh train` preflights `hf auth whoami` — must print your username |

`HF_TOKEN` **overrides** any cached `hf auth login`. A revoked or typo’d token
always returns `401` even if the website says “You have been granted access”.

### B. Download demos (task 0 = `turning_on_radio`)

```bash
scripts/docker/download_demos.sh 0
# EXPECT:
ls data/demos/meta/info.json
ls data/demos/data/chunk-000 | head
```

Full dataset is ~3.27 TB (100 chunks). Start with chunk `0` only.

### C. Deploy modality (automatic on train)

GR00T needs `data/demos/meta/modality.json`. Either:

```bash
scripts/docker/train_groot.sh deploy-modality
```

or just run `train` — the wrapper creates `modality.json` if it is missing.

### D. Train

```bash
export HF_TOKEN=hf_...
export HF_CACHE=$HOME/.cache/huggingface   # optional
scripts/docker/train_groot.sh train
```

Defaults inside the container:

| Env | Default | Meaning |
|---|---|---|
| `TASK_NAME` | `turning_on_radio` | Experiment / data association name |
| `EXP_NAME` | `b1k-$TASK_NAME` | Checkpoint folder name |
| `NUM_GPUS` | `1` | `>1` switches to `torchrun` |
| `GLOBAL_BATCH_SIZE` | `128` | Official doc uses `2048` on 8 big GPUs |
| `MAX_STEPS` | `150000` | Cap training steps |

Examples:

```bash
TASK_NAME=picking_up_trash scripts/docker/train_groot.sh train
NUM_GPUS=2 GLOBAL_BATCH_SIZE=256 scripts/docker/train_groot.sh train
MAX_STEPS=5000 scripts/docker/train_groot.sh train   # short pilot
```

### E. Serve + eval

```bash
# pick a step directory written under CKPT_ROOT
ls data/checkpoints/groot/b1k-turning_on_radio/

export PATH_TO_CKPT=$PWD/data/checkpoints/groot/b1k-turning_on_radio/checkpoint-XXXX
scripts/docker/train_groot.sh serve
# EXPECT: listening on 0.0.0.0:8000
```

On the host (second terminal), OmniGibson eval — same as the visual pilot:

```bash
conda activate behavior392
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES

python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir outputs/groot_eval \
  --max-steps 2000 \
  --no-headless
```

---

## Step-by-step: π0.5

```bash
scripts/docker/build_pi05.sh
scripts/docker/download_demos.sh 0

export EXP_NAME=turning_on_radio_pilot
# export HF_TOKEN=...   # recommended for HF assets

scripts/docker/train_pi05.sh norm-stats          # required once
BATCH_SIZE=8 scripts/docker/train_pi05.sh train  # raise if you have VRAM

find data/checkpoints/pi05 -type d | head
export PATH_TO_CKPT=/absolute/path/to/step_dir
export TASK_NAME=turning_on_radio
scripts/docker/train_pi05.sh serve
```

Then the same OmniGibson eval command as GR00T (§E), pointed at `:8000`.

---

## Architecture (why two places)

```
┌────────────────────────────┐         WS :8000         ┌──────────────────────────┐
│  Docker: b1k-pi05          │ ◀──────────────────────▶ │  Host: behavior392       │
│  or b1k-groot              │   (msgpack obs/action)   │  OmniGibson eval / LIVE  │
│  TRAIN + SERVE only        │                          │                          │
└────────────────────────────┘                          └──────────────────────────┘
```

| Workload | Where |
|---|---|
| Fine-tune π0.5 / GR00T | Docker (`b1k-pi05` / `b1k-groot`) |
| Serve policy on `:8000` | Same Docker image (`serve`) |
| LIVE Isaac / score Q | Host conda `behavior392` |
| This repo’s `echo` MVP | Host / root `Dockerfile` (not for real Q) |

Isaac Sim is **not** in the train images (size, EULA, GUI). That matches the
challenge’s policy-server ↔ simulator split.

---

## Repo layout

```
docker/pi05/     Dockerfile + entrypoint (norm-stats | train | serve)
docker/groot/    Dockerfile + entrypoint (deploy-modality | train | serve)
scripts/docker/
  build_pi05.sh / build_groot.sh
  download_demos.sh
  train_pi05.sh / train_groot.sh
  _persist_paths.sh          # shared DATA_ROOT / CKPT_ROOT / HF_CACHE
docs/docker-training.md      # this file
```

| Official CLI (baselines.html) | This repo |
|---|---|
| `uv run … compute_norm_stats.py pi05_b1k` | `train_pi05.sh norm-stats` |
| `uv run … train_b1k.py pi05_b1k` | `train_pi05.sh train` |
| `uv run … serve_b1k.py` | `train_pi05.sh serve` |
| `python … deploy_modality.py $DATA` | `train_groot.sh deploy-modality` |
| `torchrun … train_b1k.py` (GR00T) | `train_groot.sh train` |
| `python … serve_b1k.py` (GR00T) | `train_groot.sh serve` |

Skip training entirely: organizers publish a `turning_on_radio` checkpoint on the
[baselines page](https://behavior.stanford.edu/challenge/baselines.html#provided-checkpoints)
→ download → `serve` → eval.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FAIL: missing …/meta/info.json` | `scripts/docker/download_demos.sh 0` then `train` |
| `FileNotFoundError: …/meta/modality.json` | `scripts/docker/train_groot.sh deploy-modality` (or re-run `train` — it auto-deploys now) |
| `FileNotFoundError: 'hf'` on download | `pip install -U 'huggingface_hub[cli]'` then re-run `download_demos.sh` |
| `Invalid user token` / `401` gated repo | New Read token; accept Cosmos + GR00T gates; `HF_TOKEN` must be valid (`whoami` preflight). Revoke any token you pasted into chat |
| Website says access granted, still 401 | Bad/expired `HF_TOKEN` overrides login — unset and export a fresh token |
| Re-downloads multi‑GB models every run | Confirm mount line `persist HF models … → /root/.cache/huggingface`; set `HF_CACHE` to the dir that already has the blobs |
| `uv sync` timeout (`nvidia-cusparse` / pypi.nvidia.com) | Re-run `build_groot.sh` (BuildKit cache + retries) |
| OOM | Lower `GLOBAL_BATCH_SIZE` (GR00T) or `BATCH_SIZE` (π0.5) |
| Want LIVE window during **train** | Training has no scene UI — only **eval** is visual (`--no-headless`) |
| Eval can’t connect | `curl localhost:8000/healthz`; serve binds `0.0.0.0` |

---

## Checklist

- [ ] `docker … nvidia-smi` sees the GPU  
- [ ] `b1k-groot` and/or `b1k-pi05` image built  
- [ ] `pip install 'huggingface_hub[cli]'`  
- [ ] `download_demos.sh 0` → `data/demos/meta/info.json` exists  
- [ ] GR00T: gates accepted + `HF_TOKEN` passes whoami  
- [ ] GR00T: `deploy-modality` then `train`  
- [ ] π0.5: `norm-stats` then `train`  
- [ ] `serve` + host OmniGibson eval  

**After a green train:** wire into this repo’s serving stack or package a
submission — see `docs/solution.md` and `docs/gpu-simulation.md`.
