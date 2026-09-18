# Docker training for challenge baselines (π0.5 + GR00T) — no host conda/uv

**Official source of truth for the algorithms:**  
[BEHAVIOR Challenge Baselines — Train](https://behavior.stanford.edu/challenge/baselines.html#train)

**What this repo adds:** two Docker images that bake the official OpenPI and
Isaac-GR00T training stacks at **build time**, so on the host you only need:

- Docker Engine + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- A GPU
- The LeRobot demo chunks on disk

You do **not** install `conda`, `uv`, or a local OpenPI/GR00T venv on the host.

---

## 1. Why these images exist

The official walkthrough forces two host-side package managers:

| Official step | Host pain |
|---|---|
| OpenPI / GR00T setup | `uv sync` + project venvs + submodules |
| OmniGibson eval | separate `conda` env (`behavior` / `behavior392`) + Isaac Sim |

That split is intentional in the challenge design (policy server ↔ simulator
over WebSocket), but it is painful to reproduce on a laptop.

**Our split (same architecture, less pain):**

```
┌────────────────────────────┐         WS :8000         ┌──────────────────────────┐
│  Docker: b1k-pi05          │ ◀──────────────────────▶ │  Host: behavior392       │
│  or b1k-groot              │   (msgpack obs/action)   │  OmniGibson eval / LIVE  │
│  TRAIN + SERVE only        │                          │  (already set up here)   │
└────────────────────────────┘                          └──────────────────────────┘
```

| Workload | Where | Image / env |
|---|---|---|
| Fine-tune π0.5 | Docker | `b1k-pi05` |
| Fine-tune GR00T N1.7 | Docker | `b1k-groot` |
| Serve trained policy | Docker | same images (`serve`) |
| Simulate / score / LIVE window | **Host** | `conda activate behavior392` (Isaac Sim does not belong in these slim train images) |
| This repo’s `echo` MVP | Host or existing `Dockerfile` | not for real Q |

**Why not one mega-image with Isaac Sim?** Isaac Sim 5.1 + assets is tens of
GB, needs the NVIDIA EULA, and GUI/`DISPLAY` plumbing fights containers. The
challenge itself expects OmniGibson **outside** the policy container
(see root `Dockerfile` comments). Keep sim on the host env you already have.

**Why bake `uv` into the image?** So *you* never run it. Build once; runtime
`PATH` is already the project `.venv`.

---

## 2. What you train (and why)

Both baselines learn from the same public demos:
[`behavior-1k/2026-challenge-demos`](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos)
(LeRobot v3.0, ~3.27 TB full; download **one chunk** to pilot).

### 2.1 π0.5 (OpenPI `behavior` fork)

| | |
|---|---|
| **Why** | Official VLA baseline; this repo’s `policy.backend: vla` path targets the same family |
| **Code** | [`wensi-ai/openpi` @ `behavior`](https://github.com/wensi-ai/openpi) |
| **Config** | `pi05_b1k` — R1Pro, 32-step action horizon, base weights `gs://openpi-assets/checkpoints/pi05_base/params` |
| **Does** | Fine-tunes π0.5 on challenge demos; serves WebSocket policy for the evaluator |
| **Good when** | You want the primary challenge baseline / VLA route |

Flow:

1. **Norm stats** — dataset mean/std for actions/state (required before train).
2. **Train** — `scripts/b1k/train_b1k.py pi05_b1k …`
3. **Serve** — `scripts/b1k/serve_b1k.py` on port 8000.
4. **Eval** — host `omnigibson.eval.eval` → that port (same as `LIVE=1` pilot).

### 2.2 GR00T N1.7 (Isaac-GR00T challenge fork)

| | |
|---|---|
| **Why** | Second official baseline; strong alternative VLA stack |
| **Code** | [`wensi-ai/Isaac-GR00T`](https://github.com/wensi-ai/Isaac-GR00T) |
| **Base** | `nvidia/GR00T-N1.7-3B` + gated backbone [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) |
| **Does** | Fine-tunes projector + diffusion action head (vision/LLM frozen); 16-step horizon; R1Pro via `examples/b1k/r1pro.py` |
| **Good when** | You want the GR00T baseline or a second opinion vs π0.5 |

Flow:

1. **Accept HF gates** + set `HF_TOKEN`.
2. **deploy-modality** — write `meta/modality.json` into the demo tree.
3. **Train** — `scripts/b1k/train_b1k.py …`
4. **Serve** — `scripts/b1k/serve_b1k.py` on port 8000.
5. **Eval** — same host OmniGibson client as π0.5.

### 2.3 Skip training?

Organizers ship a ready checkpoint for `turning_on_radio` (π0.5 and GR00T) on
the [baselines page](https://behavior.stanford.edu/challenge/baselines.html#provided-checkpoints).
Download → `serve` with that path → eval. Training is only required when you
want your own weights / more tasks.

---

## 3. Layout in this repo

```
docker/
  pi05/
    Dockerfile          # clones OpenPI behavior, uv sync at build, PATH=.venv
    entrypoint.sh       # norm-stats | train | serve | shell
  groot/
    Dockerfile          # clones wensi-ai/Isaac-GR00T, uv sync at build
    entrypoint.sh       # deploy-modality | train | serve | shell
scripts/docker/
  build_pi05.sh
  build_groot.sh
  download_demos.sh
  train_pi05.sh         # thin docker run wrapper
  train_groot.sh
docs/docker-training.md # this file
```

Mounts used by the wrappers:

| Host path | Container | Purpose |
|---|---|---|
| `DATA_ROOT` (default `data/demos`) | `/data/demos` | LeRobot demos |
| `CKPT_ROOT` | `/checkpoints` | Norm assets + checkpoints |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HF / model cache |

---

## 4. One-time host prerequisites

```bash
# NVIDIA driver + Docker + GPU runtime
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# EXPECT: same GPU visible inside the container
```

Optional: `hf` CLI on the host only for downloading demos (or use
`scripts/docker/download_demos.sh`, which falls back to this repo’s Python
downloader).

---

## 5. Download demos (pilot = task 0)

```bash
cd /home/luis/Documents/Behavior_Challenge
chmod +x scripts/docker/*.sh

# Optional: hf auth login   # or export HF_TOKEN=...
scripts/docker/download_demos.sh 0
# EXPECT: data/demos/data/chunk-000/ … + meta/ + videos/
```

Full 100-task dump is multi-TB — start with one chunk.

---

## 6. Build the images (once)

```bash
# π0.5 — long first build (clone + uv sync + CUDA stack)
scripts/docker/build_pi05.sh
# EXPECT: "built b1k-pi05"

# GR00T — similarly long
scripts/docker/build_groot.sh
# EXPECT: "built b1k-groot"
```

Build uses `uv` **inside Docker only**. Host never needs `uv`.

---

## 7. Train π0.5 (step by step)

Defaults are sized for a **single ~16–24 GB GPU** (`BATCH_SIZE=8`). The official
doc uses `batch_size=64` on large multi-GPU boxes — override when you have the
VRAM.

```bash
export DATA_ROOT=$PWD/data/demos
export CKPT_ROOT=$PWD/data/checkpoints/pi05
export EXP_NAME=turning_on_radio_pilot
# export HF_TOKEN=...   # optional but recommended

# 1) Norm stats (writes under the OpenPI outputs tree / mounted cache)
scripts/docker/train_pi05.sh norm-stats

# 2) Fine-tune
BATCH_SIZE=8 scripts/docker/train_pi05.sh train
# Checkpoints: typically under CKPT_ROOT / OpenPI outputs/checkpoints/pi05_b1k/$EXP_NAME/<step>
```

Find a step dir:

```bash
find data/checkpoints/pi05 -type d -name '*000*' | head
# set PATH_TO_CKPT to a concrete step directory
```

Serve:

```bash
export PATH_TO_CKPT=/absolute/path/to/step_dir
export TASK_NAME=turning_on_radio
scripts/docker/train_pi05.sh serve
# EXPECT: listening on :8000; curl localhost:8000/healthz → 200
```

Eval on the host (second terminal) — same contract as the visual pilot:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior392
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES

python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir outputs/pi05_eval \
  --max-steps 2000 \
  --no-headless          # LIVE window; omit and use --headless for MP4-only
```

Or point this repo’s echo pilot script at an already-running server (serve first,
then run evaluator only).

---

## 8. Train GR00T N1.7 (step by step)

```bash
# 1) Accept the gate on Hugging Face for:
#    https://huggingface.co/nvidia/Cosmos-Reason2-2B
#    https://huggingface.co/nvidia/GR00T-N1.7-3B
export HF_TOKEN=hf_xxx

export DATA_ROOT=$PWD/data/demos
export CKPT_ROOT=$PWD/data/checkpoints/groot
export TASK_NAME=turning_on_radio

# 2) Deploy modality config into the demo tree
scripts/docker/train_groot.sh deploy-modality

# 3) Fine-tune (1 GPU defaults; raise NUM_GPUS / GLOBAL_BATCH_SIZE on big boxes)
NUM_GPUS=1 GLOBAL_BATCH_SIZE=128 scripts/docker/train_groot.sh train
# Checkpoints → data/checkpoints/groot/b1k-turning_on_radio/checkpoint-<step>/

# 4) Serve
export PATH_TO_CKPT=$PWD/data/checkpoints/groot/b1k-turning_on_radio/checkpoint-XXXX
scripts/docker/train_groot.sh serve
```

Eval: identical OmniGibson command as §7 (point at `:8000`).

---

## 9. How the entrypoints map to the official CLI

| Official (baselines.html) | This image |
|---|---|
| `uv run scripts/compute_norm_stats.py pi05_b1k …` | `b1k-pi05 norm-stats` |
| `uv run scripts/b1k/train_b1k.py pi05_b1k …` | `b1k-pi05 train` |
| `uv run scripts/b1k/serve_b1k.py …` | `b1k-pi05 serve` |
| `python scripts/b1k/deploy_modality.py $DATA_ROOT` | `b1k-groot deploy-modality` |
| `torchrun … scripts/b1k/train_b1k.py …` | `b1k-groot train` (`NUM_GPUS>1` uses `torchrun`) |
| `python scripts/b1k/serve_b1k.py …` | `b1k-groot serve` |

Raw `docker run` examples (wrappers just set mounts):

```bash
docker run --rm -it --gpus all \
  -v $PWD/data/demos:/data/demos \
  -v $PWD/data/checkpoints/pi05:/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  b1k-pi05 train
```

---

## 10. Honest limits / troubleshooting

| Issue | Fix |
|---|---|
| Want LIVE Isaac window during train | Training has no scene UI — only **eval serve** is visual; use `--no-headless` on host eval |
| OOM on 16 GB | Lower `BATCH_SIZE` (π0.5) or `GLOBAL_BATCH_SIZE` (GR00T); close other GPU apps |
| GR00T 401 / gated model | Accept HF gates + `export HF_TOKEN=…` |
| `scripts/b1k` missing in image | Rebuild from `wensi-ai/Isaac-GR00T` (challenge fork), not a generic NVIDIA GR00T tree |
| `uv sync` timeout on `nvidia-cusparse-cu12` / pypi.nvidia.com | Flaky CDN while pulling multi‑GB CUDA wheels. Re-run `scripts/docker/build_groot.sh` (BuildKit caches successful downloads; Dockerfile retries 6× with `UV_HTTP_TIMEOUT=600`) |
| Norm-stats / checkpoints not on host | Confirm `CKPT_ROOT` mount; browse with `find data/checkpoints -type d \| head` |
| Eval can’t connect | `curl localhost:8000/healthz`; serve must bind `0.0.0.0` (GR00T entrypoint does; π0.5 uses OpenPI’s server) |
| Still need this repo’s System-2 server | After training, either use OpenPI/GR00T `serve_b1k.py` **or** wire weights into `configs/server.yaml` `policy.backend: vla` (OpenPI integration seam) |

---

## 11. Checklist

- [ ] `docker run --gpus all … nvidia-smi` works  
- [ ] `scripts/docker/download_demos.sh 0` populated `data/demos/data/chunk-000`  
- [ ] `scripts/docker/build_pi05.sh` and/or `build_groot.sh` finished  
- [ ] π0.5: `norm-stats` then `train` then `serve` + host eval  
- [ ] GR00T: HF gates + `deploy-modality` then `train` then `serve` + host eval  
- [ ] (Optional) compare Q vs the provided Drive checkpoints on the baselines page  

**Next after a green train:** package submission / wire into this repo’s serving
stack — see `docs/solution.md` and `docs/gpu-simulation.md`.
