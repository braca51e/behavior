# 2026 BEHAVIOR Challenge — Requirements Spec

**Source:** https://behavior.stanford.edu/challenge/index.html and linked pages (Dataset, Baselines, Evaluation & Rules, Submission Guidelines, Demo Gallery, Updates), plus the canonical `StanfordVL/BEHAVIOR-1K` v3.9.2 evaluator source and the public Hugging Face demo dataset.
**Compiled:** 2026-09-11 (task `t_b5a9c819`).
**Purpose of this doc:** complete enough to build, train, and submit a winning entry without re-reading the challenge site. Where the site and the evaluator code disagree, the code (`eval_utils.py`, `score_utils.py`, v3.9.2) is treated as authoritative and flagged.

---

## 1. What the challenge is

The **2026 BEHAVIOR Challenge** (2nd edition) is a single-track embodied-AI benchmark. The goal:

> **Solve 100 full-length household tasks in the house-scale BEHAVIOR-1K / OmniGibson simulator** by writing a policy that combines long-horizon navigation, high-level reasoning, and dexterous bimanual manipulation, using **only onboard robot observations (RGB + depth + proprioception)**.

This is a *foundation-model / imitation-learning* challenge, not a one-off scripting contest. The 2026 edition consolidates to **one evaluation track** (the old 2025 "standard" vs "privileged" split is gone); the track is RGB + depth + proprioception only.

**At a glance**

| Item | Value |
|---|---|
| Tasks | 100 full-length household tasks |
| Environments | 7 scenes (4 new in 2026) |
| Evaluation track | One: RGB + depth + proprioception (robot onboard only) |
| Demonstrations | 20,000 human teleop demos, ~1,950 h total, 31 unique skills |
| Robot embodiment | **Not fixed** — default R1Pro, or any OmniGibson-supported custom robot |
| Baselines | π0.5 (OpenPI fork) and GR00T N1.7 (Isaac-GR00T fork) |
| Ranking metric | Average task success score (Q) with BDDL partial credit |
| Prize pool | **$11,000** — 1st $5,000 · 2nd $3,000 · 3rd $2,000 · Outstanding open-source $1,000 |

**Important dates**

| Milestone | Date |
|---|---|
| Challenge launch | 2026-07-02 |
| **Submission deadline** | **2026-10-16** |
| Winners announcement | 2026-11-04 |

Support: Discord community + office hours every **Monday 5–6 pm PT** over Zoom. A **participant registration form** was introduced (Aug 2026) and is required.

---

## 2. Inputs (what you get)

### 2.1 Demonstration dataset (training data)
Hosted on Hugging Face, **public**, MIT-licensed.

| Repo | Format | Size | Use |
|---|---|---|---|
| `behavior-1k/2026-challenge-demos` | **LeRobot v3.0** | **3.27 TB** | Primary training set: 20,000 teleop demos across 100 tasks (200 per task) |
| `behavior-1k/2026-challenge-rawdata` | Raw **HDF5** | 1.44 TB | Exact original trajectories; replay via `OmniGibson/scripts/learning/replay_obs.py` to collect extra visual obs |

Tasks are stored as **numbered chunks**: `chunk-000` = task 0, `chunk-001` = task 1, … `chunk-099` = task 99. You may download only the chunks you need (the full set is 3.27 TB). Canonical download pattern:

```bash
export TASK_ID=0
CHUNK=$(printf "chunk-%03d" $TASK_ID)
huggingface-cli download behavior-1k/2026-challenge-demos \
  --repo-type dataset --local-dir "$DATA_ROOT" \
  --include "data/$CHUNK/**" \
  --include "meta/episodes/$CHUNK/**" \
  --include "videos/*/$CHUNK/**" \
  --include "meta/info.json" \
  --include "meta/stats.json" \
  --include "meta/tasks.parquet"
```

Dataset layout (LeRobot v3): `annotations/`, `data/`, `meta/`, `videos/`.

Key `meta` files:
- `meta/info.json` — schema, fps, episode/frame counts (see §4).
- `meta/stats.json` — normalization statistics (recomputed after the Aug velocity fix).
- `meta/tasks.parquet` — per-task metadata.
- `meta/tasks.jsonl` — **natural-language description for all 100 tasks** (added Jul 2026). Tasks 0–49 reuse the 2025 wording (with typo fixes); 50–99 are new 2026-derived descriptions.
- Per-episode **language annotations** were also released (Jul 2026).
- `meta/episodes/chunk-*/file-*.parquet` — episode index → length mapping.

### 2.2 Observation modalities available during *training* (you may use any of these)
Privileged information is allowed during training as long as it is **not** used at challenge-track eval time. Available: RGB, depth, proprioception, actions, task/language info, skill/subtask annotations.

### 2.3 Observation modalities allowed during *evaluation* (hard restriction)
**Only RGB + depth + proprioception.** Forbidden at eval time:
- ground-truth segmentation
- object state / object pose
- target object pose
- full-scene point cloud
- **robot global pose**
- any other simulator-privileged information

You may **not** manipulate the environment directly (teleporting the robot, setting object states, etc.) during evaluation. The organizers manually inspect your wrapper code to enforce this.

### 2.4 BDDL task definitions
Usable and **identical during training and evaluation**. Each task's goal is a set of BDDL predicates; success is measured by how many of those predicates are satisfied at episode end (see §5).

---

## 3. Outputs / what a submission must produce

A submission is a **served policy** (not a monolithic script). The challenge is a client–server loop:

1. Your **policy server** listens on a WebSocket.
2. The **OmniGibson evaluator** (`omnigibson.eval.eval`, run v3.9.2) drives the sim, sends flattened observations to your server each step.
3. Your server returns a **msgpack-encoded** response containing a single **action array** (`robot.action_dim` floats).
4. Per successful rollout the evaluator writes a metrics JSON; with `--write-video` it also records head + both wrist camera MP4s.

**The final deliverable package (zip) must contain:**
- Your policy server code (`.py`) implementing the WebSocket interface.
- The **exact** robot config (`.yaml`) — the bundled `omnigibson/eval/r1pro.yaml` or your custom `--robot-config`.
- The **evaluation wrapper** code (proves obs restrictions are honored).
- The `eval.camera_sensor_names` mapping (head / left_wrist / right_wrist roles) — required for video writing.
- **Evaluation outputs**: one JSON per rollout, named `<task_name>_<instance_id>_<rollout_id>.json`, inside a folder named `<track>.<testset>.<team>.<affiliation>.<date>/json/`.
- A **portal link** to **all rollout MP4 videos** (one per rollout, up to **1,000** videos).

**You must not modify** the output JSON files or rollout videos in any way.

**Serving model options (choose one):**
- **Docker (recommended):** submit a Docker image that serves the policy; organizers run OmniGibson *outside* the container and connect over the WebSocket policy client. Must run on a **single 24 GB VRAM GPU** (final eval hardware: RTX 3090, A5000, Titan RTX).
- **IP-based:** host the policy yourself and expose an IP; must open **at least 50 ports** for parallel evaluation. Typical stacks: TorchServe, LitServe, vLLM, NVIDIA Triton, or equivalent.

**Confidentiality:** submissions stay confidential unless you explicitly grant disclosure. Open-source strongly encouraged (and separately eligible for the $1,000 Outstanding Open-Source prize). If your solution depends on external model/API calls, you must provide the credentials, quota, and serving config; **organizers do not cover external API costs.**

---

## 4. Data format — exact tensor schema (R1Pro, from `meta/info.json`)

Dataset is **30 fps**, LeRobot codebase version **v3.0**, `total_episodes = 20,000`, `total_frames = 210,916,774`, `total_tasks = 100`, `chunks_size = 1000`.

Per-frame features:

| Feature | Shape | dtype |
|---|---|---|
| `action` | **23** | float32 |
| `observation.state` | **61** (proprio) | float32 |
| `observation.rgb.head (zed_link)` | **720×720×3** | video (H.264) |
| `observation.rgb.left_wrist` | **480×480×3** | video |
| `observation.rgb.right_wrist` | **480×480×3** | video |
| `observation.depth_linear.head (zed_link)` | **720×720×1** | video (linear depth) |
| `observation.depth_linear.left_wrist` | **480×480×1** | video |
| `observation.depth_linear.right_wrist` | **480×480×1** | video |
| `observation.robot2cam_pose.*_camera_0` (×3 cams) | **7** (pos + quat) | float32 |
| `next.reward` | 1 | float32 |
| `next.terminated` / `next.truncated` | 1 | bool |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | 1 | float/int |

### 4.1 Action space — 23-dim (`ACTION_QPOS_INDICES["R1Pro"]`)
| Slice | Group | DoF |
|---|---|---|
| `[0:3]` | base (holonomic; velocity) | 3 |
| `[3:7]` | torso / trunk (position) | 4 |
| `[7:14]` | left arm (position) | 7 |
| `[14:15]` | left gripper (single scalar) | 1 |
| `[15:22]` | right arm (position) | 7 |
| `[22:23]` | right gripper (single scalar) | 1 |

Base is a `HolonomicBaseJointController` (velocity, limits ±1.0 cmd / ±0.75 x-y, ±1.0 yaw out). Arms + trunk are `JointController` (position, kp=150). Grippers are `MultiFingerGripperController` (smooth mode). `grasping_mode: assisted`. `action_normalize: false`.

### 4.2 Proprioception — 61-dim (`PROPRIOCEPTION_INDICES["R1Pro"]`)
| Slice | Field |
|---|---|
| `[0:3]` | `base_qvel` (robot-local vx, vy, yaw-rate — see §9 fixes) |
| `[3:10]` | `arm_left_qpos` |
| `[10:17]` | `arm_left_qvel` |
| `[17:20]` | `eef_left_pos` |
| `[20:24]` | `eef_left_quat` |
| `[24:26]` | `gripper_left_qpos` |
| `[26:28]` | `gripper_left_qvel` |
| `[28:35]` | `arm_right_qpos` |
| `[35:42]` | `arm_right_qvel` |
| `[42:45]` | `eef_right_pos` |
| `[45:49]` | `eef_right_quat` |
| `[49:51]` | `gripper_right_qpos` |
| `[51:53]` | `gripper_right_qvel` |
| `[53:57]` | `trunk_qpos` |
| `[57:61]` | `trunk_qvel` |

### 4.3 Camera intrinsics (R1Pro, `CAMERA_INTRINSICS`)
- **Head (zed, 720×720):** fx=fy=306.0, cx=cy=360.0
- **Wrists (left/right realsense, 480×480):** fx=fy=388.6639, cx=cy=240.0

### 4.4 Demo statistics (site "Dataset Statistics")
270,600 skills total, **31 unique skills**, **27.06 skills/trajectory**, **avg trajectory duration 351.54 s ≈ 5.9 min**. Representative skill verbs include: attach, chop, close door/drawer/lid, hand over, hang, hold, ignite, insert, move to, open door/drawer/lid, pick up from, place in/next to/on/under, pour, press, push to, release, spray, sweep surface, tip over, turn off/on switch, turn to, wipe hard. Expect open/close/pour/wipe/spray/toggle/cook/slice style multi-skill compositions.

---

## 5. Scoring — exactly how Q is computed

### 5.1 Primary metric (the only ranking metric): **Q = task success score**
- **Per rollout:** `q_score.final` = *(# goal BDDL predicates satisfied at episode end) / (# total goal predicates)*. Range 0.0–1.0. Partial credit is built in — a half-done task scores 0.5.
- **Per task:** average `q_score.final` over the task's scored instances.
- **Overall Q** = mean of the per-task averages across all **100 tasks**. This single number ranks the leaderboard.

Also reported (not used for ranking beyond ties): `task_sr` = fraction of scored instances with `q_score.final == 1.0` (full success rate).

### 5.2 Secondary / efficiency metrics (tie-breakers)
Normalized against **human averages from 200 demos per task** (so 1.0 = human-typical efficiency; lower distance/time is better):
- **Simulated time** → `time_score`. Computed as:
  `time_score = 1.5/(1.5−1) − 1/((1.5−1)·normalized_time) = 3 − 2/normalized_time`
  (i.e., faster-than-human → >1; at 1.5× human time → 0; slower → negative.) `EVAL_TIMEOUT_MULTIPLIER = 1.5`.
- **Distance navigated** (base body accumulated travel) → `normalized_agent_distance.base`.
- **End-effector displacement** (left & right hands accumulated) → `normalized_agent_distance.left` / `.right`.

Ties on Q are broken by these secondary metrics.

### 5.3 Episode timeout
`--max-steps` optional. **If omitted**, per-episode timeout = **1.5× the mean human demonstration length** (in simulator steps) for that task.

---

## 6. Evaluation protocol & instance split (important for self-scoring)

Constants (`eval_utils.py`): `NUM_TEST_INSTANCES = 40`, `NUM_PUBLIC_TEST_INSTANCES = 20`, `NUM_HIDDEN_TEST_INSTANCES = 20`, `TEST_INSTANCE_IDS = [301, 302, …, 340]`.

So **each task has 40 pre-sampled test instances** (sim instance IDs 301–340). The `--instance-indices` flag is **0-based into this 40-entry list** (0→301 … 19→320 … 39→340).

| Phase | Instances used | `--instance-indices` |
|---|---|---|
| **Self-evaluation / leaderboard reporting** | first **10 public** | `0 1 2 3 4 5 6 7 8 9` |
| Scoring supports `testset=public` | 20 public (301–320) | indices 0–19 |
| Scoring supports `testset=hidden` | 20 hidden (321–340) | indices 20–39 |
| **Final evaluation** (organizers) | held-out instances; leaderboard frozen, **top-5 re-run** | hidden set |

- **`--num-rollouts 1`** per scored instance for reported/final results.
- Each instance differs in **initial object states and initial robot pose**.
- The simulator is **nondeterministic** — the same policy can give different results across rollouts of the same instance. This is expected; **do not cherry-pick** rollouts.
- **Partial scenes:** evaluation loads only the exact rooms specified for the task in `B100_task_misc.csv` (column "Rooms to inlcude"), keeping the eval scene consistent with task metadata (fixed in v3.9.2).

**⚠ Discrepancy to note:** the human-facing docs say "20 extra instances per task; report on the first 10 public; hold out **10** more for final," but the shipped scorer defines **20 public + 20 hidden**. Plan against the **code** (40 instances/task, 1-rollout-each scoring on either the 20-public or 20-hidden set) and treat the docs' "10" as the minimal reporting slice. Confirm at office hours.

---

## 7. Reference implementation commands

**Repo:** `StanfordVL/BEHAVIOR-1K`, **tag `v3.9.2`** (not v3.9.0). Install:
```bash
git clone -b v3.9.2 https://github.com/StanfordVL/BEHAVIOR-1K.git $PATH_TO_BEHAVIOR_1K
cd $PATH_TO_BEHAVIOR_1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
```

**Run the evaluator** (after starting your policy server on `--host/--port`):
```bash
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 1 2 3 4 5 6 7 8 9 \
  --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir outputs/b1k_eval \
  --write-video
```

Evaluator flags (reference):
- `--task-name` — BEHAVIOR task id (e.g. `turning_on_radio`).
- `--host`/`--port` — WebSocket policy server address (default port 8000). Evaluator waits for a `/healthz` health check, then opens the WebSocket.
- `--instance-indices` — 0-based indices into the task's 40 test instances (see §6).
- `--num-rollouts` — rollouts per instance (use 1 for reporting/final).
- `--max-steps` — optional episode timeout; default = 1.5× human mean length.
- `--env-wrapper` — full target path. `omnigibson.eval.wrappers.DefaultWrapper` (224×224 RGB + proprio, debug, no depth) or `omnigibson.eval.wrappers.RGBDFullResWrapper` (official 720/480 RGB + depth). **Use RGBDFullResWrapper for the real challenge track.**
- `--output-dir` — JSON metrics go to `<output-dir>/json/`.
- `--write-video` — **required** for challenge outputs; writes head+wrist MP4s to `<output-dir>/videos/`.
- `--video-fps`, `--headless`/`--no-headless`, `--robot-config` (custom robot yaml).

**Custom robot:** pass `--robot-config path/to/my_robot.yaml`. Must be a complete canonical OmniGibson robot dict (`model` [lowercase model id, not the deprecated `type`], `name` [default `robot_r1`], `controller_config`, `obs_modalities`, `proprio_obs`, `sensor_config`, `action_normalize`, `grasping_mode`, plus `eval.camera_sensor_names` for the head/left_wrist/right_wrist roles). The evaluator **overwrites `position`/`orientation` at runtime** with the instance's prescribed start pose. The returned action array must match the config's `robot.action_dim`. If your robot isn't in OmniGibson yet, use the custom-robot-import tutorial.

**WebSocket protocol:** evaluator client = `omnigibson.eval.policies.WebsocketPolicy`; helper server = `omnigibson.eval.utils.network_utils.WebsocketPolicyServer`. Server receives flattened obs, returns a msgpack-encoded action array. Baseline servers (OpenPI / GR00T) expose compatible endpoints.

### 7.1 Per-rollout metrics JSON schema (do not edit)
```json
{
  "task": "turning_on_radio",
  "instance_id": 0,
  "rollout_id": 0,
  "steps": 500,
  "success": false,
  "agent_distance":   { "base": 2.0, "left": 1.5, "right": 1.2 },
  "normalized_agent_distance": { "base": 1.5, "left": 1.2, "right": 1.1 },
  "q_score": { "final": 0.4 },
  "time": { "simulator_steps": 500, "simulator_time": 16.666, "normalized_time": 1.6 }
}
```

### 7.2 Baselines (reference pipelines — same common setup + eval; only train/serve differ)
- **π0.5** via challenge fork of **OpenPI** (`github.com/wensi-ai/openpi`, branch `behavior`): config `pi05_b1k`, R1Pro, 32-step action horizon; `compute_norm_stats.py`, `scripts/b1k/train_b1k.py`/`.sh`/`.sbatch.sh`; serve `scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/$TASK --control_mode receding_horizon --action_horizon 16 --port 8000`.
- **GR00T N1.7** via challenge fork of **Isaac-GR00T** (`github.com/wensi-ai/Isaac-GR00T`): fine-tunes `nvidia/GR00T-N1.7-3B` (backbone `nvidia/Cosmos-Reason2-2B` is **gated** — accept the gate), 16-step action horizon, `NEW_EMBODIMENT` tag, `examples/b1k/r1pro.py`; `deploy_modality.py`; serve `scripts/b1k/serve_b1k.py` (temporal ensembling on by default).
- Fine-tuned **checkpoints are provided** per task (e.g. `turning_on_radio`) so you can validate the full eval/submission pipeline without training.
- Both use `uv` for training deps and a `behavior` conda env for the OmniGibson evaluator.

---

## 8. Hardware / performance expectations

**Benchmark machine:** RTX 4090 (24 GB), Ryzen 9 7950X (16-core/32-thread), 128 GB RAM, Ubuntu 22.04.5.
- **Scene load:** ~150–300 s one-time per trial (varies by scene).
- **FPS with random actions** (reference): RGB-only 224 = 24.55; RGB 720/480 = 20.62; RGB+depth 224 = 16.55; **RGB+depth 720/480 (official track) = 13.52 FPS**.
- **Final eval hardware:** single **24 GB** GPUs (RTX 3090 / A5000 / Titan RTX). Your policy must fit in 24 GB.

Budget for sim time: 100 tasks × 10 instances × ~1 rollout, each up to 1.5× a ~6-min human trajectory ≈ 1000 episodes. At ~13.5 sim-FPS with scene-load overhead, a full self-eval is a **multi-hour, GPU-cluster-sized** job — plan the rollout harness for parallelism (and 50-port serving if you go the IP route).

---

## 9. Known issues / mandatory fixes (from the Updates page) — build these in

**Use `v3.9.2` for everything** (evaluation + replay). v3.9.0 is deprecated for the challenge.

**Aug 2026 (08/24):**
- Demo dataset **velocity fields corrected** to raw sim joint velocities; `meta/stats.json` recomputed. Affected `observation.state` slices: `[10:17]` arm_left_qvel, `[26:28]` gripper_left_qvel, `[35:42]` arm_right_qvel, `[51:53]` gripper_right_qvel, `[57:61]` trunk_qvel. **Re-sync your local demo dataset and recompute norm stats.**
- **Partial-scene eval** now loads the exact rooms from `B100_task_misc.csv`.
- **RGBDFullResWrapper** obs loading fixed (refresh sim handles after changing camera resolution).
- Leaderboard + submission form bug fixes; **new participant registration form**.

**Jul 2026 (07/27):**
- `observation.state[0:3]` now records **R1Pro base velocity in the robot-local frame** (previously raw holonomic joint velocities). `holonomic_base_qvel_to_robot_frame` rotates x/y by base yaw and keeps yaw-rate as component 3. **Matches the action convention** — do the same transform if you build your own base controller.
- **Depth videos fixed** (see HF discussion `behavior-1k/2026-challenge-demos` #2). Re-sync depth tracks.
- `meta/tasks.jsonl` (NL descriptions) and per-episode language annotations added.

---

## 10. Edge cases & failure modes to handle in your solution

1. **Partial credit dominates the score.** Since Q averages over all 100 tasks with fractional BDDL credit, a policy that reliably satisfies *some* goal predicates on every task beats one that fully solves 30 and 0-scores 70. Design for robustness across the task distribution, not max-coverage on a few.
2. **Long horizon (avg ~6 min, timeout 1.5×).** You need memory / subgoal structure to survive multi-skill compositions without forgetting the goal or the robot's own position (note: robot **global pose is NOT observable** at eval — you cannot read the base position from obs; you must estimate it, e.g. via SLAM/integration).
3. **Non-privileged navigation only.** With RGB+depth+proprio and no global pose, cross-room search is the hard part. Expect to build SLAM or an occupancy/traversal estimate from onboard vision.
4. **Nondeterministic sim.** Do not cherry-pick; score on the fixed instance sets, single rollout each.
5. **Object-state transitions.** Tasks require state changes (open/close, pour, heat/cook, wipe, toggle, slice, attach). These are BDDL predicates; your policy must physically cause the transition, not just reach a pose.
6. **Efficiency tie-breakers.** Once Q is high, beating rivals comes from beating the human-normalized time and EEF/base displacement — i.e., don't wander; take direct routes and minimal hand motion.
7. **Embodiment freedom.** You may switch robots/controllers to improve actionability, but the action array must match your submitted config and the wrapper must still expose only RGB/depth/proprio. Any custom robot must be OmniGibson-supported and included verbatim.
8. **24 GB VRAM cap at final eval.** If you train a big model, you must quantize/distill/serve within 24 GB, or self-host with 50+ ports.
9. **Dataset size (3.27 TB).** Per-task chunk downloads + `--decode-only-used-frames`-style optimizations are essential; you cannot naively stream all videos.
10. **Registration required** before you can report to the leaderboard / submit.

---

## 11. What a *winning* solution must satisfy (acceptance checklist)

- [ ] Policy consumes **only** RGB (head 720×720 + wrists 480×480) + depth + 61-dim proprio at eval; no privileged info, no env manipulation; wrapper code included and inspectable.
- [ ] Runs on **v3.9.2**, with **RGBDFullResWrapper** for the official track; `/healthz` WebSocket server returning msgpack action arrays matching `robot.action_dim` (23 for R1Pro default).
- [ ] Scored correctly: `--instance-indices 0–9`, `--num-rollouts 1`, `--write-video` on, outputs in `<track>.<testset>.<team>.<affiliation>.<date>/json/` + videos.
- [ ] **Fits a single 24 GB GPU** (or self-hosted ≥50 ports).
- [ ] Maximizes **Q = mean over 100 tasks of (BDDL goal-predicate satisfaction fraction)**; secondary: beat human-normalized time/distance.
- [ ] Robust across all 100 task families (attach, open/close, pour, wipe, spray, ignite, chop/slice, toggle, rearrange, place variants, etc.), with multi-room search and object-state transitions.
- [ ] Reproducible end-to-end from submitted files; registered participant; external-API credentials provided if used.
- [ ] Optional but scoring-relevant: open-source (→ $1,000 prize).

**The 100 task ids** (chunk index 0–99) are listed in `raw/tasks_100.txt` and `raw/tasks.jsonl` (also in the repo `docs/` if desired). First 12: turning_on_radio, picking_up_trash, putting_away_Halloween_decorations, cleaning_up_plates_and_food, can_meat, setting_mousetraps, hiding_Easter_eggs, picking_up_toys, rearranging_kitchen_furniture, putting_up_Christmas_decorations_inside, set_up_a_coffee_station_in_your_kitchen, putting_dishes_away_after_cleaning; last 3: setup_a_bar_for_a_cocktail_party, laying_tile_floors, sorting_books_on_shelf.

---

## 12. Where to find things (canonical links)

- Repo + evaluator (v3.9.2): https://github.com/StanfordVL/BEHAVIOR-1K
- Demo dataset: https://huggingface.co/datasets/behavior-1k/2026-challenge-demos
- Raw HDF5: https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata
- Leaderboard: https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard
- Task list / Demo Gallery: https://behavior.stanford.edu/challenge/tasks/index.html
- Rules / Submission / Dataset / Baselines: under https://behavior.stanford.edu/challenge/
- Discord: https://discord.gg/bccR5vGFEx
- Baseline forks: `github.com/wensi-ai/openpi` (branch `behavior`), `github.com/wensi-ai/Isaac-GR00T`
- Cite: Li et al., "BEHAVIOR-1K: A Human-Centered Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation."

### Raw evidence captured locally (this workspace `raw/`)
`challenge_index.html/.txt` (landing), `dataset.txt`, `baselines.txt`, `evaluation.txt`, `submission.txt`, `tasks_index.html`, `updates.txt`, `tasks.jsonl` + `tasks_100.txt` (100 NL descriptions), `demo_info.json` (LeRobot schema), `r1pro.yaml` (default robot), `eval_utils.py` (constants: resolutions, action/proprio slices, instance ids, camera matrices), `score_utils.py` (Q + time_score formulas, submission folder contract), `2025_test_instances.csv` + `2025_misc.csv` (instance/room metadata format).
