# 2026 BEHAVIOR Challenge — Winning Solution Architecture

**Task:** `t_08f25ebb` · **Inputs:** `docs/challenge-spec.md` (authoritative), `raw/eval_utils.py`, `raw/score_utils.py`, `raw/r1pro.yaml`, `raw/demo_info.json`, `docs/task-ids-100.txt`
**Deliverable of this doc:** strategy, tradeoffs, module boundaries, concrete interfaces, and a 5-week implementation plan that fits the 2026-10-16 deadline.

---

## 1. What winning actually requires (scoring decomposition)

The only ranked metric is

```
Q = mean over 100 tasks of ( mean over scored instances of ( #goal_BDDL_predicates_satisfied / #total_goal_predicates ) )
```

Three consequences drive the whole architecture:

1. **Partial credit dominates.** A 0.5 on every task beats a 1.0 on 30 tasks and 0.0 on 70. We must earn *some* predicates on *all 100* task families, not max out a few. Every capability we add must have non-zero value across the long tail.
2. **Robustness is the differentiator, not peak dexterity.** The 200 demos/task give us per-skill statistics; the failure modes to beat are drift, forgetting the goal over ~6-minute horizons, getting lost across rooms (robot global pose is **not observable** at eval), and stuck loops. A policy that survives 10,000+ sim steps per episode without degradation is worth more than one that executes harder motions.
3. **Efficiency tie-breakers are a free second prize.** `time_score = 3 − 2/normalized_time` and normalized base/EEF displacement reward (a) early stopping the moment all BDDL predicates are satisfied, and (b) direct routes. Both come almost for free from a working progress tracker (see §6) — implement progress tracking from day one.

**Target:** Q ≥ 0.55 by deadline with a clean margin on the efficiency metrics. (Baselines fine-tune one VLA per task; a single shared, skill-conditioned, recovery-equipped system with per-task adapters should comfortably clear them on mid/hard tasks and match them on easy ones.)

---

## 2. Strategy overview (and why)

**Chosen approach: hierarchical "System 2 / System 1" embodied agent on the default R1Pro embodiment, built from the two provided baselines (π0.5 OpenPI fork = primary VLA; GR00T N1.7 = backup).**

Four layers:

| Layer | Role | Implementation |
|---|---|---|
| **Planner** | Task NL + BDDL goal → ordered skill plan; tracks which skill is active; emits language subgoals | Rule-based skill-sequence miner over the 20k demos' per-episode language annotations (31-skill taxonomy) + per-task plan templates. No LLM at eval. |
| **Perception** | Self-localization, room state, predicate progress | (a) Odometry: integrate `observation.state[0:3]` robot-local base velocity, corrected by depth-based visual odometry; (b) Room prior: `B100_task_misc.csv` "Rooms to include" (task metadata, identical at train/eval) tells us which rooms are loaded; (c) BDDL predicate detectors: small vision classifiers trained with **privileged supervision from the demo data** (segmentation/pose allowed at training time), inferring onboard RGB+depth+proprio only. |
| **Policy** | Subgoal language + RGB(720 head / 480 wrists) + depth + 61-dim proprio → 23-dim action chunk | **π0.5 fine-tuned once on all 20k demos, skill-conditioned** (subgoal text = conditioning), 32-step action horizon, receding-horizon serving. Per-task LoRA/task-embedding adapters for the weakest quartile. |
| **Control / Serving** | WebSocket contract, action chunking, safety limits, watchdogs | Thin deterministic wrapper: msgpack protocol, action-horizon cache, per-skill time budget, stuck-retry logic, early-stop on full predicate satisfaction. |

**Why this beats "just fine-tune a VLA per task" (the baseline strategy):**
- One shared policy amortizes 100 fine-tunes → ~10× less training, and shares manipulation skills across the many tasks that reuse them (pick-and-place, open/close, pour appear in dozens of tasks).
- The planner decouples *what to do next* from *how to do it*, so a ~6-minute episode never asks the VLA to hold a 27-skill goal in context — the classic long-horizon forgetting failure.
- Predicate detectors give us (i) progress gating between subgoals, (ii) stuck detection, (iii) early stopping — the three things that convert raw VLA competence into Q and efficiency.
- Everything above the VLA is small, deterministic, and testable without the GPU; iteration speed on planner/progress bugs is minutes, not hours.

**Why π0.5 over GR00T N1.7 (primary):** open weights (no gated backbone), the challenge fork already has `pi05_b1k` config, `compute_norm_stats.py`, `serve_b1k.py` with receding horizon — i.e., the full train/serve path is proven on R1Pro and fits 24 GB in bf16. GR00T N1.7 stays as the backup path if π0.5 skill-conditioning underperforms on manipulation-heavy tasks (its temporal ensembling is a nice robustness feature we can also port to the serving layer).

**Why not an external LLM / VLM planner at eval:** the 24 GB single-GPU cap, the ~13.5 sim-FPS cadence, confidentiality of serving config, and organizer inspection of the wrapper make deterministic local planning strictly safer and faster. External calls are only a fallback for the *training-time* annotation-mining step (allowed, no cost at eval).

**Embodiment decision: keep R1Pro.** The demo distribution, controller configs, action/proprio slices, and camera intrinsics all match it exactly; switching robots invalidates the 20k demos we would have to re-collect or domain-adapt.

---

## 3. Repository layout

```
behavior_challenge/
├── docs/
│   ├── challenge-spec.md          # requirements (done)
│   ├── design.md                  # this doc
│   ├── task-ids-100.txt
│   ├── task-descriptions-100.jsonl
│   └── plans/                     # mined per-task skill plans (100 files)
│       └── <task_id>.json         # {plan: [{skill, args, mean_len_s, p90_len_s}], pred_templates: [...]}
├── src/
│   ├── b1k/
│   │   ├── __init__.py
│   │   ├── server.py              # WebSocket policy server (entrypoint: python -m b1k.server)
│   │   ├── protocol.py            # obs parsing, msgpack encode/decode, /healthz
│   │   ├── controller.py          # action-chunk cache, receding horizon, safety clamps
│   │   ├── planner/
│   │   │   ├── skill_tax.py       # 31-skill taxonomy (names, verb patterns)
│   │   │   ├── miner.py           # demos → per-task skill plans (training-time)
│   │   │   ├── task_planner.py    # eval-time: BDDL goal → active skill queue
│   │   │   └── progress.py        # subgoal progress state machine + watchdogs
│   │   ├── perception/
│   │   │   ├── odometry.py        # velocity integration + depth VO fusion
│   │   │   ├── rooms.py           # room prior from task metadata csv
│   │   │   └── detectors/
│   │   │       ├── labels.py      # privileged demo → predicate-label teacher (training)
│   │   │       ├── pred_models.py # per-predicate-family vision classifiers (eval: onboard only)
│   │   │       └── state.py       # PredicateTracker: current satisfaction map
│   │   ├── policy/
│   │   │   ├── vla.py             # π0.5 inference wrapper (skill-conditioned)
│   │   │   ├── norm.py            # norm-stats load/apply (recomputed post-Aug fix)
│   │   │   └── chunking.py        # horizon blending (temporal ensembling port)
│   │   └── config.py              # dataclass for all runtime config (paths, ports, model)
│   ├── training/
│   │   ├── download.py            # per-chunk HF download (100 tasks, 3.27 TB plan)
│   │   ├── lerobot_io.py          # LeRobot v3.0 frame/episode/video reader
│   │   ├── stats.py               # recompute meta/stats.json (post velocity fix)
│   │   ├── skill_annotations.py   # per-episode language → skill-segment parse
│   │   ├── vla_finetune.py        # π0.5 skill-conditioned finetune (OpenPI fork cfg)
│   │   ├── adapters.py            # per-task LoRA / task-embedding adapters
│   │   └── detector_train.py      # predicate classifier training (privileged teacher)
│   └── evalharness/
│       ├── run_eval.py            # wraps omnigibson.eval.eval (v3.9.2) per task/instance
│       ├── parallel.py            # multi-port server fan-out (≥50 ports) + job queue
│       ├── aggregate.py           # parse metrics JSON → Q, task_sr, time_score per task
│       └── report.py              # leaderboard-format table + regressions vs last run
├── configs/
│   ├── r1pro.yaml                 # verbatim from raw/r1pro.yaml (do not modify)
│   ├── server.yaml                # ports, model path, horizon, watchdog thresholds
│   └── training.yaml              # data roots, batch sizes, adapters
├── tests/
│   ├── test_protocol.py           # msgpack round-trip, obs key parsing vs eval_utils
│   ├── test_planner.py            # plan mining on synthetic + real annotations; gating
│   ├── test_progress.py           # watchdog/early-stop state machine
│   ├── test_odometry.py           # velocity integration vs synthetic ground truth
│   ├── test_aggregate.py          # Q computation vs score_utils formula on fixture JSONs
│   └── fixtures/                  # canned obs dicts, metrics JSONs, annotation segments
├── scripts/
│   ├── bootstrap.sh               # v3.9.2 clone+setup, conda envs (behavior + uv training)
│   ├── self_eval.sh               # full 100-task × 10-instance local eval
│   └── make_submission.sh         # package zip per submission contract
├── data/                          # (gitignored) demo chunks, stats, checkpoints
└── outputs/                       # (gitignored) self-eval metrics + videos
```

Module dependency rule (enforced by tests/CI import check): `server → controller → {planner, perception, policy}`; `planner` and `perception` must import nothing from `policy` (keeps the brain testable on CPU); `training/*` is never imported at eval time (no privileged-path leakage by construction); `evalharness` only talks to `server` over the WebSocket (identical to how the organizers' evaluator will).

---

## 4. External + internal interfaces

### 4.1 WebSocket protocol (hard contract — mirror `omnigibson.eval.policies.WebsocketPolicy`)

- **Endpoint:** one server per port; evaluator hits `GET /healthz` until 200, then opens the WS.
- **Request (evaluator → server):** msgpack `dict[str, np.ndarray]` of the **flattened** obs produced by `omnigibson.eval.utils.flatten_obs_dict` (keys joined by `::`, e.g. `robot_r1::...::rgb`, `...::depth`, plus a 61-dim proprio vector). Exact key strings must be confirmed against v3.9.2 source in week 1 (`tests/test_protocol.py` does this from a recorded fixture rollout).
- **Response (server → evaluator):** msgpack containing a single **action array of `robot.action_dim` = 23 floats** (R1Pro layout: `[0:3]` base velocity cmd, `[3:7]` trunk pos, `[7:14]` L-arm pos, `[14:15]` L-grip, `[15:22]` R-arm pos, `[22:23]` R-grip). No privileged fields, no env calls.
- **Server signature:**
  ```python
  class B1KServer:                      # src/b1k/server.py
      def __init__(self, cfg: ServerConfig): ...
      def handle_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray:
          """One eval step. Returns exactly one 23-dim action."""
  ```
- **Multi-instance:** `parallel.py` spawns N `B1KServer` workers on ports `8000..8049`; each is stateless across tasks (task identity arrives via the evaluator; server resets on WS reconnect / new task start message or healthz cycle — confirmed from v3.9.2 client behavior).

### 4.2 Planner

```python
# src/b1k/planner/task_planner.py
@dataclass
class Subgoal:
    skill: str                 # one of the 31 canonical skills (skill_tax.SKILL_IDS)
    args: dict                 # e.g. {"object": "cup", "container": "fridge"} (template slots)
    text: str                  # rendered language subgoal fed to the VLA
    budget_steps: int          # p90 demo duration of this subgoal × 1.5, in 30 Hz steps

class TaskPlanner:
    def __init__(self, task_id: int, task_text: str, bddl_goal: str, plan: MinedPlan): ...
    def initial_subgoals(self) -> list[Subgoal]: ...
    def current(self) -> Subgoal: ...
    def advance(self, pred_state: PredicateState) -> None:
        """Called by progress.py when the current subgoal's exit predicates are satisfied
        (or budget exhausted with best-effort marking)."""
    def replan(self, pred_state: PredicateState) -> list[Subgoal]:
        """Re-derive the remaining queue from the BDDL goal minus satisfied predicates."""

# src/b1k/planner/miner.py  (training-time)
def mine_plans(demo_root: Path, tasks: list[int]) -> dict[int, MinedPlan]:
    """Scan per-episode language annotations (LeRobot v3.0) for all 100 tasks;
    cluster the 27.06 avg skills/trajectory into the 31-skill taxonomy; emit
    per-task ordered skill templates + duration statistics (mean/p90)."""
```

**MinedPlan** (stored at `docs/plans/<task_id>.json`) is the planner's static knowledge: ordered skill sequence with slot templates, per-subgoal duration stats, and the list of BDDL predicate *families* each subgoal is expected to flip (e.g. `open_container`, `contains`, `lit`).

### 4.3 Progress & watchdog (the Q- and efficiency-engine)

```python
# src/b1k/perception/detectors/state.py
class PredicateTracker:
    """Maintains satisfaction of every goal predicate from onboard obs only."""
    def update(self, frame: Frame) -> "PredicateState":
        """frame: head/wrist RGB+depth + proprio. Returns {pred_id: (sat: bool, conf: float)}.
        Smoothing: predicate flips require K=3 consistent frames (conf ≥ 0.6) to fight noise."""

# src/b1k/planner/progress.py
class Progress:
    def tick(self, pred_state, subgoal: Subgoal, step: int) -> Action:
        """Returns one of:
        CONTINUE | ADVANCE (subgoal done) | RETRY (stuck: no progress Δpred for
        stuck_steps, max = min(budget, 3× demo-mean subgoal length)) | REPLAN
        | FINISH (all goal predicates satisfied → early stop; also enforces
        global time budget 1.5× human mean episode length from task metadata)."""
```

`FINISH` is what buys the efficiency tie-breakers: the evaluator keeps running until our server stops responding meaningfully (or `max_steps`), so **early termination policy** = when all predicates are satisfied, command a safe "park" (hold pose, close grippers, zero base) and keep serving no-op actions; `time_score` is then computed on the much shorter episode. *Verification item (week 1): confirm how `omnigibson.eval.eval` ends an episode when the agent does nothing — if it runs to `max_steps` regardless, fall back to the `--max-steps` flag per task = human-mean length × 1.45 (still under the 1.5 timeout).*

### 4.4 Odometry & navigation

```python
# src/b1k/perception/odometry.py
class Odometry:
    """Fuses: (1) proprio base velocity observation.state[0:3] (robot-local
    vx, vy, yaw-rate — the July 2026 convention) integrated per step,
    (2) head-depth visual odometry (pretrained DPVO/DPV2, ~10 Hz sub-sample,
    used only for drift correction: per-step blend with exponential smoothing,
    gain tuned on demo trajectories where privileged pose is available).
    Exposes: relative pose since episode start, and a coarse 0.5 m grid
    occupancy built from head depth (walls/furniture) for room-scale search."""
    def update(self, base_qvel: np.ndarray, depth_head: np.ndarray) -> None
    @property
    def pose(self) -> np.ndarray            # 4x4 (relative, robot frame at t0)
    @property
    def occupancy(self) -> OccupancyGrid    # for search heuristics

# src/b1k/perception/rooms.py
class RoomPrior:
    """Static per-task list of loaded rooms from B100_task_misc.csv
    (task metadata — legal at eval). Orders subgoals and biases search
    ('move to <room X> then <object>' when the target object is not visible
    in the head camera)."""
```

Navigation policy (no VLN needed): for "move to X / open X" subgoals the VLA receives the subgoal text plus a **visually-grounded** prompt; if the target object is not detected (detector confidence < τ for 5 s), `Progress` emits SEARCH: follow occupancy-frontier search biased toward the RoomPrior order, with wall-following fallback. This is the single biggest long-horizon risk (cross-room search without global pose), so it gets its own integration test with synthetic episodes (week 3).

### 4.5 Policy (VLA)

```python
# src/b1k/policy/vla.py
class SkillConditionedVLA:
    """π0.5 (OpenPI 'behavior' fork, pi05_b1k cfg) fine-tuned once on all
    20k demos with language conditioning = task_text + '· subgoal: ' + subgoal.text.
    Inputs: head RGB 720, L/R wrist RGB 480, L/R depth 480 (head depth optional
    at eval — depth is in the wrapper; include it, it is free), proprio 61.
    Output: 32-step action chunk (23-dim), receding horizon 16 (serve_b1k convention)."""
    def act(self, frame: Frame, subgoal_text: str, task_text: str) -> np.ndarray  # (32, 23)

# src/b1k/controller.py
class ActionController:
    """Owns the action-chunk cache. Each eval step: step=1 from current chunk
    (receding); every K=16 steps or on subgoal ADVANCE: re-query VLA.
    Blends chunk boundaries (temporal-ensembling port from GR00T serving):
    a_t = (1-α)·old_t + α·new_t, α ramp 0→1 over the chunk.
    Clamps to r1pro.yaml command limits (±0.75 x-y, ±1.0 yaw; joint ranges from
    the model file); deadbands gripper commands < 0.02 change to avoid chatter."""
```

**Adapters (phase 3):** per-task LoRA (r=8 on attention) or a learned task-embedding head on the weakest ~20 tasks (identified by self-eval); selected at serve time by `task_id` from the evaluator; same 24 GB budget (one base model + ≤20 tiny adapters, swapped on task boundary).

### 4.6 Training pipeline

```python
# src/training/  (training-time only)
def download_chunks(task_ids: list[int], data_root: Path) -> None      # HF, chunk-000..099
def recompute_norm_stats(data_root: Path) -> None                       # MUST run post Aug-2026 velocity fix
def parse_skill_segments(data_root: Path) -> SkillSegments              # language annotations → (t0,t1,skill,object)
def train_detectors(data_root: Path, out: Path) -> dict[str, Path]      # privileged teacher labels from demo
def finetune_vla(cfg: TrainingConfig) -> Path                            # skill-conditioned π05_b1k
def train_adapter(task_id: int, base: Path, out: Path) -> Path           # per-task LoRA/task-emb
```

Detector label families (privileged → onboard classifiers): `open/closed(container)`, `lit/unlit(appliance)`, `contains/empty(vessel)`, `object_in_gripper(hand)`, `contact(surface)`, `positioned(above/on/in)` — each a small conv classifier on head + wrist crops at 30 Hz over demo frames (teacher = BDDL evaluation of the privileged sim state at that frame — free, dense, exact). This converts "did I actually do it?" from guesswork into a learned observation, which is the backbone of progress gating and early stopping.

### 4.7 Eval harness

```python
# src/evalharness/run_eval.py
def run_task(task_id: int, instance_indices: list[int], port: int,
             output_dir: Path, max_steps: int | None = None) -> list[Path]:
    """Subprocess: python -m omnigibson.eval.eval --task-name <id> --host 127.0.0.1
    --port <port> --instance-indices ... --num-rollouts 1
    --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --write-video ..."""
# src/evalharness/aggregate.py
def aggregate(json_dir: Path) -> Summary:
    """Recompute Q / task_sr / time_score / normalized distances exactly per
    score_utils.py (unit-tested against the fixture JSONs)."""
```

Self-eval protocol: 100 tasks × instances `0..9` × 1 rollout ≈ 1,000 episodes × ~9 min sim ≈ multi-day on one 4090 → `parallel.py` fans out over 50 ports across whatever GPUs are available (target: 4×24 GB machines → < 1.5 days); every full self-eval produces `outputs/<date>/report.md` with per-task Δ vs previous run (regression gate before any model swap).

---

## 5. Inference pipeline (per eval step)

```
obs dict (flattened) ──protocol.py──▶ Frame {rgb_head 720², rgb_wl/wr 480², depth_*, proprio 61}
  1. PredicateTracker.update(frame)            # ~5 ms (small CNNs, 256px crops, batched)
  2. Progress.tick(pred_state, subgoal, t)     # state machine, <1 µs
     ── CONTINUE:
  3.   Odometry.update(base_qvel, depth_head)  # ~2 ms (subsampled VO every 3rd frame)
  4.   controller.step() → cached chunk action # VLA call every 16 steps: 300–800 ms on 24 GB bf16
     ── ADVANCE/RETRY/SEARCH: update subgoal text; controller.blend; odometry.search() if target unseen
     ── FINISH: park pose; no-op actions
  5. msgpack encode(23 floats) → ws
```

Latency budget vs 13.52 sim FPS: VLA amortized over a 16-step chunk ≈ 40–50 ms/step; everything else < 10 ms/step → policy is not the sim bottleneck. VRAM budget (24 GB): π0.5 bf16 ≈ 6–7 GB, activations at 720²+2×480² ≈ 4–6 GB (chunk=1, offload KV between re-queries), detector CNNs < 0.5 GB, VO model (DPV2) < 1.5 GB (CPU offload acceptable, runs at 10 Hz) → **fits with ≥ 6 GB headroom**.

---

## 6. Tradeoffs & rejected alternatives

| Option | Verdict | Reason |
|---|---|---|
| Per-task VLA fine-tune only (baseline style) | **Rejected as primary; kept as fallback per task** | 100× training cost, no cross-task skill sharing, no robustness layer, no early stop. |
| GR00T N1.7 primary | **Rejected as primary; backup** | Gated Cosmos-Reason2-2B backbone (approval friction), heavier serving; temporal ensembling ported instead. |
| External LLM/VLM planner at eval | **Rejected** | 24 GB + 13.5 FPS + no guaranteed API quota at final eval; deterministic local planner is cheaper and more reliable. |
| Custom embodiment | **Rejected** | R1Pro matches all 20k demos exactly; any robot switch forfeits the data advantage. |
| End-to-end RL on top | **Rejected** | No reward oracle at eval (partial credit is the only signal, BDDL not observable privileged); detector+watchdog gets most of the robustness for a fraction of the risk. |
| Learning-based navigation (VLN) | **Deferred** | Subgoal-VLA + occupancy-frontier + room prior covers cross-room search; VLN head is a documented swap-in if search fails > 20 % of episodes in self-eval. |
| 20 public + 20 hidden split | **Plan on code (20/20); confirm at office hours** | Spec flags the docs-vs-code discrepancy; the instance indices we self-score on (0–9) are unaffected either way. |
| Docker vs IP serving | **Docker (recommended by organizers)** | One 24 GB box reproduces final-eval hardware; 50-port IP route is the documented fallback if Docker infra fails in week 4. |

**Top risks (ranked):** (1) cross-room search without global pose — mitigated by odometry+occupancy+RoomPrior, integration-tested week 3; (2) long-horizon VLA drift on novel instances — mitigated by subgoal receding + chunk blending + adapters; (3) detector noise causing bad ADVANCE/FINISH — mitigated by 3-frame hysteresis + confidence floors + budget caps (a subgoal can never exceed 1.5× its p90 demo duration without RETRY/REPLAN); (4) 3.27 TB data logistics — mitigated by per-chunk download in week 1 and `--include` patterns from the spec; (5) early-termination semantics (how the evaluator handles a parked robot) — verified in week 1 with a no-op server fixture.

---

## 7. Implementation plan (5 weeks to 2026-10-16)

**Week 1 (Sep 12–18) — Foundations, all verifiable without the big model.**
- `bootstrap.sh`: BEHAVIOR-1K v3.9.2 clone + `setup.sh` envs (this machine or a 4090 box); confirm WebSocket msgpack shape, episode-end semantics, and `flatten_obs_dict` key strings via one fixture rollout with the provided π0.5 checkpoint for `turning_on_radio` (no training).
- Start full dataset download (per-chunk, 3.27 TB) on fast storage; recompute `meta/stats.json` (post-velocity-fix) — **blocking for all training.**
- Implement + unit-test: `protocol.py`, `controller.py` (clamps/blend/deadband), `skill_tax.py`, `miner.py` (on first 10 chunks), `aggregate.py` (vs `score_utils` fixtures), `parallel.py` skeleton.
- **Milestone M1:** end-to-end *valid* submission artifact with the baseline π0.5 checkpoint serving 10 tasks × 10 instances through our server + harness, even if Q is low. (Guarantees the pipeline works; also satisfies the open-source-prize requirement if released.)

**Week 2 (Sep 19–25) — The shared brain.**
- Skill-conditioned π0.5 finetune on all 20k demos (skill text + task text conditioning; 32-step horizon; norm stats from week 1).
- Detector training (privileged teacher) for all 6 predicate families; evaluate on held-out demo episodes (F1 per family ≥ 0.85).
- Odometry: DPV2 on head depth, blend gain tuned on 5k demo episodes with privileged pose (ATE target < 0.3 m over 60 s).
- **Milestone M2:** full self-eval (100 tasks × 10 instances) of *shared VLA + planner + detectors + odometry*. Expected Q ≈ 0.40–0.50.

**Week 3 (Sep 26 – Oct 2) — Robustness & navigation.**
- Watchdog tuning (RETRY/REPLAN budgets from mined duration stats); search integration tests (synthetic cross-room episodes); `RoomPrior` wiring.
- Identify bottom-quartile tasks from M2 → launch per-task adapters (batch 1 of ~20).
- **Milestone M3:** self-eval round 2, expected Q ≈ 0.50–0.55 with FINISH-based early stopping engaged (time_score > 0 on most tasks).

**Week 4 (Oct 3–9) — Efficiency & hardening.**
- Adapter batch 2; efficiency pass: direct-route bias (occupancy A* from odometry pose to detected target), minimize SEARCH wander (counts against normalized distance).
- 24 GB soak test on final-eval-class hardware (RTX 3090/Titan RTX if borrowable); Docker image build + cold-start < 300 s; registration form + leaderboard submission.
- **Milestone M4:** full self-eval round 3 on the *exact* submission Docker image. Regression-gated.

**Week 5 (Oct 10–16) — Freeze & submit.**
- No model changes after Oct 12 (only serving/config fixes). Final self-eval round 4; package zip per §7.1 of the spec (`<track>.<testset>.<team>.<affiliation>.<date>/json/` + 1,000-video portal + wrapper code + r1pro.yaml + server code); open-source release (repo, configs, training scripts) → Outstanding Open-Source prize.
- Submit before Oct 16 deadline with buffer for portal/video upload.

**Team notes:** office hours Monday PT (confirm 20/20 instance split + episode-end semantics); Discord for dataset issues (depth-video fix already noted — we re-sync).

---

## 8. Acceptance mapping (design → spec checklist)

| Spec §11 item | Where this design satisfies it |
|---|---|
| Onboard-only obs at eval | `perception/` + `pred_models.py` consume `Frame` (RGB/depth/proprio) only; privileged code quarantined in `training/` (never importable at serve time — CI check) |
| v3.9.2 + RGBDFullResWrapper + /healthz + msgpack 23-dim | `server.py`/`protocol.py`, `bootstrap.sh`, fixture test in week 1 |
| Scored on instances 0–9, 1 rollout, videos, folder contract | `evalharness/run_eval.py` + `make_submission.sh` |
| Single 24 GB GPU | §5 VRAM budget; M4 soak test |
| Maximize Q (partial credit) + efficiency | §1 decomposition; `Progress` (early stop, budgets), `RoomPrior`/occupancy (direct routes) |
| Robust across 100 task families | shared skill-conditioned VLA (31 skills mined from demos) + adapters; per-task regression reports |
| Reproducible from submitted files | Docker image, `configs/`, `docs/plans/` baked in, training scripts open-sourced |

**Concrete interfaces for the implementation task (t_21f882e2):** every public class/function in §4 with its signature is the contract; `tests/fixtures/` must contain (a) a recorded flattened-obs dict + 23-float action from one baseline rollout, (b) 5 canned metrics JSONs covering Q fractions 0/0.5/1.0 with and without early stop, and (c) a synthetic skill-annotation file exercising `miner.py`. The MVP entrypoint is `python -m b1k.server --config configs/server.yaml --port 8000`, and the reproducible artifact is `scripts/self_eval.sh` producing `outputs/<date>/report.md`.
