# Research Plan 2027 — Engineering Appendix (implementation, infrastructure, tickets)

Companion to [`research-plan-2027.md`](research-plan-2027.md), which holds the research direction. This file holds everything about *how* to build and run the studies. Read the research plan first.

## A. Platform facts the studies run on

| Asset | Where |
|---|---|
| Single-GPU Docker-only node (Brev) training GR00T N1.7 (~2.3 s/it, global batch 128) and serving on :8000 | `docker/groot/`, `scripts/docker/train_groot.sh`, `docs/docker-training.md` |
| π0.5 Docker train/serve | `docker/pi05/`, `scripts/docker/train_pi05.sh` |
| Two-container evaluation (policy ↔ `stanfordvl/behavior:3.9.2`), headless + MP4 | `scripts/docker/eval.sh` |
| Organizer single-task checkpoints (π0.5, GR00T; `turning_on_radio`) | `docs/docker-training.md` §E.1 |
| Demos: 200 episodes/task, LeRobot v3.0, RGB+depth+proprio, skill-segment annotations, `task_instance_id` | `src/training/download.py`, `src/training/lerobot_io.py` |
| 40 official test instances/task (301–340), per-rollout JSON with `q_score.final` | `docs/challenge-spec.md` §6 |
| Planner/watchdog stack (31 skills, miner, CONTINUE/ADVANCE/RETRY/REPLAN/SEARCH/FINISH, predicate tracker, odometry/frontier, chunk blending, controller) | `src/b1k/planner/*`, `src/b1k/perception/*`, `src/b1k/policy/chunking.py`, `src/b1k/controller.py` |
| Eval harness reproducing official scoring | `src/evalharness/*` |
| Training seams (detector trainer, LoRA adapters, skill-annotation parser) | `src/training/detector_train.py`, `adapters.py`, `skill_annotations.py` |

**Throughput assumption:** ~13.5 sim-FPS + 150–300 s scene load → full-length episode ≈ 20–25 min with policy and sim on one GPU → **~50 full-length episodes / GPU-day** (unverified; see A2 below).

## B. Assumptions to verify in week 1

| # | Assumption | How to verify | Owner | If false |
|---|---|---|---|---|
| A1 | Two-container eval (`train_groot.sh serve` + `eval.sh`) produces an official JSON + MP4 for the organizer checkpoint | Run once, headless, on the Brev box; archive the JSON | infra | Everything is blocked — fix first |
| A2 | ~50 full-length episodes / GPU-day | Time 5 consecutive `eval.sh` runs incl. scene load → `docs/research/throughput.md` | infra | Re-scale matrices; decide on second box |
| A3 | Demo-composition ("seen") instances are evaluable: demo instance files (1–200) exist in `OMNIGIBSON_DATA_PATH`, or public 301–320 overlap in composition | Inspect `task_instance_id` in `meta/episodes` vs. instance files in the sim image | Study 1 | C-seen := public 301–320, C-unseen := hidden 321–340; state the weaker definition |
| A4 | Privileged state (goal predicates, poses, held objects, contacts, rooms) is readable from a custom `--env-wrapper` inside `stanfordvl/behavior:3.9.2` | 40-line logger stub, 1 episode, open the parquet | Study 1 | Parse the official per-predicate JSON (coarser onsets) |
| A5 | Per-frame `task_index` changes are honoured by the GR00T LeRobot v3 loader | Unit test across a segment boundary | Study 2 | Use `V-both`; inject skill text as a second language field |
| A6 | K = 8 batched flow-matching samples fit the 30 Hz / 16-step re-query budget | `scripts/bench_k_samples.sh` | Study 3 | K = 4; else D-disp offline-only |
| A7 | Released artifacts are license-compatible with the demo dataset | Read the HF dataset card | all | Release labels + parquet, not videos |

## C. Implementation per study

### Study 1 — Where the Demos End

1. `src/research/sim/privileged_logger.py` — subclass of `omnigibson.eval.wrappers.RGBDFullResWrapper` writing per-step `{step, goal_predicates, base_pose, gripper_l/r, held_objects, contacts, nearest_goal_dist, room_id}` to `/out/<task>_<inst>_<roll>.priv.parquet`. Mount `-v $PWD/src:/work/src -e PYTHONPATH=/work`; add `EVAL_WRAPPER` env var to `scripts/docker/eval.sh`.
2. `src/research/anatomy/label_failures.py` — rule table (below) → `labels.jsonl` (class, onset step, active skill). CPU; `tests/test_label_failures.py` on synthetic traces.
3. `src/training/concat_tasks.py` — 20-task LeRobot v3 dataset (symlinked videos, concatenated `meta/episodes`, re-indexed `task_index`).
4. Trained predicate detectors via `src/training/detector_train.py` → `data/detectors/<family>.pt`; gate F1 ≥ 0.85 held-out.
5. GR00T backend in `src/b1k/policy/vla.py` (`_act_vla` → in-process `Gr00tPolicy` or call to the serve container) so the watchdog wraps the same checkpoint.
6. `src/research/common/run_matrix.py` — resumable (task, condition, instance, arm) enumerator over `eval.sh`; `manifest.csv`.
7. `src/research/anatomy/analyze.py` — joins official JSONs with `labels.jsonl`; tables/figures with bootstrap CIs.
8. Release: HF dataset `b1k-anatomy` (labels + priv parquet + videos) + labeler.

**Automatic labeling rules (privileged, analysis-time), mapped to the 10-class schema of Rasouli et al.:**

| Class | Rule |
|---|---|
| Grasp failure | ≥ 2 gripper-close events within 0.3 m of a goal object, no `held` transition |
| Improper object handling | held → not-held outside any goal receptacle |
| Placement failure | release within 0.5 m of goal receptacle, goal predicate not satisfied |
| Skill failure (articulated/state) | within 1 m of target, ≥ 10 s arm motion, predicate never flips |
| Semantic confusion | grasp of object not in BDDL goal nor in mined plan |
| Navigation failure | target room never entered (pose vs. `rooms.py` polygons) or base displacement < 0.5 m for > 30 s mid-plan |
| Collision | contact events with non-task objects above impulse threshold |
| Execution-order confusion | predicate flips out of order w.r.t. mined plan |
| Task confusion | ≥ 2 skills executed that belong to another task's mined plan |
| Abrupt termination | no action-norm change and no predicate change for > 60 s with budget remaining |

### Study 2 — What to Say vs. When to Switch

1. `src/training/skill_relabel.py` — per-frame `task_index` rewrite from annotation segments (`lerobot_io`, `skill_annotations.py`), `meta/tasks.parquet` with skill texts, symlinked videos; `--mode {task,skill,shuffled,both}`.
2. Loader test: 3 frames across a segment boundary get different language in `examples/b1k/r1pro.py` data config.
3. `src/b1k/policy/vla.py` — GR00T backend sends `Subgoal.text` per query; `controller.py` already re-queries on ADVANCE.
4. `src/b1k/protocol.py` — optional `privileged` field; server refuses it under `--strict-onboard` (challenge path stays legal).
5. `progress.py` — `switch: {oracle, onboard, stage, openloop, vlm}`; `miner.py` exposes p50/p90 durations.
6. `src/research/interface/stage_head.py` — RLC-style stage classifier + voting; `src/research/interface/vlm_verifier.py` — Semantic-Handoff-style verifier (Qwen-VL on head + wrist frames every 2 s).
7. `src/research/interface/switch_metrics.py` — aligns ADVANCE events with oracle flips.
8. `run_matrix.py --study interface`.

### Study 3 — Do Runtime Failure Detectors Pay for Themselves

1. Batched K-sample decoding in the GR00T serve path (backbone once, action head batch K) and π0.5 (`sample_actions` with split rng); flag `--k-samples`; `scripts/bench_k_samples.sh` for K ∈ {1,4,8,16}.
2. `src/b1k/policy/uncertainty.py` — dispersion stats + medoid; `chunking.py` returns boundary residual; `controller.py` writes a feature row per step (non-blocking) to `features.parquet`.
3. Forward hooks: action-head last-layer pooled features (SAFE-style probe); low-rank weight-perturbation wrapper (PFD-style) — off by default.
4. `src/research/alarm/train_detectors.py` — per-family detector (LightGBM / linear / MLP), task-grouped CV, calibration; `tests/test_alarm_features.py`.
5. `progress.py` — `trigger: {rule, predictive:<family>, oracle}`; shared ladder L1 RETRY → L2 REPLAN → L3 SEARCH → L4 safe pose, budget-capped at 1.5× p90; trigger = score > τ for ≥ 1 s, τ set at 10 % false alarms per family on the pilot.
6. `run_matrix.py --study alarm`; reuse `vlm_verifier.py`.

## D. Shared infrastructure (build first, weeks of Oct 20 and Oct 27)

1. `privileged_logger.py` + `EVAL_WRAPPER` in `eval.sh`.
2. `label_failures.py` + tests.
3. `run_matrix.py`, `analyze_common.py` (bootstrap CI; Q math from `src/evalharness/aggregate.py`).
4. GR00T backend in `src/b1k/policy/vla.py`.
5. Feature tap + K-sample decoding + hooks — **before** Study 1's matrix so it doubles as Study 3's training data.
6. `concat_tasks.py`, `skill_relabel.py`.
7. `vlm_verifier.py` (shared baseline for Studies 2 and 3).
8. Reproducibility: every run records image digest, checkpoint hash, instance ids, seeds, git SHA into the output dir; n-per-cell tables are generated, never typed.

Rule: nothing in `src/research/` may be imported from `src/b1k/`. The challenge serving path stays unchanged and onboard-only.

## E. Compute and money (single node + one rented twin)

| Item | GPU-days |
|---|---|
| Study 1 finetunes (GR00T + π0.5, 20 tasks) | 3 |
| Study 1 matrix (1,400 episodes) | 20 |
| Study 2 finetunes (4 GR00T + 2 π0.5) | 8.5 |
| Study 2 matrix (1,750 episodes) | 22 |
| Study 3 intervention matrix (1,900 episodes) | 25 |
| Pilots, re-runs, audits (20 %) | 16 |
| **Total** | **~95 GPU-days** |

Two boxes Nov 1–Feb 10 ≈ 200 GPU-days. One box ≈ 100: Studies 1 + 2 to IROS 2027 with reduced Study 2 (drop S-vlm, S-stage → 1,200 eps), Study 3 to ICRA 2028. Disk: 2 TB per box (20-task demos ≈ 0.7 TB; relabeled datasets symlinked; videos ≈ 0.3 TB per 1,000 episodes).

## F. Tickets (first four weeks)

| Ticket | Deliverable | Depends on | Study | Target |
|---|---|---|---|---|
| T1 | A1 done; `docs/research/throughput.md` with A2 numbers | — | shared | Oct 23 |
| T2 | `privileged_logger.py` + `EVAL_WRAPPER`; 1 parquet from 1 episode (A4) | T1 | shared | Oct 27 |
| T3 | `label_failures.py` + tests | T2 | 1 | Oct 31 |
| T4 | `run_matrix.py` + `analyze_common.py` | T1 | shared | Oct 31 |
| T5 | GR00T backend in `vla.py`; watchdog arm serves organizer checkpoint end-to-end | T1 | shared | Nov 3 |
| T6 | `concat_tasks.py`; 20-task download started (disk ≥ 2 TB) | A3 | 1 | Nov 3 |
| T7 | `skill_relabel.py` + loader test (A5); relabeled `turning_on_radio` | — | 2 | Oct 31 |
| T8 | 5k-step V-task vs V-skill pilots + 20 oracle-switched episodes each → **Study 2 pilot gate (Nov 10)** | T2, T5, T7 | 2 | Nov 10 |
| T9 | K-sample + feature tap + bench (A6); 60 labeled episodes → **Study 3 pilot gate (Nov 8)** | T2, T3, T5 | 3 | Nov 8 |
| T10 | 40-episode seen/unseen pilot on organizer checkpoints with labels → first ΔQ | T2, T3, T4 | 1 | Oct 31 |
| T11 | GR00T + π0.5 20-task finetunes launched; detectors trained → **Study 1 pilot gate (Nov 15)** | T6 | 1 | Nov 15 |
| T12 | Decision: rent second box (yes if A2 < 60 eps/day and all three studies are go) | T1 | lead | Oct 24 |

Staffing for 2–3 people: one owner per study; shared tickets (T1, T2, T4, T5) to the Study 1 owner since its matrix is the substrate. Every ticket lands as a PR to `main` with a result note in `docs/research/log.md`.

**Definition of "ready to run a matrix":** A1–A4 verified, T2–T5 merged, T10 produced labeled JSONs, and the per-episode output directory records image digest, checkpoint hash, instance ids, seeds and git SHA.
