# Research Plan 2027 — Three Publishable Ideas Built on This Repo

**Date:** 2026-10-19 · **Target venues:** IROS 2027 (paper deadline **Mar 1, 2027**, Florence) as the primary target; ICRA 2028 (deadline ~Sep 15, 2027) as the second slot; CoRL 2027 (~May/June 2027) as an alternative for Idea 3.
**Supersedes** `docs/icra-iros-2027-ideas.md` / `docs/icra-iros-2027-final-scope.md` where they conflict. Those documents assumed a 4–8 GPU cluster and a physical R1Pro. **Neither exists.** This plan is sized to what we actually have today:

| Asset we have today | Where |
|---|---|
| One single-GPU Docker-only node (Brev) that trains GR00T N1.7 (`scripts/docker/train_groot.sh`, measured ~2.3 s/it at global batch 128) and serves it on :8000 | `docker/groot/`, `docs/docker-training.md` |
| π0.5 Docker train/serve path | `docker/pi05/`, `scripts/docker/train_pi05.sh` |
| Two-container evaluation (policy container ↔ `stanfordvl/behavior:3.9.2` sim container), headless with video | `scripts/docker/eval.sh` |
| Organizer-provided single-task checkpoints (π0.5 and GR00T, `turning_on_radio`) → a zero-training starting point | `docs/docker-training.md` §E.1 |
| 2026 challenge demos: 200 teleop episodes/task, LeRobot v3.0, RGB+depth+proprio, per-episode **language skill-segment annotations**, `task_instance_id` per episode | `src/training/download.py`, `src/training/lerobot_io.py` |
| 40 official test instances per task (ids 301–340), per-rollout JSON with `q_score.final` (partial credit over BDDL goal predicates) | `docs/challenge-spec.md` §6 |
| A deterministic planner/watchdog stack: 31-skill taxonomy, plan miner, subgoal progress state machine (CONTINUE/ADVANCE/RETRY/REPLAN/SEARCH/FINISH), predicate tracker with hysteresis, odometry + occupancy frontier, chunk blending, receding-horizon controller | `src/b1k/planner/*`, `src/b1k/perception/*`, `src/b1k/policy/chunking.py`, `src/b1k/controller.py` |
| Eval harness that reproduces official scoring, multi-port fan-out, aggregation/report | `src/evalharness/*` |
| Training seams: predicate-detector trainer (privileged teacher), LoRA adapters, skill-annotation parser | `src/training/detector_train.py`, `adapters.py`, `skill_annotations.py` |

**The one number that sizes everything.** The sim runs at ~13.5 FPS with 150–300 s scene load; a 2,000-step episode is ~5–8 min of wall-clock, a full-length (~16k-step) episode ~20–25 min. Policy and sim share the one GPU. Realistic sustained throughput: **~100 short or ~50 full-length episodes per GPU-day**. Every matrix below is sized against that and fits in **Oct 20, 2026 → Feb 10, 2027 (~110 GPU-days)** if we rent a second identical box for the evaluation waves (recommended, ~$1.5–3k total). Without a second box, run Idea 1 + Idea 2 to IROS and push Idea 3 to ICRA 2028.

**No real robot.** All three ideas are sim-only by design and are written so the abstract never carries a real-world claim (the top reviewer rejection reason). Each has a durable released artifact (dataset / protocol / code) so it survives the "no novel contribution" desk screen.

---

## Idea 1 — *Anatomy of Failure*: a skill-level failure benchmark for house-scale VLA policies

**Type:** benchmark + diagnostic study. **Venue:** IROS 2027. **Verdict: GO — primary. Start now.**

### 1.1 The claim

> Nobody has measured *what* limits VLA policies on multi-minute, mobile, bimanual household tasks. We show that for two state-of-the-art open policy families (π0.5, GR00T N1.7) on BEHAVIOR-1K, (i) most Q-loss comes from a small number of **robustness** failure classes (stuck loops, search failure, wrong object) rather than low-level skill execution; (ii) the loss is concentrated in navigation-heavy and multi-skill tasks; (iii) a simple **no-retrain watchdog** recovers a measurable fraction of it. We release the labeled rollouts, the auto-labeler, and the protocol.

Falsifiable hypotheses:

- **H1** ≥ 50 % of the Q-loss relative to a per-skill "best-case" reference is attributable to robustness classes (`stuck_loop`, `search_fail`, `goal_forget`, `wrong_object`), not `skill_exec_fail`.
- **H2** Q drops ≥ 10 points (0–100 scale) from *seen* instances (demo layouts) to *unseen* official instances, and the drop is ≥ 2× larger on navigation-heavy / ≥ 4-skill tasks than on single-room, ≤ 3-skill tasks.
- **H3** The watchdog (Idea 1's intervention arm, no retraining) recovers ≥ 20 % of the robustness-attributable gap in both mean and worst-case (min-over-task) Q.

Why it is publishable: it is the first empirical decomposition of long-horizon VLA failure at house scale (gap G1.5 in `docs/icra-iros-2027-trends.md`), it releases a ~1.5k-episode labeled dataset nobody else has, and it is policy-agnostic (two policy families).

### 1.2 Experimental design

**Tasks (20).** Stratify the 100 tasks by mined skill-plan statistics (`src/b1k/planner/mine_plans_cli.py` output): 5 single-room ≤ 3 skills (e.g. `turning_on_radio`), 5 single-room ≥ 4 skills, 5 multi-room navigation-heavy, 5 bimanual/articulated-heavy. The stratification table is Table 1 of the paper and is produced by one script (`src/research/anatomy/select_tasks.py` → `docs/research/anatomy_tasks.json`).

**Policies (3 arms).**

| Arm | What | How we get it |
|---|---|---|
| P-GR00T | GR00T N1.7 multi-task finetune on the 20 tasks | `train_groot.sh train` on a 20-task concatenated dataset (see 1.4). ~50k steps ≈ 32 h once. For `turning_on_radio` we also report the organizer checkpoint as a sanity row. |
| P-π0.5 | π0.5 multi-task finetune on the same 20 tasks, same demo count | `train_pi05.sh train`, same data split. |
| P-GR00T+WD | P-GR00T served through the repo's watchdog (`progress.py` RETRY/REPLAN/SEARCH) with **trained** predicate detectors | The intervention arm for H3. Same checkpoint as P-GR00T — the only difference is the wrapper. |

**Instance conditions (2).** C-seen = instances whose object/layout composition matches the demo episodes (link via `task_instance_id` in `meta/episodes/*.parquet`; evaluate on those instance files if present in `OMNIGIBSON_DATA_PATH`, else on the public set 301–320 whose composition overlaps). C-unseen = official hidden instances 321–340. State precisely in the paper that demo instances (1–200) and test instances (301–340) are disjoint by id; "seen" is defined by **composition overlap**, not id.

**Matrix.** 20 tasks × 2 conditions × 10 instances × 3 arms = **1,200 episodes** + 200 repeat episodes (same instance, 2 extra rollouts on 5 tasks) to quantify sim nondeterminism. Full-length episodes (official 1.5× human-mean budget). ≈ **1,400 × ~20 min ≈ 470 GPU-h ≈ 20 GPU-days.**

**Failure taxonomy (auto-labeled from privileged sim state, legal at analysis time).**

| Label | Rule (computed from the per-step privileged log) |
|---|---|
| `stuck_loop` | no goal-predicate change **and** base displacement < 0.5 m **and** gripper state unchanged for > 30 s inside a skill budget |
| `search_fail` | target object's room never entered (robot pose vs. room polygons from `src/b1k/perception/rooms.py`) before timeout |
| `goal_forget` | a goal predicate that was satisfied becomes unsatisfied and is never re-satisfied |
| `wrong_object` | grasp event on an object not in the BDDL goal nor in the mined plan for the active skill |
| `skill_exec_fail` | robot within 1 m of the right object, ≥ 2 grasp attempts, predicate never flips |
| `transition_error` | predicate flips in the wrong order relative to the mined plan (e.g. container closed before item placed) |
| `timeout_clean` | none of the above; ran out of time while making monotone progress |

Validation: 2 annotators label 300 random episodes from video (`--write-video` output) blind to the auto label; report Cohen's κ (gate: ≥ 0.7, else collapse to 5 classes).

**Metrics.** Q (official math, n per cell, 95 % bootstrap CI), task success rate, time score, **per-skill success** (31 skills × 20 tasks, binomial CI), horizon-decay curve (Q vs. number of skills remaining), ΔQ(seen−unseen) per stratum, failure-class frequency per arm, **fixability map** = per class: ΔQ(P-GR00T+WD − P-GR00T). Worst-case Q = min over tasks.

### 1.3 What to implement (repo-specific)

1. **Privileged per-step logger inside the sim container** — `src/research/sim/privileged_logger.py`: a subclass of `omnigibson.eval.wrappers.RGBDFullResWrapper` that, at every step, appends `{step, t, goal_predicates: {name: bool}, robot_base_pose, gripper_l/r, held_objects, nearest_goal_object_dist, room_id}` to a Parquet file in `/out/<task>_<inst>_<roll>.priv.parquet`. Mounted into the eval container with `-v $PWD/src:/work/src -e PYTHONPATH=/work` and selected via `--env-wrapper research.sim.privileged_logger.PrivilegedLoggerWrapper`. Add an `EVAL_WRAPPER` env var to `scripts/docker/eval.sh`.
2. **Auto-labeler** — `src/research/anatomy/label_failures.py`: reads the `.priv.parquet` + mined plan (`docs/plans/<task>.json`) → one `labels.jsonl` row per episode with the class, onset step, and the skill active at onset. Pure pandas/numpy, CPU, unit-tested on synthetic traces (`tests/test_label_failures.py`).
3. **Multi-task dataset builder** — `src/training/concat_tasks.py`: builds a LeRobot v3 dataset pointing at the 20 downloaded chunks (symlink videos, concatenate `meta/episodes`, re-index `task_index`) so `train_b1k.py` sees one dataset. Reuse `download_demos.sh <ids...>`.
4. **Trained predicate detectors** — run `src/training/detector_train.py` on the 20 tasks (privileged teacher labels from `labels.py`); export `data/detectors/<family>.pt`; `TrainedPredicateModel` already loads them. Gate: per-family F1 ≥ 0.85 held-out.
5. **Watchdog serving arm** — the `b1k.server` path already does this when `policy.backend: vla` + detectors are configured (`configs/server.yaml`). Add a GR00T backend to `src/b1k/policy/vla.py` (`_act_vla` → HTTP/websocket call to the GR00T serve container or in-process `Gr00tPolicy`). The watchdog's RETRY = immediate re-query, REPLAN = re-derive remaining subgoals from BDDL goal minus satisfied predicates, SEARCH = frontier-biased base velocity override from `odometry.py`.
6. **Matrix runner** — `src/research/anatomy/run_matrix.py`: enumerates (task, condition, instance, arm) → calls `scripts/docker/eval.sh` with `EVAL_WRAPPER`, `OUT_DIR=outputs/anatomy/<arm>/<task>/...`, resumable (skips done JSONs). One process per GPU box.
7. **Analysis notebook/script** — `src/research/anatomy/analyze.py`: joins official JSONs (`q_score.final`, per-predicate satisfaction) with `labels.jsonl` → all tables/figures; bootstrap CIs; writes `paper/figs/*.pdf`.
8. **Release** — HF dataset `b1k-anatomy` (labels + priv parquet + videos) and the labeler code.

### 1.4 Testing / gates / timeline

| When | Milestone | Gate (go/no-go) |
|---|---|---|
| Oct 20–31 | Logger + labeler on the organizer `turning_on_radio` checkpoint, 40 episodes (20 seen/20 unseen), both policies | Pipeline runs end-to-end in Docker; first ΔQ(seen−unseen) number; labeler produces non-degenerate class histogram |
| Nov 1–15 | 20-task data download (~0.7 TB), `concat_tasks.py`, GR00T 20-task finetune (50k steps ≈ 32 h), π0.5 finetune; detectors trained | **G1 (Nov 15):** 5-task pilot (100 episodes) shows ΔQ(seen−unseen) ≥ 5 on ≥ 1 policy **and** ≥ 3 failure classes each ≥ 10 % frequency. Else: redesign conditions (add BDDL-resampled perturbations) before scaling |
| Nov 16–Dec 31 | Full 1,400-episode matrix (20 GPU-days; 10 if two boxes) | **G2 (Dec 31):** taxonomy κ ≥ 0.7 on 300-episode audit |
| Jan 1–Jan 31 | Analysis, fixability map, figures | **G3 (Jan 31):** H1/H2 hold directionally; if H3 fails, the paper is still a benchmark + decomposition paper (reframe title, drop H3) |
| Feb 1–Mar 1 | Draft, internal review, **submit IROS 2027** | 8 pages incl. references; video ≤ 180 s from `--write-video` output |

**Risks.** (a) Both policies collapse to ~0 Q on unseen instances → no separation; caught at G1 on 100 episodes. (b) Finetunes are weak (Q < 0.1 everywhere) → use the organizer checkpoints on the tasks they cover and reduce to a 10-task matrix. (c) Disk: 20 tasks ≈ 0.7 TB demos + videos; budget 2 TB on the box.

---

## Idea 2 — *Mind the Interface*: does skill-level language conditioning help house-scale VLAs, and where does the hierarchy break?

**Type:** controlled method study (training-side + runtime), the "brain ↔ cerebellum interface" evaluation that the literature lacks (gap G1.5). **Venue:** IROS 2027 (co-submission) or ICRA 2028 if the finetunes slip. **Verdict: GO — second.**

### 2.1 The claim

> Hierarchical VLA systems (Hi Robot, HAMSTER, π0.5's own high-level/low-level split) assume that conditioning the low-level policy on a *sub-goal* instruction ("open the cabinet") rather than the *task* instruction ("put the groceries away") improves long-horizon execution — but no one has measured this with the progress signal held fixed. Using BEHAVIOR-1K's 270k language skill-segment annotations, we train task-conditioned and skill-conditioned variants of the same VLA on identical data, and evaluate them under three progress-gating regimes — **oracle** (privileged predicate ADVANCE), **onboard** (trained detectors + hysteresis), and **open-loop** (demo p50 time budgets). This isolates three quantities: the value of skill conditioning itself, the cost of imperfect progress detection, and the residual low-level execution gap.

Hypotheses:

- **H1** Skill-conditioned + oracle gating beats task-conditioned by ≥ 10 Q points on ≥ 4-skill tasks and ≤ 3 points on ≤ 3-skill tasks (the benefit scales with horizon).
- **H2** Onboard gating recovers ≥ 60 % of the oracle gain; open-loop timing recovers < 30 % (i.e. *when* to switch matters more than *what* to say).
- **H3** Skill-conditioned policies generalize better across tasks sharing skills: on 5 **held-out tasks** composed of seen skills, skill-conditioned Q ≥ 2× task-conditioned Q.
- **H4** (negative control) Shuffled skill text (wrong but well-formed sub-goals) performs no better than task conditioning — the gain comes from the semantics, not from a shorter prompt.

Why publishable: it is a clean controlled experiment with an oracle upper bound; H3 is a genuine compositional-generalization result on a public benchmark; it reuses the repo's planner stack as the "brain" and reports where the interface breaks.

### 2.2 Experimental design

**Tasks.** 15 training tasks (reuse Idea 1's strata; 5 per stratum excluding the bimanual one) + 5 held-out tasks whose mined plans use only skills present in the training 15 (`select_tasks.py --heldout-by-skill-coverage`).

**Policies (one base model: GR00T N1.7; π0.5 replicated on 5 tasks for policy-agnosticism).**

| Variant | Training language field |
|---|---|
| V-task | the dataset's task instruction (as today) |
| V-skill | per-frame skill-segment text from the annotations, rendered through `skill_tax.render_text` (canonical 31-skill templates with the object slot filled) |
| V-skill-shuffled | same segmentation, skill text drawn from a *different* segment of the same episode (H4 control) |
| V-both | `"<task>. Now: <skill>"` (tests whether task context is additive) |

All variants: identical demos, identical steps (50k), identical hyper-parameters; one GR00T finetune ≈ 32 h → **4 variants ≈ 5.5 GPU-days** (+ 2 π0.5 variants ≈ 3 GPU-days).

**Gating regimes at eval (for V-skill / V-both / V-shuffled).**

| Regime | ADVANCE signal | Implementation |
|---|---|---|
| G-oracle | privileged BDDL predicate flip for the active skill's exit family | the sim-side `PrivilegedLoggerWrapper` publishes predicate state to the policy over a side-channel (extra WebSocket message); the policy server's `progress.py` consumes it instead of the detector map |
| G-onboard | `PredicateTracker` over trained detectors (3-frame hysteresis) | existing `src/b1k/perception/detectors/state.py` |
| G-openloop | advance after the mined p50 skill duration from demo statistics | `TaskPlanner` budgets only, detectors disabled |
| (V-task has no gating) | — | — |

**Matrix.** 20 tasks (15 train + 5 held-out) × 10 unseen instances × {V-task, V-skill×3 regimes, V-both×G-onboard, V-shuffled×G-oracle} = 20 × 10 × 6 = **1,200 episodes** ≈ 17 GPU-days (full-length) — or 9 GPU-days capped at 2× the human mean. π0.5 replication: 5 tasks × 10 × 3 arms = 150 episodes.

**Metrics.** Q (primary), per-skill success, **sub-goal switching accuracy** (G-onboard ADVANCE time − oracle ADVANCE time; precision/recall of ADVANCE events), horizon-decay curves per variant, held-out-task Q (H3), and the decomposition Q(V-skill, oracle) − Q(V-task) = *interface value*; Q(oracle) − Q(onboard) = *detection cost*; 1 − Q(oracle) = *execution residual*.

### 2.3 What to implement (repo-specific)

1. **Skill relabeler** — `src/training/skill_relabel.py`: for each episode, read the annotation segments via `lerobot_io` (already parses `annotation_path`), map each segment to a canonical skill (`skill_annotations.py` already does this; log unmatched fragments), render text, and write a derived LeRobot v3 dataset in which `task_index` changes per segment and `meta/tasks.parquet` holds the skill texts. Videos are symlinked, not copied. Flags: `--mode {task,skill,shuffled,both}`.
2. **GR00T data config** — `examples/b1k/r1pro.py` uses the per-frame `task_index` → language already; verify the loader honours per-frame changes (it does for LeRobot v3; add a unit test reading 3 frames across a segment boundary).
3. **Eval-time subgoal prompting** — `src/b1k/policy/vla.py:_act_vla` already receives `subgoal_text`; wire the GR00T backend so the language sent each query is the active `Subgoal.text`. Re-query immediately on ADVANCE (`controller.py` already honours `requery=True`).
4. **Oracle side-channel** — extend `src/b1k/protocol.py` with an optional `privileged` field (only present in research runs; the server refuses it when `--strict-onboard` is set so the challenge path stays legal). Sim wrapper fills it from BDDL state.
5. **Open-loop budgets** — `src/b1k/planner/miner.py` already stores per-skill duration stats; expose p50 in `MinedPlan` and add `progress.py` mode `openloop`.
6. **Switching-accuracy metric** — `src/research/interface/switch_metrics.py`: aligns ADVANCE events (from server logs) with oracle flips (from priv parquet).
7. Reuse Idea 1's matrix runner and analysis scaffolding (`run_matrix.py --study interface`).

### 2.4 Testing / gates / timeline

| When | Milestone | Gate |
|---|---|---|
| Oct 20–Nov 10 | `skill_relabel.py` + loader test; relabel `turning_on_radio`; **5k-step pilot finetunes** of V-task vs V-skill (≈ 3 h each); 20 episodes each with G-oracle | **G1 (Nov 10):** V-skill+oracle ≥ V-task by any positive margin on the pilot, and the relabeler covers ≥ 90 % of segments with a canonical skill. If V-skill is *worse*, debug prompt format before scaling (try V-both) |
| Nov 11–Dec 10 | 15-task relabeled datasets; 4 GR00T variants at 50k steps (5.5 GPU-days, interleave with Idea 1's matrix on the second box) | finetune losses comparable across variants (sanity) |
| Dec 11–Jan 20 | 1,200-episode matrix + π0.5 replication | **G2 (Jan 20):** H1 directionally true on ≥ 10 of 15 tasks; if H2 fails (onboard ≈ open-loop) the detection-cost finding is itself the result — keep the paper, retitle |
| Jan 21–Feb 15 | Analysis, decomposition figures | **G3:** H3 held-out result; if null, report as a boundary result in §Limitations |
| Feb 15–Mar 1 | Draft + **submit IROS 2027** (or ICRA 2028 if G2 slipped past Jan 31) | |

**Risks.** (a) GR00T ignores the language field after single-task finetuning (known for small-data finetunes) → H1 null; the 5k-step pilot detects this for ~6 GPU-h; mitigation: V-both, freeze fewer backbone layers, or add the language-dropout trick. (b) Annotation coverage < 90 % → extend the 31-skill taxonomy regexes (`skill_tax.py`), log in paper. (c) Compute collision with Idea 1 → that is why the second box is recommended.

---

## Idea 3 — *Disagreement Is a Free Alarm*: flow-matching sample dispersion as a predictive failure signal and a zero-retrain recovery trigger

**Type:** runtime algorithm, policy-agnostic, cheap to train (CPU), rides on Idea 1's substrate. **Venue:** ICRA 2028 (or IROS 2027 if Idea 1's matrix finishes by Dec 31 and the pilot gate passes). **Verdict: CONDITIONAL GO — pilot in November, commit on gate.**

### 3.1 The claim

> Both baseline VLAs decode actions by flow matching from Gaussian noise. Sampling K action chunks per query and measuring their dispersion costs one extra batched action-head pass (the VLM backbone runs once) and gives an **intrinsic, label-free uncertainty signal**. We show that on house-scale tasks this dispersion — combined with the receding-horizon chunk disagreement already computed in `chunking.py` — predicts sub-skill failure **≥ 5 s before** the predicate-level evidence appears, outperforming post-hoc progress watchdogs, and that a trigger-matched recovery ladder driven by the predictive signal raises Q over the same ladder driven by the post-hoc rule.

Hypotheses:

- **H1** A detector on {K-sample dispersion, chunk-boundary disagreement, progress velocity, proprio deviation} reaches AUROC ≥ 0.80 at 5 s lead on held-out *tasks* (not just held-out episodes); dispersion alone ≥ 0.70.
- **H2** Predictive-trigger ladder beats the post-hoc rule-trigger ladder (identical ladder, identical budgets) by ≥ 3 Q points and no-intervention by ≥ 6, with false-alarm rate < 15 %.
- **H3** Transfer: a detector trained on GR00T features loses < 5 AUROC points when applied to π0.5 (feature set is policy-agnostic by construction).
- **H4** **Medoid chunk selection** (execute the K-sample medoid instead of a single sample) alone gives +1–3 Q at zero extra latency beyond the batched pass — a "free lunch" baseline we must beat and report.

Positioning: SAFE (2025) and FAIL-Detect (2025) learn failure detectors from VLA internal features on short tabletop tasks; neither reports lead time at multi-minute horizons, partial-credit recovery value, or a trigger-matched intervention comparison. Cite them explicitly and compare to an internal-feature probe as a baseline.

### 3.2 Experimental design

**Features (all onboard, < 10 ms CPU/step).**
(i) K-sample dispersion: per-horizon-step std across K = 8 chunks, summarized as mean/max over the 16-step horizon and per action group (base, torso, arms, grippers); (ii) chunk-boundary disagreement: ‖old_tail − new_head‖ from `blend_chunks`; (iii) progress velocity: predicate flips/s vs mined p90 budget (from `progress.py`); (iv) proprio deviation from per-skill demo mean/std (`src/training/stats.py`); (v) odometry displacement over the last 10 s (`odometry.py`); (vi) skill id + plan position + predicates remaining. Logged every step to `features.parquet` next to the episode JSON.

**Labels.** From Idea 1's `labels.jsonl`: positive at step t if a failure class onset occurs in (t, t + lead] with lead ∈ {3, 5, 10} s, for the currently active skill; episodes with `timeout_clean` and successes supply negatives. Sensitivity table over lead and over the stuck-window definition.

**Models.** Primary: gradient-boosted trees (LightGBM) + a 3 s temporal MLP, trained on CPU; probe: a linear probe on GR00T backbone features (SAFE-style baseline); ablations leave-one-channel-out; **oracle** (privileged) upper bound, labeled as such.

**Recovery ladder (identical for both triggers).** L1 RETRY (re-query with fresh noise) → L2 REPLAN (remaining subgoals from goal minus satisfied) → L3 SEARCH (frontier-biased base override) → L4 reset-to-safe-pose. Budget-capped at 1.5× p90 like the existing watchdog. Trigger A: existing rule (zero predicate delta over the stuck window). Trigger B: detector probability > τ for ≥ 1 s (τ picked on the pilot for 10 % false alarms).

**Matrix.** Detector training: Idea 1's 1,400 episodes (already have features if the tap is in place before the matrix starts — **do this first**). Intervention: 20 tasks × 10 unseen instances × {none, medoid-only, rule-ladder, predictive-ladder, oracle-ladder} = **1,000 episodes** ≈ 14 GPU-days full-length. Transfer: 5 tasks × 10 × 2 arms on π0.5 = 100 episodes.

### 3.3 What to implement (repo-specific)

1. **Batched K-sample decoding** — in the GR00T serving path (`serve_b1k.py` policy wrapper, or our `src/b1k/policy/vla.py` GR00T backend): run the backbone once, tile the embeddings K times along the batch dim, sample K noise tensors, run the flow-matching action head once (batch K), return `(K, 16, 23)`. Same for π0.5 (`openpi` `sample_actions` with `rng` split K ways). Flag `--k-samples 8`. Measure latency; it must stay under the 30 Hz step budget with chunk re-query every 16 steps.
2. **Feature tap** — `src/b1k/policy/uncertainty.py`: dispersion statistics + medoid selection; `chunking.py` returns the boundary residual (one-line change); `controller.py` logs a feature row per step to `features.parquet` via a non-blocking writer.
3. **Detector training** — `src/research/alarm/train_detector.py` (LightGBM + MLP; task-grouped CV; AUROC/AP vs lead; calibration), `tests/test_alarm_features.py` on synthetic streams.
4. **Trigger plumbing** — `progress.py` gets a `trigger: {rule, predictive, oracle}` option; the predictive trigger consumes `uncertainty.py` output; the ladder code is shared, so H2 is trigger-matched by construction.
5. **Latency report** — `scripts/bench_k_samples.sh`: ms per query for K ∈ {1, 4, 8, 16} on the box; goes in the paper.
6. Reuse Idea 1's `run_matrix.py --study alarm` and analysis code.

### 3.4 Testing / gates / timeline

| When | Milestone | Gate |
|---|---|---|
| Oct 25–Nov 8 | K-sample decoding + feature tap on the organizer `turning_on_radio` checkpoints (both policies); latency bench; 60 episodes with features + priv labels | **G0 (Nov 8):** K = 8 fits the step budget; dispersion distribution differs between eventually-failed and successful episodes (Mann-Whitney p < 0.01 on 60 episodes). If no separation → stop; fold the medoid-selection result into Idea 1 as a paragraph |
| Nov 9–Dec 31 | Features logged during Idea 1's full matrix (zero marginal sim cost); detector trained task-grouped | **G1 (Dec 31):** AUROC ≥ 0.70 at 3 s on held-out tasks **and** predictive-trigger beats rule-trigger on a 5-task, 100-episode probe. Else: publish the detector-only result as a workshop paper and defer |
| Jan 1–Feb 10 | 1,000-episode intervention matrix (needs the second box, or runs Feb–Apr for ICRA 2028) | **G2:** H2 margin ≥ 3 Q on ≥ 12/20 tasks |
| IROS path | Feb 10–Mar 1 draft | only if G2 done by Feb 10 |
| ICRA 2028 path | Mar–Aug 2027: add π0.5 transfer, lead-time sensitivity, longer matrix; submit ~Sep 15, 2027 | |

**Risks.** (a) The post-hoc watchdog already recovers most of the recoverable gap → H2 null; the 100-episode probe at G1 catches this before 1,000 episodes are spent, and the detection + lead-time result (H1) plus medoid "free lunch" (H4) still make a workshop/short paper. (b) Flow-matching samples collapse to near-identical chunks after finetuning (low dispersion everywhere) → check at G0; fall back to dropout-MC or two checkpoints as the ensemble. (c) Latency: K = 8 may not fit on a smaller GPU → K = 4.

---

## Shared infrastructure to build first (weeks of Oct 20 and Oct 27)

These unblock all three ideas and should be done before any large run:

1. `src/research/sim/privileged_logger.py` + `EVAL_WRAPPER` support in `scripts/docker/eval.sh` (Idea 1.3.1).
2. `src/research/anatomy/label_failures.py` with tests (Idea 1.3.2).
3. `src/research/common/run_matrix.py` (resumable; one JSON per episode; writes `manifest.csv`) and `analyze_common.py` (bootstrap CI, Q math from `src/evalharness/aggregate.py`).
4. GR00T backend in `src/b1k/policy/vla.py` so the watchdog/prompting/trigger arms wrap the same checkpoint (Ideas 1–3).
5. Feature tap + K-sample decoding (Idea 3.3.1–2) — **before** Idea 1's matrix so the matrix doubles as Idea 3's training data.
6. `src/training/concat_tasks.py` and `src/training/skill_relabel.py` (Ideas 1, 2).
7. Reproducibility discipline from day one: every run records image digests (`docker inspect --format '{{.Id}}'`), checkpoint hashes, instance ids, seeds, and git SHA into the episode JSON directory; the paper's "n per cell" tables are generated, never typed.

## Compute and money (single node + one rented twin)

| Item | GPU-days |
|---|---|
| Idea 1 finetunes (GR00T 20-task, π0.5 20-task) | 3 |
| Idea 1 matrix (1,400 full-length episodes) | 20 |
| Idea 2 finetunes (4 GR00T + 2 π0.5 variants) | 8.5 |
| Idea 2 matrix (1,350 episodes) | 17 |
| Idea 3 intervention matrix (1,100 episodes) | 15 |
| Pilots, re-runs, audits (20 %) | 13 |
| **Total** | **~77 GPU-days** |

Two boxes from Nov 1 to Feb 10 give ~200 GPU-days of capacity — comfortable. One box gives ~100: enough for Ideas 1 + 2 to IROS 2027 with Idea 3 moving to ICRA 2028. Disk: plan 2 TB per box (20-task demos ≈ 0.7 TB, relabeled datasets are symlinked, videos ≈ 0.3 TB per 1,000 episodes at 720p).

## Paper plan

| Paper | Venue | Submit | Primary artifact |
|---|---|---|---|
| Anatomy of Failure (Idea 1) | IROS 2027 | Mar 1, 2027 | `b1k-anatomy` labeled dataset + labeler + protocol |
| Mind the Interface (Idea 2) | IROS 2027 (fallback ICRA 2028) | Mar 1, 2027 | relabeled skill datasets + 4 checkpoints + switching metrics |
| Disagreement Is a Free Alarm (Idea 3) | ICRA 2028 (stretch IROS 2027) | Sep 15, 2027 | `alarm` detector + feature dataset + K-sample serving code |

All three share one evaluation protocol, one task stratification, and one pair of policy families, so the writing cost of the second and third papers is roughly half of the first. Submit the three as a "diagnose → interface → predict" line; each must stand alone (no cross-dependence in claims).

## Backup idea (if an idea dies at its gate)

**"Which 50 demos?" — demonstration selection for long-horizon VLA finetuning.** Using the 200 demos/task, finetune at budgets {25, 50, 100, 200} under three selection rules (random, max instance/layout diversity via `task_instance_id`, shortest-duration) on 5 tasks; 60 short finetunes × 5k steps ≈ 15 GPU-days + 1,200 short episodes ≈ 10 GPU-days. Answers gap G1.4 (data quality vs. scale) on a public house-scale benchmark; low method risk; slots into either venue.
