# Research Plan 2027 — Three Publishable Ideas Built on This Repo

**Date:** 2026-10-19 (rev. 2, after literature overlap check) · **Target venues:** IROS 2027 (paper deadline **Mar 1, 2027**, Florence) primary; ICRA 2028 (~Sep 15, 2027) second slot; CoRL 2027 (~May/June 2027) alternative for Idea 3.
**Supersedes** `docs/icra-iros-2027-ideas.md` / `docs/icra-iros-2027-final-scope.md`. Those assumed a 4–8 GPU cluster and a physical R1Pro; **neither exists**, and several of their novelty claims have since been taken by published work (§0). This plan is sized to what we have and positioned against what exists as of today.

| Asset we have today | Where |
|---|---|
| One single-GPU Docker-only node (Brev) that trains GR00T N1.7 (`scripts/docker/train_groot.sh`, ~2.3 s/it at global batch 128) and serves it on :8000 | `docker/groot/`, `docs/docker-training.md` |
| π0.5 Docker train/serve path | `docker/pi05/`, `scripts/docker/train_pi05.sh` |
| Two-container evaluation (policy container ↔ `stanfordvl/behavior:3.9.2` sim container), headless with video | `scripts/docker/eval.sh` |
| Organizer-provided single-task checkpoints (π0.5 and GR00T, `turning_on_radio`) → zero-training starting point | `docs/docker-training.md` §E.1 |
| 2026 challenge demos: 200 teleop episodes/task, LeRobot v3.0, RGB+depth+proprio, per-episode language skill-segment annotations, `task_instance_id` per episode | `src/training/download.py`, `src/training/lerobot_io.py` |
| 40 official test instances per task (ids 301–340), per-rollout JSON with `q_score.final` (partial credit over BDDL goal predicates) | `docs/challenge-spec.md` §6 |
| Deterministic planner/watchdog stack: 31-skill taxonomy, plan miner, progress state machine (CONTINUE/ADVANCE/RETRY/REPLAN/SEARCH/FINISH), predicate tracker with hysteresis, odometry + occupancy frontier, chunk blending, receding-horizon controller | `src/b1k/planner/*`, `src/b1k/perception/*`, `src/b1k/policy/chunking.py`, `src/b1k/controller.py` |
| Eval harness reproducing official scoring, multi-port fan-out, aggregation/report | `src/evalharness/*` |
| Training seams: predicate-detector trainer (privileged teacher), LoRA adapters, skill-annotation parser | `src/training/detector_train.py`, `adapters.py`, `skill_annotations.py` |

**The one number that sizes everything.** Sim runs at ~13.5 FPS plus 150–300 s scene load; a full-length (~16k-step) episode is ~20–25 min wall-clock with policy and sim sharing one GPU. Sustained throughput: **~50 full-length episodes per GPU-day**. All matrices below fit in **Oct 20, 2026 → Feb 10, 2027** if we rent a second identical box for the evaluation waves (~$1.5–3k). Without it: Ideas 1 + 2 to IROS, Idea 3 to ICRA 2028.

**No real robot.** All three ideas are sim-only by design; abstracts never carry real-world claims. Each releases a durable artifact (labeler / datasets / protocol / code).

---

## 0. Novelty check (literature as of 2026-10-19)

Searched arXiv/web for each idea's headline claim. Findings that change the plan:

| Prior work | What it did | Collides with | What remains open (and is now our claim) |
|---|---|---|---|
| **Behavior-Skill** [1] (Aug 31, 2026) | Skill-level benchmark on BEHAVIOR-1K: 235k skill instances, 34 skill types, π0.5 + GR00T N1.7; evaluates each skill **in isolation from a restored intermediate state**; shows skill-conditioned training beats task-conditioned (+6.0 / +5.6 TSCR) | Idea 1 (skill-level profiles), Idea 2 (skill vs task conditioning) | Authors state they "do not cover task planning, automatic decomposition, or sequential dependencies" and name sequential evaluation as future work. Our Idea 2 is exactly that: **closed-loop, sequential execution with the switching signal controlled**. |
| **How VLAs (Really) Work** [2] (Apr 2026, Huawei) | 8 experts hand-label 500 episodes (50 tasks × 10) of the 2025 winner into a **10-class failure taxonomy**; reproducibility and trial-variance study; progress-aware metric variants | Idea 1 (taxonomy) | Manual, one policy, test instances only. Open: **automatic privileged-state labeling at 10× scale**, **seen-vs-unseen instance generalization**, two policy families, and an **intervention arm** (what is fixable without retraining). We adopt their 10 classes as the human-annotation schema and report κ against it. |
| **Semantic Handoff** [3] (Jul 2026, SimpleAI) | π0.5 skill checkpoints chained by an agent + VLM verifier (advance/retry/replan) on BEHAVIOR-1K; snapshot-start vs chained-start comparison; failure attribution to grounding / execution / readiness | Idea 1 (watchdog arm), Idea 2 (gating), Idea 3 (VLM trigger) | A system with a learned verifier; no oracle/onboard/open-loop decomposition, no controlled conditioning ablation, no detector-family comparison. Their VLM verifier becomes a **baseline gating/trigger arm** in Ideas 2 and 3. |
| **RLC 1st-place report** [4], **Comet 2nd-place report** [5] (Dec 2025) | RLC: System-2 stage tracking (learned onboard progress), multi-sample flow matching at training, hand correction rules; Comet: task-coverage scaling (1→7→10→50 tasks), RFT with perturbed initial poses | Idea 2 (stage tracking ≈ onboard gating), Idea 3 (multi-sample) | Neither measures instance-level generalization or attributes failures automatically. RLC stage tracking becomes a **gating baseline** in Idea 2. |
| **CycleVLA** [6] (Jan 2026), **CARE** [7], **FAR** [8], **AgentChord** [9], **ProAct-VLM** [10] (2026) | Proactive failure prediction + backtracking + **MBR decoding** (= medoid chunk selection); stage-conditioned corrective data; failure-aware retry; task-graph recovery branches; pre-failure VLM replanning. All on LIBERO/tabletop/short real tasks, binary success | Idea 3 (medoid "free lunch", recovery ladder) | MBR/medoid is **known** — we report it as a baseline, not a contribution. None evaluates on multi-minute mobile bimanual tasks with partial credit, none compares detector *families* under one trigger-matched recovery protocol. |
| **SAFE** [11], **FAIL-Detect** [12], **FIPER** [13], **PFD** [14], **Diff-DAgger** [15] (2024–2026) | Runtime failure detectors: VLA internal features (SAFE, with K-sample action variance as a weaker baseline), OOD + flow-density (FAIL-Detect), OOD + action entropy (FIPER), low-rank weight perturbation beating stochastic sampling (PFD), diffusion-loss vs ensemble disagreement (Diff-DAgger: disagreement confounds multimodality) | Idea 3's original "dispersion is a novel alarm" claim | Dispersion is **not** novel and is known to be weaker than internal-feature / perturbation signals. Open: **do any of these signals convert into Q at house scale**, with what lead time, on which failure classes? Idea 3 becomes a comparative detection→recovery study. |
| **ATOM-Bench** [16] (Jun 2026, real-world tabletop) | Atomic vs held-out compositional tasks; Atomic Score / Compositional Failure Share | Idea 2 H3 (compositional held-out) | Real tabletop, no mobile/long-horizon. Compositional held-out on BEHAVIOR with controlled switching is open. |
| **AtomVLA** [17], **RoboTwin hierarchical study** [18], **HiRoC** [19], **Anticipation-VLA** [20], **Steerable Policies** [21] | Subtask-instruction ablations on LIBERO-Long (+2.2 pts), RoboTwin oracle subtask prompts (45.4→53.8 %), hierarchical post-training, subgoal generation, richer command interfaces | Idea 2 H1 | All use oracle or learned switching without isolating *when-to-switch* from *what-to-say*. Our oracle/onboard/open-loop/shuffled design is the missing control. |
| **Silent-failure observability** [22], **SSC** [23], **RoboMD** [24] | Privileged-state labeling on ALOHA; structured subtask labels on BEHAVIOR; RL-driven vulnerability discovery | Idea 1 methodology | Supporting, not competing: cite as precedent for privileged-state labeling and structured skill annotation. |

**Net effect.** Idea 1 narrows from "first failure taxonomy" to **automatic attribution + instance-generalization gap + fixability**; Idea 2 narrows from "does skill conditioning help" (answered: yes, in isolation) to **does it survive closed-loop sequencing, and is the bottleneck what-to-say or when-to-switch**; Idea 3 drops the dispersion-novelty claim and becomes **"Do runtime failure detectors pay for themselves at house scale?"** — a detector-family comparison measured in recovered Q. All three remain publishable; the bar is now "clean controlled measurement nobody has run" rather than "first to look".

---

## Idea 1 — *Where the Demos End*: automatic failure attribution and the instance-generalization gap of house-scale VLA policies

**Type:** measurement study + released labeler. **Venue:** IROS 2027. **Verdict: GO — primary. Start now.**

### 1.1 The claim

> Hand-labeled failure studies of BEHAVIOR-1K policies exist for one policy on a few hundred episodes [2]. We replace expert viewing with an **automatic attribution pipeline** driven by privileged simulator state (goal predicates, poses, grasps, room occupancy) that labels every rollout step into the 10-class taxonomy of [2] plus onset time and active skill, validated at κ ≥ 0.7 against blind human labels. We use it to answer two questions nobody has measured: (i) how much Q is lost moving from instances whose object/layout composition appears in the demos to the official unseen instances, and in which skill strata; (ii) which failure classes a no-retrain watchdog recovers. Two policy families (π0.5, GR00T N1.7), 20 tasks, ~1,400 labeled episodes released.

Hypotheses:

- **H1** The automatic labeler agrees with blind human annotation at κ ≥ 0.7 on the 10-class schema of [2] (≥ 0.8 after collapsing to 5 super-classes).
- **H2** Q drops ≥ 10 points (0–100) from seen-composition to unseen instances; the drop is ≥ 2× larger on multi-room / ≥ 4-skill tasks than on single-room / ≤ 3-skill tasks, and is dominated by navigation-failure and semantic-confusion classes rather than grasp failure.
- **H3** The watchdog (RETRY/REPLAN/SEARCH, no retraining) recovers ≥ 20 % of the Q lost to {abrupt termination, execution-order confusion, navigation failure} and < 5 % of the Q lost to {grasp, placement, improper handling} — i.e. the fixability map separates "planning-side" from "data-side" loss.

### 1.2 Experimental design

**Tasks (20).** Stratify the 100 tasks by mined skill-plan statistics (`src/b1k/planner/mine_plans_cli.py`): 5 single-room ≤ 3 skills, 5 single-room ≥ 4 skills, 5 multi-room navigation-heavy, 5 bimanual/articulated-heavy. Script `src/research/anatomy/select_tasks.py` → `docs/research/anatomy_tasks.json` (Table 1 of the paper).

**Policies (3 arms).**

| Arm | What | How |
|---|---|---|
| P-GR00T | GR00T N1.7 multi-task finetune on the 20 tasks | `train_groot.sh train` on a concatenated dataset (1.3); ~50k steps ≈ 32 h once. Organizer `turning_on_radio` checkpoint reported as a sanity row |
| P-π0.5 | π0.5 multi-task finetune, same split | `train_pi05.sh train` |
| P-GR00T+WD | P-GR00T served through `progress.py` with trained predicate detectors | H3 arm; same checkpoint, wrapper differs only |

**Instance conditions (2).** C-seen = instances whose composition matches demo episodes (via `task_instance_id` in `meta/episodes/*.parquet`; evaluate on those instance files if available in `OMNIGIBSON_DATA_PATH`, else on public 301–320 with composition overlap). C-unseen = hidden 321–340. State in the paper that demo instances (1–200) and test instances (301–340) are disjoint by id; "seen" is composition overlap, not id. This axis is absent from [1], [2], [4], [5].

**Matrix.** 20 tasks × 2 conditions × 10 instances × 3 arms = 1,200 episodes + 200 repeats (sim-nondeterminism estimate, following [2]'s variance finding). Full-length. ≈ **470 GPU-h ≈ 20 GPU-days.**

**Labeler rules (privileged, analysis-time only), mapped to the schema of [2]:**

| Class [2] | Automatic rule |
|---|---|
| Grasp failure | ≥ 2 gripper-close events within 0.3 m of a goal object, no `held` transition |
| Improper object handling | `held` object dropped (held → not held) outside any goal receptacle |
| Placement failure | object released inside 0.5 m of goal receptacle, goal predicate not satisfied |
| Skill failure (articulated/state) | robot within 1 m of target, ≥ 10 s of arm motion, predicate never flips |
| Semantic confusion (wrong object) | grasp of an object not in the BDDL goal nor in the mined plan |
| Navigation failure | target room never entered (pose vs. `rooms.py` polygons) or base displacement < 0.5 m for > 30 s mid-plan |
| Collision | contact events with non-task objects above impulse threshold (OmniGibson contact API) |
| Execution-order confusion | predicate flips out of order w.r.t. mined plan (e.g. door closed before item placed) |
| Task confusion | ≥ 2 skills executed that belong to a different task's mined plan |
| Abrupt termination | no action-norm change and no predicate change for > 60 s with budget remaining |

Validation: 2 annotators blind-label 300 random episodes from `--write-video` output using [2]'s instructions; report κ (gate H1).

**Metrics.** Q (official math, n per cell, bootstrap CI), task success, time score, per-skill success (31 skills × 20 tasks), horizon-decay (Q vs. skills remaining), ΔQ(seen−unseen) per stratum, failure-class frequency per arm and condition, fixability map ΔQ(P-GR00T+WD − P-GR00T) per class, worst-case (min-over-task) Q, labeler κ.

### 1.3 What to implement

1. `src/research/sim/privileged_logger.py` — subclass of `omnigibson.eval.wrappers.RGBDFullResWrapper` writing per-step `{step, goal_predicates, base_pose, gripper_l/r, held_objects, contacts, nearest_goal_dist, room_id}` to `/out/<task>_<inst>_<roll>.priv.parquet`. Mounted with `-v $PWD/src:/work/src -e PYTHONPATH=/work`; add `EVAL_WRAPPER` env var to `scripts/docker/eval.sh`.
2. `src/research/anatomy/label_failures.py` — rules above → `labels.jsonl` (class, onset step, active skill). CPU, unit-tested on synthetic traces (`tests/test_label_failures.py`).
3. `src/training/concat_tasks.py` — 20-task LeRobot v3 dataset (symlinked videos, concatenated `meta/episodes`, re-indexed `task_index`).
4. Trained detectors via `src/training/detector_train.py` → `data/detectors/<family>.pt`; gate F1 ≥ 0.85 held-out.
5. GR00T backend in `src/b1k/policy/vla.py` (`_act_vla` → in-process `Gr00tPolicy` or call to the serve container) so the watchdog wraps the same checkpoint.
6. `src/research/common/run_matrix.py` — resumable (task, condition, instance, arm) enumerator over `eval.sh`; `manifest.csv`.
7. `src/research/anatomy/analyze.py` — joins official JSONs with `labels.jsonl`; all tables/figures with bootstrap CIs.
8. Release: HF dataset `b1k-anatomy` (labels + priv parquet + videos) + labeler.

### 1.4 Gates / timeline

| When | Milestone | Gate |
|---|---|---|
| Oct 20–31 | Logger + labeler on organizer `turning_on_radio` checkpoints, 40 episodes (20 seen / 20 unseen) | Pipeline runs in Docker end-to-end; non-degenerate class histogram; first ΔQ(seen−unseen) |
| Nov 1–15 | 20-task download (~0.7 TB), `concat_tasks.py`, GR00T + π0.5 finetunes, detectors | **G1 (Nov 15):** 5-task pilot (100 eps): ΔQ(seen−unseen) ≥ 5 on ≥ 1 policy **and** 50-episode κ ≥ 0.6. Else: add BDDL-resampled perturbation instances / refine rules before scaling |
| Nov 16–Dec 31 | Full 1,400-episode matrix | **G2 (Dec 31):** κ ≥ 0.7 on 300-episode audit |
| Jan 1–31 | Analysis, fixability map | **G3:** H2 holds directionally; if H3 null, the paper is labeler + generalization gap (still novel vs. [2]) |
| Feb 1–Mar 1 | Draft, **submit IROS 2027** | |

**Risks.** (a) Both policies ≈ 0 Q on unseen → no separation (G1 catches on 100 eps). (b) Weak finetunes → use organizer checkpoints on their tasks, 10-task matrix. (c) Reviewer: "incremental over [2]" → answer is scale (1,400 vs 500 episodes, automatic), the instance axis, two policy families, and the intervention arm; cite [2] as the schema we validate against.

---

## Idea 2 — *What to Say vs. When to Switch*: does skill-level conditioning survive closed-loop sequential execution?

**Type:** controlled method study. **Venue:** IROS 2027 (co-submission) or ICRA 2028. **Verdict: GO — second.**

### 2.1 The claim

> Behavior-Skill [1] shows skill-conditioned training improves isolated skill execution on BEHAVIOR-1K (+6 points), and hierarchical VLA systems [3, 4, 17–21] assume the gain carries into full tasks. But in closed-loop execution the low-level policy is only as good as the signal that tells it *which* sub-goal is active. We train task-, skill-, shuffled-skill- and both-conditioned variants of the same VLA on identical data and evaluate **full sequential tasks** under four switching regimes — **oracle** (privileged predicate flip), **onboard-detector** (trained predicate classifiers), **learned stage tracker** (RLC-style [4]), **open-loop** (demo p50 budgets) — plus a VLM-verifier regime [3] as a system baseline. This decomposes task Q into *interface value* (what to say), *switching cost* (when to switch), and *execution residual*, and tests compositional generalization to held-out tasks built from seen skills.

Hypotheses:

- **H1** Skill-conditioned + oracle switching beats task-conditioned by ≥ 8 Q points on ≥ 4-skill tasks and ≤ 3 on ≤ 3-skill tasks; the gain in full tasks is **smaller** than [1]'s isolated-skill gain (sequencing eats part of it — the quantity we report).
- **H2** Switching matters more than wording: open-loop switching recovers < 30 % of the oracle gain; onboard detectors ≥ 60 %; learned stage tracking between them. Late/early switch time (vs. oracle) explains ≥ 50 % of the variance in per-episode Q loss.
- **H3** On 5 held-out tasks composed of seen skills, skill-conditioned Q ≥ 2× task-conditioned Q (compositional reuse; cf. ATOM-Bench [16] real-tabletop result).
- **H4** Shuffled skill text ≈ task conditioning (semantics, not prompt length, carry the gain).

### 2.2 Experimental design

**Tasks.** 15 training tasks (Idea 1 strata minus bimanual) + 5 held-out tasks whose mined plans use only skills in the training 15 (`select_tasks.py --heldout-by-skill-coverage`).

**Variants (GR00T N1.7; π0.5 replicated on 5 tasks).**

| Variant | Training language field |
|---|---|
| V-task | dataset task instruction |
| V-skill | per-frame skill-segment text via `skill_tax.render_text` |
| V-skill-shuffled | same segmentation, text from a different segment of the same episode |
| V-both | `"<task>. Now: <skill>"` |

Identical demos / steps (50k) / hyper-parameters. 4 GR00T finetunes ≈ 5.5 GPU-days; 2 π0.5 ≈ 3.

**Switching regimes (for skill-conditioned variants).**

| Regime | ADVANCE signal | Implementation |
|---|---|---|
| S-oracle | privileged predicate flip of active skill's exit family | sim-side `PrivilegedLoggerWrapper` publishes predicate state on a side-channel; `progress.py` consumes it |
| S-onboard | `PredicateTracker` over trained detectors | existing `detectors/state.py` |
| S-stage | learned stage head on VLA features + voting (reimplementation of [4]) | small MLP on backbone pooled features, trained on demo segment labels |
| S-openloop | mined p50 skill duration | `TaskPlanner` budgets, detectors off |
| S-vlm (system baseline) | VLM verifier advance/retry/replan as in [3] | Qwen-VL on head + wrist frames every 2 s |

**Matrix.** 20 tasks × 10 unseen instances × {V-task; V-skill × {oracle, onboard, stage, openloop}; V-both × onboard; V-shuffled × oracle; V-skill × vlm} = 20 × 10 × 8 = **1,600 episodes** ≈ 22 GPU-days full-length (12 if capped at 2× human mean). π0.5 replication: 5 × 10 × 3 = 150.

**Metrics.** Q, per-skill success, **switch-time error** (ADVANCE − oracle flip; precision/recall of ADVANCE events), horizon-decay per variant, held-out Q (H3), decomposition: Q(skill, oracle) − Q(task) = interface value; Q(oracle) − Q(regime) = switching cost; 1 − Q(oracle) = execution residual. Report alongside [1]'s TSCR on the same tasks where available.

### 2.3 What to implement

1. `src/training/skill_relabel.py` — per-frame `task_index` rewrite from annotation segments (`lerobot_io`, `skill_annotations.py`), `meta/tasks.parquet` with skill texts, symlinked videos; `--mode {task,skill,shuffled,both}`.
2. Loader test: 3 frames across a segment boundary get different language in `examples/b1k/r1pro.py` data config.
3. `src/b1k/policy/vla.py` — GR00T backend sends `Subgoal.text` per query; `controller.py` already re-queries on ADVANCE.
4. `src/b1k/protocol.py` — optional `privileged` field; server refuses it under `--strict-onboard` (challenge path stays legal).
5. `progress.py` — `switch: {oracle, onboard, stage, openloop, vlm}`; `miner.py` exposes p50/p90 durations.
6. `src/research/interface/stage_head.py` — RLC-style stage classifier + voting; `src/research/interface/vlm_verifier.py` — [3]-style verifier.
7. `src/research/interface/switch_metrics.py` — aligns ADVANCE events with oracle flips.
8. Reuse `run_matrix.py --study interface`.

### 2.4 Gates / timeline

| When | Milestone | Gate |
|---|---|---|
| Oct 20–Nov 10 | `skill_relabel.py` + loader test; relabel `turning_on_radio`; 5k-step pilots V-task vs V-skill (~3 h each); 20 episodes each under S-oracle | **G1 (Nov 10):** V-skill+oracle > V-task by any margin; relabeler covers ≥ 90 % of segments. If V-skill is worse, debug prompt format / try V-both before scaling |
| Nov 11–Dec 10 | 15-task relabeled datasets; 4 GR00T variants at 50k; stage head; detectors (shared with Idea 1) | losses comparable across variants |
| Dec 11–Jan 25 | 1,600-episode matrix + π0.5 replication | **G2 (Jan 25):** H1 directionally on ≥ 10/15 tasks; if H2 inverts (openloop ≈ onboard) that *is* the finding — retitle |
| Jan 26–Feb 15 | Analysis, decomposition figures | **G3:** H3; null → Limitations |
| Feb 15–Mar 1 | Draft + **submit IROS 2027** (ICRA 2028 if G2 slips past Feb 1) | |

**Risks.** (a) GR00T ignores language after small-data finetune → 5k-step pilot catches it (~6 GPU-h); mitigate with V-both, fewer frozen layers, language dropout. (b) Annotation coverage < 90 % → extend `skill_tax.py` regexes, log in paper. (c) Reviewer: "[1] already showed skill conditioning helps" → our abstract states the isolated result as the premise and the closed-loop decomposition as the contribution.

---

## Idea 3 — *Do Runtime Failure Detectors Pay for Themselves at House Scale?* A trigger-matched detection→recovery study

**Type:** comparative empirical study of runtime monitors, policy-agnostic, rides on Idea 1's substrate. **Venue:** ICRA 2028 (IROS 2027 stretch). **Verdict: CONDITIONAL GO — pilot in November, commit on gate.**

### 3.1 The claim

> Runtime failure detectors for generative policies — internal-feature probes [11], OOD/density scores [12, 13], weight-perturbation disagreement [14], action-sample dispersion [11, 15], progress rules, and VLM verifiers [3, 6] — are evaluated on tabletop tasks with binary success and a "stop or ask for help" action. Nobody has asked whether any of them **convert into recovered partial credit on multi-minute mobile bimanual tasks**. We run six detector families through one trigger-matched recovery ladder on BEHAVIOR-1K and report, per family and per failure class: AUROC/AP at 3/5/10 s lead, false-alarm rate, latency, and **ΔQ over no-intervention and over the post-hoc progress rule**. Detector quality and recovery value turn out to be different rankings — that is the paper.

Hypotheses:

- **H1** Internal-feature (SAFE-style) and perturbation (PFD-style) detectors reach AUROC ≥ 0.80 at 5 s lead on held-out tasks; dispersion alone ≤ 0.70 (replicating [11, 14] at house scale).
- **H2** At least one predictive trigger beats the post-hoc progress rule by ≥ 3 Q and no-intervention by ≥ 6 Q with false-alarm < 15 %; the gain is concentrated in {navigation, execution-order, abrupt-termination} classes and ≈ 0 for grasp/placement classes (recovery ladder cannot fix data-side failures — ties to Idea 1 H3).
- **H3** Detection rank ≠ recovery rank: the best-AUROC family is not the best-ΔQ family (lead time and false-alarm cost dominate AUROC once a ladder is attached).
- **H4** Features that are policy-agnostic (dispersion, progress, proprio) transfer GR00T→π0.5 with < 5 AUROC loss; internal-feature probes do not transfer without retraining.

MBR/medoid chunk selection [6] is included as a **known** zero-trigger baseline, not a contribution.

### 3.2 Experimental design

**Detector families (all onboard, logged every step to `features.parquet`).**

| Family | Signal | Source |
|---|---|---|
| D-rule | zero predicate delta over stuck window (existing watchdog) | `progress.py` |
| D-disp | K = 8 flow-matching samples: per-horizon std, per action group; + chunk-boundary residual | `uncertainty.py`, `chunking.py` |
| D-feat | linear/MLP probe on action-head last-layer features (SAFE [11]) | GR00T / openpi hooks |
| D-pert | low-rank weight perturbation disagreement (PFD [14]), K = 4 | hook on action-head attention weights |
| D-prog | progress-velocity + proprio-deviation + odometry features, GBDT | `progress.py`, `stats.py`, `odometry.py` |
| D-vlm | VLM verifier every 2 s (CycleVLA / Semantic-Handoff style [3, 6]) | `vlm_verifier.py` (shared with Idea 2) |
| D-oracle | privileged label | upper bound |

**Labels.** From Idea 1's `labels.jsonl`: positive at step t if a failure onset occurs in (t, t + lead], lead ∈ {3, 5, 10} s, for the active skill; successes and clean timeouts are negatives. Sensitivity table over lead and stuck-window definition.

**Recovery ladder (identical for all triggers).** L1 RETRY (fresh-noise re-query) → L2 REPLAN (remaining subgoals from goal − satisfied) → L3 SEARCH (frontier-biased base override) → L4 safe pose. Budget-capped at 1.5× p90. Trigger = detector score > τ for ≥ 1 s, τ set on the pilot at 10 % false alarms per family.

**Matrix.** Detector training: Idea 1's 1,400 episodes with the feature tap live (zero marginal sim cost). Intervention: 20 tasks × 10 unseen instances × {none, medoid-only, D-rule, D-disp, D-feat, D-pert, D-prog, D-vlm, D-oracle} = **1,800 episodes** ≈ 25 GPU-days full-length (reduce to 6 arms / 1,200 episodes if single box). Transfer: 5 tasks × 10 × {D-disp, D-feat} on π0.5 = 100.

**Metrics.** AUROC/AP vs lead (task-grouped CV), false-alarm/hour, lead-time distribution, ms/step latency (K ∈ {1,4,8,16}), ΔQ per arm with CI, ΔQ per failure class (Idea 1 labels), worst-case Q, transfer loss.

### 3.3 What to implement

1. Batched K-sample decoding in the GR00T serve path (backbone once, action head batch K) and π0.5 (`sample_actions` with split rng); flag `--k-samples`. Latency bench `scripts/bench_k_samples.sh`.
2. `src/b1k/policy/uncertainty.py` — dispersion stats + medoid; `chunking.py` returns boundary residual; `controller.py` writes a feature row per step (non-blocking).
3. Feature hooks: action-head last-layer pooled features (D-feat); low-rank perturbation wrapper (D-pert) — both as forward hooks, off by default.
4. `src/research/alarm/train_detectors.py` — per-family detector (LightGBM / linear / MLP), task-grouped CV, calibration; `tests/test_alarm_features.py`.
5. `progress.py` — `trigger: {rule, predictive:<family>, oracle}`; shared ladder.
6. Reuse `run_matrix.py --study alarm`, `vlm_verifier.py` from Idea 2.

### 3.4 Gates / timeline

| When | Milestone | Gate |
|---|---|---|
| Oct 25–Nov 8 | K-sample + hooks on organizer checkpoints (both policies); latency bench; 60 episodes with features + labels | **G0 (Nov 8):** K = 8 fits the 30 Hz / 16-step re-query budget; ≥ 2 families separate failed vs. successful episodes (Mann-Whitney p < 0.01). If none → stop; fold into Idea 1 |
| Nov 9–Dec 31 | Features logged through Idea 1's matrix; detectors trained | **G1 (Dec 31):** best family AUROC ≥ 0.70 at 3 s held-out tasks **and** a predictive trigger beats D-rule on a 5-task / 100-episode probe. Else: detector-comparison-only workshop paper |
| Jan 1–Feb 10 | Intervention matrix (second box) — or Feb–Apr for ICRA 2028 | **G2:** H2 on ≥ 12/20 tasks |
| IROS path | Feb 10–Mar 1 draft | only if G2 done by Feb 10 |
| ICRA 2028 path | Mar–Aug 2027: full 9 arms, π0.5 transfer, lead sensitivity; submit ~Sep 15, 2027 | |

**Risks.** (a) D-rule already recovers most recoverable Q → H2 null; the G1 probe catches it before 1,800 episodes; H1/H3/H4 still make a comparison paper. (b) Flow samples collapse after finetuning (low dispersion) → documented, D-pert and D-feat carry. (c) Reviewer: "SAFE/CycleVLA did this" → neither measured ΔQ at house scale or compared families under one ladder; our Related Work says so with their numbers.

---

## Shared infrastructure (weeks of Oct 20 and Oct 27)

1. `src/research/sim/privileged_logger.py` + `EVAL_WRAPPER` in `scripts/docker/eval.sh`.
2. `src/research/anatomy/label_failures.py` + tests (10-class schema of [2]).
3. `src/research/common/run_matrix.py`, `analyze_common.py` (bootstrap CI; Q math from `src/evalharness/aggregate.py`).
4. GR00T backend in `src/b1k/policy/vla.py`.
5. Feature tap + K-sample decoding + feature hooks — **before** Idea 1's matrix so it doubles as Idea 3's training data.
6. `src/training/concat_tasks.py`, `src/training/skill_relabel.py`.
7. `src/research/interface/vlm_verifier.py` (shared baseline for Ideas 2 and 3).
8. Reproducibility: every run records image digest, checkpoint hash, instance ids, seeds, git SHA into the output dir; n-per-cell tables are generated, never typed.

## Compute and money (single node + one rented twin)

| Item | GPU-days |
|---|---|
| Idea 1 finetunes (GR00T + π0.5, 20 tasks) | 3 |
| Idea 1 matrix (1,400 episodes) | 20 |
| Idea 2 finetunes (4 GR00T + 2 π0.5) | 8.5 |
| Idea 2 matrix (1,750 episodes) | 22 |
| Idea 3 intervention matrix (1,900 episodes) | 25 |
| Pilots, re-runs, audits (20 %) | 16 |
| **Total** | **~95 GPU-days** |

Two boxes Nov 1–Feb 10 ≈ 200 GPU-days capacity. One box ≈ 100: Ideas 1 + 2 to IROS 2027 with reduced Idea 2 matrix (drop S-vlm, S-stage → 1,200 eps), Idea 3 to ICRA 2028. Disk: 2 TB per box.

## Paper plan

| Paper | Venue | Submit | Primary artifact |
|---|---|---|---|
| Where the Demos End (Idea 1) | IROS 2027 | Mar 1, 2027 | `b1k-anatomy` labeled dataset + automatic labeler + instance-generalization protocol |
| What to Say vs. When to Switch (Idea 2) | IROS 2027 (fallback ICRA 2028) | Mar 1, 2027 | relabeled skill datasets + 4 checkpoints + switching metrics + oracle side-channel |
| Do Runtime Failure Detectors Pay for Themselves (Idea 3) | ICRA 2028 (stretch IROS 2027) | Sep 15, 2027 | detector zoo + feature dataset + trigger-matched ladder |

Each must stand alone; shared protocol/strata/policies halve the writing cost of papers 2 and 3.

## Kickoff — how the team starts (week of Oct 20)

### Reading order for a new team member
1. `README.md` → `docs/docker-training.md` (get GR00T serve + `scripts/docker/eval.sh` running on the organizer `turning_on_radio` checkpoint; this is the smoke test for everything else).
2. This document, §0 first, then the idea you own.
3. `docs/challenge-spec.md` §6 (instance ids, Q math) and `docs/design.md` §4 (planner/watchdog architecture the research arms wrap).
4. The three papers your idea is positioned against ([1][2] for Idea 1; [1][3][4] for Idea 2; [6][11][14] for Idea 3). Read them before writing code.

### Assumptions this plan rests on that nobody has measured yet — verify in week 1

| # | Assumption | How to verify | Owner | If false |
|---|---|---|---|---|
| A1 | Two-container eval (`train_groot.sh serve` + `eval.sh`) produces an official JSON + MP4 for the organizer checkpoint | Run it once, headless, on the Brev box; archive the JSON | infra | Every matrix is blocked — fix first |
| A2 | ~50 full-length episodes / GPU-day with policy and sim on one GPU | Time 5 consecutive `eval.sh` runs incl. scene load; record `episodes/day` in `docs/research/throughput.md` | infra | Re-scale all matrices by the measured ratio; decide on the second box |
| A3 | Demo-composition instances (C-seen) are evaluable: either demo instance files (ids 1–200) exist in `OMNIGIBSON_DATA_PATH`, or public 301–320 overlap in composition | Inspect `task_instance_id` in `meta/episodes` vs. the instance files in the sim image | Idea 1 | Define C-seen as public 301–320 and C-unseen as hidden 321–340; state the weaker definition in the paper |
| A4 | Privileged state (goal predicates, poses, held objects, contacts, rooms) is readable from a custom `--env-wrapper` inside `stanfordvl/behavior:3.9.2` | Write the 40-line logger stub, run 1 episode, open the parquet | Idea 1 | Fall back to parsing the official per-predicate JSON (coarser onset times) |
| A5 | Per-frame `task_index` changes are honoured by the GR00T LeRobot v3 loader | Unit test across a segment boundary | Idea 2 | Use `V-both` with the task text and inject skill text via a second language field |
| A6 | K = 8 batched flow-matching samples fit the 30 Hz / 16-step re-query budget on the box | `scripts/bench_k_samples.sh` | Idea 3 | K = 4; if still over budget, D-disp becomes offline-only |
| A7 | Released artifacts (videos, labels) are license-compatible with the demo dataset terms | Read the HF dataset card license | all | Release labels + parquet only, not videos |

### Work breakdown (first four weeks)

| Ticket | Deliverable | Depends on | Idea | Target |
|---|---|---|---|---|
| T1 | A1 done; `docs/research/throughput.md` with A2 numbers | — | shared | Oct 23 |
| T2 | `src/research/sim/privileged_logger.py` + `EVAL_WRAPPER` in `eval.sh`; 1 parquet from 1 episode (A4) | T1 | shared | Oct 27 |
| T3 | `src/research/anatomy/label_failures.py` + `tests/test_label_failures.py` (10-class schema of [2]) | T2 | 1 | Oct 31 |
| T4 | `src/research/common/run_matrix.py` (resumable) + `analyze_common.py` (bootstrap CI over official JSONs) | T1 | shared | Oct 31 |
| T5 | GR00T backend in `src/b1k/policy/vla.py`; watchdog arm serves the organizer checkpoint end-to-end | T1 | shared | Nov 3 |
| T6 | `src/training/concat_tasks.py`; 20-task download started (disk ≥ 2 TB) | A3 | 1 | Nov 3 |
| T7 | `src/training/skill_relabel.py` + loader test (A5); relabeled `turning_on_radio` | — | 2 | Oct 31 |
| T8 | 5k-step V-task vs V-skill pilot finetunes + 20 oracle-switched episodes each → **G1-Idea 2 (Nov 10)** | T2, T5, T7 | 2 | Nov 10 |
| T9 | K-sample decoding + feature tap + `bench_k_samples.sh` (A6); 60 labeled episodes → **G0-Idea 3 (Nov 8)** | T2, T3, T5 | 3 | Nov 8 |
| T10 | 40-episode seen/unseen pilot on organizer checkpoints with labels → first ΔQ number | T2, T3, T4 | 1 | Oct 31 |
| T11 | GR00T + π0.5 20-task finetunes launched; detectors trained → **G1-Idea 1 (Nov 15)** | T6 | 1 | Nov 15 |
| T12 | Decision: rent second box (yes if A2 < 60 eps/day and all three ideas are go) | T1 | lead | Oct 24 |

Suggested staffing for a 2–3 person team: one owner per idea; the shared tickets (T1, T2, T4, T5) go to whoever owns Idea 1 since its matrix is the substrate. Every ticket lands as a PR to `main` with a one-paragraph result note in `docs/research/log.md` (create on first use). Nothing in `src/research/` may be imported from `src/b1k/` — the challenge serving path must stay unchanged and onboard-only.

### Definition of "ready to run a matrix"
A1–A4 verified, T2–T5 merged, the organizer-checkpoint pilot (T10) produced labeled JSONs, and the per-episode output directory records image digest, checkpoint hash, instance ids, seeds and git SHA.

## Backup idea

**"Which 50 demos?"** — demo selection for long-horizon finetuning: budgets {25, 50, 100, 200} × rules {random, max layout diversity via `task_instance_id`, shortest duration} × 5 tasks; 60 finetunes × 5k steps ≈ 15 GPU-days + 1,200 short episodes ≈ 10. Comet [5] scaled *tasks*; nobody has scaled *demos per task* with a diversity criterion on BEHAVIOR. Low method risk.

---

## References (verified 2026-10-19)

1. Ma, C. et al. *Behavior-Skill: A Fine-Grained Benchmark for Evaluating Vision-Language-Action Policies in Long-Horizon Tasks.* arXiv:2608.30536, Aug 31 2026. https://arxiv.org/abs/2608.30536 — code https://github.com/nubot-nudt/Behavior-Skill
2. Rasouli, A. et al. (Huawei Noah's Ark). *How VLAs (Really) Work In Open-World Environments.* arXiv:2604.21192, Apr 2026. https://arxiv.org/abs/2604.21192
3. Rui, K. et al. (SimpleAI). *Diagnosing Semantic Handoff Failures in Agent-Orchestrated Vision-Language-Action Skill Composition.* arXiv:2607.06256, Jul 2026. https://arxiv.org/abs/2607.06256
4. Larchenko, I., Zarin, G., Karnatak, A. *Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge.* arXiv:2512.06951, Dec 2025. https://arxiv.org/abs/2512.06951
5. Bai, J. et al. (NVIDIA). *Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge.* arXiv:2512.10071, Dec 2025. https://arxiv.org/abs/2512.10071
6. Ma, C. et al. (Oxford). *CycleVLA: Proactive Self-Correcting Vision-Language-Action Models via Subtask Backtracking and Minimum Bayes Risk Decoding.* arXiv:2601.02295, Jan 2026. https://arxiv.org/abs/2601.02295
7. *CARE: Experience-Guided Atomic Corrective Execution for Vision-Language-Action Policies.* arXiv:2609.24118, Sep 2026. https://arxiv.org/abs/2609.24118
8. Hao, H. et al. (CMU). *FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement.* arXiv:2607.01111, Jul 2026. https://arxiv.org/abs/2607.01111
9. *From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation (AgentChord).* arXiv:2605.11951, May 2026. https://arxiv.org/abs/2605.11951
10. *ProAct-VLM: Pre-Failure Vision-Language Task Replanning with Continuous Perception Feedback.* arXiv:2609.37681, Sep 2026. https://arxiv.org/abs/2609.37681
11. Gu, Q. et al. *SAFE: Multitask Failure Detection for Vision-Language-Action Models.* arXiv:2506.09937, Jun 2025. https://arxiv.org/abs/2506.09937 — https://vla-safe.github.io/
12. Xu, C. et al. (TRI). *Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies (FAIL-Detect).* arXiv:2503.08558, Mar 2025. https://arxiv.org/abs/2503.08558
13. *FIPER: Failure Prediction at Runtime for Generative Robot Policies.* arXiv:2510.09459, Oct 2025. https://arxiv.org/abs/2510.09459
14. Lee, Y., Har, D. (KAIST). *Perturbation-Based Epistemic Uncertainty for Failure Detection in Vision-Language-Action Models (PFD).* arXiv:2606.20754, Jun 2026. https://arxiv.org/abs/2606.20754
15. Lee, S.-W., Kang, X., Kuo, Y.-L. *Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation.* arXiv:2410.14868, 2024/2025. https://arxiv.org/abs/2410.14868
16. Wu, Z. et al. (BAAI). *ATOM-Bench: A Real-World Benchmark for Atomic Skills and Compositional Generalization in Manipulation Policies.* arXiv:2606.16826, Jun 2026. https://arxiv.org/abs/2606.16826
17. *AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models.* arXiv:2603.08519, Mar 2026. https://arxiv.org/abs/2603.08519
18. *Enhancing Long-Horizon Robot Manipulation with Hierarchical Subtask Conditioning (RoboTwin study).* MLHMI 2026. https://stargazerzj.github.io/RoboTwin/materials/MLHMI_2026_Paper.pdf
19. *Beyond Flat Policies: Hierarchical Post-Training for Embodied Agents in Robotic Manipulation (HiRoC).* arXiv:2608.05999, Aug 2026. https://arxiv.org/abs/2608.05999
20. *Anticipation-VLA: Solving Long-Horizon Embodied Tasks via Anticipation-based Subgoal Generation.* arXiv:2605.01772, May 2026. https://arxiv.org/abs/2605.01772
21. *Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control.* arXiv:2602.13193, Feb 2026. https://arxiv.org/abs/2602.13193
22. *How Visible Are Silent Manipulation Failures? An Observability Study of False-Success Detection in Simulated Robot Episodes.* arXiv:2606.03134, Jun 2026. https://arxiv.org/abs/2606.03134
23. *SSC: A Verifiable Structured Representation for Bimanual Manipulation Labelling.* arXiv:2608.04425, Aug 2026. https://arxiv.org/abs/2608.04425
24. Sagar, S. et al. *RoboMD: Uncovering Robot Vulnerabilities through Semantic Potential Fields.* ICLR 2025, arXiv:2412.02818. https://arxiv.org/abs/2412.02818
25. Li, C. et al. *BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation.* arXiv:2403.09227. https://arxiv.org/abs/2403.09227
26. 2025 BEHAVIOR Challenge leaderboard (RLC 0.2599 / Comet 0.2514 held-out Q). https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html

Earlier background (VLA surveys, π0.5, GR00T N1, Hi Robot, HAMSTER, LIBERO, venue facts) is in `docs/icra-iros-2027-trends.md` (63 sources, Sep 2026). **Re-run this overlap check in January 2027 before drafting**: the BEHAVIOR-1K failure/recovery space produced ≥ 6 directly relevant papers in the last 7 months.
