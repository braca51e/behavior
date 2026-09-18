# Three Scoped Research Ideas for ICRA/IROS 2027-28 (BEHAVIOR House-Scale Manipulation)

**For:** task `t_aeffa76c` (3 scoped ideas) → consumed by `t_0423f72f` (feasibility, robotics-engineer) and `t_f4013e5a` (final scope doc, writer).
**Written:** 2026-09-13 by ml-researcher.
**Inputs:** `docs/icra-iros-2027-trends.md` (trend/gap map, 63 verified sources; gaps cited as Gx.y), `docs/design.md` + `docs/solution.md` (challenge architecture and real asset numbers), `raw/` (v3.9.2 evaluator sources, r1pro.yaml).

## Venue framing (binding for all 3 ideas)

ICRA 2027's paper deadline (Sep 15, 2026, 11:59pm PST) is 2 days from today and unreachable for new research. Per the trend map §0:

- **Primary target: IROS 2027** (Florence, Sep 26–Oct 1, 2027; paper deadline **Mar 1, 2027 PST**). Each plan below is scoped so the core experiments finish by mid-Feb 2027.
- **Secondary target: ICRA 2028** (~Sep 15, 2027, per annual pattern). This lands *after* the 2026 BEHAVIOR Challenge winners (Nov 4, 2026) — challenge-adjacent follow-ups (full-matrix extensions, real-robot completions) fit it.
- Reviewer bar (trend map §0, applied to every idea below): quantified gap claim in the intro; evidence ladder matched to claim altitude; n per condition + verbatim success criterion + failure taxonomy; hardware/sensor-matched tuned baselines including a classical option; numeric sim-real deltas where applicable; contribution must survive a "no novel contribution" desk screen (a claim + mechanism + measurement, not a system assembly of known parts).

## Asset fit (what all 3 ideas ride on; no new hardware programs)

20k BEHAVIOR teleop demos (3.27 TB, LeRobot v3.0, RGB head 720² / wrists 480² + depth + 61-dim proprio; 200 demos per task; MIT-licensed); OmniGibson/BEHAVIOR-1K v3.9.2 stack; π0.5 (OpenPI `behavior` fork, skill-conditioned finetune path proven) + GR00T N1.7 (backup); single R1Pro-class mobile dual-arm robot (23-dim action: base vel, trunk, 2×7 arm, 2 grippers); 100 house-scale task families (~6 min episodes, 27 skills/trajectory); official Q-score partial-credit metric; 40 test instances per task (ids 301–340); 31-skill taxonomy mined from demo language annotations; per-task BDDL goal predicates; onboard-only BDDL predicate detectors (privileged teacher labels at training time); progress watchdog (RETRY/REPLAN/SEARCH/FINISH); evalharness reproducing exact official scoring (`aggregate.py`).

## The three ideas at a glance

| # | One-liner | Type | Core gaps | Primary venue |
|---|---|---|---|---|
| 1 | **Anatomy of Failure** — first systematic long-horizon generalization & failure decomposition of VLA policies on BEHAVIOR-1K Q-score | Benchmark + diagnostic study (no new policy training) | G1.5, G5.1, G6.1 | IROS 2027 |
| 2 | **Predict-Before-You-Fail** — predictive sub-skill failure detection + graceful degradation layer on top of any house-scale VLA | Algorithm (learned early-warning + intervention ladder), sim-first | G6.3, G6.4 | IROS 2027 |
| 3 | **How Far Is Simulation From Home** — controlled sim-real delta quantification for house-scale manipulation + residual adaptation with ≤N min on-robot data | Measurement protocol + method (real robot central) | G2.3, G2.4 | IROS 2027 (protocol+delta); ICRA 2028 (full adaptation, if hardware slips) |

**Distinctness (acceptance criterion).** Idea 1 contributes a *diagnosis*: a protocol, dataset, and fixability map — it trains no new policy. Idea 2 contributes a *runtime mechanism*: a learned early-warning model and degradation policy that sits above an existing VLA (works on both π0.5 and GR00T N1.7). Idea 3 contributes a *cross-domain measurement + transfer method*: the first numeric sim-real deltas for house-scale tasks and a per-skill residual adapter. They target different gap clusters, different evaluation settings (sim matrix / sim failure sweeps / sim+real), and different deliverables (annotated dataset + harness / algorithm + code / protocol + deltas table + adapter). They are also synergistic by construction: Idea 1's rollout matrix and failure taxonomy generate the positive labels for Idea 2; Idea 3's matched-instance protocol reuses Idea 1's stratified task/instance selection. Any one of the three is a complete paper standing alone.

---

# Idea 1 — Anatomy of Failure: a long-horizon generalization & failure benchmark for house-scale VLA policies

## 1.1 Problem statement

Every current VLA manipulation benchmark measures *single-task, tabletop, short-horizon* competence (LIBERO 130 tasks, SimplerEnv, RoboCasa). House-scale tasks — ~6-minute episodes, 27 skills per trajectory, mobile base, bimanual, 100+ object states, cross-room navigation without global pose observability — are the operating regime the field is converging on (π0.5's open-world positioning, BEHAVIOR Challenge), but **nobody has published a systematic study of what actually limits policies in that regime**: whether failures are low-level execution deficits or robustness/composition deficits, and how performance degrades as instance diversity (objects, layout, task family) grows. BEHAVIOR-1K's Q-score (partial credit over BDDL goal predicates) is the only existing metric that supports this, yet no generalization/failure decomposition on it exists (gaps G1.5 + G5.1 + G6.1).

## 1.2 Research question

**RQ:** Across 100 house-scale task families and stratified instance diversity, how does VLA policy performance decompose into (a) per-skill execution failures and (b) long-horizon robustness failures (stuck loops, goal forgetting, cross-room search failure), and which failure classes are addressable by low-level data vs. by planning/robustness layers, *without retraining the policy*?

Sub-questions: How does Q decay as remaining horizon lengthens (horizon decay)? Which instance-perturbation axes (object identity, scene layout, task family) cause the largest Q drops? Is the ranking of policies (π0.5-shared vs π0.5-per-task vs GR00T N1.7 vs skill-scripted classical) stable across conditions?

## 1.3 Hypotheses (quantified, falsifiable)

- **H1 (robustness dominates):** for every VLA variant, ≥ 60% of Q-loss relative to a perfect-skill-execution reference is attributable to robustness failure classes (stuck loops, search failure, goal forgetting, bad subgoal transitions) rather than to per-skill execution failure.
- **H2 (generalization gap is large and condition-specific):** Q drops ≥ 15 points from in-distribution instances to scene-perturbed instances (novel layout / swapped objects within BDDL constraints), with the drop concentrated in navigation and multi-skill tasks; pure single-skill tasks drop < 10 points.
- **H3 (no-retrain recovery):** adding a watchdog layer (our challenge Progress state machine: RETRY/REPLAN/SEARCH/FINISH, zero VLA retraining) recovers ≥ 20% of the robustness-attributable gap, and this recovery is measurable in Q *and* worst-case task Q.

## 1.4 Novelty relative to prior work

- **vs LIBERO / SimplerEnv / RoboCasa [20][21]:** those measure in-distribution single-task tabletop success; none exposes horizon, scene-perturbation, or robustness-failure axes at house scale. Contribution is a *different evaluation protocol*, not more tabletop rows.
- **vs BEHAVIOR-1K [31]:** BEHAVIOR-1K defines the benchmark and Q-score but the published work uses it for task coverage, not for a systematic policy failure decomposition; no paper has published the generalization sweep + failure taxonomy + per-skill breakdown on Q.
- **vs π0.5 / GR00T [10][39]:** those report open-world *claims* with few-task evidence; we provide the missing house-scale evidence ladder with n reported everywhere.
- **vs VLA surveys [17][18]:** they catalog ~200+ papers; none resolves "what limits long-horizon completion" empirically.
- Desk-screen defense: the contribution is a **claim (H1–H3) + mechanism (decomposition protocol, intervention ablation) + measurement (annotated 5k+ rollout dataset with per-skill failure labels)**, not a system assembly. The released dataset + taxonomy is the durable artifact.

## 1.5 Proposed method

**Protocol design.**
1. *Policy panel* (5 variants, all served through the identical wrapper, same 24 GB budget, identical evalharness — hardware-matched by construction):
   - P1: per-task π0.5 finetune (organizer baseline recipe; run on a stratified 20-task subset for cost control, reported as such);
   - P2: shared skill-conditioned π0.5 (our challenge M2 policy — one finetune on all 20k demos, 31-skill taxonomy conditioning);
   - P3: GR00T N1.7 (provided backup baseline, temporal-ensemble serving);
   - P4: classical/skill-scripted reference — mined per-task skill plans executed with scripted primitives + our predicate detectors (System-2-only lower bound, "no learned policy");
   - P5: P2 + watchdog (the robustness layer ablated on top — this is the intervention arm for H3).
2. *Instance conditions* (3, per task, from the 40 official test instances 301–340):
   - C1 **in-distribution**: instances whose object/layout composition was seen during demo mining (train-time privileged sim access identifies the demo composition);
   - C2 **scene-perturbed**: instances resampled within BDDL constraints to novel room layouts / object swaps (OmniGibson instance generator); n = 20–40 per task;
   - C3 **cross-task**: each policy additionally run on 10 task families held out of its finetuning set (only meaningful for P1; reported per panel member where applicable).
3. *Rollout matrix*: full 100 tasks × 10 instances × 5 policies for C1/C2 (≈ 10k episodes; the "headline table"), plus a 20-task deep subset at 40 instances for C2/C3 and failure-taxonomy mining (≈ 4k episodes). Total ≈ 14k episodes.
4. *Failure auto-labeling*: during rollouts (privileged sim state is legal at training/analysis time), label every episode window with BDDL-grounded failure classes: `stuck_loop` (no predicate change > 30 s within a skill budget), `search_fail` (occupancy frontier exhausted / target room never reached), `goal_forget` (predicate regression after satisfaction), `wrong_object`, `skill_exec_fail` (per-skill detector negative at subgoal end), `timeout`, `transition_error` (bad subgoal advance). Taxonomy validated by human audit on a random n ≥ 300 episodes (2 annotators, Cohen's κ reported).
5. *Analysis*: per-skill success rates (27 skills × 100 tasks), horizon-decay curves (Q vs skills remaining), generalization-gap table (ΔQ per condition × per task family), failure-class frequency table per policy, fixability map: each failure class → which intervention (watchdog arm / per-task adapter / more low-level data) measurably recovers it (from P5 vs P2 and P1-vs-P2 deltas).

**Deliverables.** (i) `hse-bench` eval harness (stratified instances + auto-labeler + aggregate) released; (ii) ≈ 14k annotated rollout episodes with per-skill failure labels on HF; (iii) failure taxonomy + fixability map; (iv) headline decomposition table (5 policies × 3 conditions × 100 tasks).

## 1.6 Required data / simulation environment

OmniGibson v3.9.2 + BEHAVIOR-1K v3.9.2 (already bootstrapped via `scripts/bootstrap.sh`); 20k demos for P1/P2 finetunes (3.27 TB, chunked download — challenge infrastructure); no new *demonstration* data collection (rollouts only); 4–8 × 24 GB GPUs for the 50-port fan-out evalharness (`parallel.py`); ~15 min of human audit time per 100 episodes for taxonomy validation. CPU box suffices for all analysis (repo already validates the full scoring path on CPU).

## 1.7 Baselines

The policy panel *is* the baseline set: P1 (per-task π0.5 — the organizer-provided baseline strategy, hardware-matched via identical serving wrapper), P3 (GR00T N1.7 — organizer-provided), P4 (classical skill-scripted — the required "strong classical option"), P5 (P2+watchdog — intervention ablation). Context rows: LIBERO-90 and SimplerEnv success rates for the same policy families (tabletop reference scale, clearly labeled as different regime).

## 1.8 Evaluation metrics

Q (official score_utils math, partial credit — the primary metric, reported with n per cell); task_sr (binary success); time_score (efficiency tie-breaker); per-skill success rate with 95% binomial CIs; failure-class frequency table; generalization gap ΔQ = Q(C1) − Q(C2) per task family; horizon-decay slope (ΔQ per additional skill remaining); worst-case Q (min over tasks) — the reviewer-demanded complement to mean Q; κ for taxonomy audit.

## 1.9 Expected contribution

1. The first empirical answer to "what limits VLAs on 10-minute household tasks, and which limit is fixable by better low-level data vs. better planning/robustness" — directly actionable for policy design.
2. A reusable house-scale evaluation protocol (stratified instances + BDDL-grounded auto failure labels) that future policies can benchmark against, with n and success criteria stated per cell.
3. A released, audited failure taxonomy + fixability map (≈ 14k labeled episodes) — the durable dataset artifact.
4. A quantified, honest generalization-gap table for the two strongest open house-scale policies (π0.5-class, GR00T-class), which the field currently lacks.

## 1.10 12-month execution plan (2026-09 → 2027-08)

| Window | Milestone |
|---|---|
| Sep 13–Oct 16 | (Parallel with challenge) lock stratified-instance generator + auto-labeler on 10 pilot tasks; P2/P3 checkpoints ready from challenge M2/M3; pilot matrix (10 tasks × 10 instances × 5 policies, ≈ 500 episodes) → protocol sanity check (do conditions separate?). |
| Oct 16 | Challenge submitted; P4 scripted-reference completed; labeler frozen. |
| Oct 20–Nov 30 | Full C1/C2 matrix waves 1–2 (50 of 100 tasks × 2 conditions × 5 policies ≈ 5k episodes) on 8×24 GB; C2 perturbation generator regression-tested. |
| Dec 1–Jan 15 | Matrix waves 3–4 (remaining 50 tasks) + 20-task deep subset C2/C3 (≈ 4k episodes); human taxonomy audit (2 annotators, n ≥ 300); failure-class stability checks. |
| Jan 15–Feb 10 | Analysis: per-skill/horizon/generalization tables; fixability map; H1–H3 verdicts; negative-result triage (if H2 fails on C2, re-scope perturbation axes before writing). |
| Feb 10–Mar 1 | Draft → internal review → **IROS 2027 submission (Mar 1, 2027 PST)**. |
| Mar–May | Rebuttals/camera-ready; release `hse-bench` + HF dataset (v1); challenge winners (from Nov 4, 2026, public context) added as external-reference rows if disclosed submissions are analyzable. |
| Jun–Aug 2027 | ICRA 2028 (if submitted): v2 with full 40-instance depth, GR00T N1.7 refresh, and cross-venue comparisons. |

**Go/no-go gates.** G1 (Oct 20): pilot conditions must separate (ΔQ(C1−C2) ≥ 5 points for ≥ 3 of 5 policies) — else redesign perturbations. G2 (Jan 15): taxonomy κ ≥ 0.7 — else shrink taxonomy to 5 classes. G3 (Feb 10): H1/H3 must hold directionally — else reposition as pure benchmark (still publishable at IROS, lower risk).

## 1.11 Risk assessment

- **Technical uncertainty — Medium.** Biggest risk is *no signal*: policies could perform too uniformly across conditions (VLAs collapsing to 0 on C2) making decomposition uninformative. Mitigation: pilot gate G1 on 500 episodes before committing to 14k; C1 baseline always gives the decomposition a reference point even if C2 collapses.
- **Data availability — Low.** All data (demos, instances, BDDL) already exists in the challenge stack; no collection. Taxonomy audit is the only human data cost (~60 person-hours).
- **Compute — Medium-High.** ≈ 14k × ~9 min sim episodes ≈ 2,100 GPU-hours ≈ 2–3 weeks on 8×24 GB (spread over Oct–Jan, off-peak vs challenge work). P1 per-task finetunes on 20 tasks ≈ 20 × one-pi0.5-LoRA-cost (bounded by design; full-100 variant deferred to ICRA 2028).
- **Venue fit — High.** Benchmark/decomposition studies are IROS-mainstream (LIBERO, R2R line); the desk-screen defense is the *mechanism* (intervention ablation P5 vs P2) plus the released dataset, not just tables.

---

# Idea 2 — Predict-Before-You-Fail: predictive failure detection & graceful degradation for house-scale VLA policies

## 2.1 Problem statement

Learned house-scale policies fail *late and catastrophically*: by the time a stuck loop, slip, or wrong-object grasp is visible in BDDL predicate trajectories, the sub-skill budget is gone and the 6-minute episode's partial credit has already eroded. Existing failure-aware work is (a) post-hoc (ARMADA, IROS 2025 workshop best paper — detects after the fact; ICRA 2026 HRI finalists use diffusion-policy uncertainty for when-to-ask-human on *single* tasks), or (b) low-level only (CBF safety filters certify the action, not the VLA policy — gap G6.1), and nothing predicts an *about-to-fail sub-skill* k steps before it manifests, at house scale, with measured task-level recovery. Graceful degradation for long-horizon tasks (intervene on 5–10% of steps, keep the rest autonomous, with measured Q) is missing entirely (G6.4; ICRA 2025 HRI winner was the *data-collection* version of shared control, not deployment).

## 2.2 Research question

**RQ:** Can a lightweight early-warning model, using onboard observations only, predict imminent sub-skill failure in a house-scale VLA episode with ≥ 5 s lead time, and does a degradation ladder triggered by that prediction (RETRY → REPLAN → SEARCH → handoff/abort-and-restart) measurably increase Q — in particular worst-case Q — compared to no intervention and to post-hoc rule-based intervention?

## 2.3 Hypotheses (quantified, falsifiable)

- **H1 (predictability):** an early-warning model (EWR) over onboard-only features predicts BDDL-grounded imminent failure (next expected predicate flip does not occur within 1.5× p90 sub-skill demo budget) with AUC ≥ 0.85 / AP ≥ 0.70 at ≥ 5 s lead, on held-out tasks.
- **H2 (value of lead time):** EWR-triggered intervention raises Q by ≥ 8 points over no-intervention and by ≥ 3 points over post-hoc rule-triggered intervention (identical ladder, different trigger — isolating the prediction lead), with false-alarm rate < 15% (intervention on sub-goals that would have succeeded).
- **H3 (worst-case benefit):** min-over-task Q improves ≥ 3× more than mean-over-task Q (degradation protects the tail, which is where partial credit bleeds).
- **H4 (policy-agnostic):** EWR transfers to a second VLA (GR00T N1.7) with < 3 points of H2's gain lost (early-warning features are not VLA-specific).

## 2.4 Novelty relative to prior work

- **vs ARMADA (IROS 2025):** post-hoc failure detection + shared control on single-task manipulation; EWR is *predictive* (lead time is the measured quantity) and evaluated at house scale on Q partial credit, where intervention value is defined by preserved predicates, not by task restarts only.
- **vs ICRA 2026 HRI finalists (diffusion-policy uncertainty → when-to-ask):** uncertainty of the action distribution is a *symptom*; EWR conditions on task structure (remaining predicates, skill state, progress velocity) — predicting *which* sub-skill is about to fail and *why* (failure-class-conditional early warning), with the ablation "uncertainty-features-only" as a baseline.
- **vs CBF/safety-filter line [42][43][45]:** those certify/modify the *action*; EWR sits above the policy and decides *when to change behavior regime* (retry/replan/search/handoff) — complementary, and we show the CBF-filter variant as a low-level arm in the ladder for the safety claim.
- **vs ReWiND/VLA-RL [22][62]:** those retrain with rewards; EWR is a no-retrain runtime layer (cheaper, policy-agnostic) — positioned as the alternative when you cannot retrain at deployment.
- Desk-screen defense: H2's trigger-matched design (same ladder, learned vs rule trigger) is the *mechanism*; lead-time and false-alarm are the *measurements*. No one can reproduce the claim without our EWR.

## 2.5 Proposed method

**Signal design (onboard-only, per step).** Feature vector: (i) predicate-detector confidence trajectories (6 families, 3-frame hysteresis outputs); (ii) progress velocity Δpred/s and sub-skill elapsed fraction vs mined p90 budget; (iii) VLA action-chunk disagreement: temporal-ensemble variance + receding-horizon chunk-blend alpha residual (already computed by `controller.py` — free); (iv) gripper load/proprio deviation from per-skill demo statistics; (v) odometry drift error (velocity-integration vs depth-VO residual); (vi) skill identity + remaining-task structure (plan position, predicates remaining). ~40-dim, < 10 ms CPU inference budget (must not break the 13.5 sim-FPS cadence).

**Labels.** BDDL-grounded failure events auto-labeled from privileged sim state during rollouts (reuse Idea 1's auto-labeler; a "failure" = the event window in which the next expected predicate flip fails to materialize within its budget, with failure class attached). Positive density: ≈ 10–25 failure events per failed episode; successful episodes provide hard negatives (windows where the sub-skill *looks* risky but succeeds — critical for the false-alarm metric).

**Model.** (a) primary: skill-conditioned gradient-boosted trees + a small temporal MLP over the last 3 s feature window (interpretable, CPU-trainable, ablatable per feature); (b) comparison: transformer over last-8-frame wrist/head crops + features (the "vision is necessary" probe); (c) baselines: uncertainty-only (action-chunk variance), rule-only (our challenge Progress watchdog — the post-hoc trigger arm), oracle (privileged perfect detection — upper bound for H2).

**Degradation ladder.** EWR output (failure-class-conditional probability + lead estimate) gates a 4-level ladder: L0 continue; L1 RETRY (same sub-skill, fresh plan); L2 REPLAN (re-derive remaining queue from BDDL goal minus satisfied predicates); L3 SEARCH (occupancy-frontier, RoomPrior-biased); L4 HANDOFF (sim: abort-and-restart sub-skill with fresh instance state; real: request human). Level selection is failure-class-conditional (search failures → L3, not L1), and the ladder respects budget caps (a sub-skill can never exceed 1.5× p90 — inherited from the challenge watchdog, so H2's comparison is trigger-matched by construction).

**Experiments.** 5 policies × 3 conditions × 100 tasks × 10 instances (rollout substrate = Idea 1's C1/C2 matrix, so Idea 2 rides Idea 1's compute; if Idea 1 is not executed, a reduced 40-task substrate suffices for a standalone paper). Intervention arms: no-intervention, rule-triggered ladder, EWR-triggered ladder, EWR + CBF low-level filter, oracle (upper bound). Feature ablations: leave-one-channel-out (predicate conf / progress vel / chunk disagreement / proprio / odometry). Policy-agnostic check: EWR trained on π0.5 episodes applied to GR00T N1.7 episodes (H4).

## 2.6 Required data / simulation environment

OmniGibson v3.9.2 (no real hardware required for the paper; a 10-task real-robot pilot is optional stretch → ICRA 2028). Failure substrate: ≈ 5k episodes from the rollout matrix (Idea 1 waves) + ≈ 2k *degradation-sweep* episodes where we intentionally induce stress (novel-instance C2, perturbed object properties where BDDL allows, induced stuck starts) to raise positive density — ≈ 7k episodes total, ~1,050 GPU-hours. EWR training: CPU / 1 × 24 GB. 20k demos for per-skill statistics. No human annotation beyond a 100-episode audit of auto-labels.

## 2.7 Baselines

No-intervention (raw Q — reference); **rule-triggered ladder** (our challenge Progress watchdog: identical ladder, post-hoc trigger — the key ablation isolating prediction value); uncertainty-only trigger (action-chunk variance threshold); vision-only EWR (transformer, no task-structure features — probes necessity of the privileged-informed feature design); oracle trigger (privileged — upper bound, labeled as such); CBF low-level filter arm (safety-complement); per-VLA transfer arm (π0.5-trained EWR on GR00T). All arms served through the identical wrapper at identical budget caps.

## 2.8 Evaluation metrics

Lead time (s) with CIs; AUC / AP of imminent-failure prediction (per failure class); ΔQ vs no-intervention (primary, official scoring, n per arm); ΔQ vs rule-trigger (the H2 isolation); false-alarm rate (interventions on sub-goals that would have succeeded); recovery rate (fraction of would-have-failed sub-skills that reach their exit predicate after intervention); worst-case Q (min over tasks) and its improvement ratio (H3); intervention budget (fraction of steps intervened — the "5–10%" claim, reported honestly); efficiency penalty (time_score delta — interventions cost time; net Q+efficiency reported); per-class breakdown (search failures vs stuck vs slip).

## 2.9 Expected contribution

1. First *predictive* (vs post-hoc) failure model for VLA-driven manipulation, with lead time as a reported, ablated quantity — directly answering G6.3.
2. A measured degradation recipe for long-horizon tasks: which ladder level, for which failure class, at what trigger threshold, with net Q effect — answering the deployment side of G6.4 (the ICRA 2025 HRI winner's data-collection result has no deployment counterpart).
3. Proof of policy-agnosticism (H4): one EWR above two different VLAs — making the layer a reusable runtime component for any future house-scale VLA.
4. An open early-warning dataset (feature stream + BDDL-grounded failure labels from ≈ 7k episodes) + `ewr` code release.

## 2.10 12-month execution plan (2026-09 → 2027-08)

| Window | Milestone |
|---|---|
| Sep 13–Oct 31 | EWR feature pipeline implemented on the challenge serving path (CPU, < 10 ms/step verified against 13.5 FPS budget); BDDL-grounded auto-labeler shared with Idea 1; 1k-episode pilot (10 tasks) → H1 sanity (AUC ≥ 0.70 at 3 s lead on pilot). |
| Nov 1–Dec 15 | Degradation sweeps (2k stress episodes) + substrate matrix completion (5k, shared with Idea 1 waves); EWR primary + all ablation arms trained. |
| Dec 16–Jan 20 | Intervention experiments: 5 arms × 100 tasks × 10 instances; H1–H4 verdicts; worst-case analysis; efficiency-penalty accounting. |
| Jan 21–Feb 10 | Draft; optional 10-task real-robot pilot if hardware is free (stretch only — paper scope is sim, evidence ladder stated honestly); camera-ready for IROS. |
| **Feb 10–Mar 1** | **IROS 2027 submission.** |
| Mar–May | Rebuttals; release `ewr` + early-warning dataset v1. |
| Jun–Aug 2027 | ICRA 2028 v2: real-robot validation (30–50 tasks), GR00T refresh, cross-paper integration (Idea 1 taxonomy + Idea 2 EWR as a combined "diagnose-then-degrade" story). |

**Go/no-go gates.** G1 (Oct 31): pilot AUC ≥ 0.70 @ 3 s lead — else add channels / re-examine label window before scaling compute. G2 (Jan 20): H2 isolation (EWR > rule trigger) must hold on ≥ 60 tasks jointly — else reposition as "detection + ladder" paper (weaker, still viable). G3 (Feb 10): false-alarm < 25% — else threshold-tuning story only.

## 2.11 Risk assessment

- **Technical uncertainty — Medium-High.** "Imminent failure" is label-design-dependent; if the BDDL-grounded window is wrong, AUC claims shift. Mitigation: window defined *from demo statistics* (1.5× p90 budget), not ad-hoc; 100-episode human audit; sensitivity table over window sizes in the appendix. Second risk: the challenge watchdog (our own post-hoc system) may already be so strong that EWR's marginal gain (H2) is small — this is the paper's core tension, and G2 exists to catch it; a null result with a full trigger-matched ablation is still a publishable negative/boundary result at IROS.
- **Data availability — Low.** Sim-only; all substrate comes from the challenge rollout infrastructure. Failure-positive density is the only data risk, addressed by intentional degradation sweeps.
- **Compute — Low-Medium.** ≈ 1,050 GPU-hours (substrate shared with Idea 1 if that runs; standalone it is the full matrix ≈ 1,500 GPU-hours). EWR itself trains on CPU. Cheapest of the three ideas.
- **Venue fit — High.** IROS 2026 has a dedicated safety award category; predictive failure (G6.3) is a named open gap; trigger-matched ablations match the reviewer bar exactly. Sim-only scope is acceptable for an *algorithm + benchmark* paper as long as the evidence ladder is stated (no real-world claims in the abstract).

---

# Idea 3 — How Far Is Simulation From Home? Controlled sim-real delta quantification + residual adaptation for house-scale manipulation

## 3.1 Problem statement

Sim-to-real transfer is largely settled for *single-task, single-arm, tabletop* manipulation and for *locomotion* (domain randomization → RMA residual adaptation [24] → Isaac-Gym real-world RL). It has never been measured for **house-scale composition**: mobile + bimanual + long-horizon + 100+ object states, where sim2real methods tuned per-task do not obviously compose (gap G2.4; the BEHAVIOR Challenge exists precisely because baselines are far from solving it *in sim*, and no real-robot counterpart of BEHAVIOR-1K exists). Compounding this: there is no cheap, standard way to measure "how wrong is my simulator" for a given robot/task — papers report end-task deltas only, which are confounded by policy quality (gap G2.3). Reviewers' top rejection reason ("sim results carry real-world claims") is unanswerable for house-scale work because *nobody has reported the numbers*.

## 3.2 Research question

**RQ:** (a) How much of a sim-trained house-scale policy's Q survives in the real world — measured per skill family (navigation vs single-arm vs contact-rich vs bimanual/articulated) under a controlled matched-instance protocol — and which failure modes appear *only* in the real world (the sim-fidelity gap signal)? (b) Can a per-skill residual action adapter, trained on ≤ N minutes of on-robot data, close a measurable fraction of that delta without retraining the VLA?

## 3.3 Hypotheses (quantified, falsifiable)

- **H1 (delta structure):** sim-real Q-delta is skill-family-specific and ordered: ΔQ(navigation) ≤ 5 < ΔQ(single-arm non-contact) ≤ 15 < ΔQ(contact-rich) ≤ 25 < ΔQ(bimanual/articulated) ≤ 30 points (Q in [0,100] convention for deltas; official Q ∈ [0,1] reported alongside).
- **H2 (real-only failures):** ≥ 30% of real-world failure episodes are attributable to failure classes that occur at < 5% frequency in sim on the same policy (slip-on-grasp, contact-force deficits, drift accumulation) — i.e., a real-only failure mode is the dominant delta driver, not uniformly worse execution.
- **H3 (residual closes the gap cheaply):** a per-skill residual adapter r(o) → Δa (32-step, RMA-style [24] but for manipulation action chunks) trained on ≤ 10 min of on-robot data per skill family reduces the contact-rich and bimanual ΔQ by ≥ 40%, with a monotone data-budget curve (1/5/10/20/30 min) saturating ≤ 30 min.
- **H4 (fidelity probe predicts delta):** a cheap sim-only probe — distributional divergence between sim and real on identical observation prefixes (KL on policy action-chunk confidence + proprio deviation statistics) — predicts per-task ΔQ with R² ≥ 0.5, letting practitioners estimate expected real performance *before* deploying.

## 3.4 Novelty relative to prior work

- **vs single-task manipulation sim2real [32][9][13]:** those report end-task deltas on one task/arm; we report a *per-skill-family delta table over 20 matched house-scale tasks* — the composition result (G2.4) no prior sim2real paper addresses.
- **vs RMA / residual adaptation [24] and soft-robot residual physics (RAL 2025 best paper [63]):** RMA is locomotion, per-episode, proprio-only; our residual is for VLA action chunks at house scale with visual+task-structure conditioning, and — critically — we *measure the gap it closes* under a controlled protocol rather than claiming transfer.
- **vs world-model data-engine papers (Cosmos/Genie/FLIP [25][26][27]):** those never report a real-robot delta; H4's fidelity probe is the missing cheap measurement they implicitly need (positioned as a companion metric, not a competing method).
- **vs RoboCasa/BEHAVIOR-1K [20][31]:** sim-only; the real-robot counterpart does not exist — this paper *is* the protocol that creates it (20-task scale, deliberately small so it becomes reusable).
- Desk-screen defense: contribution = (i) the matched-instance protocol (method), (ii) the first numeric house-scale delta table (measurement, H1/H2), (iii) the residual + budget curve (method, H3), (iv) the fidelity probe (measurement, H4). Numeric sim-real deltas are *explicitly* on the reviewer-want list.

## 3.5 Proposed method

**Controlled matched-instance protocol.**
1. *Task selection:* 20 tasks stratified from the 100 by skill-family composition: 5 navigation-heavy (e.g., `loading_the_car`, `carrying_in_groceries`), 5 single-arm non-contact (e.g., `setting_mousetraps`), 5 contact-rich (e.g., `turning_on_radio`, `can_meat`), 5 bimanual/articulated (e.g., `rearranging_kitchen_furniture`, `outfit_a_basic_toolbox`). Selection documented with the mined skill-plan statistics (`docs/plans/`).
2. *Instance matching:* for each task, 3 real-world layout variants (L1/L2/L3) in the group's real apartment/lab testbed, each with BDDL predicate states manually grounded (annotator marks predicate satisfaction on rollout video at 0.5× speed; ~1–2 min/rollout; detector-assisted pre-labeling cut by 50%); sim side: 3 matched OmniGibson instances per layout (same task, object set, room topology — the C2 perturbation machinery from Idea 1, constrained to match the real layout). Matching table (real layout ↔ sim instance) released.
3. *Rollout arms (identical policy, P2 shared skill-conditioned π0.5 from the challenge):* 10 sim instances (3 layouts × 3–4 instances + 2 held-out) and 10 real rollouts (3 layouts × 3–4 + 2 held-out), n = 10 per layout-condition, 1 rollout each (challenge convention; no cherry-picking; videos archived). Total ≈ 20 tasks × 20 = 400 rollouts ≈ 40 real-robot hours + 400 sim episodes (~60 GPU-hours).
4. *Delta analysis:* Q_sim vs Q_real per task and per family (H1 table); failure-class frequency per condition using Idea 1's auto-labeler (sim) + manually-audited real labels (H2: real-only class identification, κ reported); per-episode alignment (first K steps where sim/real diverge) to localize *where* the gap opens.
5. *Residual adaptation:* per-skill-family residual head r(o_t, skill, remaining-predicates) → Δa_{t:t+32}, RMA-style [24]: policy action a' = clamp(a + r(o)). On-robot data per family: 5 min autonomous rollouts of the sim policy (collecting mismatch) + 5 min teleop correction episodes (collecting fix) ≈ 10 min/family, 40 min total robot time, repeated at 1/5/10/20/30 min budgets for the H3 curve. Trained per family (small MLP/GRU, 1 × 24 GB or CPU). Ablations: whole-policy residual vs per-skill-family; sim+real mix vs real-only; data-budget curve; no-residual zero-shot (the delta reference).
6. *Fidelity probe (H4):* on each task, compute from sim-only runs the policy's action-chunk confidence distribution + proprio-deviation statistics; train a regressor to real ΔQ (target from step 4); report R² + calibration curve; the probe is a *practitioner artifact*: "estimate your expected house-scale delta before you mount the robot."

**Deliverables.** (i) 20-task matched-instance protocol + released matching table; (ii) first numeric house-scale sim-real delta table (20 tasks × 2 arms × failure classes); (iii) per-family residual adapters + data-budget curves; (iv) fidelity probe model + calibration; (v) `simreal-house` code release.

## 3.6 Required data / simulation environment

OmniGibson v3.9.2 + the challenge P2 policy; **real robot: the R1Pro-class mobile dual-arm platform** (challenge asset) for ≈ 10–15 robot-days (400 delta rollouts ≈ 40 h + 40 min residual data + retries ≈ 2× buffer); real testbed: one apartment/lab kitchen with 3 re-configurable layouts per task family (furniture rearrangement is the matching knob — no new construction); manual predicate grounding pipeline (2 annotators, ~20 person-hours); 1 × 24 GB GPU (residual training is small); safety: dual-arm mobile operation with spotter + e-stop, no elevated-robot work, contact tasks limited to the robot's working envelope.

## 3.7 Baselines

Zero-shot sim-trained policy on real (the delta reference — not a "baseline" to beat but the measured quantity); real-only per-task finetune of π0.5 (cost reference: what full adaptation would need — run on 3 tasks to bound the curve's upper end, reported as expensive reference); RMA-proper style proprio-only residual (ablation: is visual/task conditioning necessary?); per-family residual (ours); no-residual with challenge watchdog (robustness-only reference); skill-scripted classical execution on real where feasible (5 tasks; the "how far is *any* sim training from real" floor). All arms use the identical serving wrapper and identical predicate-grounding.

## 3.8 Evaluation metrics

Q_sim, Q_real, ΔQ per task/family with 95% CIs (n = 10 per arm per layout-condition, stated verbatim); real-only failure-class frequency (H2); divergence-localization step distribution; residual: ΔQ_real after adaptation, gap-closed % (H3), data-budget curve (Q vs robot-minutes, saturation point), ablation deltas (proprio-only vs full conditioning); fidelity probe: R², calibration (predicted vs observed ΔQ scatter), leave-one-task-out (H4); time-to-deploy (how much sim work the probe saves: expected robot-hours of delta prediction accuracy); per-rollout artifact: video + predicate annotation + Q breakdown (released).

## 3.9 Expected contribution

1. The first *numeric* sim-real delta table for house-scale manipulation, per skill family — the missing evidence rung for every future house-scale sim paper (reviewers currently reject on the exact absence of these numbers).
2. A controlled matched-instance protocol that turns "sim vs real" from an anecdote into a repeatable measurement (matching table + predicate grounding released).
3. A per-skill residual adaptation recipe with an honest data-budget curve (≤ N min on-robot data → X% gap closed) — the composition answer to G2.4 that RMA-style locomotion results do not provide.
4. A cheap sim-only fidelity probe (H4) that predicts per-task delta before deployment — a practitioner tool the world-model literature implicitly needs (G2.3).

## 3.10 12-month execution plan (2026-09 → 2027-08)

| Window | Milestone |
|---|---|
| Sep 13–Oct 16 | (Parallel with challenge) 20-task selection + matching table design; predicate-grounding SOP written; **pilot: 3 tasks × 2 layouts × 6 rollouts on the real robot** (≈ 6 robot-hours) — validates the protocol end-to-end and gives first real ΔQ numbers (even 3-task deltas are a teaser for the community; internal only). |
| Oct 17–Nov 30 | Layout reconfiguration of testbed (3 variants per family); full real delta rollouts wave 1 (10 of 20 tasks ≈ 200 rollouts ≈ 20 robot-days); sim matched instances generated + 200 sim rollouts (≈ 30 GPU-hours); real-only failure audit ongoing. |
| Dec 1–Jan 15 | Delta rollouts wave 2 (remaining 10 tasks); H1/H2 verdicts; residual training starts on completed families (10 min/family budget); data-budget curve at 1/5 min points. |
| Jan 16–Feb 10 | Residual arms complete (10/20/30 min points, 4 families); real re-rollouts of adapted policy (≈ 150 rollouts ≈ 15 robot-days); H3 verdict; fidelity probe trained + leave-one-task-out; H4 verdict. |
| **Feb 10–Mar 1** | Draft → **IROS 2027 submission** (scope: protocol + delta table + contact/bimanual residual results + probe; navigation/single-arm residual deferred to v2 if hardware time ran short). |
| Mar–May | Camera-ready; release protocol + matching table + `simreal-house`; challenge winners (Nov 4, 2026) cited as sim-side context. |
| Jun–Aug 2027 | ICRA 2028 v2 (full 20-task × 4-family residual completion, GR00T N1.7 real arm, probe calibration on second layout set). |

**Go/no-go gates.** G1 (Oct 16): pilot protocol must yield a *measurable* ΔQ on 3 tasks (ΔQ ≥ 5 points on ≥ 2 of 3) and predicate grounding κ ≥ 0.8 — else fix SOP before wave 1 (this is the make-or-break gate for the whole idea). G2 (Jan 15): H1 ordering must hold directionally on wave-1 families — else reposition as "delta table + probe" (H1/H4 are the paper; residual becomes a methodological appendix). G3 (Feb 10): H3 gap-closed ≥ 25% at 10 min on ≥ 2 families — else claim the protocol + measurement only (still a paper: the delta table is novel regardless).

## 3.11 Risk assessment

- **Technical uncertainty — High (the highest of the three).** Real-world BDDL predicate grounding is the load-bearing weakness: sim predicates have no automatic real counterpart, and manual grounding at n = 10/layout-condition is where annotator error lives. Mitigation: SOP with detector pre-labels + 2-annotator κ gate (G1); divergence-localization analysis cross-checks labels against video; per-rollout artifacts released so errors are auditable. Second risk: real testbed drift between waves (furniture moves, lighting) — mitigate with fixed layout photos per condition and same-day sim/real pairs.
- **Data availability — Medium.** Dependent on the R1Pro-class robot being available Nov–Feb for ≈ 35 robot-days. If hardware slips past Jan 15, the paper degrades to delta-table-on-completed-tasks + probe (G2 reposition) — the sim side and protocol are never hardware-gated. Real-only finetune reference is bounded to 3 tasks precisely to cap this risk.
- **Compute — Low.** ≈ 120 GPU-hours (sim rollouts + residual training). The cost is robot-time, not FLOPs — the idea is the cheapest on compute and the most expensive on hardware/logistics.
- **Venue fit — High.** Numeric sim-real deltas are on the explicit reviewer-want list; the "sim carries real claims" rejection reason is *answered* by this paper's existence. IROS accepts protocol+measurement papers (Robo-DM won ICRA 2025 Robot Learning as a data-infrastructure paper — precedent). Real-robot requirement is met by construction (unlike Ideas 1–2, the abstract can carry real-world claims *because* the real arm is central).

---

## Cross-cutting notes for feasibility review (t_0423f72f)

1. **Shared substrate, staggered launches.** All three reuse the challenge serving wrapper, predicate detectors, auto-labeler, evalharness, and P2 policy. Idea 1's rollout matrix (Oct–Jan) is Idea 2's failure-label substrate; Idea 3's matched-instance protocol reuses Idea 1's C2 perturbation machinery. Recommended sequencing if all three run: Idea 3 pilot *first* (Oct, 3 tasks, gates the highest-risk item — real predicate grounding — before IROS deadline), Idea 1 waves *continuous* (Oct–Jan, compute-bound, no hardware), Idea 2 *last* (rides Idea 1's data, CPU-bound).
2. **Minimal viable experiment per idea** (the smallest credible validation): I1 → 500-episode pilot matrix proving conditions separate (ΔQ(C1−C2) ≥ 5 pts on ≥ 3 policies); I2 → 1k-episode pilot proving AWR ≥ 0.70 @ 3 s lead; I3 → 3-task × 2-layout real pilot proving measurable ΔQ + grounding κ ≥ 0.8.
3. **IROS 2027 deadline (Mar 1, 2027)** is the binding constraint for all three; ICRA 2028 is the fallback for each, and the natural home for the combined "diagnose → predict → transfer" trilogy if the group runs all three.
4. **Compute totals if all three run fully:** ≈ 4,200 GPU-hours (sim) + ≈ 35 robot-days + ≈ 40 person-hours annotation — a 2-person research load over 5 months, not a multi-year program. Each idea alone: I1 ≈ 2,100 GPU-h; I2 ≈ 1,050 GPU-h (shared substrate); I3 ≈ 120 GPU-h + 35 robot-days.
5. **What the three ideas are NOT:** no new VLA backbone, no dexterous-hand or full-humanoid hardware, no world-model training, no RL post-training at scale — by design, to survive desk screen on mechanism rather than compute show, and to stay inside the group's existing asset envelope (per parent handoff constraint #2).
