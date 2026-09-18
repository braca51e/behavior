# Research Scope — Three House-Scale VLA Ideas for IROS 2027 (Final)

**Date:** 2026-09-13 · **For:** the research team deciding what to start immediately
**Synthesizes:** `docs/icra-iros-2027-trends.md` (literature/gap map, 63 verified sources), `docs/icra-iros-2027-ideas.md` (scoped ideas), `docs/feasibility-assessment-2027.md` (grounded feasibility review). Where the ideas doc and the feasibility review disagree (compute estimates), the feasibility review wins — it re-derived the numbers from the v3.9.2 evaluator constants.

## TL;DR — the decision this document settles

- **Venue:** IROS 2027 (paper deadline **2027-03-01 23:59 PST**, Florence — verified via three sources) is the binding target. ICRA 2027 (Sep 15, 2026) is unreachable; ICRA 2028 (~Sep 2027) is the fallback for deferred scope.
- **Start in October:** I1 “Anatomy of Failure” 500-episode pilot **and** I3 “How Far Is Simulation From Home” ~6-robot-hour real pilot — **I3 pilot first**, because it resolves the portfolio’s highest technical risk by Oct 16. Run the I2 “Predict-Before-You-Fail” 1k-episode pilot in parallel at near-zero marginal cost and **commit I2 only if its H2-margin gate passes (mid-Nov decision)**.
- **Do not run all three at full scope simultaneously.** Corrected total: ~7–9k GPU-hours + ~35 robot-days + 40–60 person-hours — a 2-person, 5-month load competing for the same substrate and people as the Oct 16 BEHAVIOR Challenge submission.
- **Two preconditions to confirm by end of September (both currently unverified):** **(A)** 4–8×24 GB GPU cluster access Oct–Jan — without it, I1/I2 as scoped are NO-GO (single 24 GB box ⇒ ~5–6 months of rollout time). **(B)** physical R1Pro + re-configurable real testbed Nov–Feb — without it, I3 degrades to its delta-table + fidelity-probe scope.
- **Shared dependency:** all three ride on the challenge’s M2/M3 (shared skill-conditioned π0.5 “P2”, trained predicate detectors, 100-plan miner). Treat challenge milestones as a hard gate on the October pilots.

## 1. Shared context (what all three assume)

**Assets (verified in the challenge repo).** 20k BEHAVIOR teleop demos (3.27 TB, LeRobot v3.0, RGB 720²/480² + depth, 200 demos/task, MIT-licensed); OmniGibson/BEHAVIOR-1K v3.9.2 — official RGB+depth track steps at **13.52 sim-FPS + 150–300 s scene load** (the source of the corrected compute); 100 house-scale task families (~6-min episodes, 27 skills/trajectory); 40 official test instances per task (ids 301–340); official Q-score partial-credit metric + an evalharness that reproduces exact official scoring; π0.5 (OpenPI `behavior` fork, skill-conditioned finetune path proven) + GR00T N1.7 (backup); one R1Pro-class mobile dual-arm robot (23-dim action).

**Reviewer bar (applied to every idea).** Quantified gap claim in the intro; an evidence ladder matched to claim altitude (sim work carries no real-world claims); n per condition + verbatim success criterion + disclosed failure tables; hardware/sensor-matched tuned baselines including a strong classical option; numeric sim-real deltas where applicable; a contribution that survives the “no novel contribution” desk screen (a claim + mechanism + measurement, not a system assembly of known parts).

**Repo state.** The challenge repo is at Week 1 (echo VLA backend, heuristic detectors, 3 of 100 plans). P2, trained detectors, and the full plan miner land with the challenge’s M2/M3 deliverables on the **Oct 16** deadline.

---

## 2. Idea 1 — Anatomy of Failure: a long-horizon generalization & failure benchmark for house-scale VLA policies

**Type:** benchmark + diagnostic study (no new policy training). **Verdict: GO — primary.**

### Abstract (paper-ready)

House-scale manipulation is the operating regime that VLA policies are converging on, yet no systematic study of what actually limits them exists. We introduce `hse-bench`, a protocol that decomposes long-horizon generalization failure on BEHAVIOR-1K: a 5-policy panel (per-task and shared π0.5, GR00T N1.7, a skill-scripted classical reference, and shared π0.5 + watchdog) evaluated across 100 task families and 3 instance conditions (in-distribution, scene-perturbed, cross-task), yielding ~14k rollout episodes. We auto-label failure windows from privileged BDDL state into a validated taxonomy (stuck loops, search failure, goal forgetting, wrong object, skill-execution failure, transition error, timeout) and build a fixability map showing which classes a no-retrain watchdog intervention recovers and which require new low-level data. Headline results: robustness failures account for ≥60% of Q-loss relative to a perfect-skill-execution reference; the ≥15-point generalization gap to perturbed instances is concentrated in navigation and multi-skill tasks; and the watchdog recovers ≥20% of the robustness gap.

### Contribution

1. The first empirical answer to “what limits VLAs on multi-minute household tasks, and which limit is addressable by better low-level data vs. better planning/robustness” — directly actionable for policy design.
2. A reusable house-scale evaluation protocol (stratified instances + BDDL-grounded auto labels + intervention ablation) with n and success criteria stated per cell.
3. Released, durable artifacts: `hse-bench` harness + ~14k labeled episodes (HF) + taxonomy + fixability map — the desk-screen defense.
4. A quantified, honest generalization-gap table for the two strongest open house-scale policy families (π0.5-class, GR00T-class).

### Method outline

- **Policy panel (all served through the identical wrapper, same 24 GB budget, identical evalharness — hardware-matched by construction):** P1 per-task π0.5 finetune (stratified 20-task subset, reported as such; deferrable); P2 shared skill-conditioned π0.5 (challenge M2); P3 GR00T N1.7; P4 classical skill-scripted reference (mined skill plans + predicate detectors — the required strong-classical baseline and “no learned policy” lower bound); P5 = P2 + watchdog (the intervention arm for H3).
- **Instance conditions:** C1 *in-distribution* — **defined by composition overlap with the demo-mining instances, not by instance id** (demo instances 1–200 are disjoint from official test instances 301–340; linkage is implementable today via `task_instance_id` in episode metadata); C2 *scene-perturbed* — resampled within BDDL to novel layouts/object swaps; C3 *cross-task* — P1 evaluated on task families held out of its finetuning set.
- **Rollout matrix:** 100 tasks × 10 instances × 5 policies for C1/C2 (~10k episodes, the headline table) + a 20-task deep subset at 40 instances for C2/C3 (~4k episodes). Total ≈ 14k episodes.
- **Auto failure labeling** from privileged sim state (legal at analysis time): `stuck_loop` (no predicate change >30 s within skill budget), `search_fail` (frontier exhausted / target room never reached), `goal_forget` (predicate regression after satisfaction), `wrong_object`, `skill_exec_fail`, `timeout`, `transition_error`. Taxonomy validated by 2 annotators on n ≥ 300 random episodes; Cohen’s κ reported.
- **Analysis:** per-skill success (27 skills × 100 tasks), horizon-decay curves (Q vs skills remaining), ΔQ per condition × task family, failure-class frequency per policy, and the fixability map (P5-vs-P2 and P1-vs-P2 deltas → which intervention recovers which class).

### Evaluation plan

- **Metrics:** Q (primary; official `score_utils` math; n per cell), task success rate, time score, per-skill success with 95% binomial CIs, failure-class tables, ΔQ(C1−C2) per family, horizon-decay slope, **worst-case (min-over-task) Q**, taxonomy κ.
- **Hypotheses (falsifiable):** H1 ≥60% of Q-loss (vs perfect-skill-execution reference) is attributable to robustness classes; H2 ΔQ(C1→C2) ≥15 points, concentrated in navigation/multi-skill tasks (single-skill <10); H3 the watchdog (P5, no retrain) recovers ≥20% of the robustness-attributable gap in both mean and worst-case Q.
- **Baselines:** the panel itself (P4 satisfies the classical-baseline requirement); LIBERO/SimplerEnv rows included as a clearly-labeled different-regime reference.
- **Gates:** G1 (Oct 20) — 500-episode pilot (10 tasks × 10 instances × 5 policies, ~100 GPU-h): ΔQ(C1−C2) ≥5 on ≥3 of 5 policies, else redesign perturbations. G2 (Jan 15) — taxonomy κ ≥0.7, else shrink to 5 classes. G3 (Feb 10) — H1/H3 hold directionally, else reposition as a pure benchmark (still publishable).

### Timeline and compute

Pilot locked Sep 13–Oct 16 (parallel with the challenge; labeler frozen Oct 16) → matrix waves 1–2 Oct 20–Nov 30 (~5k episodes) → waves 3–4 + deep subset + human audit Dec 1–Jan 15 (~4k episodes) → analysis Jan 15–Feb 10 → draft and **IROS 2027 submission Mar 1** → ICRA 2028 v2 (full 40-instance depth, GR00T refresh) Jun–Aug 2027. Compute (corrected): **~4–5k GPU-hours** (the ideas doc’s 2,100 used sim-episode duration as wall-clock) ≈ 5–6 weeks of sustained 8×24 GB occupancy spread over Oct–Jan, including P1 finetunes.

### Risks

(a) **No signal** — policies collapse uniformly on C2 so conditions don’t separate; the 500-episode pilot gate catches this on ~100 GPU-h before a 14k commitment. (b) **Compute starvation** — single 24 GB box makes the matrix ~5–6 months: NO-GO without precondition A. (c) **P1 finetune slippage** — defer P1; the headline table still works with P2–P5. Data risk is low: no collection; every input already exists in the challenge stack.

### Next actions (launch checklist)

1. Confirm cluster access (precondition A) this week — it is the idea’s only hard blocker.
2. Stand up the 500-episode pilot the week the M2 P2 checkpoint lands; P4 classical reference by Oct 16.
3. Freeze the auto-labeler on 10 pilot tasks by Oct 16.
4. Reserve 2 annotators × ~30 person-hours for the Dec–Jan taxonomy audit wave.

---

## 3. Idea 2 — Predict-Before-You-Fail: predictive sub-skill failure detection & graceful degradation for house-scale VLA policies

**Type:** runtime algorithm (learned early-warning + trigger-matched degradation ladder), sim-first. **Verdict: CONDITIONAL GO — third; cheap parallel pilot, commit only on gate.**

### Abstract (paper-ready)

Learned house-scale policies fail late and catastrophically: by the time a stuck loop or wrong-object grasp is visible in predicate trajectories, the sub-skill budget — and the episode’s partial credit — is already gone. Existing failure-aware work is post-hoc or low-level only. We present EWR, an onboard-only early-warning model that predicts imminent sub-skill failure with ≥5 s lead time, and a **trigger-matched graceful-degradation ladder** — identical ladder, learned vs. rule trigger — that isolates the value of prediction itself. EWR conditions on ~40 onboard features (predicate-detector confidence, progress velocity against mined p90 budgets, VLA action-chunk disagreement, proprio deviation, odometry drift, remaining task structure), trains on BDDL-grounded failure labels auto-extracted from privileged sim state, and drives a budget-capped ladder (continue → retry → replan → search → handoff) with failure-class-conditional level selection. On a ~7k-episode house-scale substrate across π0.5 and GR00T N1.7, EWR reaches AUC ≥0.85 at ≥5 s lead and raises Q by ≥8 points over no intervention and ≥3 points over a post-hoc watchdog trigger, with worst-case Q improving ≥3× the mean.

### Contribution

1. The first *predictive* (vs post-hoc) failure model for VLA-driven manipulation, with lead time as a reported and ablated quantity.
2. A measured degradation recipe: per failure class → ladder level → trigger threshold → net Q (the deployment side of long-horizon shared control).
3. Proof of policy-agnosticism: one EWR above two different VLAs.
4. Released `ewr` code + early-warning dataset (~7k episodes: feature stream + BDDL failure labels).

### Method outline

- **Features (onboard-only, <10 ms CPU per step, must not break the 13.5 sim-FPS cadence):** (i) predicate-detector confidence trajectories (3-frame hysteresis); (ii) progress velocity Δpred/s and sub-skill elapsed fraction vs mined p90 budget; (iii) action-chunk disagreement — temporal-ensemble variance + receding-horizon chunk-blend residual (already computed in `src/b1k/policy/chunking.py`; one-line tap to expose it as a logged feature); (iv) gripper load/proprio deviation from per-skill demo statistics; (v) odometry drift (velocity-integration vs depth-VO residual; pose/occupancy exist in `src/b1k/perception/odometry.py`); (vi) skill identity + plan position + predicates remaining.
- **Labels:** “imminent failure” = the expected predicate flip does not occur within 1.5× the p90 sub-skill demo budget — defined from demo statistics, not ad-hoc. Auto-labeled from privileged sim state, reusing I1’s auto-labeler; ~100-episode human audit; positive density raised by ~2k intentional *degradation-sweep* episodes (novel C2 instances, BDDL-allowed property perturbations, induced stuck starts); successful episodes supply hard negatives (critical for the false-alarm metric).
- **Models:** primary = skill-conditioned gradient-boosted trees + a small temporal MLP over the last 3 s (interpretable, CPU-trainable, ablatable per feature); probe = transformer over last-8-frame crops (is vision necessary?); baselines = uncertainty-only (chunk variance), rule-only (the challenge Progress watchdog — identical ladder, post-hoc trigger; its six actions are verified in `src/b1k/planner/progress.py`), oracle (privileged upper bound, labeled as such).
- **Ladder:** L0 continue → L1 RETRY → L2 REPLAN (re-derive the remaining queue from the BDDL goal minus satisfied predicates) → L3 SEARCH (occupancy-frontier, room-prior-biased) → L4 HANDOFF (sim: restart sub-skill fresh; real: request human). Budget-capped at 1.5× p90 (inherited from the challenge watchdog, so the H2 comparison is trigger-matched by construction); level selection is failure-class-conditional (search failures go to L3, not L1).

### Evaluation plan

- **Arms:** no-intervention (raw Q reference), rule-triggered ladder, EWR-triggered ladder, EWR + CBF low-level filter, oracle. Substrate = I1’s C1/C2 matrix (fallback if I1 does not run: standalone 40-task matrix, ~1,500 GPU-h). 5 arms × 100 tasks × 10 instances; leave-one-feature-channel-out ablations; policy transfer (train on π0.5, evaluate on GR00T).
- **Hypotheses:** H1 AUC ≥0.85 / AP ≥0.70 at ≥5 s lead on held-out tasks; H2 ΔQ ≥+8 over no-intervention **and** ≥+3 over rule-trigger, false-alarm rate <15%; H3 min-over-task Q improves ≥3× more than mean; H4 transfer to GR00T loses <3 points of H2’s gain.
- **Gates:** G1 (Oct 31) — 1k-episode pilot (10 tasks, run in parallel with I1’s pilot on the same substrate): AUC ≥0.70 @3 s **and** a first look at the H2 margin (does the rule-trigger already recover most of the gap?). G2 (Jan 20) — H2 isolation holds on ≥60 tasks jointly, else reposition as “detection + ladder” (weaker, still viable). G3 (Feb 10) — false-alarm <25%, else a threshold-tuning story only.

### Timeline and compute

Feature pipeline + shared auto-labeler by Oct 31; 1k-episode pilot in parallel with I1 wave 1 (near-zero marginal cost) → **commit decision mid-Nov**; if committed: degradation sweeps + substrate Nov 1–Dec 15, intervention experiments Dec 16–Jan 20, draft and submit Feb 10–Mar 1. Compute (corrected): **~1,800–2,200 marginal GPU-hours** (the ideas doc’s 1,050 already assumed the shared substrate; standalone is higher); EWR itself trains on CPU. If the gate fails: fold into the ICRA 2028 trilogy.

### Risks

(a) **Core tension — H2 null:** the challenge’s own watchdog may already be so strong that a predictive trigger adds little; a trigger-matched null is a publishable boundary result but a weaker paper — the Oct 31 margin probe exists to catch this before ~2k stress episodes are spent. (b) **Label-window design:** “imminent” is defined from demo statistics (1.5× p90); wrong window shifts AUC claims — mitigated by the sensitivity table + 100-episode audit. (c) **Schedule coupling** to I1’s substrate — tightest timeline of the three.

### Next actions

1. Implement the one-line chunk-disagreement tap in the serving path this week.
2. Run the pilot with I1’s wave 1 (shared substrate, parallel).
3. Do **not** pre-commit the full 7k-episode program; reserve ~2.5 weeks of November compute only if the mid-Nov gate passes.

---

## 4. Idea 3 — How Far Is Simulation From Home: controlled sim-real delta quantification + residual adaptation for house-scale manipulation

**Type:** measurement protocol + transfer method (real robot central). **Verdict: CONDITIONAL GO — co-primary; pilot first.**

### Abstract (paper-ready)

Sim-to-real transfer is largely settled for single-task, single-arm manipulation and locomotion, but has never been measured for house-scale composition — mobile + bimanual + long-horizon + 100+ object states — and no real-robot counterpart of BEHAVIOR-1K exists. We introduce a **controlled matched-instance protocol** (20 tasks stratified by skill family; 3 real layout variants per task; 3 matched sim instances per layout; identical policy; n = 10 rollouts per layout-condition) and report the first numeric per-skill-family sim-real delta table for house-scale manipulation. We then close a measured fraction of the gap with a per-skill residual adapter r(o) → Δa over VLA action chunks — RMA-style, trained on ≤10 min of on-robot data per family — with an honest data-budget curve (1/5/10/20/30 min), and present a cheap sim-only **fidelity probe** that predicts per-task ΔQ (R² ≥0.5) before deployment. Headline structure: the delta is ordered by family (navigation ≤5 < single-arm ≤15 < contact-rich ≤25 < bimanual ≤30 points), and ≥30% of real failures come from real-only classes (slip, contact-force deficits, drift) that occur at <5% frequency in sim.

### Contribution

1. The matched-instance protocol: turns “sim vs real” from anecdote into repeatable measurement (matching table + predicate grounding released).
2. The first numeric house-scale sim-real delta table per skill family — the missing evidence rung reviewers currently reject on (numeric deltas are explicitly on the reviewer-want list; no published real-robot BEHAVIOR-1K counterpart was found).
3. Per-skill residual adaptation for VLA action chunks with an honest data-budget curve — the composition answer RMA-style locomotion results do not provide.
4. The sim-only fidelity probe: a practitioner tool (“estimate your expected house-scale delta before you mount the robot”) that the world-model literature implicitly needs.

### Method outline

- **Task selection:** 20 tasks stratified by skill-family composition — 5 navigation-heavy, 5 single-arm non-contact, 5 contact-rich, 5 bimanual/articulated (selection documented from mined skill-plan statistics).
- **Instance matching:** 3 real layout variants (L1–L3) per task in the group’s re-configurable apartment/lab-kitchen testbed (furniture rearrangement is the matching knob — no new construction); BDDL predicate states **manually grounded** on rollout video at 0.5× speed (2 annotators, detector-assisted pre-labeling, κ-gated); sim side: 3 matched OmniGibson instances per layout (same task, object set, room topology) generated with I1’s C2 perturbation machinery constrained to match the real layout. Matching table released.
- **Rollouts:** identical policy (P2 shared skill-conditioned π0.5); 10 sim instances + 10 real rollouts per task (n = 10 per layout-condition; no cherry-picking; videos archived) ≈ 400 rollouts ≈ 40 robot-hours + 400 sim episodes.
- **Delta analysis:** Q_sim vs Q_real per task/family with 95% CIs; failure-class frequencies (I1’s auto-labeler on sim; manually audited labels on real); per-episode divergence localization (first K steps where sim/real diverge) to localize *where* the gap opens.
- **Residual adaptation:** r(o_t, skill, remaining-predicates) → Δa_{t:t+32} over 32-step action chunks, a′ = clamp(a + r(o)); on-robot data per family = 5 min autonomous rollouts (collect mismatch) + 5 min teleop correction (collect fix) ≈ 10 min/family, 40 min total, repeated at 1/5/10/20/30-min budgets for the H3 curve; small MLP/GRU on 1×24 GB or CPU. Ablations: per-family vs whole-policy; real-only vs sim+real mix; proprio-only (RMA-proper) vs full conditioning; no-residual zero-shot (the delta reference).
- **Fidelity probe (H4):** from sim-only runs, distributional divergence on identical observation prefixes (KL on action-chunk confidence + proprio-deviation statistics) regressed to real ΔQ; report R² + calibration + leave-one-task-out.

### Evaluation plan

- **Hypotheses:** H1 delta ordering as above (Q in [0,100] convention for deltas; official Q ∈ [0,1] reported alongside); H2 ≥30% of real failure episodes attributable to real-only classes (<5% sim frequency); H3 residual closes ≥40% of contact-rich/bimanual ΔQ at ≤10 min, with the data-budget curve saturating ≤30 min; H4 probe R² ≥0.5.
- **Baselines:** zero-shot sim-trained policy on real (the measured quantity, not a target); real-only per-task π0.5 finetune on 3 tasks (cost reference, capped by design); proprio-only residual (ablation); watchdog-only robustness reference; skill-scripted classical execution on 5 tasks (the “how far is *any* sim training from real” floor). All arms: identical serving wrapper, identical predicate grounding.
- **Gates:** G1 (Oct 16) — **pilot: 3 tasks × 2 layouts × 6 rollouts (~6 robot-hours + ~30 GPU-h):** measurable ΔQ ≥5 on ≥2 of 3 tasks **and** grounding κ ≥0.8. This is the make-or-break gate: it validates the protocol end-to-end and resolves the load-bearing manual-grounding risk six weeks before any large commitment. G2 (Jan 15) — H1 ordering holds directionally on wave-1 families, else reposition as “delta table + probe” (H1/H4 carry the paper; residual becomes an appendix). G3 (Feb 10) — H3 gap-closed ≥25% at 10 min on ≥2 families, else claim protocol + measurement only (the delta table is novel regardless).

### Timeline and compute

Pilot Oct 1–16 → layout reconfiguration + real delta wave 1 (10 of 20 tasks ≈ 200 rollouts ≈ 20 robot-days) + 200 sim rollouts Oct 17–Nov 30 → wave 2 Dec 1–Jan 15 + residual training on completed families → residual arms complete + ~150 adapted-policy re-rollouts (≈15 robot-days) Jan 16–Feb 10 → **submit Mar 1** (scope: protocol + delta table + contact/bimanual residual + probe; navigation/single-arm residual deferred to ICRA 2028 if hardware runs short). Total: ~40 robot-hours of delta rollouts, **~35 robot-days Nov–Feb** including residual re-rollouts and retries, and **~150–200 GPU-hours** sim — the cheapest idea on compute, the most expensive on logistics. Safety: spotter + e-stop, no elevated-robot work, contact tasks within the working envelope — a safety incident stops the program, so it is treated as a dependency, not a footnote.

### Risks

(a) **Manual predicate grounding error** — the load-bearing weakness (sim predicates have no automatic real counterpart); the κ gate at the Oct 16 pilot + divergence-localization cross-check + released per-rollout artifacts keep it auditable. (b) **Real testbed drift** between waves — fixed layout photos per condition + same-day sim/real pairs. (c) **Hardware slip past Jan 15** — degrades to G2 scope (delta table + probe only), still a paper. (d) **No robot/testbed at all** — NO-GO on the real arm; only the sim-only probe survives (precondition B).

### Next actions

1. **Confirm physical R1Pro + re-configurable testbed + 1-GPU access by end of September (precondition B)** — the single most important action in the portfolio right now.
2. If confirmed, run the pilot in the first week of October.
3. Write the predicate-grounding SOP and train the 2 annotators *before* the pilot.
4. Fix the layout-photography discipline from day one (it is the drift mitigation).

---

## 5. Comparison and recommendation

| Axis | I1 — Anatomy of Failure | I3 — Sim-Real Delta | I2 — Predict-Before-Fail |
|---|---|---|---|
| Type | Benchmark + diagnostic | Measurement + transfer | Runtime algorithm |
| **Novelty** | Med-High: first house-scale decomposition + released labeled dataset | **Very High:** no real-robot BEHAVIOR-1K counterpart (verified); numeric deltas on the explicit reviewer-want list | Med-High: first predictive (vs post-hoc) model, lead time ablated |
| **Feasibility** | **High** (given cluster A): lowest method risk, no data collection | Medium: hardware-gated, highest technical risk; 6-robot-h pilot is cheap insurance | Med-High: cheapest marginal cost, but real null-result risk on H2 |
| **Impact** | High: durable artifact; substrate for I2 (labels) and I3 (C2 machinery); doubles as the challenge’s robustness study | **Very High:** answers the #1 reviewer rejection reason; the only idea whose abstract can carry real-world claims | Med-High: IROS safety-category fit; policy-agnostic reusable runtime layer |
| **Time-to-paper** | ~5.5 mo: pilot gate Oct 20 → submit Mar 1; ~6 weeks margin if cluster lands in October | ~5.5 mo: pilot gate Oct 16 → submit Mar 1; hardware window Nov–Feb | ~5 mo: pilot Oct 31, commit mid-Nov; **tightest** (depends on I1) |
| Compute (corrected) | ~4–5k GPU-h | ~150–200 GPU-h + ~35 robot-days | ~1,800–2,200 marginal GPU-h |
| **Rank** | **1** | **2** | **3** |
| **Verdict** | **GO — primary** | **CONDITIONAL GO — co-primary (pilot first)** | **CONDITIONAL GO — third (pilot-gated)** |

**Recommendation:** pursue **I1 + I3 as the two primary IROS 2027 papers**; run **I2 as a cheap parallel pilot and commit only if its gate passes** (mid-Nov). This maximizes (a) expected IROS 2027 submissions, (b) novelty coverage (benchmark + sim-real), and (c) synergy (I1 feeds both). **Do not run all three at full scope simultaneously** — see §7.

**Why, in one paragraph.** I1 is the only idea with no hardware or logistics risk, and it is the substrate the other two consume (its rollout matrix is I2’s positive labels; its C2 perturbation machinery is I3’s instance generator) — so it is ranked first on feasibility and is the portfolio’s load-bearing wall. I3 is ranked second on feasibility but first on novelty and reviewer fit: it is the only idea that structurally defeats the “sim results carry real-world claims” rejection, and its 6-robot-hour pilot resolves the portfolio’s highest technical risk (manual BDDL grounding) by Oct 16, six weeks before any large commitment. I2 is third because its core tension — the challenge’s own watchdog may already recover most of the recoverable gap, making the prediction value (H2) a null — is a genuine null-result risk that the trigger-matched pilot probe resolves cheaply; pre-committing the full program before that probe would burn ~2k stress episodes on faith.

## 6. Sequencing (Oct 2026 → Mar 2027)

| Window | Actions | Gates / decisions |
|---|---|---|
| Now → Sep 30 | Confirm precondition **A** (cluster) and **B** (robot + testbed); track challenge M2/M3 (P2 checkpoint, trained detectors, 100-plan miner) | Go/no-go inputs for the whole portfolio |
| Oct 1–16 | **I3 real pilot** (3 tasks × 2 layouts × 6 rollouts; ~6 robot-h + ~30 GPU-h) | G1-I3 (Oct 16): ΔQ ≥5 on ≥2/3 tasks **and** κ ≥0.8 |
| Oct 16 | Challenge submission (M2/M3 land); I1 auto-labeler frozen; P4 classical reference done | Shared-substrate gate for all three |
| Oct 16–20 | **I1 500-episode pilot** (10 tasks × 10 instances × 5 policies; ~100 GPU-h) | G1-I1 (Oct 20): ΔQ(C1−C2) ≥5 on ≥3/5 policies |
| → Oct 31 | **I2 1k-episode pilot** in parallel (shared substrate, near-zero marginal cost) + H2-margin probe (EWR vs rule-trigger) | G1-I2 (Oct 31): AUC ≥0.70 @3 s **and** visible H2 margin |
| Mid-Nov | I2 commit decision | Commit / reposition / defer to ICRA 2028 |
| Nov 1 → Jan 15 | I1 full matrix (14k episodes; ~5–6 wks of 8×24 GB continuous occupancy spread); I3 waves 1–2 (~35 robot-days, if B); I2 stress sweeps + full program (if committed) | G2-I1 (Jan 15) κ ≥0.7; G2-I3 (Jan 15) H1 ordering; G2-I2 (Jan 20) H2 on ≥60 tasks |
| Jan 15 → Feb 10 | Analysis, H-verdicts, negative-result triage, drafting | G3 gates (Feb 10) |
| **Mar 1, 2027** | **IROS 2027 submission** | I1 + I3 (+ I2 if gated in) |
| H2 2027 | ICRA 2028 (~Sep 15): v2 / deferred scope — P1 full-100, I2 real pilot, I3 navigation/single-arm residual, or the combined “diagnose → predict → transfer” trilogy | Fallback venue |

## 7. Preconditions and gating facts (read before starting)

1. **Precondition A — GPU cluster (I1, I2).** ~4–8×24 GB GPUs usable Oct–Jan, plus the 3.27 TB demo download. The corrected compute matters: the official track steps at 13.52 sim-FPS with 150–300 s scene load, so a 9-min episode is ~20 min of GPU wall-clock — I1’s matrix is **~4–5k GPU-hours**, not 2,100. On a single 24 GB box the matrix is ~5–6 months: **I1/I2 as scoped are NO-GO without A**.
2. **Precondition B — physical robot + testbed (I3).** A mobile dual-arm R1Pro-class platform and a re-configurable apartment/lab-kitchen testbed for ~35 robot-days Nov–Feb. The “challenge asset” robot is the **sim embodiment**; physical access is unconfirmed (the feasibility review ran on a Jetson Orin and could not verify A or B). Without B, I3 → G2 scope (delta table on completed tasks + fidelity probe) or ICRA 2028.
3. **Shared dependency — challenge M2/M3.** All three assume P2 (shared skill-conditioned π0.5), trained predicate detectors, and the 100-plan miner. The repo is at Week 1 today (echo backend, heuristic detectors, 3 of 100 plans). If the challenge submission slips past Oct 16, or P2 lands weak, substrate quality degrades for **all three at once** — track M2/M3 as a hard gate on the October pilots.
4. **C1 definition (I1; state precisely in the paper).** “In-distribution” = **composition overlap** with the demo-mining instances, *not* instance id: demo instances (1–200) are disjoint from official test instances (301–340). Implementable today via `task_instance_id` in episode metadata (verified against the live dataset) — state it to preempt the obvious reviewer objection.
5. **Verified grounding that may be cited.** IROS 2027 deadline 2027-03-01 23:59 PST (three independent sources); no published real-robot BEHAVIOR-1K counterpart (web search); watchdog RETRY/REPLAN/SEARCH/FINISH, odometry pose/occupancy, and the chunk-blend residual exist in `src/b1k`.
6. **Honest portfolio math (all three, corrected).** ~7–9k GPU-h + ~35 robot-days + ~40–60 person-hours ≈ a 2-person, 5-month load — not a multi-year program, and it competes with the challenge itself for the same people and substrate. Stagger per §6; **let the October pilots, not the plan, decide what gets committed.**

## 8. Sources

- `docs/icra-iros-2027-trends.md` — literature/gap map: gaps cited above (G1.5, G2.3, G2.4, G5.1, G6.1, G6.3, G6.4), 63 live-verified sources, reviewer bar, venue facts.
- `docs/icra-iros-2027-ideas.md` — full per-idea specs: RQ, novelty vs named prior work, baselines, metrics, 12-month plans (this document compresses them and applies the feasibility corrections).
- `docs/feasibility-assessment-2027.md` — grounded feasibility review: corrected compute (§1 there), MVEs, gates, verdicts, preconditions, sequencing (this document’s §5–§7 follow it).
