# Feasibility Assessment — 3 ICRA/IROS 2027 Research Ideas
**Task:** `t_0423f72f` (feasibility + robotics scope) → feeds `t_f4013e5a` (final scope doc, writer).
**Reviewed:** `/home/luis/behavior_challenge/docs/icra-iros-2027-ideas.md` (3 ideas), grounded against the repo's real code, the BEHAVIOR-1K 2026 challenge spec, the v3.9.2 evaluator constants, and the live 20k-demo dataset schema.
**Date:** 2026-09-13 (robotics-engineer). Verdicts are go/no-go per idea + the smallest credible experiment (MVE) for each.

---

## 0. Method & what I actually verified (grounding)

I did not trust the ideas doc at face value. For every load-bearing claim I either read the repo source, checked the challenge spec/evaluator constants, or queried the live dataset. Findings:

| Claim in ideas doc | Verdict | Evidence |
|---|---|---|
| IROS 2027 deadline = Mar 1, 2027 (PST), Florence Sep 26–Oct 1 | **CONFIRMED** | mldeadlines.com + aconf.org + IEEE-RAS CFP all list 2027-03-01 23:59 PST. ICRA 2027 (Sep 15 2026) is 2 days past today → unreachable. ICRA 2028 (~Sep 2027) is the fallback. |
| Demos link to sim instances (needed for I1's in-distribution vs novel split) | **CONFIRMED** | `meta/episodes/chunk-000/file-000.parquet` (live HF) has columns `task_instance_id`, `raw_episode_id`, `demo_index_within_task`, `task_index`, `annotation_path`. I read the first 5 rows of task 0. |
| `controller.py` computes chunk-blend residual (I2's "free" feature iii) | **CONFIRMED (partial)** | `src/b1k/policy/chunking.py:blend_chunks` exists and computes the alpha-ramp blend; the raw "disagreement/variance" is not currently *exposed* as a logged feature — I2 must add a one-line tap. |
| Watchdog RETRY/REPLAN/SEARCH/FINISH exists (I1's P5, I2's rule-trigger arm) | **CONFIRMED** | `src/b1k/planner/progress.py` implements all six actions (CONTINUE/ADVANCE/RETRY/REPLAN/SEARCH/FINISH) with budget caps. This is the challenge's M2/M3 brain. |
| Odometry pose + occupancy frontier (I3/I2 nav claims) | **CONFIRMED** | `src/b1k/perception/odometry.py` has `pose`, `position`, `yaw`, `occupancy.frontier()`. |
| No published real-robot counterpart of BEHAVIOR-1K (I3's novelty) | **CONFIRMED (not found)** | Web search returns only the sim benchmark (arXiv 2403.09227, CoRL 2022). No physical-robot house-scale delta study published. |
| **GPU-hour figures (I1 ≈ 2,100 h, I2 ≈ 1,050 h)** | **UNDERSTATED ~2×** | The doc treats a "~9 min sim episode" as 9 min of GPU wall-clock. Spec §8: official RGB+depth 720/480 track steps at **13.52 sim-FPS** → a 9-min (16,200-step) episode takes **~20 min of GPU wall-clock**, plus **150–300 s one-time scene load** per trial. Corrected I1 matrix ≈ **4–5k GPU-hours**, not 2,100. See §3. |

**Critical context — the challenge repo is at Week 1, fixture-level.** The VLA backend currently defaults to `echo` (replays recorded demo actions); there is **no real π0.5/GR00T checkpoint, no trained detectors (heuristics only), and only 3 of 100 mined plans** (`docs/plans/{0,1,3}.json`). The `data/demos/` tree holds fixture `actions.npy` for tasks 0 and 1 only. **All three ideas ride on the challenge's M2/M3 deliverables** (shared skill-conditioned π0.5 = "P2", trained predicate detectors, the 50-port evalharness, and the full 100-plan miner) which land on the **Oct 16 challenge deadline**. The ideas are challenge-adjacent *by construction* and **inherit the challenge's schedule risk** as their single biggest dependency.

---

## 1. The one number that changes the plan: corrected compute

The ideas doc's "~4,200 GPU-hours if all three run" understates the sim work by roughly half because it used sim-episode *duration* as if it were GPU *wall-clock*. Recomputing from the spec's measured 13.52 sim-FPS:

- Per episode (9-min timeout, 16,200 steps): ≈ 20 min GPU + ≈ 3–5 min scene-load amortized ≈ **~23 min wall per episode per GPU**.
- I1 matrix (~14k episodes): **≈ 4,600–5,400 GPU-hours** (was 2,100).
- I2 degradation sweeps + substrate (~7k episodes, shared): marginal **≈ 1,800–2,200 GPU-hours** on top of I1's matrix (the doc's 1,050 was already assuming the share; standalone is higher).
- I3 sim side (~400 episodes): **≈ 150–200 GPU-hours** (compute is genuinely cheap — the cost is robot-time, unchanged).

**Implication, not a blocker:** on 8×24 GB GPUs the I1 matrix is ~660–900 GPU-hours each → **~2.5–4 weeks continuous**, plus the ~20 per-task π0.5 finetunes (P1) which run alongside. The ideas doc's "2–3 weeks on 8 GPUs" is roughly right *for the matrix only*; add a week for P1 finetunes and I2's stress sweeps and the honest number is **~5–6 weeks of sustained 8-GPU occupancy** spread over Oct–Jan. This fits a 2-person team **only if a modest GPU cluster is actually available** (§2). On a single 24 GB box it is ~5–6 months — infeasible for the Mar 1 deadline.

---

## 2. The two gating unknowns I could not confirm (do not proceed on these blindly)

These are the facts that decide go/no-go and I have **no way to verify from the repo** (this box is a Jetson Orin: 7 GB RAM, 22 GB free disk — it cannot host the rollouts or the 3.27 TB demo set). **Get answers before committing compute:**

- **GPU cluster access (I1, I2):** Do we have ~4–8×24 GB GPUs usable Oct–Jan for the rollout matrix + P1 finetunes + the 3.27 TB demo download? If **no multi-GPU access**, I1/I2 as scoped are **NO-GO** (the corrected §1 math makes single-GPU infeasible). I3's sim side survives on 1 GPU.
- **Physical R1Pro + real testbed (I3):** Do we actually have the mobile dual-arm robot **and** a re-configurable real apartment/lab kitchen for ~35 robot-days Nov–Feb? The ideas doc lists the robot as a "challenge asset" but a *challenge asset* is the **sim embodiment** — physical access is unconfirmed. If **no robot/testbed**, I3 degrades to its G2 path (delta-table + fidelity probe only) — still publishable but loses the real-robot claim that makes its abstract strong.

Everything below is written **conditional on** these two answers.

---

## 3. Per-idea feasibility

### Idea 1 — Anatomy of Failure (benchmark + diagnostic). **GO (primary).**

**Feasibility: HIGH (given GPU cluster).** This is the most self-contained idea and the substrate the other two ride on.

- **Hardware/sim:** OmniGibson v3.9.2 + 20k demos + P1–P5 policy panel. No real hardware. No *new data collection* (rollouts only). All data already exists in the challenge stack (verified: instance linkage via `task_instance_id`).
- **Data effort:** Low. The only human cost is the taxonomy audit (~60 person-hours, 2 annotators, n≥300, κ gate). Instance C1/C2 split is **implementable today** — I confirmed `task_instance_id` lets us reconstruct which object/layout compositions the 200 demos/task covered, so C1 "composition seen in demos" vs C2 "novel composition (resampled within BDDL)" is a real, defensible split. *Subtlety to state in the paper:* demo instances (ids 1–200) and official test instances (301–340) are **disjoint by construction**, so C1 must be defined by *composition overlap*, not by "same instance id" — which is exactly the doc's definition, so no redesign needed, just state it precisely.
- **Compute:** ~4–5k GPU-hours (corrected, §1). Medium-High but bounded and off-peak vs the challenge.
- **Safety:** none (sim-only).
- **Reproducibility:** strong — releases `hse-bench` harness + ~14k labeled episodes + taxonomy. Durable dataset artifact is the desk-screen defense.
- **Timeline:** pilot (500 episodes) by Oct 20 gate G1; matrix Oct–Jan; analysis Jan–Feb; **submit Mar 1**. Fits with ~6 weeks margin if the cluster is available in October.
- **Most likely failure modes:** (a) **no signal** — policies collapse uniformly to ~0 on C2 so conditions don't separate (G1 catches this on 500 episodes, cheap); (b) **compute starved** — no cluster ⇒ NO-GO (§2); (c) P1 per-task finetunes are slow to produce (20 finetunes) — mitigate by running them as a background wave, and P1 can be deferred to ICRA 2028 if it slips (the headline table still works with P2/P3/P4/P5).
- **Smallest credible experiment (MVE):** the **500-episode pilot matrix** — 10 tasks × 10 instances × 5 policies on C1 vs C2 — proving the two conditions separate: **ΔQ(C1−C2) ≥ 5 points for ≥ 3 of 5 policies** (gate G1). If this passes on ~100 GPU-hours, the full 14k matrix is a mechanical scale-up with no further method risk. *This is the cheapest gate in the whole portfolio and it front-loads the idea's biggest methodological risk.*

**Verdict: GO.** Launch the pilot immediately (October). Lowest method risk, highest reusability (I2 and I3 both consume its outputs), and it doubles as the challenge's own robustness study.

### Idea 2 — Predict-Before-You-Fail (algorithm). **CONDITIONAL GO — launch as a low-cost parallel pilot, commit only on a gate.**

**Feasibility: MEDIUM-HIGH (cheapest marginal cost, real null-result risk).**

- **Hardware/sim:** sim-only; rides I1's substrate. EWR itself trains on CPU (verified: features come from `progress.py`, `chunking.py`, detectors, odometry — all CPU). No real hardware required.
- **Data effort:** Low (failure labels auto-generated from privileged sim state, reusing I1's auto-labeler; ~100-episode audit). **The 2k "degradation sweeps" (intentionally-induced stress) are a real extra compute line the doc under-weights** — that is where the positive-density comes from and it does not come for free.
- **Compute:** ~1,800–2,200 marginal GPU-hours (shared substrate), corrected §1.
- **Safety:** none.
- **Reproducibility:** strong (releases `ewr` + ~7k-episode feature-stream dataset + degradation recipe).
- **Timeline:** pilot by Oct 31 (G1: AUC ≥ 0.70 @ 3 s on 1k episodes); if I1 is running, substrate is free; commit H2–H4 Jan–Feb; submit Mar 1. **Tightest schedule of the three** because it depends on I1's data.
- **Most likely failure modes — this is the idea's core tension:** (a) **the challenge's own watchdog (P5) is already so strong that a *predictive* trigger (EWR) adds little over a *post-hoc* rule trigger** → H2 (EWR > rule trigger by ≥3 pts) collapses to a null. The doc acknowledges this and has gate G2 for it, but a null H2 means the paper's headline ("prediction has measurable value beyond the post-hoc watchdog") fails. A trigger-matched *null* is publishable as a boundary result, but it is a weaker paper. (b) **label-window design** — "imminent failure" is defined from 1.5× p90 demo budget; if that window is wrong the AUC claims shift (mitigate with the sensitivity table + 100-ep audit). (c) **schedule coupling** — if I1 slips, I2 loses its substrate and must run a standalone 40-task matrix (~1,500 GPU-h, the doc's fallback).
- **Smallest credible experiment (MVE):** the **1k-episode pilot (10 tasks)** proving **EWR AUC ≥ 0.70 at 3 s lead** on held-out tasks (gate G1) *and*, critically, a **first look at the H2 margin**: does the rule-trigger (watchdog) arm already recover most of the recoverable gap on this pilot? If EWR beats the rule trigger by a clear margin on 10 tasks, commit; if the two are statistically indistinguishable, **reposition** (drop to "detection + ladder" or defer to the ICRA 2028 trilogy) rather than burning 2k stress episodes. *Run this pilot in parallel with I1's 500-episode pilot — same substrate, near-zero marginal cost — so the decision is made by mid-November, not in February.*

**Verdict: CONDITIONAL GO.** Do not commit the full 7k-episode program on faith. Start the cheap pilot alongside I1; the H2-marginal probe in the pilot is the real go/no-go. It is the natural **third** item — strong synergy, real null-result risk, tightest schedule.

### Idea 3 — How Far Is Simulation From Home (measurement + transfer). **CONDITIONAL GO — the pilot is cheap insurance that resolves the top risk; the full program is hardware-gated.**

**Feasibility: MEDIUM (highest technical risk, highest novelty, cheapest compute, most expensive logistics).**

- **Hardware/sim:** **real robot is load-bearing** (R1Pro-class + re-configurable real testbed, ~35 robot-days Nov–Feb). Sim side is cheap (~150–200 GPU-h, 1 GPU suffices — §1). This is the **only one of the three that needs physical hardware**, and it is the one where I have the least confirmed access (§2).
- **Data effort:** **MEDIUM-HIGH and it is the load-bearing weakness.** Sim predicates have no automatic real counterpart; the real side requires **manual BDDL predicate grounding** (2 annotators, ~0.5× video marking, κ gate). This is where annotator error and schedule slip live, and it is *not* cheap even though the doc rates "data availability Medium." The matched-instance protocol (real layout ↔ sim instance, 3 layouts × 3–4 instances) is a genuine new artifact but a real one to build and get right.
- **Compute:** Low (~120–200 GPU-h corrected). Cost is **robot-time, not FLOPs** — the cheapest idea on compute, the most expensive on logistics.
- **Safety:** real mobile dual-arm operation — spotter + e-stop, no elevated-robot work, contact tasks within working envelope. Standard but mandatory; it is a *dependency* (a safety incident stops the program).
- **Reproducibility:** strong (releases matching table + protocol + `simreal-house` + per-rollout video/predicate artifacts). Numeric sim-real deltas are **explicitly on the reviewer-want list** (§0 trend map), and because the real arm is central the abstract *can* carry real-world claims — the one idea that structurally defeats the "sim carries real claims" rejection reason.
- **Timeline:** pilot (3 tasks × 2 layouts × 6 real rollouts ≈ 6 robot-hours) by **Oct 16** — this is the make-or-break gate (G1: measurable ΔQ ≥ 5 on ≥2 of 3 tasks **and** grounding κ ≥ 0.8). Waves Oct–Jan; residual + probe Jan–Feb; **submit Mar 1** (scope: protocol + delta table + contact/bimanual residual + probe; navigation/single-arm residual deferred to ICRA 2028 if hardware runs short).
- **Most likely failure modes:** (a) **manual predicate grounding is error-prone** → κ fails G1 → the whole measured quantity is untrustworthy (this is *the* risk; the Oct 16 pilot exists precisely to catch it before 35 robot-days are spent); (b) **real testbed drift** between waves (furniture moves, lighting) — mitigate with fixed layout photos + same-day sim/real pairs; (c) **hardware slip past Jan 15** → degrades to G2 path (delta table + fidelity probe only, no residual) — still a paper, the protocol+measurement is novel regardless; (d) **no robot/testbed at all** → NO-GO on the real arm; only the sim-only fidelity probe survives.
- **Smallest credible experiment (MVE):** the **3-task × 2-layout × 6-rollout real pilot (≈ 6 robot-hours + ~30 GPU-h)** validating the protocol end-to-end and yielding **first real ΔQ numbers** with **grounding κ ≥ 0.8**. This is a *teaser-grade* result in its own right (even 3-task deltas are publishable internally as a pre-study) and it resolves the idea's single biggest risk 6 weeks before any large commitment. **Run it first** (October) — it is the highest-risk item in the whole portfolio and the pilot is cheap.

**Verdict: CONDITIONAL GO.** If the robot + testbed + a 1-GPU box are confirmed available, this is a **co-primary** with I1: highest novelty, best reviewer fit, and the pilot de-risks the top failure mode before the deadline. If physical access is not confirmed by mid-October, it downgrades to the G2 scope (delta-table-on-completed-tasks + probe) or is deferred to ICRA 2028.

---

## 4. Cross-cutting findings (apply to all three)

1. **Single shared dependency: the challenge's M2/M3.** All three assume a real shared skill-conditioned π0.5 (P2), trained predicate detectors, and the 100-plan miner — none of which exist yet (repo is at Week 1, `echo` backend, 3 plans, heuristic detectors). If the challenge submission slips past Oct 16 or the P2 policy is weak, the substrate quality degrades for *all three* at once. **Recommendation: the ideas team should track the challenge milestones M2/M3 as a hard gate on I1's pilot and on I2/I3's launches.**
2. **Corrected compute (§1) is ~2× the doc's figures** for I1/I2 because sim-FPS ≠ episode-duration. Re-plan October GPU occupancy on the corrected numbers.
3. **C1 must be defined by composition overlap, not instance-id** (demos 1–200 are disjoint from test 301–340). Implementable today via `task_instance_id`; state it precisely in the paper to preempt a reviewer objection.
4. **Sequencing (refines the doc's "I3 pilot first, I1 continuous, I2 last"):**
   - **Oct (now):** I3 real pilot (3 tasks, 6 robot-h) — de-risks the top portfolio risk. *In parallel:* I1 500-ep pilot + I2 1k-ep pilot on the same substrate (cheap, shares compute).
   - **Nov–Jan:** I1 full matrix (continuous, compute-bound, no hardware). I3 waves (if robot confirmed). I2 stress sweeps + full program (if its H2-marginal pilot passed).
   - **Jan–Feb:** analysis, H-verdicts, drafting.
   - **Mar 1:** I1 + I3 (co-primary) submit; I2 submits if its pilot gates passed, else folds into the ICRA 2028 trilogy.
5. **Synergy is real and by construction:** I1's rollout matrix + failure taxonomy = I2's positive labels; I1's C2 perturbation machinery = I3's matched-instance generator. Running I1 first makes I2 and I3 cheaper. The "diagnose → predict → transfer" trilogy is the natural ICRA 2028 home if all three complete.

---

## 5. Ranking (feasibility × publication potential, conditional on §2)

| Rank | Idea | Feasibility | Publication potential | Gate to pass (when) | One-line call |
|---|---|---|---|---|---|
| **1** | **I1 — Anatomy of Failure** | **High** (GPU cluster) | **High** (durable dataset + first house-scale decomposition; IROS-mainstream) | ΔQ(C1−C2)≥5 on ≥3/5 policies (Oct 20) | **GO — primary, launch Oct** |
| **2** | **I3 — Sim-Real Delta** | **Medium** (robot + testbed) | **Very High** (novelty + reviewer-demand fit + real claims in abstract) | Pilot ΔQ≥5 on ≥2/3 tasks + κ≥0.8 (Oct 16) | **CONDITIONAL GO — co-primary; run pilot first** |
| **3** | **I2 — Predict-Before-Fail** | **Med-High** (cheapest if I1 runs) | **Med-High** (safety-category fit, but null-result risk on H2) | Pilot AUC≥0.70 @3s **and** H2 margin visible (Oct 31) | **CONDITIONAL GO — parallel pilot, commit on gate** |

---

## 6. Recommendation (concrete go/no-go + next actions)

**Pursue I1 and I3 as the two primary papers; run I2 as a cheap parallel pilot and commit it only if its gate passes.** This maximizes (a) expected number of IROS 2027 submissions, (b) novelty coverage (benchmark + sim-real), and (c) synergy (I1 feeds both).

- **I1 — GO.** The 500-episode pilot is the cheapest, lowest-risk gate in the portfolio and it de-risks the idea's core method question (do conditions separate?). Start it the week the challenge M2 P2 checkpoint lands.
- **I3 — CONDITIONAL GO (co-primary).** Its pilot is **6 robot-hours** and resolves the portfolio's highest technical risk (manual BDDL grounding) by **Oct 16**, six weeks before any large commitment. **Action: confirm robot + testbed + 1-GPU access by end of September; if yes, run the pilot in the first week of October.** If no, the idea degrades to G2 scope or defers to ICRA 2028 — decide that at the Oct 16 pilot review, not earlier.
- **I2 — CONDITIONAL GO (third).** Do **not** pre-commit the 7k-episode program. Run the 1k-episode pilot **in parallel with I1's** (shared substrate, near-zero marginal cost) and include the **H2-marginal probe** (does the rule-trigger watchdog already recover most of the gap?). Commit only if EWR clearly beats the rule trigger; otherwise reposition or fold into the ICRA 2028 trilogy. Decide by **mid-November**.
- **Blocking action for the whole portfolio:** answer the two §2 unknowns (GPU cluster? physical robot+testbed?) **before end of September**. Every go/no-go above is conditional on them. On a single 24 GB box with no robot, the honest portfolio is **I1 only** (and even that is a stretch at the corrected §1 compute), I2 rides it, and I3 drops to the sim-only probe — a materially smaller program.

**Do NOT run all three at full scope simultaneously.** That is ~4,200+ GPU-hours (doc) / ~7–9k (corrected §1, incl. I2's stress sweeps) + 35 robot-days + 40+ person-hours = a 2-person, 5-month load where the challenge's own Oct 16 submission is also competing for the same people and the same substrate. Stagger per §4.3 and let the October pilots — not the plan — decide what gets committed.
