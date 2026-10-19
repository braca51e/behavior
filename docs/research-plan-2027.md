# Research Direction 2027 — Three Studies on Long-Horizon Household VLA Policies

**Date:** 2026-10-19 (rev. 3 — research direction only; engineering detail lives in [`research-implementation-2027.md`](research-implementation-2027.md)).
**Venues:** IROS 2027 (deadline Mar 1, 2027) primary; ICRA 2028 (~Sep 15, 2027) second; CoRL 2027 alternative for Study 3.
**Substrate:** BEHAVIOR-1K / 2026 BEHAVIOR Challenge (100 household tasks, ~6-minute mobile bimanual episodes, 200 teleop demos per task with skill-segment annotations, 40 official test instances per task, partial-credit Q-score), with π0.5 and GR00T N1.7 as the two policy families. Simulation only; no real-robot claims.
**Supersedes** the Sep-2026 `icra-iros-2027-*` documents (see banners there).

---

## 0. The research programme in one paragraph

Household-scale VLA policies now exist — the 2025 BEHAVIOR Challenge winner reaches Q ≈ 0.26, the runner-up ≈ 0.35 post-challenge — but the field understands *why they fail* only through hand-labeled case studies, understands *how to structure them hierarchically* only through isolated-skill or oracle-switched experiments, and understands *how to monitor them at runtime* only on tabletop tasks with binary success. The three studies below attack those three blind spots on the same substrate, the same 20 stratified tasks, and the same two policy families, so that each is a standalone paper and together they form a **diagnose → interface → recover** line. The unifying scientific stance: long-horizon failure is a *measurement* problem before it is a *method* problem, and the partial-credit Q-score on a 100-task benchmark is the first instrument fine-grained enough to measure it.

---

## 1. Where the field is (literature position, verified 2026-10-19)

A fresh search found that the BEHAVIOR-1K failure/recovery space produced at least six directly relevant papers in the last seven months. What they establish, and what they leave open:

| Established by prior work | By whom | What remains open |
|---|---|---|
| A 10-class expert failure taxonomy exists for BEHAVIOR-1K; the 2025 winner's failures are dominated by grasp failure, collision, and order/task confusion; trial-to-trial variance is large (policies tend to fully succeed or fully fail per instance) | Rasouli et al., *How VLAs (Really) Work* [2] — 500 hand-labeled episodes, one policy | Scale beyond hand labeling; more than one policy family; **what changes between instances the demos covered and instances they did not**; which failures are fixable without retraining |
| Skill-conditioned finetuning improves isolated skill execution (+6.0 π0.5, +5.6 GR00T) when each skill starts from a restored valid state; contact-rich skills are the shared bottleneck | Ma et al., *Behavior-Skill* [1] | The authors state they do not cover planning, decomposition, or "the sequential dependencies between constituent skills during execution" — i.e. **whether the gain survives closed-loop sequencing when the switching signal is imperfect** |
| Chaining π0.5 skill checkpoints with a VLM verifier works from clean snapshots (77–100 %) but stalls from chained states; failures attribute to readiness, grounding, execution | Rui et al., *Semantic Handoff* [3] | A controlled decomposition: how much of the stall is *what the policy is told* vs *when it is told to move on* |
| Learned stage tracking with voting (RLC [4]) and task-coverage scaling (Comet [5]) raise Q in the challenge | 2025 challenge reports | Neither measures instance-level generalization or attributes failures automatically |
| Subtask-level prompting helps on LIBERO-Long (+2.2) and RoboTwin (45→54 %, oracle switching) | AtomVLA [17], RoboTwin study [18], HiRoC [19], Anticipation-VLA [20], Steerable Policies [21] | All use oracle or learned switching without isolating when-to-switch from what-to-say |
| Runtime failure detectors: internal-feature probes beat K-sample action variance (SAFE [11]); OOD/density scores (FAIL-Detect [12], FIPER [13]); weight-perturbation disagreement beats stochastic sampling (PFD [14]); sample disagreement confounds multimodality (Diff-DAgger [15]); proactive backtracking + MBR decoding (CycleVLA [6]); corrective-data and retry schemes (CARE [7], FAR [8], AgentChord [9], ProAct-VLM [10]) | 2024–2026, all tabletop/LIBERO, binary success, recovery = stop/backtrack/ask | **Whether any detector converts into recovered partial credit on multi-minute mobile bimanual tasks**; which signal family wins for which failure class; whether detection quality and recovery value even rank the same way |
| Atomic vs. compositional generalization on real tabletop: strong atomic skill ≠ compositional success | ATOM-Bench [16] | Same question at house scale with controlled switching |

The honest consequence: none of our three studies can claim to be "first to look". Each claims to be the **first controlled measurement** of a quantity the above papers name but do not isolate.

---

## Study 1 — *Where the Demos End*: instance generalization and automatic failure attribution for household VLA policies

**Type:** measurement study with a released instrument. **Venue:** IROS 2027. **Priority: first — it produces the labeled substrate the other two consume.**

### 1.1 Research questions

- **RQ1.** When a household VLA policy moves from instances whose object/layout composition it saw in demonstrations to instances it did not, how much partial credit does it lose, and in which kinds of tasks?
- **RQ2.** Can failure be attributed *automatically* from privileged simulator state with agreement comparable to expert viewing, so that failure studies scale from hundreds to tens of thousands of episodes?
- **RQ3.** Which failure classes are recoverable by a planning-side intervention that does not touch the policy weights, and which are not?

### 1.2 Why these questions matter

The 2026 challenge evaluates on 40 pre-sampled instances per task that are disjoint from the 200 demo instances. Every leaderboard number therefore mixes two things: how well the policy learned the skills, and how far the test instances sit from the demo distribution. Nobody has separated them. If RQ1 shows a large, task-structure-dependent gap, that is evidence that household VLA performance is bottlenecked by *coverage* rather than *capacity* — which argues for demo-selection and instance-diversity work over bigger models. If the gap is small, the opposite.

RQ2 is methodological but load-bearing for the field: [2] needed eight experts for 500 episodes. The instrument that replaces them is what lets anyone run failure studies at the scale the matrices of Studies 2 and 3 require.

RQ3 is the question a practitioner actually has. "Grasp failure is the most common class" [2] does not say whether to collect more grasp data or to add a supervisor; the fixability map does.

### 1.3 Hypotheses and what each outcome would mean

| | Hypothesis | If confirmed | If refuted |
|---|---|---|---|
| H1 | Automatic attribution agrees with blind human labeling at κ ≥ 0.7 on the 10-class schema of [2] (≥ 0.8 after collapsing to 5 super-classes) | Failure studies become a software artifact; [2]'s taxonomy becomes a benchmark metric | Report where rules and humans disagree; the disagreement classes are themselves a finding about what is observable from state alone |
| H2 | Q drops ≥ 10 points (0–100) from seen-composition to unseen instances; the drop is ≥ 2× larger on multi-room / ≥ 4-skill tasks; it is dominated by navigation and semantic-confusion classes, not grasp failure | Generalization at house scale is a *where-am-I / which-object* problem, not a dexterity problem; coverage over layouts is the lever | Household VLA failure is mostly intrinsic skill failure; the data-scale story holds and Study 2's hierarchy question becomes more important |
| H3 | A no-retrain watchdog recovers ≥ 20 % of the Q lost to {abrupt termination, order confusion, navigation failure} and < 5 % of the Q lost to {grasp, placement, handling} | A clean planning-side vs. data-side split of the loss — the practitioner's answer | If the watchdog recovers nothing anywhere, planning-side supervision is not the lever at this policy quality; if it recovers grasp failures too, retries are a cheaper fix than data |

### 1.4 Study design (scientific level)

- **Tasks:** 20 of the 100, stratified by mined skill-plan structure into four strata × 5: single-room ≤ 3 skills; single-room ≥ 4 skills; multi-room navigation-heavy; bimanual/articulated-heavy. The stratification table is Table 1 and is the backbone of all three studies.
- **Policies:** multi-task finetunes of GR00T N1.7 and π0.5 on the same 20 tasks (same demos, same steps), plus the GR00T policy wrapped by the watchdog as the intervention arm. The organizer-provided single-task checkpoint is a sanity row.
- **Instance conditions:** *seen* = official instances whose object/layout composition matches demo episodes (via `task_instance_id` linkage; demo ids 1–200 and test ids 301–340 are disjoint by construction, so "seen" is composition overlap, stated explicitly); *unseen* = the hidden official instances.
- **Scale:** 20 tasks × 2 conditions × 10 instances × 3 arms = 1,200 full-length episodes, plus 200 repeated rollouts to quantify simulator nondeterminism (following [2]'s variance finding). ~1,400 episodes.
- **Attribution:** every step labeled from privileged state (goal predicates, poses, grasps, contacts, room occupancy) into the 10 classes of [2] with onset time and active skill; 300 episodes blind-labeled by two humans for κ.
- **Measures:** Q with bootstrap CIs and n per cell; per-skill success across 31 skills; horizon-decay (Q vs. skills remaining); ΔQ(seen − unseen) per stratum; failure-class frequency per arm × condition; the fixability map ΔQ(watchdog − none) per class; worst-case (min-over-task) Q; κ.

### 1.5 Expected figures

1. Stratum × condition Q bars for both policies (the generalization gap).
2. Failure-class frequency, seen vs. unseen, both policies — the "where the loss moves" plot.
3. Horizon-decay curves per stratum.
4. The fixability map: per class, Q recovered by the watchdog, with CIs.
5. Confusion matrix of automatic vs. human labels.

### 1.6 Contribution statement

1. The first instance-level generalization measurement on BEHAVIOR-1K for two policy families, with the instance axis defined by demo-composition overlap.
2. An automatic failure-attribution instrument validated against the expert schema of [2], released with ~1,400 labeled episodes (≈3× the largest existing hand-labeled set).
3. A fixability map separating planning-side from data-side Q-loss.

### 1.7 Threats to validity and how they are handled

- *Both policies collapse to ≈0 on unseen instances* → no separation. Caught by a 100-episode pilot before scaling; fallback is to add BDDL-resampled perturbation instances as an intermediate condition.
- *Weak finetunes* → use organizer checkpoints where available, shrink to a 10-task matrix, report as such.
- *"Incremental over [2]"* → the answer is scale, automation, the instance axis, two families, and the intervention arm; [2] is cited as the schema we validate against.
- *Simulator nondeterminism* → explicitly measured (200 repeats) and reported as a noise floor against which ΔQ is judged.

### 1.8 Scientific gates

- Nov 15: 5-task pilot shows ΔQ(seen − unseen) ≥ 5 on at least one policy and κ ≥ 0.6 on 50 episodes. Otherwise redesign the conditions before spending the matrix.
- Dec 31: κ ≥ 0.7 on the 300-episode audit.
- Jan 31: H2 holds directionally. If H3 is null the paper is instrument + generalization gap, still novel.
- Mar 1: submit.

---

## Study 2 — *What to Say vs. When to Switch*: does skill-level conditioning survive closed-loop sequential execution?

**Type:** controlled method study. **Venue:** IROS 2027 (co-submission) or ICRA 2028. **Priority: second — cleanest open gap.**

### 2.1 Research questions

- **RQ1.** Behavior-Skill [1] shows skill-conditioned training improves *isolated* skill execution. How much of that gain survives when skills are executed *in sequence* inside a full task, with the policy depending on an external signal to tell it which sub-goal is active?
- **RQ2.** In a hierarchical VLA system, which matters more: the *content* of the sub-goal instruction (what to say) or the *timing* of the switch between sub-goals (when to switch)?
- **RQ3.** Does skill-level conditioning buy compositional generalization — success on tasks never seen in training but composed of seen skills?

### 2.2 Why these questions matter

Every hierarchical VLA design — π0.5's own high/low split, Hi Robot, HAMSTER, the RoboTwin and AtomVLA ablations, Semantic Handoff's agent harness — rests on an unexamined assumption: that the interface between the "brain" (which sub-goal now) and the "cerebellum" (how to execute it) is cheap and the gain from decomposition flows through it. Behavior-Skill measured the gain with the interface removed (restored states, isolated skills). Semantic Handoff observed that chaining stalls but could not say why. Nobody has held the policy fixed and varied *only the switching signal*, so nobody knows whether hierarchical systems are limited by the low-level policy's language grounding or by the high-level progress estimate. RQ2 is that experiment. Its answer directs an entire line of work: if switching dominates, the field should invest in progress estimation and sub-goal verification; if content dominates, in richer command interfaces (Steerable Policies' direction).

RQ3 is the household-scale counterpart of ATOM-Bench's real-tabletop finding that atomic competence does not imply compositional success.

### 2.3 Hypotheses and what each outcome would mean

| | Hypothesis | If confirmed | If refuted |
|---|---|---|---|
| H1 | Skill-conditioned + oracle switching beats task-conditioned by ≥ 8 Q on ≥ 4-skill tasks and ≤ 3 on ≤ 3-skill tasks; the full-task gain is smaller than [1]'s isolated-skill gain | Decomposition pays, scales with horizon, and sequencing taxes it by a measurable amount — the first such number | Skill conditioning does not transfer from isolated to sequential execution; [1]'s result is an artifact of restored states — a strong negative result |
| H2 | Switching dominates: open-loop timing recovers < 30 % of the oracle gain, onboard predicate detectors ≥ 60 %, learned stage tracking in between; switch-time error explains ≥ 50 % of per-episode Q variance | The hierarchy's bottleneck is progress estimation; research effort should move to verification | Content dominates; even badly-timed sub-goals help; invest in command interfaces |
| H3 | On 5 held-out tasks composed of seen skills, skill-conditioned Q ≥ 2× task-conditioned Q | Skill conditioning is a compositional-generalization mechanism at house scale | Compositional reuse fails as in ATOM-Bench; held-out task structure is the hard part |
| H4 | Shuffled (wrong but well-formed) skill text ≈ task conditioning | The gain is semantic, not a prompt-length or format effect | The gain is partly a format effect — must be reported |

### 2.4 Study design (scientific level)

- **Tasks:** 15 training tasks (Study 1 strata minus the bimanual stratum) + 5 held-out tasks whose mined plans use only skills present in the training 15.
- **Conditioning variants (same base model, same demos, same steps):** task-instruction; per-segment skill instruction (from the dataset's skill annotations rendered through the 31-skill taxonomy); shuffled skill instruction (negative control); task + skill combined. GR00T N1.7 for all four; π0.5 replication on 5 tasks for policy-agnosticism.
- **Switching regimes (vary only the signal that tells the policy to advance):** oracle (privileged predicate flip); onboard predicate detectors with hysteresis; learned stage tracker with voting (RLC-style); open-loop demo-median budgets; VLM verifier (Semantic-Handoff-style) as a system baseline.
- **Scale:** 20 tasks × 10 unseen instances × 8 arms ≈ 1,600 full-length episodes + 150 for the π0.5 replication.
- **Measures:** Q; per-skill success; **switch-time error** relative to oracle (and precision/recall of advance events); horizon-decay per variant; held-out Q; and the decomposition **Q(skill, oracle) − Q(task) = interface value**, **Q(oracle) − Q(regime) = switching cost**, **1 − Q(oracle) = execution residual**.

### 2.5 Expected figures

1. Stacked decomposition bars per stratum: interface value / switching cost / execution residual.
2. Q vs. switch-time error scatter (the H2 variance plot).
3. Held-out compositional Q, skill vs. task conditioned (H3).
4. Horizon-decay curves per switching regime.

### 2.6 Contribution statement

1. The first closed-loop measurement of skill-conditioning gain in full household tasks, with the isolated-skill result of [1] as the premise.
2. The what-to-say / when-to-switch decomposition, isolating the hierarchy's bottleneck for the first time.
3. Compositional-generalization evidence at house scale.

### 2.7 Threats to validity

- *The finetuned VLA ignores the language field* (known for small-data finetunes) → a 5k-step pilot detects this cheaply; mitigations are the combined variant, fewer frozen layers, language dropout. If unresolved, that too is a reportable finding about current finetuning recipes.
- *Annotation coverage* — skill segments that map to no canonical skill are logged and reported.
- *"[1] already showed this"* → the abstract states [1]'s result as the premise and the closed-loop decomposition as the contribution.

### 2.8 Scientific gates

- Nov 10: skill-conditioned + oracle beats task-conditioned by any margin on a 5k-step pilot. Otherwise debug the prompt format before training four full variants.
- Jan 25: H1 directional on ≥ 10/15 tasks. If H2 inverts (open-loop ≈ onboard), that is the finding; retitle.
- Mar 1: submit IROS 2027, or ICRA 2028 if the matrix slips past Feb 1.

---

## Study 3 — *Do Runtime Failure Detectors Pay for Themselves at House Scale?* Detection quality vs. recovery value

**Type:** comparative empirical study. **Venue:** ICRA 2028 (IROS 2027 stretch). **Priority: third — conditional on a November pilot.**

### 3.1 Research questions

- **RQ1.** Runtime failure detectors for generative policies — internal-feature probes, OOD/density scores, weight-perturbation disagreement, action-sample dispersion, progress rules, VLM verifiers — have been ranked by detection quality on tabletop tasks. Do those rankings hold on multi-minute mobile bimanual tasks, and at what lead time?
- **RQ2.** When each detector drives the *same* recovery ladder (retry → replan → search → safe pose) under the same budget, how much partial credit does each recover, and on which failure classes?
- **RQ3.** Is detection quality (AUROC) even the right thing to optimise? Or do lead time and false-alarm cost reorder the families once recovery is attached?

### 3.2 Why these questions matter

The failure-detection literature evaluates detectors as classifiers and stops: the downstream action is "stop" or "ask for help", and the benchmark is binary success on a 30-second task. In a six-minute household task the robot cannot stop — it has partial credit to protect and time budget to spend — and the detector's value is the Q it recovers minus the Q its false alarms destroy. That quantity has never been measured. If RQ3 shows detection and recovery rankings diverge, the field has been optimising the wrong metric. If a cheap policy-agnostic signal recovers as much as an expensive internal-feature probe, that changes deployment choices. And if nothing recovers grasp-class failures (Study 1 H3), the boundary of what runtime monitoring can do at this policy quality is established.

This study deliberately makes **no claim to a new detector**. Medoid / MBR decoding [6] and K-sample dispersion [11, 15] are included as known baselines.

### 3.3 Hypotheses and what each outcome would mean

| | Hypothesis | If confirmed | If refuted |
|---|---|---|---|
| H1 | Internal-feature and perturbation detectors reach AUROC ≥ 0.80 at 5 s lead on held-out tasks; dispersion alone ≤ 0.70 (replicating [11, 14] at house scale) | Tabletop detector rankings transfer to long-horizon | Long-horizon failure has different signatures; a result in itself |
| H2 | At least one predictive trigger beats the post-hoc progress rule by ≥ 3 Q and no-intervention by ≥ 6 Q at < 15 % false alarms, with gains concentrated in navigation / order / termination classes and ≈ 0 in grasp / placement | Prediction has measurable value beyond post-hoc rules, and its limits are the data-side classes | The post-hoc rule already captures the recoverable Q; prediction does not pay at this policy quality — a publishable boundary |
| H3 | Detection rank ≠ recovery rank | AUROC is the wrong objective for deployable monitors; propose recovered-Q as the metric | AUROC is a fine proxy; simpler story |
| H4 | Policy-agnostic features transfer GR00T → π0.5 with < 5 AUROC loss; internal-feature probes do not without retraining | A practical case for cheap signals in multi-policy fleets | Internal features generalise across architectures — surprising and interesting |

### 3.4 Study design (scientific level)

- **Detector families:** post-hoc progress rule (the existing watchdog); K-sample flow-matching dispersion + chunk-boundary disagreement; internal-feature probe (SAFE-style); low-rank weight-perturbation disagreement (PFD-style); progress-velocity + proprioceptive deviation features; VLM verifier; privileged oracle as upper bound. Medoid chunk selection as a zero-trigger baseline.
- **Labels:** failure onsets from Study 1's attribution, with lead windows of 3 / 5 / 10 s; successes and clean timeouts as negatives; sensitivity over window definitions.
- **Recovery ladder:** identical for every trigger — retry with fresh noise → replan remaining sub-goals → frontier search → safe pose; identical budget cap; threshold per family set at 10 % false alarms on the pilot. This makes the comparison *trigger-matched by construction*: differences in recovered Q are attributable to the trigger alone.
- **Scale:** detectors trained on Study 1's ~1,400 episodes (features logged during that matrix at zero extra simulation cost); intervention matrix 20 tasks × 10 unseen instances × 9 arms ≈ 1,800 episodes; GR00T → π0.5 transfer on 5 tasks.
- **Measures:** AUROC / AP vs. lead (task-grouped cross-validation); false alarms per hour; lead-time distribution; per-step latency; ΔQ per arm with CIs; ΔQ per failure class; worst-case Q; transfer loss.

### 3.5 Expected figures

1. Detection rank vs. recovery rank (the H3 plot — two orderings side by side).
2. ΔQ per detector family per failure class, heatmap.
3. Lead-time vs. recovered-Q curves.
4. Transfer: AUROC on GR00T vs. π0.5 per feature family.

### 3.6 Contribution statement

1. The first measurement of runtime failure detectors in recovered partial credit on multi-minute household tasks.
2. A trigger-matched protocol that separates detector value from ladder value.
3. Evidence on whether detection quality predicts recovery value — and a proposed metric if it does not.

### 3.7 Threats to validity

- *The post-hoc rule already recovers most recoverable Q* → H2 null; caught by a 100-episode probe before the matrix; H1 / H3 / H4 remain a comparison paper.
- *Flow samples collapse after finetuning* (low dispersion everywhere) → documented; feature and perturbation families carry.
- *"SAFE / CycleVLA did this"* → neither measured recovered Q at house scale or compared families under one ladder; stated with their numbers.

### 3.8 Scientific gates

- Nov 8: at least two detector families separate eventually-failed from successful episodes on 60 pilot episodes (Mann-Whitney p < 0.01). Otherwise stop; fold the medoid result into Study 1.
- Dec 31: best family AUROC ≥ 0.70 at 3 s on held-out tasks *and* a predictive trigger beats the rule on a 5-task probe. Otherwise a detector-comparison workshop paper.
- Feb 10: H2 on ≥ 12/20 tasks → IROS 2027; else ICRA 2028 with the full matrix.

---

## 4. How the three studies fit together

| | Study 1 | Study 2 | Study 3 |
|---|---|---|---|
| Question | What fails, where, and is it fixable without retraining? | Is the hierarchy limited by what it says or when it switches? | Does predicting failure recover partial credit? |
| Role | Diagnose | Interface | Recover |
| Shared | 20 stratified tasks, two policy families, official instances, Q-score, automatic attribution | | |
| Feeds | Labels → Study 3 training; strata → Studies 2, 3; fixability map → Study 3's expected ceiling | Onboard-detector switching → Study 3's progress-rule arm | — |
| Standalone claim | yes | yes | yes |

Each paper must stand without the others; the shared protocol halves the writing cost of the second and third.

## 5. Open questions for the team to debate before committing

1. **Which paper leads?** Study 1 produces the substrate but is the most exposed to "incremental over [2]". Study 2 has the cleanest named gap (Behavior-Skill's own future-work sentence). An argument exists for leading with Study 2 and treating Study 1's labeler as a tool paper or appendix.
2. **Is Q the right dependent variable?** [2] proposes progress-aware adjustments to Q. Adopting one of them alongside official Q would strengthen all three studies and align with the only existing failure study.
3. **Instance definition.** If demo-composition instances are not evaluable (Appendix A3), "seen vs. unseen" degrades to "public vs. hidden" and Study 1 H2 weakens. Decide the fallback framing now.
4. **Second policy family.** π0.5 replication on 5 tasks is the minimum for policy-agnosticism claims; full replication doubles compute. Decide per study.
5. **Backup direction** if a gate fails: *"Which 50 demos?"* — demo-selection for long-horizon finetuning (budget × diversity rule × 5 tasks). Comet [5] scaled tasks; nobody has scaled demos-per-task with a layout-diversity criterion on BEHAVIOR.

Re-run the literature check (§1) in January 2027 before drafting.

---

## References (verified 2026-10-19)

1. Ma, C. et al. *Behavior-Skill: A Fine-Grained Benchmark for Evaluating Vision-Language-Action Policies in Long-Horizon Tasks.* arXiv:2608.30536, Aug 2026. https://arxiv.org/abs/2608.30536
2. Rasouli, A. et al. (Huawei Noah's Ark). *How VLAs (Really) Work In Open-World Environments.* arXiv:2604.21192, Apr 2026. https://arxiv.org/abs/2604.21192
3. Rui, K. et al. (SimpleAI). *Diagnosing Semantic Handoff Failures in Agent-Orchestrated Vision-Language-Action Skill Composition.* arXiv:2607.06256, Jul 2026. https://arxiv.org/abs/2607.06256
4. Larchenko, I., Zarin, G., Karnatak, A. *Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge.* arXiv:2512.06951, Dec 2025. https://arxiv.org/abs/2512.06951
5. Bai, J. et al. (NVIDIA). *Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge.* arXiv:2512.10071, Dec 2025. https://arxiv.org/abs/2512.10071
6. Ma, C. et al. (Oxford). *CycleVLA: Proactive Self-Correcting Vision-Language-Action Models via Subtask Backtracking and Minimum Bayes Risk Decoding.* arXiv:2601.02295, Jan 2026. https://arxiv.org/abs/2601.02295
7. *CARE: Experience-Guided Atomic Corrective Execution for Vision-Language-Action Policies.* arXiv:2609.24118, Sep 2026. https://arxiv.org/abs/2609.24118
8. Hao, H. et al. (CMU). *FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement.* arXiv:2607.01111, Jul 2026. https://arxiv.org/abs/2607.01111
9. *From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation (AgentChord).* arXiv:2605.11951, May 2026. https://arxiv.org/abs/2605.11951
10. *ProAct-VLM: Pre-Failure Vision-Language Task Replanning with Continuous Perception Feedback.* arXiv:2609.37681, Sep 2026. https://arxiv.org/abs/2609.37681
11. Gu, Q. et al. *SAFE: Multitask Failure Detection for Vision-Language-Action Models.* arXiv:2506.09937, Jun 2025. https://arxiv.org/abs/2506.09937
12. Xu, C. et al. (TRI). *Can We Detect Failures Without Failure Data? (FAIL-Detect).* arXiv:2503.08558, Mar 2025. https://arxiv.org/abs/2503.08558
13. *FIPER: Failure Prediction at Runtime for Generative Robot Policies.* arXiv:2510.09459, Oct 2025. https://arxiv.org/abs/2510.09459
14. Lee, Y., Har, D. (KAIST). *Perturbation-Based Epistemic Uncertainty for Failure Detection in Vision-Language-Action Models (PFD).* arXiv:2606.20754, Jun 2026. https://arxiv.org/abs/2606.20754
15. Lee, S.-W., Kang, X., Kuo, Y.-L. *Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation.* arXiv:2410.14868. https://arxiv.org/abs/2410.14868
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
26. 2025 BEHAVIOR Challenge leaderboard. https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html

Background on venues, reviewer expectations, and the broader VLA / sim-to-real / hierarchy literature (63 sources through Sep 2026): `docs/icra-iros-2027-trends.md`.
