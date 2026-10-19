# Conditions for Scientific Success

**Date:** 2026-10-19. Companion to [`research-plan-2027.md`](research-plan-2027.md). This document states what must be *scientifically true* for each of the three studies to yield a publishable result — the validity of the questions, instruments, comparisons, and evidence — and what the team must understand before running anything. It contains no engineering or operations; those are in `research-implementation-2027.md` and are the team's business.

---

## 1. Programme-level conditions

A study in this programme is publishable if and only if all seven hold. Check each study against them before committing compute.

| # | Condition | Meaning | How we know it holds |
|---|---|---|---|
| C1 | **The question is open and named** | A specific published paper either poses the question as future work or stops short of it in a way we can quote | Plan §1 positioning table: one row per study with the quote or the omission |
| C2 | **The claim is falsifiable and pre-stated** | Each hypothesis has a direction, an effect size, and a stated meaning for both confirmation and refutation | Plan hypothesis tables; hypotheses written down before the matrix runs |
| C3 | **The instrument measures the construct** | The dependent variable actually captures what the claim is about (construct validity) | §2 per study |
| C4 | **The comparison isolates the cause** | Arms differ in exactly one factor; every confound is held fixed or measured (internal validity) | §2 per study |
| C5 | **The design can detect the effect** | The minimum effect that would matter is larger than the design's noise floor at the stated n (statistical power) | §3 |
| C6 | **The scope is stated and honest** | Simulation-only, two policy families, 20 stratified tasks; no claim leaks beyond its evidence | Abstract and limitations drafted first, not last |
| C7 | **The contribution survives the desk screen** | Reviewer can name what is new relative to the five closest papers in one sentence; a released instrument or dataset backs it | Contribution statements in the plan; artifact named per study |

---

## 2. Per-study scientific conditions

### Study 1 — Where the Demos End

**The construct.** The claim is about *distribution shift from demonstrated to novel instance compositions*. For the instrument to be valid:

- "Seen" must mean *composition overlap with the demonstrations* (which objects, which layout, which rooms) — not instance id, which is disjoint by construction. If composition overlap cannot be established, the axis degrades to "public vs. hidden official instances", which measures something weaker (sampling variance within the test distribution) and the claim must be rewritten accordingly. **This decision determines whether Study 1 is a generalization paper or a variance paper.**
- Partial-credit Q must be the right dependent variable for *where* failure happens. Q is progress-agnostic: it scores final predicate states, not the path. Rasouli et al. [2] show this hides order violations and lucky end-states. Report official Q (comparability with the leaderboard) **and** a progress-aware variant alongside it, so the attribution does not reward accidental completion.
- The failure taxonomy must be *the field's*, not ours. Adopting the 10 classes of [2] makes the automatic labeler a measurement of an established construct rather than a new, unvalidated one. κ against human labels on that schema is the validity test; without it, the labeler is an opinion.

**The comparisons that must be clean.**
- Seen vs. unseen must hold policy, checkpoint, task, time budget, and serving path fixed; only the instance changes.
- Watchdog vs. none must be the *same weights*; the wrapper is the only difference. Otherwise the fixability map confounds intervention with policy.
- The two policy families must be finetuned on identical demos for identical steps; otherwise "both families show X" is a data-size effect.

**The theory the result must speak to.** If the gap is dominated by navigation and semantic-confusion classes on multi-room tasks, the interpretation is that household VLA failure at this quality is a *coverage* problem (which layouts the demos spanned), which is a distinct claim from Behavior-Skill's *capacity* finding (contact-rich skills are the bottleneck). The paper must engage with that disagreement directly: both can be true at different horizons, and the horizon-decay curve is the figure that reconciles them.

**Result that makes the paper:** a stratum-dependent gap with an attribution profile that shifts between conditions, plus a fixability map that separates planning-side from data-side loss.
**Result that still makes a paper:** no gap, but a validated automatic labeler at 3× the scale of [2] with a published confusion analysis of what is and is not observable from state.
**Result that kills it:** the labeler cannot reach κ ≥ 0.6 even on collapsed classes. Then the instrument is invalid and nothing downstream is trustworthy. This is why κ is the first gate.

**What the team must understand deeply before starting:** [2] (its taxonomy, its variance finding, its progress-aware metric proposals); the BDDL goal-predicate semantics and how Q is computed from them; what `task_instance_id` actually encodes about an instance.

---

### Study 2 — What to Say vs. When to Switch

**The construct.** The claim is about *the interface between sub-goal selection and execution* in hierarchical policies. Validity requires:

- The *only* manipulated factors are (a) the language the low-level policy is conditioned on and (b) the signal that advances to the next sub-goal. The policy architecture, demos, training steps, and action chunking must be identical across variants. If the skill-conditioned variant also sees shorter prompts or more frequent re-queries, the effect is confounded — hence the shuffled-text control (H4) and re-query timing held fixed across regimes.
- "Oracle switching" must be a true upper bound: the privileged predicate flip for the active skill's exit condition, applied instantly. Anything less (e.g. a delayed or noisy oracle) makes the decomposition uninterpretable.
- The decomposition identity *interface value + switching cost + execution residual = 1 − Q(task-conditioned)* only holds if the arms are nested as designed. Verify the arithmetic on the pilot before trusting it on the matrix.

**The comparisons that must be clean.**
- Behavior-Skill's +6 isolated-skill gain [1] is the *premise*, so our task set should overlap theirs where possible and our skill taxonomy should map onto their 34 categories, so the isolated-vs-sequential comparison is like-for-like.
- Switching regimes are compared *at the same policy*; the VLM-verifier arm is a system baseline, labeled as such, not a controlled factor.
- Held-out compositional tasks must be verified to contain *only* skills present in training; one novel skill contaminates H3.

**The theory the result must speak to.** Hierarchical VLA work assumes decomposition gains flow through the interface. If H2 holds (switching dominates), the field's bottleneck is progress estimation — a perception/verification problem, not a language-grounding one — and work like Steerable Policies [21] is optimising the wrong side. If content dominates, the opposite. The discussion must state which research line each outcome favours; that is the paper's value beyond the numbers.

**Result that makes the paper:** a clean decomposition with a clear dominant term and a horizon dependence.
**Result that still makes a paper:** skill conditioning does *not* survive sequencing — a negative result that qualifies [1] and all oracle-switched ablations [17–21].
**Result that kills it:** the finetuned policy ignores the language field entirely, so no variant differs. This is a known failure of small-data VLA finetunes and must be ruled out on a cheap pilot before any further training; if it cannot be fixed, the study is not runnable on these policies.

**What the team must understand deeply before starting:** [1] in full (its evaluation protocol from restored states, exactly what it did not measure); [3] (how its verifier decides advance/retry/replan and why chaining stalls); [4]'s stage-tracking mechanism; the hierarchical-VLA ablations [17–21] and which switching signal each used.

---

### Study 3 — Do Runtime Failure Detectors Pay for Themselves

**The construct.** The claim is about *recovery value*, not detection quality. Validity requires:

- The dependent variable is **recovered Q**: Q(detector-triggered ladder) − Q(no intervention), net of false-alarm damage. AUROC is a secondary, explanatory variable. A study that reports AUROC and infers value has not measured the construct.
- The recovery ladder must be *identical* across triggers, with identical budgets and identical thresholds-at-fixed-false-alarm-rate. Then and only then is ΔQ attributable to the trigger. This is the trigger-matched protocol; any per-family tuning of the ladder breaks it.
- "Failure onset" labels must come from Study 1's validated attribution with a stated lead window; the lead-window sensitivity table is part of the result, not an appendix, because every detector's AUROC depends on it.

**The comparisons that must be clean.**
- Detector families must include the field's current best (internal-feature probe [11], perturbation disagreement [14]), the cheap policy-agnostic signals (dispersion, progress), the rule the system already has, a VLM verifier, and an oracle. Omitting the strong ones makes the comparison uninteresting; omitting the oracle leaves the ceiling unknown.
- Known zero-trigger improvements (medoid / MBR decoding [6]) are baselines, not contributions. Claiming them is a desk-reject.
- Held-out *tasks* for detector evaluation; detectors that only generalise across episodes of seen tasks are not deployable and the field knows it [11].

**The theory the result must speak to.** The failure-detection literature optimises classification metrics with "stop or ask" as the only action. In long-horizon tasks the robot must keep going; the detector's worth is the partial credit it protects. If detection and recovery rankings diverge (H3), the field's metric is wrong and the paper proposes recovered-Q as the replacement. If they agree, the paper is a clean replication at a new scale plus the first per-class recovery map.

**Result that makes the paper:** a predictive trigger recovering Q beyond the post-hoc rule, with class-resolved limits consistent with Study 1's fixability map.
**Result that still makes a paper:** no trigger beats the post-hoc rule — a boundary result that says prediction does not pay at this policy quality — provided the detector comparison itself is thorough.
**Result that kills it:** no detector family separates failed from successful episodes at all on the pilot. Then there is nothing to recover from and the study folds into Study 1.

**What the team must understand deeply before starting:** SAFE [11] (features, conformal thresholds, and its own finding that sample variance is weak); PFD [14]; Diff-DAgger's multimodality argument [15]; CycleVLA [6] (what MBR does and does not do); the existing watchdog's trigger logic, since it is the control arm.

---

## 3. Statistical power — what the designs can and cannot detect

Assume per-episode Q has standard deviation ≈ 0.30 (consistent with [2]'s bimodal "fully succeed or fully fail" finding). Then, for a difference of means between two arms:

| Comparison level | Episodes per arm | Standard error of ΔQ | Smallest reliably detectable ΔQ (≈ 2 SE) |
|---|---|---|---|
| Whole matrix (20 tasks × 10 instances) | 200 | ≈ 0.030 | **≈ 6 points** |
| One stratum (5 tasks × 10 instances) | 50 | ≈ 0.060 | **≈ 12 points** |
| One task (10 instances) | 10 | ≈ 0.13 | ≈ 27 points — per-task claims are not supportable |

Consequences for the hypotheses:
- Whole-matrix claims of ≥ 6–8 points (Study 2 H1, Study 3 H2 "≥ 6 over none") are detectable. The "≥ 3 over the post-hoc rule" margin in Study 3 H2 is **at the edge**; it needs either paired analysis (same instances across arms — which the design provides, roughly halving the SE) or more instances on the subset where the rule and the predictor disagree.
- Stratum-level claims (Study 1 H2 "≥ 2× larger on multi-room tasks") require stratum differences of ≥ 12 points or paired analysis across conditions on the same instances. State this; do not claim stratum ordering on point estimates alone.
- Per-task results are descriptive only (the horizon-decay curve, the worst-case Q); never test them.
- Simulator nondeterminism must be measured (repeat rollouts) and subtracted conceptually: an effect smaller than the repeat-to-repeat spread on the same instance is not an effect.
- Pairing: every arm runs on the same instance set, so analyse ΔQ *per instance* and bootstrap over instances. This is the single largest free gain in power the design offers; use it.

---

## 4. The evidence standard every claim is held to

1. **Numbers in the abstract have a table.** Each has n, CI, and the condition it was measured in.
2. **Every "better than" names the baseline and the matching.** Same weights, same budget, same instances, same serving path — or the difference is stated.
3. **Every hypothesis is reported**, confirmed or not, in the order it was pre-stated.
4. **The failure table is in the paper**, not the supplement. Reviewers at these venues read it as a strength.
5. **Prior results are reproduced where they are the premise.** Study 2 should show [1]'s isolated-skill gain on our tasks before showing what sequencing does to it; Study 3 should show [11]'s variance-vs-features ordering before showing recovery.
6. **Scope sentences are in the abstract.** "In simulation, on 20 BEHAVIOR-1K tasks, for π0.5 and GR00T N1.7."
7. **A claim not supported at the detectable effect size (§3) is downgraded to an observation**, in the text, by us, before a reviewer does it.

---

## 5. Intellectual risks (not operational ones)

| Risk | Which study | Why it threatens validity | Guard |
|---|---|---|---|
| Construct drift: measuring public-vs-hidden variance and calling it generalization | 1 | Overclaims distribution shift | Decide the instance definition first; name it in the title if weak |
| Q rewards accidental end-states | 1, 2, 3 | Attribution and recovery credited to luck | Report a progress-aware Q variant alongside official Q |
| Labeler validated only against our own intuitions | 1 | Circular instrument | κ against human labels on an external schema [2] |
| Conditioning confounded with prompt format or re-query frequency | 2 | Interface value overstated | Shuffled-text control; fixed re-query cadence across arms |
| Oracle switching not a true upper bound | 2 | Decomposition arithmetic breaks | Verify Q(oracle) ≥ every other regime on the pilot |
| Held-out "compositional" tasks contain an unseen skill | 2 | H3 contaminated | Audit mined plans skill-by-skill |
| Detector evaluated on seen tasks | 3 | Inflated AUROC, non-deployable | Task-grouped cross-validation only |
| Ladder tuned per detector | 3 | ΔQ no longer attributable to the trigger | Freeze ladder and budgets before any detector exists |
| Claiming medoid/MBR or dispersion as novel | 3 | Desk reject with citation | Positioned as baselines in the plan; keep it so |
| Being scooped on the exact framing | all | Novelty gone | Weekly literature watch; Jan 10 full re-check; ability to re-position each study around its *measurement* rather than its *topic* |
| Both policies at Q ≈ 0 on hard strata | all | Floor effect; no variance to explain | Stratification includes easy strata; report where the floor is as a finding |
| Over-interpreting two policy families as "VLAs in general" | all | External validity | Say "two families"; replication on the second is the minimum |

---

## 6. Scientific decision points

These are the moments where the *science* changes course (operational dates are in the implementation appendix).

| Decision | Trigger | Options |
|---|---|---|
| Is Study 1 a generalization study or a variance study? | Whether seen-composition instances are evaluable | Keep the strong framing / rewrite H2 and the title |
| Is the labeler an instrument or an opinion? | κ on the first 50 human-labeled episodes | Proceed / refine rules / collapse classes / stop |
| Does the policy read the language at all? | Pilot: skill- vs task-conditioned differ under oracle switching | Proceed / change conditioning recipe / Study 2 not runnable on these policies |
| Does any detector see failure coming? | Pilot separation test | Proceed / fold Study 3 into Study 1 |
| Does prediction beat the post-hoc rule? | Probe on 5 tasks | Full study / boundary-result paper / detector-comparison only |
| Which study leads the IROS submission? | Jan 10 literature re-check + pilot effect sizes | Study 1 / Study 2 / both |
| Is official Q enough? | Pilot attribution shows accidental completions | Add progress-aware variant as co-primary |

---

## 7. Required background (read, not skim)

Per person, before touching their study. Each reading ends with a one-page note in `docs/research/reading/` answering: what did they measure, what did they explicitly not measure, which of our hypotheses does it bear on.

- **Everyone:** [2] Rasouli et al.; [1] Behavior-Skill; [25] BEHAVIOR-1K; the 2025 challenge reports [4][5]; `docs/challenge-spec.md` §5–6 (how Q is computed).
- **Study 1:** the BDDL predicate families used by the 20 tasks; [22] (privileged-state labeling precedent); [16] ATOM-Bench's AS/CFS metrics as a comparison point for "skill vs. composition" attribution.
- **Study 2:** [3] Semantic Handoff; [17]–[21] hierarchical conditioning ablations, noting the switching signal each used; [4] stage tracking.
- **Study 3:** [11] SAFE, [12] FAIL-Detect, [13] FIPER, [14] PFD, [15] Diff-DAgger, [6] CycleVLA, [7]–[10] recovery schemes — with a table of task horizon, success metric, and recovery action for each (this table becomes Related Work).

---

## 8. What success looks like in March

Two papers whose abstracts a reviewer in this area cannot summarise as "they ran π0.5 and GR00T on BEHAVIOR and reported numbers", because each contains a measurement that did not exist before: a validated instrument and an instance-generalization gap; a decomposition of hierarchy loss into content and timing. A third study either queued for ICRA 2028 with a pilot result in hand, or honestly closed with a boundary result. Every hypothesis — confirmed or refuted — reported as pre-stated. Nothing claimed that the design could not detect.
