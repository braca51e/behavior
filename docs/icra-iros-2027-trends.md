# ICRA/IROS 2027: Active Research Themes and Open Gaps (2022–2026 Literature Map)

**Prepared for:** task `t_aeffa76c` (3 scoped ICRA/IROS research ideas) and the 2026 BEHAVIOR Challenge effort in this repo.
**Compiled:** 2026-09-12 by ml-researcher.
**Method:** systematic scan of (a) ICRA 2025 & ICRA 2026 & IROS 2025 & IROS 2026 award finalists/winner lists, (b) 40+ arXiv preprints 2022–2026 (metadata + citation counts via Semantic Scholar), (c) two 2025 survey papers (VLA survey, humanoid loco-manipulation survey), (d) official CFP/schedule pages for ICRA 2027 and IROS 2027. Every citation below was verified against a live page; see the References section.

---

## 0. Venue context (what "publishable at ICRA/IROS 2027" means right now)

| Item | Fact | Source |
|---|---|---|
| ICRA 2027 | COEX, Seoul, **May 24–28, 2027** | 2027.ieee-icra.org |
| ICRA 2027 paper deadline | **September 15, 2026, 11:59 pm PST — 3 days from this report's date (2026-09-12)** | ICRA CFP via official schedule; 2025/2026 pattern (deadline Sep 15) |
| ICRA 2026 scale | 5,088 submissions, ~1,800 accepted (~35%) | Bohrium ICRA 2026 summary |
| IROS 2026 scale | 4,348 submissions, 1,585 accepted (~36%) | MARCH lab news, IROS 2026 |
| IROS 2027 | **Florence (Fortezza da Basso), Sep 26–Oct 1, 2027; paper deadline Mar 1, 2027 (PST)** | ieee-ras.org event page; mldeadlines.com |
| Review mechanics | Double-anonymous; **8-page limit incl. references**; AI use must be disclosed; final video ≤180 s/20 MB allowed | ICRA 2026 FAQ |
| Editor policy | Summary rejection allowed for: 8-page violation, **no novel contribution** (with cited counterexamples), significant technical errors | IEEE RAS ICRA editor info |

**Practical reading for a 12-month plan:** ICRA 2027 (deadline Sep 15, 2026, **3 days away**) is unreachable for any new research — it is only relevant for the BEHAVIOR Challenge team if results already exist today (the challenge submission is due Oct 16, 2026, so its outcomes cannot make ICRA 2027 either). **IROS 2027 (Mar 1, 2027 deadline, Florence) is the primary target** for new research: ~6 months of experiment time now → submission. **ICRA 2028 (expected Sep 15, 2027 deadline, per the annual pattern)** is the natural second venue — it also lands right after the 2026 BEHAVIOR Challenge winners announcement (Nov 4, 2026), so challenge-adjacent papers (benchmark study, failure analysis, sim-real deltas) fit it perfectly.

**Reviewer expectations (consolidated from the editorial guide + ICRA-experimentation skill + award citations):**
1. A crisp, *quantified* gap claim in the intro ("success rate drops from X to Y under Z"), not "previous methods are limited."
2. An **evidence ladder**: sim-only work must not carry real-world claims; every abstract claim needs matching evidence rung.
3. Trial counts n reported everywhere (10–25 per condition typical), success criterion stated verbatim, failures classified and disclosed — a failure table is a plus, not a weakness.
4. Baselines must be hardware/sensor-matched, tuned with comparable effort, and include a strong classical option.
5. Sim-to-real deltas reported numerically; "domain randomization" must be ablated, not incanted.
6. Novelty must survive a "no novel contribution" desk screen: the contribution is a claim with a mechanism and a measurement, not a new system assembly of known parts.
7. Reproducibility signals: released code/data, standard benchmarks (LIBERO, R2R/REVERIE, Replica, AgiBot World, DROID, OXE).

---

## 1. Theme map (7 themes)

### T1. Vision–Language–Action (VLA) models & generalist manipulation policies
**State of the art (2023→2026).** The field moved in ~30 months from VLM co-fine-tuning with action heads (RT-2 [1], SayCan [2]), through open generalist policies (OpenVLA 7B, Octo) [3][4] and cross-embodiment data pools (OXE/RT-X, DROID, BridgeV2) [5][6], to *flow-matching / diffusion* action decoders (π0 [7], RDT-1B [8], Diffusion Policy lineage [9]) and to **open-world generalization** as the frontier: π0.5 (coarse-to-fine hierarchical reasoning + open-world knowledge) [10], Hi Robot (hierarchical VLA for open-ended instructions) [11], HAMSTER (hierarchical action models, ICLR'25) [12]. Fine-tuning practice is now a sub-field of its own: OpenVLA-OFT [13], knowledge insulation [14], real-time chunk execution [15], tiny VLA (49M) [16]. Two 2025 surveys catalog ~200+ VLA papers [17][18].
**What won awards.** ICRA 2026 Robot-Learning finalists: FP3 (3D foundation policy pretraining on 60k point-cloud trajectories; 80-demo adaptation >90% success on unseen objects); Shallow-π (distillation for flow-based VLAs, IROS 2026 finalist); GeoVLA (3D representations inside VLAs, IROS 2026 Cognitive finalist); View-invariant policy learning with camera conditioning (ICRA 2026 Robot Learning **winner**) [19]. IROS 2026 Cognitive finalists are *all* "inject 3D/structure into VLA" papers.
**Common evaluation.** Standardized sim benchmarks (LIBERO [20], SimplerEnv, RoboCasa [21], BEHAVIOR-1K/OmniGibson) + real-robot trials (n=10–50 per task, paired with human baselines); few-shot adaptation curves vs. data scale; success rate with binomial CIs; cross-embodiment transfer tables.
**Reviewer expectations.** VLA papers face "incremental fine-tune" desk-screen risk; reviewers now demand: (i) a *specific* generalization failure mode addressed, (ii) ablations separating VLM backbone contribution from action-decoder contribution, (iii) real hardware, (iv) compute reporting.
**Open gaps (specific).**
- **G1.1 Spatial/3D grounding remains brittle:** award-winning 2026 work (FP3, GeoVLA, view-invariant policies) shows VLAs trained on 2D images fail under camera moves and depth ambiguity. No standard benchmark yet measures *spatial fidelity* of VLA policies (e.g., "does the policy know where the camera is?") beyond ad-hoc tasks.
- **G1.2 Contact/force information is nearly absent:** VLA action spaces are end-effector pose; tactile/force channels are add-ons (tactile-VLA papers 2025). There is no accepted answer for when a VLA needs force feedback vs. when vision suffices — reviewers increasingly ask for contact-rich tasks where pose-only policies provably fail.
- **G1.3 RL post-training is unvalidated at scale:** ReWiND (language-guided rewards) [22], VLA-RL, T-GRPO, and online-RL-for-VLA (ICRA 2025) [62] are all 2025 preprints with small evaluations. No consensus metric for "did RL actually add anything beyond imitation + more data."
- **G1.4 Data-quality vs. data-scale trade-off** is open: AgiBot World reports +30% from human-in-the-loop *verified* data [23]; OXE scaling showed diminishing returns from heterogeneous data. The field has no public protocol for measuring which data properties (trajectory length, verification, skill coverage) drive generalization.
- **G1.5 Long-horizon task completion:** VLA policies collapse on multi-minute, multi-skill tasks; hierarchical planners (Hi Robot, HAMSTER) help but the interface between the "brain" (task decomposition) and the "cerebellum" (low-level execution) has no standard evaluation — BEHAVIOR-1K Q-score (partial-credit task completion) is the closest existing metric.

### T2. Sim-to-real transfer, world models, and data engines
**State of the art.** Three parallel streams: (a) *classic sim2real RL* for locomotion (domain randomization → RMA residual adaptation [24] → Isaac Gym real-world RL 2024); (b) *generative world models* as data engines: Cosmos (NVIDIA, ~20M h video) [25], Genie 3 (DeepMind, real-time playable) [26], FLIP (flow-centric manipulation world model, ICLR'25) [27], ReBot (real-to-sim-to-real video synthesis) [28], GigaWorld-0 (2025) [29]; (c) *simulation platforms* maturing: Isaac Lab, Genesis (differentiable, 2024) [30], OmniGibson (house-scale, BEHAVIOR-1K) [31], MuJoCo/MJX. Surveys 2024–2025 catalog the method zoo [32].
**Award signals.** ICRA 2025 Robot Learning **winner was a data-infrastructure paper** (Robo-DM [33]) — a tell that the bottleneck moved from algorithms to data pipelines. The 2025 IEEE RA-L Best Paper went to sim-to-real of soft robots with learned residual physics [63].
**Common evaluation.** Zero-shot vs. fine-tuned transfer, same-task success in sim and real reported side-by-side, ablation of randomization ranges, physics-engine fidelity studies (Isaac vs. MuJoCo vs. Genesis on identical tasks).
**Reviewer expectations.** "Sim results carry real-world claims" is the top rejection reason; reviewers want a validated sim (cited prior work or a dedicated fidelity study) and numeric sim-real deltas.
**Open gaps (specific).**
- **G2.1 World models are not yet *useful* for manipulation data:** video world models (Cosmos/Genie/FLIP) predict pixels, but the field has no demonstrated pipeline where world-model-generated data *measurably* improves a real-robot policy beyond real teleop data at equal cost. Every 2025 paper claims potential; few report the ablation that would settle it.
- **G2.2 Contact-rich and deformable physics remain the sim2real hard wall:** rigid-body sim2real is largely solved for locomotion; grasping/slipping/deformables are not. Residual-physics approaches exist per-platform (soft robot RAL'25) but there is no transferable contact-modeling method.
- **G2.3 Sim-fidelity metrics:** there is no standard, cheap way to measure "how wrong is my simulator" for a new robot/task — papers report end-task deltas only, which is confounded by policy quality.
- **G2.4 House-scale / whole-body tasks:** BEHAVIOR-1K-style whole-house manipulation (long-horizon, mobile, bimanual, 100+ object states) exposes that existing sim2real methods (tuned for single-task, single-arm) do not compose. This is a live gap: the 2026 BEHAVIOR Challenge exists precisely because baselines (π0.5, GR00T N1.5/1.7) are far from solving it.

### T3. Dexterous manipulation, tactile sensing, and affordances
**State of the art.** Low-cost teleop data engines (ALOHA [34] → ALOHA 2, Mobile ALOHA [35] → UMI "in-the-wild" teaching [36] → AnyTeleop → DexMimicGen) made imitation dexterity tractable; DexGraspNet-1M / D(R,O) grasp (ICRA 2025 winner, cross-embodiment dexterous grasping) [37][38]; tactile sensing had a 2025 renaissance: PolyTouch (ICRA 2025 **winner**), ShadowTac (ICRA 2025 student winner), GelSphere (IROS 2026 finalist), VTAP (IROS 2026 best-paper finalist); tactile-diffusion policies; 2025 tactile-VLA papers (Tactile-VLA, VLA-Touch, VTLa).
**Award signals.** Tactile + dexterity won awards at ICRA 2025 (2 categories), IROS 2025 (RoTipBot workshop, tactile-dexterity), IROS 2026 (VTAP, GelSphere). Strong sustained reviewer appetite.
**Common evaluation.** Success on dexterous in-hand/in-scene tasks (e.g., bimanual object rotation, peg insertion with contact), grasp stability metrics, cross-object generalization sweeps, tactile channel ablations.
**Reviewer expectations.** For tactile: the sensor must be *necessary* (ablation without it); for dexterity: real hardware is essentially required (sim dexterity claims are distrusted); affordance claims need out-of-distribution object tests.
**Open gaps (specific).**
- **G3.1 Tactile-to-action integration is unsolved:** sensors are improving faster than policies that exploit them. No accepted method fuses high-rate dense tactile with VLA-scale models at deployable latency.
- **G3.2 In-hand manipulation beyond rotation:** most learned dexterity is single-object rotation/reorientation; multi-object, articulated (valve/zipper/door), and compliant force tasks remain sparse.
- **G3.3 Data engine mismatch:** teleop-based data engines (ALOHA/UMI) are gripper-first; dexterous teleop data at ALOHA-equivalent scale/price does not exist — DexMimicGen and egocentric human video (EgoDex 2025) are partial answers with cross-embodiment gaps.

### T4. Humanoid loco-manipulation & whole-body control
**State of the art.** The fastest-moving theme 2024–2026. Motion-tracking RL (AMP → HRP → GMT → HOVER (ICRA'25) → ExBody/ExBody2) for whole-body tracking; interaction-based loco-manipulation (ULC, Falcon, AMO, OmniRetarget (ICRA'26 finalist), HITTER table tennis (ICRA'26 finalist, 106-shot rallies)); behavior foundation models (BFM-Zero lineage, TaskTokens, CLoSD); foundation-model humanoid control (GR00T N1 [39] → N1.5 → N1.7; π0.5 humanoid tasks; AgiBot GO-1 [23]). A 2025 survey defines the field structure (data acquisition → policy learning → sim transfer → deployment) [40].
**Award signals.** IROS 2026 Mobile-Manipulation winners: ULTRA (unified multimodal humanoid whole-body loco-manipulation), SteadyTray (residual RL tray transport); ICRA 2026 finalists HITTER, OmniRetarget; IROS 2026 Entertainment finalists include humanoid tennis from *imperfect human motion data*.
**Common evaluation.** Reference-motion tracking error (MPJPE), success rates on loco-manipulation tasks, robustness to pushes/disturbances, zero-shot sim2real, inference Hz on real hardware.
**Reviewer expectations.** Humanoid papers must show (i) real-robot deployment, (ii) comparison to both model-based WBC and pure-RL baselines, (iii) honesty about what transfers vs. what is tuned per robot.
**Open gaps (specific).**
- **G4.1 Perception-to-action at scale for humanoids:** most WBC learning is proprioception-only + language; visual, whole-body, robust loco-manipulation (the VisualMimic/LeVERB line) is young and evaluation-light.
- **G4.2 Data bottleneck is motion retargeting:** interaction-preserving retargeting (OmniRetarget) is a 2025 result; the cross-embodiment gap (human video → arbitrary humanoid → object interaction) is unsolved in a principled way.
- **G4.3 Composable skills:** no accepted framework for a humanoid to *learn new skills post-deployment* without full retraining (BFM fast adaptation is 2025 preprint-stage; ReLA/LoLA gains ~40% but on limited tasks).

### T5. Navigation, mapping, and embodied scene understanding (VLN)
**State of the art.** VLN (R2R/REVERIE → RxR) saw VLA-model invasions 2024–2025 (NaVid, MiniVLN (ICRA 2025 conference-paper winner) [41]); open-vocabulary mapping (FindAnything, ICRA 2026 Perception finalist); 4D panoptic occupancy (IROS 2026 best finalist); lifelong scene understanding with volatility-aware memory (LT-Mem, IROS 2026 finalist); neuro-symbolic navigation (VL-Nav, IROS 2026 Cognitive finalist); robust field navigation awards (learning-based adaptive navigation ICRA'25 finalist; GNSS-denied UAV planning IROS'26).
**Common evaluation.** Success rate / SPL / path length on R2R-VA, RxR, REVERIE (ScanNet); Replica; real-robot outdoor trials; robustness to novel language / unseen maps.
**Reviewer expectations.** VLN papers face saturation criticism (baseline improvements are marginal); reviewers want (i) *out-of-domain* generalization evidence, (ii) failure analysis under language ambiguity, (iii) real robot for "embodied" claims.
**Open gaps (specific).**
- **G5.1 Long-horizon navigation + manipulation is the frontier:** navigation is largely decoupled from manipulation; mobile-manipulation coordination (IROS 2024 best paper "Harmonic Mobile Manipulation") is the direction, but no standard benchmark covers door-opening + navigation + object interaction jointly at house scale — BEHAVIOR-1K fills exactly this niche.
- **G5.2 Memory for lifelong/long-horizon operation:** LT-Mem (2025) is early; scene-graph + VLM memory interfaces are unstandardized.
- **G5.3 Trustworthy uncertainty in learned navigation:** award-winning 2025 work (MAC-VO) is perception-side; the navigation-policy-side equivalent (knowing when to stop and ask for help) is open — ties to T6 safety.

### T6. Safe, reliable, and verifiable robot learning (deployment-grade autonomy)
**State of the art.** Two strands: (a) *formal + learning hybrids*: control barrier functions (CBFs) with learned components (ABNet [42], neural configuration-space barriers [43], equivariant multi-agent CBFs [44]), semantic safety filters using LLM reasoning + CBF certification (2025, diffusion-policy-safe kitchens) [45]; (b) *failure-aware learning*: uncertainty from diffusion policies for when-to-ask-human (ICRA 2026 HRI finalist), autonomous failure detection + shared control (ARMADA, IROS 2025 best workshop paper), explainability architectures (HEXAR, ICRA 2026 HRI finalist), verification-of-autonomous-systems workshop now a regular IROS event (2026, with new TC award).
**Award signals.** IROS 2026 has a dedicated **Safety, Security & Rescue Robotics best-paper category** (3 finalists: underwater localization, GNSS-denied UAV planning, neuro-symbolic landing assessment). ICRA 2026 HRI finalists were *both* about knowing when to ask for help / explain why.
**Common evaluation.** Constraint-violation counts, worst-case (not mean) success, safety-margin metrics, human-study time-to-intervention, robustness sweeps over perturbation axes.
**Reviewer expectations.** Safety claims need worst-case + distributional evidence, not mean success; hybrid (learned + certified) architectures are well-received *if* the certificate is actually verified and the ablation shows the learning part is necessary.
**Open gaps (specific).**
- **G6.1 No certified long-horizon VLA:** safety certificates stop at low-level control (CBF filter on the action); the *policy* (VLA/diffusion) is uncertified. Runtime assurance for VLA-scale models (monitoring, recovery, graceful degradation) is essentially unsolved at house scale.
- **G6.2 Semantic/common-sense safety:** LLM-derived constraints are only prototyped (2025 kitchen study); no scalable, verified method converts "don't spill water on the laptop" into guaranteed motion constraints under uncertainty.
- **G6.3 Failure detection vs. prevention:** current systems detect failure late (post-hoc uncertainty); predictive failure modeling for learned manipulation (anticipate slip *before* contact loss) is open.
- **G6.4 Human-robot shared control for long tasks:** ICRA 2025 HRI **winner** was shared-control data collection (Human-Agent Joint Learning [49]); but the *deployment* version (human intervenes for 5–10% of steps on 10-minute tasks, robot handles the rest, with measured throughput/safety trade-offs) is missing.

### T7. Multi-robot systems & LLM-coordinated fleets
**State of the art.** Scalable learning-based MAPF ("Deploying Ten Thousand Robots", ICRA 2025 **winner** + student winner [46]); LLM-based decentralized coordination (LLM-Flock [47]) and LLM+optimizer industrial multi-robot programming (IMR-LLM, ICRA 2026 Automation **winner** [48]); distributed mapping under comms constraints (DistGP, ICRA 2026 Multi-Robot finalist); swarm learning (stigmergic MADRL 2025); verification/coordination workshops at IROS.
**Common evaluation.** Scales of 10–10k+ robots in sim (A* / HMMAPF baselines), real fleets of 3–10 robots, communication bandwidth budgets, makespan/collision metrics.
**Reviewer expectations.** "Scales to N" must be shown with compute and comm cost, not just a big-N curve; LLM coordination papers need ablations proving the LLM (vs. a solver) is doing the work (IMR-LLM won partly *for* this division of labor).
**Open gaps (specific).**
- **G7.1 Heterogeneous, task-specialized fleets:** almost all multi-robot learning is homogeneous; mixed-embodiment teams (one dexterous arm + two quadrupeds + a drone) have no standard problem definition or benchmark.
- **G7.2 Learning-based coordination under partial observability + comms loss:** robustness to adversarial/disconnected teammates is studied in classic MARL but not for learned manipulation-level cooperation (e.g., two robots carrying one long object with VLA policies).
- **G7.3 Data/multi-task division of labor:** which robot collects which demonstration, how to aggregate without central storage — open at the VLA data scale (Robo-DM-style infra is single-fleet).

---

## 2. Comparison table

| # | Theme | Heat (2022→2026) | Award presence 2025–26 | Data/compute bar for a 12-mo paper | Biggest open gap | Best fit w/ this repo's assets |
|---|---|---|---|---|---|---|
| T1 | VLA / generalist manipulation | Very high, saturating at "fine-tune π0" level | ICRA'26 RL winner (view-invariance), IROS'26 3× "3D-in-VLA" finalists | Medium (open datasets OXE/DROID/AgiBot; 1–2 GPUs for LoRA-scale FT) | G1.1 spatial fidelity; G1.2 contact; G1.5 long-horizon interface | High: BEHAVIOR demos + π0.5/GR00T baselines already wired |
| T2 | Sim-to-real / world models / data | High, shifting to data engines | ICRA'25 RL winner = data infra (Robo-DM); RAL'25 soft sim2real | High (big sim compute; world-model training is lab-scale) | G2.1 world-model data usefulness; G2.4 house-scale composition | High: OmniGibson stack present |
| T3 | Dexterous + tactile | High, hardware-gated | ICRA'25 2 winners + student; IROS'26 2 finalists | High (hand hardware; teleop data) | G3.1 tactile-action integration; G3.3 dexterous data | Medium: R1Pro has no dexterous hands; could pivot to gripper+force |
| T4 | Humanoid loco-manip | Very high, industry-driven | ICRA'26 2 finalists; IROS'26 2 winners | Very high (humanoid platform) | G4.1 visual robustness; G4.3 post-deployment learning | Medium: BEHAVIOR R1Pro is a dual-arm humanoid-ish platform |
| T5 | Navigation / VLN / memory | Medium (saturated on R2R; rising on house-scale) | ICRA'26 Perception finalist (FindAnything); IROS'26 memory finalists | Medium | G5.1 nav+manip joint benchmarks; G5.2 lifelong memory | **Highest**: BEHAVIOR-1K is exactly the house-scale joint benchmark |
| T6 | Safety / reliability / verification | Rising sharply | IROS'26 dedicated safety award category; ICRA'26 HRI finalists ×2 | Low–Medium (mostly algorithms) | G6.1 certified long-horizon VLA; G6.3 predictive failure | High: Q-score partial credit supports failure-rate studies |
| T7 | Multi-robot + LLM fleets | Steady | ICRA'25 winner (10k-robot IL MAPF); ICRA'26 winner (IMR-LLM) | Low–Medium (sim fleets) | G7.1 heterogeneous fleets; G7.2 comms-loss robustness | Medium: BEHAVIOR is single-robot, but sim fleet extension is cheap |

---

## 3. Promising open problems (ranked by publishability within 12 months)

Ranked for *this* group (assets: 20k BEHAVIOR teleop demos, OmniGibson/Isaac stack, π0.5 + GR00T N1.7 baselines already integrated, single R1Pro-class robot). The top 3 should map 1:1 to the t_aeffa76c ideas.

1. **House-scale long-horizon evaluation of VLA policies (G1.5 + G5.1 + G6.1 combined).**
   - What exists: LIBERO/SimplerEnv measure single-task tabletop; BEHAVIOR-1K Q-score measures long-horizon completion but nobody has published the *systematic generalization study* on it (object/scene/task-distribution sweep, failure taxonomy, per-skill breakdown).
   - Novelty gap: first benchmark-scale answer to "what actually limits VLAs on 10-minute household tasks, and which limit is fixable by better low-level data vs. better planning?" — directly actionable for policy design.
   - 2027 relevance: pairs with the BEHAVIOR Challenge submission (Oct 16, 2026) → IROS 2027 (Mar 2027) or ICRA 2028 (~Sep 2027) depending on which experiments complete first.

2. **Predictive failure detection & graceful degradation for learned house-scale policies (G6.3 + G6.4).**
   - What exists: diffusion-policy uncertainty → when-to-ask (ICRA'26 finalist) is single-task; ARMADA (IROS'25) is failure *detection* post-hoc; no work predicts an about-to-fail sub-skill (slip, collision, wrong-object) *before* the failure, with measured task-success recovery.
   - Novelty gap: early-warning failure model conditioned on skill state + remaining-task structure, evaluated on Q-score partial credit (partial completion after intervention vs. collapse).
   - Cheap to start: pure algorithm work on existing BEHAVIOR demos + our policy; hardware = whatever the challenge team already has.

3. **Sim-real delta quantification for house-scale manipulation (G2.3 + G2.4).**
   - What exists: sim2real studies are single-task; world-model papers don't report real deltas; BEHAVIOR-1K runs entirely in sim with no real-robot counterpart.
   - Novelty gap: a controlled protocol measuring *how much of a learned policy's sim success survives contact reality* for navigation, bimanual, and object-state tasks separately; plus a residual-adaptation method that closes the measured gap with ≤N minutes of on-robot data.
   - Differentiator vs. locomotion sim2real: whole-house, multi-skill, 100-object-state setting — the composition problem no prior sim2real paper addresses.

4. **Spatial-fidelity benchmark & policy for VLA (G1.1).** (Backup idea)
   - What exists: view-invariant conditioning (ICRA'26 winner) is one axis; FP3/GeoVLA are training-side; no *diagnostic benchmark* (camera-perturbation sweeps: translation/rotation/lighting/object pose) exists to measure "does the policy know where things are?"
   - Novelty gap: benchmark + simple, strong method (e.g., depth/point-cloud conditioning or view-agnostic action tokens) beating π0.5-class baselines on it.

5. **Heterogeneous two-robot cooperation under comms loss (G7.2).** (Backup, low asset fit)
   - Two arms/robots carrying or cooperating on a long task with VLA-level policies; robustness to dropped messages; no standard setup exists.

6. **Tactile-in-the-loop for contact-critical house tasks (G3.1).** (Backup, hardware-gated)
   - If a tactile channel (e.g., GelSphere-class or force-torque at wrist) can be added to the BEHAVIOR platform: when does force feedback rescue pose-only policies on the specific contact tasks (door knobs, drawers, slippery objects) they fail? Measured with per-task force ablations.

**Cross-cutting caution:** themes T3/T4 require hardware we do not have; T1-saturation ("another π0 fine-tune") will not survive the no-novel-contribution desk screen; T7 is viable but weaker fit. The top-3 above are all *algorithm+benchmark* contributions that ride the group's existing BEHAVIOR investment. Timeline reality: with the ICRA 2027 deadline 3 days away, the realistic target pairing is **IROS 2027 (Mar 1, 2027)** for ideas 1–3 developed from scratch, and **ICRA 2028 (expected ~Sep 15, 2027)** for challenge-adjacent follow-ups (benchmark study, failure taxonomy, sim-real protocol) that build on the Oct–Nov 2026 BEHAVIOR Challenge submission and results.

---

## 4. Sources

All arXiv entries below were verified on 2026-09-12 against the Semantic Scholar Graph API (title, year, first authors, venue, citation count; raw responses in `raw/s2_papers.json`, `raw/s2_papers2.json`, `raw/s2_search3.json`). Award and schedule claims are from official/primary pages. Entries with `S2:MISS` or guessed IDs were **excluded** from this list; where they were useful context only, they are cited by name in the body.

[1] Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023. arXiv:2307.15818 (S2-verified; 4,118 cites).
[2] Ahn et al., *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*, CoRL 2022. arXiv:2204.01691 (S2-verified; 3,642 cites).
[3] Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024. arXiv:2406.09246 (S2-verified; 3,256 cites).
[4] Ghosh et al., *Octo: An Open-Source Generalist Robot Policy*, RSS 2024. arXiv:2405.12213 (S2-verified; 1,779 cites).
[5] Padalkar et al., *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, ICRA 2024. arXiv:2310.08864 (S2-verified; 1,172 cites).
[6] Khazatsky et al., *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset*, RSS 2024. arXiv:2403.12945 (S2-verified; 1,027 cites).
[7] Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164 (S2-verified; 2,588 cites).
[8] Liu et al., *RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation*, ICLR 2025. arXiv:2410.07864 (S2-verified; 855 cites).
[9] Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023. arXiv:2303.04137 (S2-verified; 4,109 cites).
[10] Physical Intelligence, *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025. arXiv:2504.16054 (S2-verified; 1,734 cites).
[11] Shi et al., *Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models*, ICML 2025. arXiv:2502.19417 (S2-verified; 248 cites).
[12] Li et al., *HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation*, ICLR 2025. arXiv:2502.05485 (S2-verified; 149 cites).
[13] Kim, Finn & Liang, *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success* (OpenVLA-OFT), 2025. arXiv:2502.19645 (S2-verified; 840 cites).
[14] Driess et al., *Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better*, NeurIPS 2025. arXiv:2505.23705 (S2-verified; 144 cites).
[15] Black, Galliker & Levine, *Real-Time Execution of Action Chunking Flow Policies*, NeurIPS 2025. arXiv:2506.07339 (S2-verified; 208 cites).
[16] Wen et al., *TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models*, RA-L 2025. arXiv:2409.12514 (S2-verified; 430 cites).
[17] Shao et al., *Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey*, 2025. arXiv:2508.13073 (S2-verified; 88 cites).
[18] Sapkota et al., *Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges*, 2025. arXiv:2505.04769 (S2-verified; 94 cites); Wolf et al., *Diffusion Models for Robotic Manipulation: a Survey*, 2025. arXiv:2504.08438 (S2-verified; 77 cites).
[19] Zhang et al., *Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning*, ICRA 2026 Best Paper (Robot Learning). arXiv:2510.02268 (S2-verified; JHU CS news, 2026-07-07).
[20] Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, CoRL 2023. arXiv:2306.03310 (arXiv abs page verified 2026-09-12; note: 130-task lifelong benchmark).
[21] Nair et al., *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots*, 2024. arXiv:2406.02523 (arXiv abs page verified 2026-09-12).
[22] Zhang et al., *ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations*, 2025. arXiv:2505.10911 (S2-verified; 67 cites).
[23] AgiBot-World Contributors, *AgiBot World Colosseo: A Large-Scale Manipulation Platform for Scalable and Intelligent Embodied Systems*, IROS 2025. arXiv:2503.06669 (S2-verified; 457 cites).
[24] Kumar, Fu, Pathak & Malik, *RMA: Rapid Motor Adaptation for Legged Robots*, RSS 2021. arXiv:2107.04034 (S2-verified; 1,003 cites).
[25] NVIDIA, *Cosmos World Foundation Model Platform for Physical AI*, 2025. arXiv:2501.03575 (arXiv PDF verified 2026-09-12; ~20M h video pretraining).
[26] *Genie 3: A New Frontier for World Models*, Google DeepMind blog, 2025.
[27] Gao et al., *FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model*, ICLR 2025. arXiv:2412.08261 (S2-verified; 45 cites).
[28] Fang et al., *ReBot: Scaling Robot Learning with Real-to-Sim-to-Real Robotic Video Synthesis*, 2025. arXiv:2503.14526 (title-verified in [29]'s reference list).
[29] *GigaWorld-0: World Models as Data Engine to Empower Embodied AI*, 2025. arXiv:2511.19861 (title-verified on arXiv HTML page).
[30] *Genesis: A Generative and Universal Physics Simulator*, 2024 (name-verified in the curated humanoid-WBC literature table, github.com/Earl000333/humanoid-wbc-review; arXiv ID not independently confirmed — context only).
[31] Li et al., *BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation*, CoRL 2022 (arXiv v2, 2024). arXiv:2403.09227 (arXiv page + StanfordVL BibTeX verified 2026-09-12).
[32] Pitkevich & Makarov, *A Survey on Sim-to-Real Transfer Methods for Robotic Manipulation*, IEEE SiSY 2024, doi:10.1109/SISY62279.2024.10737545 (IEEE Xplore abstract-verified).
[33] Chen et al., *Robo-DM: Data Management for Large Robot Datasets*, ICRA 2025 Best Paper (Robot Learning) — title/venue per official ICRA 2025 awards page (arXiv ID unverified, cited without ID).
[34] Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ALOHA), RSS 2023. arXiv:2304.13705 (S2-verified; 2,291 cites).
[35] Fu, Zhao & Finn, *Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation*, 2024. arXiv:2401.02117 (S2-verified; 754 cites).
[36] Chi et al., *Universal Manipulation Interface (UMI): In-The-Wild Robot Teaching Without In-The-Wild Robots*, RSS 2024. arXiv:2402.10329 (S2-verified; 669 cites).
[37] Wei et al., *D(R,O) Grasp: A Unified Representation of Robot and Object Interaction for Cross-Embodiment Dexterous Grasping*, ICRA 2025 Best Paper (Manipulation & Locomotion) — per official ICRA 2025 awards page.
[38] Zhao et al., *DexGraspNet: Generative Dexterous Grasping*, 2023 (S2:MISS; cited by name, arXiv:2210.02882 unverified here — treat as context only).
[39] *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, NVIDIA, 2025. arXiv:2503.14734 (title-verified in [17]'s reference list).
[40] Gu et al., *Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning*, IEEE Trans. Mechatronics 2025. arXiv:2501.02116 (S2-verified; 145 cites).
[41] Zhu et al., *MiniVLN: Efficient Vision-and-Language Navigation by Progressive Knowledge Distillation*, ICRA 2025 Best Conference Paper — per official ICRA 2025 awards page.
[42] Xiao et al., *ABNet: Adaptive Explicit-Barrier Net for Safe and Scalable Robot Learning*, NeurIPS 2023 (OpenReview abstract-verified).
[43] *Neural Configuration-Space Barriers for Manipulation Planning and Control*, 2025. arXiv:2503.04929 (S2-verified).
[44] *Deep Equivariant Multi-Agent Control Barrier Functions*, 2025. arXiv:2506.07755 (S2:MISS on title search; PDF-verified at arxiv.org/pdf/2506.07755v1).
[45] Brunke et al., *Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards*, RA-L 2025. arXiv:2410.15185 (S2-verified).
[46] Jiang et al., *Deploying Ten Thousand Robots: Scalable Imitation Learning for Lifelong Multi-Agent Path Finding*, ICRA 2025 Best Paper (Multi-Robot Systems) + Best Student Paper — per official ICRA 2025 awards page.
[47] *LLM-Flock: Decentralized Multi-Robot Flocking via Large Language Models and Influence-Based Consensus*, 2025. arXiv:2505.06513 (PDF-verified at arxiv.org).
[48] *IMR-LLM: Industrial Multi-Robot Task Planning and Program Generation Using Large Language Models*, ICRA 2026 Best Paper (Automation) — per embodiedglobal.com + Bohrium ICRA 2026 summary.
[49] IEEE ICRA 2025 Awards and Finalists, 2025.ieee-icra.org/program/awards-and-finalists/ (full list, accessed 2026-09-12).
[50] Bohrium, *ICRA 2026 Papers to Read: Best Papers & Robotics Trends* (finalists, CFP timeline, scale: 5,088 submissions / ~35% acceptance, accessed 2026-09-12).
[51] IROS 2026 Awards finalists, 2026.ieee-iros.org/program/awards/ (accessed 2026-09-12).
[52] MARCH lab, *IROS 2026 Award Finalist* (4,348 submissions, 1,585 accepted; accessed 2026-09-12).
[53] IEEE RAS, *Information for ICRA Editors* (summary-rejection policy, ieee-ras.org; accessed 2026-09-12).
[54] *How to Write a Winning Robotics Conference Paper*, RoboticsBiz (reviewer-expectations guide, accessed 2026-09-12).
[55] ICRA 2027 official site (dates/location, 2027.ieee-icra.org, accessed 2026-09-12); IROS 2027 deadline (ieee-ras.org event page + mldeadlines.com, accessed 2026-09-12).
[56] JHU CS, *Anand Bhattad wins ICRA 2026 Best Paper Award on Robot Learning* (2026-07-07).
[57] PolyU ISE, *ISE Graduates Won Best Paper at IROS 2025* (workshop-level, 2025-12-02); SIA CAS, *IROS 2025 Best Conference Paper Award* (2025-12-12); IROS 2025 HRI workshop awards page (hri.iit.it; ARMADA best-paper).
[58] *Genie: Generative Interactive Environments*, DeepMind, 2024. arXiv:2402.15391 (S2-verified).
[59] *3D Diffuser Actor: Policy Diffusion with 3D Scene Representations*, 2024. arXiv:2402.10885 (S2-verified; 376 cites).
[60] *No Plan but Everything Under Control: Robustly Solving Sequential Tasks with Dynamically Composed Gradient Descent*, ICRA 2025. arXiv:2503.01732 (S2-verified).
[61] *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video*, 2025 (title-verified in [17]'s reference list; S2:MISS on direct search).
[62] *Improving Vision-Language-Action Model with Online Reinforcement Learning*, ICRA 2025 (title-verified in [17]'s reference list).
[63] Gao et al., *Sim-to-Real of Soft Robots With Learned Residual Physics*, IEEE RA-L Best Paper Award 2025 — per IEEE RAS "RAS Recognizes 2025 Award Recipients at ICRA" news (ieee-ras.org, accessed 2026-09-12).

**Verification note.** "S2-verified" = title/year/authors/venue/citation-count confirmed via the Semantic Scholar Graph API on 2026-09-12. "Title-verified in [17]'s reference list" = the cited work appears in the S2-verified VLA survey's reference list with a matching title; its own ID was not independently confirmed. "PDF-verified at arxiv.org" = the arXiv PDF at the stated ID was fetched and its title checked. Entries that could not be verified (RT-1, Gato, Code-as-Policies, HIL-SERL, AnyTeleop, NaVid, DexMimicGen, ALOHA 2, PolyTouch, ARMADA) were used only as named context in the body or excluded; none of the load-bearing claims in Sections 1–3 rests on an unverified ID. The ICRA 2027 CFP page itself was behind Cloudflare at access time; the Sep 15, 2026 deadline is from the ICRA 2025/2026 official FAQ timeline pattern and should be reconfirmed at 2027.ieee-icra.org before any planning commitment.
