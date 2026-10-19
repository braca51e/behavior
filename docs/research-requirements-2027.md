# What the Research Programme Needs to Succeed

**Date:** 2026-10-19. Companion to [`research-plan-2027.md`](research-plan-2027.md) (the three studies) and [`research-implementation-2027.md`](research-implementation-2027.md) (engineering). This document lists everything *besides the ideas* that has to be in place for papers to come out the other end: people, compute, data, process, writing, release, and the decisions that must be made on time. Items marked **[gap]** do not exist yet.

---

## 1. Definition of success

| Level | Outcome | Date |
|---|---|---|
| Minimum | One IROS 2027 submission (Study 1 or 2) with released artifact, all gates passed honestly, n per cell reported | Mar 1, 2027 |
| Target | Two IROS 2027 submissions (Studies 1 + 2); Study 3 pilot gates passed and matrix scheduled | Mar 1, 2027 |
| Stretch | Studies 1 + 2 at IROS; Study 3 at ICRA 2028; the three cited as a line | Sep 15, 2027 |
| Also a success | A well-documented null result at any gate that is written up as a boundary result or workshop paper, not buried | — |

Success is **not** "all three hypotheses confirmed". Each study's hypothesis table in the plan states what a refutation means and how it is still written up.

---

## 2. People and roles

Realistic team: 2–3 people plus an advisor-level reviewer. Roles can be shared but every role must have a name against it.

| Role | Responsibility | Time | Status |
|---|---|---|---|
| Programme lead | Owns the gates, makes go/no-go calls on the dates below, owns the second-box decision, is corresponding author | ~20 % | **[gap] assign** |
| Study 1 owner | Instrument (automatic attribution), instance axis, matrix, paper 1 draft | ~60 % Oct–Mar | **[gap] assign** |
| Study 2 owner | Conditioning variants, switching regimes, matrix, paper 2 draft | ~60 % Oct–Mar | **[gap] assign** |
| Study 3 owner | Detector zoo, trigger-matched ladder, pilot; may be the same person as Study 1 until the Dec 31 gate | ~30 % Nov–Feb, 60 % after | **[gap] assign** |
| Infrastructure owner | Docker images, eval path, throughput, matrix runner, data downloads, disk | ~40 % Oct–Nov, 15 % after | **[gap] assign** (default: Study 1 owner) |
| Human annotators ×2 | 300-episode blind labeling for Study 1 κ (≈ 25 h each incl. training); must not be the Study 1 owner | 50 h total, Dec–Jan | **[gap] recruit** |
| Internal reviewer | Reads each draft two weeks before deadline as a hostile IROS reviewer; not an author | 2 × 4 h per paper | **[gap] recruit** |
| Writing lead per paper | Owns the narrative, figure list, and the 8-page budget; usually the study owner | — | = study owner |

**Skills required across the team** (check coverage; train or recruit for gaps): PyTorch + HF datasets; OmniGibson / Isaac Sim basics (enough to write an env wrapper and read privileged state); GR00T and openpi finetuning; experimental statistics (bootstrap CIs, κ, AUROC, task-grouped cross-validation, multiple-comparison awareness); scientific writing for IEEE RAS venues; Docker on a headless GPU box.

---

## 3. Compute, storage, money

| Need | Amount | Status |
|---|---|---|
| Primary GPU node (Brev, Docker-only) | continuous Oct 20 → Feb 10 (~110 GPU-days) | have |
| Second identical node for evaluation waves | Nov 1 → Feb 10 (~100 GPU-days); ≈ $1.5–3k at current hourly rates | **[decision] by Oct 24** (ticket T12) |
| Disk per node | 2 TB (20-task demos ≈ 0.7 TB; videos ≈ 0.3 TB per 1,000 episodes; checkpoints 5 × ~7 GB per finetune) | **[gap] verify free space; expand volume** |
| Off-node storage for released artifacts | ~1 TB (HF dataset or S3) | **[gap]** |
| CPU box for detector training / analysis | any laptop is enough (LightGBM, pandas) | have |
| W&B project (online) for finetunes | free tier suffices | have (see `docs/docker-training.md`) |
| VLM API budget for the verifier baselines (Studies 2, 3) | ~1 frame-pair / 2 s × ~3,000 episodes × 6 min ≈ 500k calls; or self-host Qwen-VL on the second node | **[decision] self-host recommended** |
| Measured throughput number | episodes / GPU-day (plan assumes ~50 full-length) | **[gap] measure in week 1 (A2)** |

Everything in the plan is sized to these numbers; if A2 comes in below 35 episodes/day, drop Study 3 to ICRA 2028 immediately and shrink Study 2 to five switching arms.

---

## 4. Data and assets

| Asset | Need | Status |
|---|---|---|
| Demo chunks for the 20 stratified tasks | download via `scripts/docker/download_demos.sh` (HF token, gate accepted) | **[gap] after task selection** |
| Skill-segment annotations for those tasks | ship with the demos (`annotation_path`) | have once downloaded |
| Official test instance files (301–340) in the sim image | required for all evaluation | have (in `stanfordvl/behavior:3.9.2`) |
| Demo-composition ("seen") instance files or a composition-overlap mapping | required for Study 1's instance axis | **[gap] verify (A3)**; fallback defined in plan §5 |
| Organizer-provided checkpoints (π0.5, GR00T; `turning_on_radio`) | pilots, sanity rows | have |
| Multi-task finetunes (GR00T, π0.5 on 20 tasks) | Studies 1, 3 | **[gap] ~3 GPU-days** |
| Four conditioning-variant finetunes (GR00T) + two (π0.5) | Study 2 | **[gap] ~8.5 GPU-days** |
| Trained predicate detectors (per BDDL family) | Study 1 watchdog arm, Study 2 onboard switching, Study 3 progress family | **[gap]** |
| Rasouli et al.'s labeling instructions and class definitions [2] | human-annotation protocol for Study 1 κ | **[gap] extract from paper; write a 2-page annotator SOP** |
| Behavior-Skill code and skill taxonomy [1] | align our 31-skill taxonomy with their 34 categories for comparability | **[gap] map taxonomies; document differences** |
| License check on demos and sim assets for redistribution of labels/videos | release plan | **[gap] (A7)** |

---

## 5. Process: how work is run

**Cadence.**
- Monday 30-min sync: gates, blockers, compute queue for the week.
- Friday written log entry per study in `docs/research/log.md` **[gap: create]**: what ran, n, result, decision. One paragraph. This log is the raw material for the papers' experiment sections and the defence against "we think we saw…".
- Gate meetings on the dates in §8: a go/no-go decision is recorded in the log the same day.

**Pre-registration.** Before each matrix starts, the study owner commits a short `docs/research/prereg-<study>.md` **[gap]** stating hypotheses, primary metric, n, exclusion rules, and the analysis script name. Deviations are allowed but must be logged with a reason. This is what lets us write "we hypothesised X" honestly and protects against post-hoc metric shopping.

**Experiment registry.** Every rollout directory records image digest, checkpoint hash, instance ids, seeds, git SHA (implementation §D.8). A `manifest.csv` per matrix is the single source of truth for n per cell. Tables in the paper are generated from it, never typed.

**Compute queue.** One shared text file or board **[gap]** listing what is running on which node, owner, expected finish. Prevents the two most common failures on a shared box: two jobs on one GPU, and a matrix stalled for days because nobody noticed it crashed. Add a crash alert (W&B alert or a cron that checks the manifest growth) **[gap]**.

**Literature watch.** One person checks arXiv cs.RO weekly for "BEHAVIOR", "long-horizon VLA", "failure detection VLA", "subtask conditioning" and appends to `docs/research/litwatch.md` **[gap]**. Full re-run of the §1 overlap check on Jan 10, 2027, before drafting. Six directly relevant papers appeared in the last seven months; being scooped between now and March is the single largest external risk.

**Code hygiene.** Research code lives in `src/research/`, never imported from `src/b1k/` (the challenge serving path stays legal). PRs to `main`, small, with a log entry. Tests for anything that produces a number in the paper (labeler rules, metrics, aggregation).

**Statistics standards (apply to every table).**
- n per cell stated; 95 % bootstrap CIs on Q; binomial CIs on success rates.
- Simulator nondeterminism measured (repeat rollouts) and reported as the noise floor.
- Held-out *tasks* (not just episodes) for any learned detector or classifier; task-grouped cross-validation.
- Multiple hypotheses per study → report all, do not cherry-pick; state which were pre-registered.
- Effect sizes in Q points, not only p-values.

---

## 6. Writing and submission

**Venue facts (verify on the CFP before relying on them).**
- IROS 2027: Florence, Sep 26–Oct 1, 2027; paper deadline **Mar 1, 2027 23:59 PST**; double-anonymous; 8 pages including references (6 + 2 over-length fee pattern typical — confirm); video ≤ 180 s / ≤ 20 MB; AI-use disclosure required.
- ICRA 2028: deadline expected ~Sep 15, 2027; same format family.
- Summary rejection applies for page violations and "no novel contribution with cited counterexamples" — the §1 positioning table in the plan exists to pre-empt the second.

**Per-paper checklist (the reviewer bar).**
1. Quantified gap claim in paragraph one, with the number and the citation (e.g. "the only existing failure study hand-labels 500 episodes of one policy [2]").
2. Evidence ladder: every abstract claim has a matching table; no real-world language anywhere.
3. Related work names the five closest papers and states, in one sentence each, what they did not measure.
4. n per condition, success criterion verbatim, failure table disclosed.
5. Baselines hardware/sensor-matched and tuned with comparable effort; a strong non-learned or rule baseline included (the watchdog / progress rule serves this role).
6. Limitations section that names the simulator-only scope and the two policy families.
7. Reproducibility: code, labels, protocol released; checkpoint hashes in the appendix.
8. Figures first: the expected-figures list in the plan is the paper's skeleton; draft figures by Jan 31 from pilot data so the matrix only fills them in.

**Timeline per IROS paper.**
- Jan 31: figures from completed matrix, results tables generated.
- Feb 7: full draft v1 to internal reviewer.
- Feb 14: reviewer notes back; revise.
- Feb 21: v2 frozen; video cut; supplementary assembled; anonymisation check (no repo URLs, no team names in artifacts).
- Feb 26: submit (do not use the last 72 hours).
- Mar 1: deadline.

**Authorship and credit** agreed in writing before the matrices start **[gap]**: order, who is corresponding, how annotators and infrastructure work are credited.

---

## 7. Release artifacts (the desk-screen defence)

| Study | Artifact | Where | Blocking requirement |
|---|---|---|---|
| 1 | `b1k-anatomy`: ~1,400 labeled episodes (labels + privileged parquet; videos if license permits), the automatic labeler, annotator SOP, stratification table | HF dataset + GitHub | A7 license check; anonymised release for review, de-anonymised at camera-ready |
| 2 | Relabeled skill datasets (as scripts that derive them from the public demos, not copies), four checkpoints, switching-metric code, oracle side-channel protocol | HF + GitHub | Checkpoint hosting space (~30 GB) |
| 3 | Detector zoo code, feature dataset, trigger-matched ladder, latency benchmarks | GitHub + HF | — |
| all | A `CITATION.cff`, a license (MIT for code; dataset license matching the demos), a README with exact reproduction commands | GitHub | — |

Every artifact is prepared *during* the study, not after acceptance.

---

## 8. Decision calendar (the lead owns these)

| Date | Decision | Input |
|---|---|---|
| Oct 23 | Eval path works end to end (A1); throughput measured (A2) | T1 |
| Oct 24 | Rent second node? | A2, team size |
| Oct 31 | Instance definition for Study 1 (A3): seen/unseen vs public/hidden | sim image inspection |
| Nov 8 | Study 3 pilot gate: continue or fold into Study 1 | 60-episode separation test |
| Nov 10 | Study 2 pilot gate: skill-conditioning signal present? | 5k-step pilots |
| Nov 15 | Study 1 pilot gate: instance gap and κ present? | 100-episode pilot |
| Dec 31 | Study 1 κ gate; Study 3 detector gate | 300-episode audit; AUROC on held-out tasks |
| Jan 10 | Literature re-check; which paper leads | arXiv scan |
| Jan 25 | Study 2 H1 gate | matrix |
| Jan 31 | Study 1 H2 gate; figures frozen for both IROS papers | matrices |
| Feb 10 | Study 3: IROS or ICRA 2028 | H2 probe |
| Feb 26 | Submit | — |

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Eval path never stabilises on the Docker-only box (A1) | medium | blocks everything | First ticket; escalate to building `b1k-eval` from the official Dockerfile; ask organizers' Discord | infra |
| Throughput half of assumed (A2) | medium | matrices do not fit | Second node; drop Study 3 to ICRA 2028; cap episodes at 2× human mean | lead |
| Scooped before March | medium-high | paper rejected for novelty | Weekly lit watch; Jan 10 re-check; positioning table kept current; submit early workshop versions to timestamp | lit-watch owner |
| Finetuned policies too weak to show any effect (Q ≈ 0 everywhere) | medium | no signal | Pilot gates on organizer checkpoints first; use their tasks; report honestly | study owners |
| Instance axis not evaluable (A3) | medium | Study 1 H2 weakens | Fallback definition pre-agreed (plan §5.3) | Study 1 |
| Human annotation slips (annotators unavailable in Dec) | medium | κ gate missed | Recruit by Nov 15; SOP written by Dec 1; 50 episodes labeled as a dry run | Study 1 |
| Single point of failure in people (one person leaves) | low-medium | a study dies | Every study has written pre-reg, log, and runnable scripts; shared tickets documented | lead |
| Disk full mid-matrix | high (it has already happened once) | lost days | 2 TB per node; `df` check in the matrix runner before each episode; videos optional per run | infra |
| HF token / gated-model access breaks | low | training blocked | Tokens in shell only, never in chat or git; `hf auth whoami` preflight already in scripts | infra |
| Over-scoping (trying to run all three matrices at full size on one node) | high | nothing finishes | The plan's single-node fallback is binding unless the second node is rented | lead |
| Reviewer: "sim-only, so what?" | certain | weaker scores | Abstracts make no real-world claims; limitations explicit; BEHAVIOR is the accepted house-scale sim benchmark — cite [25] and the 2025 challenge reports | writing leads |

---

## 10. Things we deliberately do not need

- A physical robot. All claims are simulation-scoped.
- A GPU cluster. Two nodes suffice for the sized matrices.
- A new policy architecture. All three studies measure; none proposes a model.
- Full 100-task coverage. 20 stratified tasks with stated strata is a defensible, reviewable scope.
- Perfect finetunes. The studies are about structure of failure and of hierarchy, which are measurable at Q ≈ 0.2–0.4.

---

## 11. Immediate actions (this week)

1. Assign the roles in §2 and write the names into this file.
2. Run A1 and A2 (implementation §B); record the throughput number.
3. Decide on the second node (Oct 24).
4. Create `docs/research/log.md`, `docs/research/litwatch.md`, and the compute queue file.
5. Write the annotator SOP skeleton from Rasouli et al. [2] and start recruiting the two annotators.
6. Agree authorship in writing.
