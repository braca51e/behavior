# Self-eval report — team `b1k`

- rollouts: **5** across **1** tasks
- **Q (mean task BDDL predicate fraction): 0.6000**
- task_sr (full success rate): 0.4000
- time_score (efficiency tie-breaker): 0.9833

## Per-task (sorted worst → best)

| task | Q | task_sr | time_score | ΔQ vs prev |
|---|---|---|---|---|
| turning_on_radio | 0.600 | 0.40 | 0.983 | — |

<!-- qjson {"q_score": {"turning_on_radio": 0.6}} -->

## Fixture verification (serving pipeline — not a leaderboard Q)

- serving pipeline: **OK**

| task | id | steps | finite 23-dim | finished(park) | confirmed predicates | error |
|---|---|---|---|---|---|---|
| turning_on_radio | 0 | 30 | yes | no | — | — |
| picking_up_trash | 1 | 30 | yes | no | — | — |
| cleaning_up_plates_and_food | 3 | 30 | yes | no | — | — |

> This section is a CPU-only verification that the WebSocket serving
> contract, action safety, and progress/park logic run end-to-end on the
> recorded obs fixture.  It is **not** a sim score.  Real Q requires
> ``--mode sim`` on an OmniGibson v3.9.2 + GPU box.

