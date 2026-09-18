"""Generate the validation-harness sample datasets + expected outputs.

Deterministic and CPU-only.  Produces, under ``validation/sample_data/``:

* ``standard.public.b1k.nous.20260912/``  — a *complete* public-testset
  submission: 100 tasks x 20 public instances (301..320) x 1 rollout = 2,000
  metrics JSONs, each named ``<task_name>_<instance_id>_<rollout_id>.json``
  with the exact spec §7.1 schema (task, instance_id, rollout_id, steps,
  success, agent_distance, normalized_agent_distance, q_score.final,
  time.{simulator_steps,simulator_time,normalized_time}).
* ``standard.public.b1k.nous.20260910/``  — a small *sample* submission
  (tasks 0..2, instances 301..305, 15 JSONs) drawn from the same values, so a
  fresh checkout has a cheap thing to score.
* ``expected_standard.public.b1k.nous.<date>.json`` — the *expected* aggregate
  scores for each sample, computed by re-parsing the written files through
  ``validator.validate_submission`` and freezing the result.  The harness
  (runner + pytest) fails if a re-score drifts from these by more than 1e-12.

Per-rollout values are seeded per task (seed = 20260912 + task_index):
``q_score.final`` is quantized to {0, .25, .5, .75, 1.0} (partial credit),
``time.normalized_time`` ~ U(0.6, 1.9) (time_score spans negative..positive),
normalized distances ~ U(0.2, 1.4).

Regenerate with::

    python3 validation/make_sample_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for p in (str(REPO / "src"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from validation import validator  # noqa: E402

BASE_SEED = 20260912
HUMAN_EP_STEPS = 9000            # for synthetic simulator_steps
SAMPLE_TASKS = (0, 1, 2)         # small sample submission
SAMPLE_INSTANCES = list(range(301, 306))
FULL_FOLDER = "standard.public.b1k.nous.20260912"
SAMPLE_FOLDER = "standard.public.b1k.nous.20260910"


def _rollout_record(task_name: str, task_index: int, instance: int) -> dict:
    rng = np.random.default_rng(BASE_SEED + task_index)
    q = float(rng.integers(0, 5)) / 4.0                    # 0/.25/.5/.75/1.0
    nt = float(rng.uniform(0.6, 1.9))                      # normalized time
    steps = int(round(nt * HUMAN_EP_STEPS))
    rec = {
        "task": task_name,
        "instance_id": int(instance),
        "rollout_id": 0,
        "steps": steps,
        "success": bool(q >= 1.0),
        "agent_distance": {
            "base": round(float(rng.uniform(0.4, 2.8)), 3),
            "left": round(float(rng.uniform(0.3, 2.2)), 3),
            "right": round(float(rng.uniform(0.3, 2.2)), 3),
        },
        "normalized_agent_distance": {
            "base": round(float(rng.uniform(0.2, 1.4)), 4),
            "left": round(float(rng.uniform(0.2, 1.4)), 4),
            "right": round(float(rng.uniform(0.2, 1.4)), 4),
        },
        "q_score": {"final": q},
        "time": {
            "simulator_steps": steps,
            "simulator_time": round(steps / 30.0, 3),
            "normalized_time": round(nt, 4),
        },
    }
    return rec


def _write_expected(folder: Path) -> Path:
    """Re-parse the written sample through the validator and freeze the scores."""
    v = validator.validate_submission(folder)
    if not v.passed:
        raise RuntimeError(f"generated sample failed validation: {v.errors[:5]}")
    out = folder.parent / f"expected_{folder.name}.json"
    obj = {
        "folder": folder.name,
        "n_rollouts": v.n_rollouts,
        "n_warnings": len(v.warnings),
        "overall_q": v.overall_q,
        "overall_task_sr": v.overall_task_sr,
        "overall_time_score": v.overall_time_score,
        "per_task_q": {t: q for t, q in v.per_task_q.items()},
    }
    out.write_text(json.dumps(obj, indent=2) + "\n")
    return out


def main() -> int:
    sample_dir = REPO / "validation" / "sample_data"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ---- full public sample (100 tasks x 20 instances) ----------------------
    task_names = validator.load_task_names()
    assert len(task_names) == 100
    full = sample_dir / FULL_FOLDER / "json"
    full.mkdir(parents=True, exist_ok=True)
    n = 0
    for idx, name in enumerate(task_names):
        for inst in validator.PUBLIC_INSTANCE_IDS:
            rec = _rollout_record(name, idx, inst)
            (full / f"{name}_{inst}_0.json").write_text(json.dumps(rec) + "\n")
            n += 1
    print(f"full sample: {n} metrics JSONs -> {sample_dir / FULL_FOLDER}")

    # ---- small sample (tasks 0..2, instances 301..305) ----------------------
    small = sample_dir / SAMPLE_FOLDER / "json"
    small.mkdir(parents=True, exist_ok=True)
    for idx in SAMPLE_TASKS:
        for inst in SAMPLE_INSTANCES:
            rec = _rollout_record(task_names[idx], idx, inst)
            (small / f"{task_names[idx]}_{inst}_0.json").write_text(json.dumps(rec) + "\n")
    print(f"sample subset: {len(SAMPLE_TASKS) * len(SAMPLE_INSTANCES)} metrics JSONs -> "
          f"{sample_dir / SAMPLE_FOLDER}")

    # ---- freeze expected outputs (round-tripped through the validator) ------
    for folder in (sample_dir / FULL_FOLDER, sample_dir / SAMPLE_FOLDER):
        exp = _write_expected(folder)
        obj = json.loads(exp.read_text())
        print(f"expected: {exp.name}  "
              f"Q={obj['overall_q']:.6f}  task_sr={obj['overall_task_sr']:.4f}  "
              f"time={obj['overall_time_score']:.6f}  rollouts={obj['n_rollouts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
