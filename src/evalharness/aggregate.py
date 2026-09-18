"""Evaluation-result aggregation (design.md section 4.7).

Recomputes Q / task_sr / time_score / normalized distances **exactly** per the
challenge's ``score_utils.compute_final_q_score`` (v3.9.2) so that our self-eval
and the organizers' leaderboard use the same math.  Unit-tested against the
five fixture metrics JSONs (``tests/fixtures/metrics_*.json``) covering Q
fractions 0 / 0.5 / 1.0 with and without early stop.

Per-rollout metrics JSON schema (challenge spec section 7.1)::

    {"task": "...", "instance_id": N, "rollout_id": N, "steps": N,
     "success": bool, "agent_distance": {"base":..,"left":..,"right":..},
     "normalized_agent_distance": {"base":..,"left":..,"right":..},
     "q_score": {"final": 0.0..1.0},
     "time": {"simulator_steps": N, "simulator_time": S, "normalized_time": N}}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

EVAL_TIMEOUT_MULTIPLIER = 1.5


def time_score_from_normalized(normalized_time: float) -> float:
    """``time_score = 3 - 2/normalized_time`` (design section 1.3 / spec 5.2).

    Equals the score_utils expression
    ``EVAL_TIMEOUT_MULTIPLIER/(EVAL_TIMEOUT_MULTIPLIER-1) -
    1/((EVAL_TIMEOUT_MULTIPLIER-1)*normalized_time)`` for the multiplier 1.5.
    Faster-than-human (normalized_time < 1) -> > 1; at 1.5x human -> 0; slower
    -> negative.  ``normalized_time <= 0`` -> 0 (defensive; the evaluator never
    emits it).
    """
    nt = float(normalized_time)
    if nt <= 1e-9:
        return 0.0
    return (EVAL_TIMEOUT_MULTIPLIER / (EVAL_TIMEOUT_MULTIPLIER - 1.0)) - \
        1.0 / ((EVAL_TIMEOUT_MULTIPLIER - 1.0) * nt)


@dataclass
class RolloutScore:
    task: str
    instance_id: int
    rollout_id: int
    q: float
    time_score: float
    base_dist: float
    left_dist: float
    right_dist: float
    steps: int = 0
    success: bool = False


@dataclass
class Summary:
    """Aggregate scores over a set of rollout JSONs."""

    rollouts: list[RolloutScore] = field(default_factory=list)
    per_task_q: dict[str, float] = field(default_factory=dict)
    per_task_sr: dict[str, float] = field(default_factory=dict)
    per_task_time: dict[str, float] = field(default_factory=dict)

    overall_q: float = 0.0
    overall_task_sr: float = 0.0
    overall_time: float = 0.0

    def __post_init__(self):
        # Always fold per-task -> overall so any construction path yields a
        # complete Summary.  Cheap and idempotent.
        self._finalize()

    @classmethod
    def from_json_dir(cls, json_dir: str | Path) -> "Summary":
        """Read every ``<task>_<instance>_<rollout>.json`` under ``json_dir``."""
        d = Path(json_dir)
        files = sorted(d.glob("*.json")) if d.is_dir() else [d]
        by_task: dict[str, list[RolloutScore]] = {}
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                o = json.load(fh)
            r = RolloutScore(
                task=str(o.get("task", f.stem)),
                instance_id=int(o.get("instance_id", -1)),
                rollout_id=int(o.get("rollout_id", 0)),
                q=float(o.get("q_score", {}).get("final", 0.0)),
                time_score=time_score_from_normalized(
                    float(o.get("time", {}).get("normalized_time", 1.5))
                ),
                base_dist=float(o.get("normalized_agent_distance", {}).get("base", 0.0)),
                left_dist=float(o.get("normalized_agent_distance", {}).get("left", 0.0)),
                right_dist=float(o.get("normalized_agent_distance", {}).get("right", 0.0)),
                steps=int(o.get("steps", 0)),
                success=bool(o.get("success", False)),
            )
            by_task.setdefault(r.task, []).append(r)

        per_task_q, per_task_sr, per_task_time = {}, {}, {}
        rollouts: list[RolloutScore] = []
        for task, rs in by_task.items():
            rollouts.extend(rs)
            n = len(rs)
            per_task_q[task] = sum(r.q for r in rs) / n
            per_task_sr[task] = sum(1 for r in rs if r.q >= 1.0) / n
            per_task_time[task] = sum(r.time_score for r in rs) / n
        return cls(
            rollouts=rollouts,
            per_task_q=per_task_q,
            per_task_sr=per_task_sr,
            per_task_time=per_task_time,
        )

    def _finalize(self):
        tasks = list(self.per_task_q.keys())
        if not tasks:
            self.overall_q = 0.0
            self.overall_task_sr = 0.0
            self.overall_time = 0.0
            return
        self.overall_q = sum(self.per_task_q.values()) / len(tasks)
        self.overall_task_sr = sum(self.per_task_sr.values()) / len(tasks)
        self.overall_time = sum(self.per_task_time.values()) / len(tasks)

    # Recompute overall (call after constructing with per-task dicts).
    @property
    def overall(self) -> dict:
        if not self.per_task_q:
            return {
                "q_score": 0.0, "task_sr": 0.0, "time_score": 0.0,
                "num_rollouts": len(self.rollouts), "num_tasks": 0,
            }
        n = len(self.per_task_q)
        return {
            "q_score": sum(self.per_task_q.values()) / n,
            "task_sr": sum(self.per_task_sr.values()) / n,
            "time_score": sum(self.per_task_time.values()) / n,
            "num_rollouts": len(self.rollouts),
            "num_tasks": n,
        }


def aggregate(json_dir: str | Path) -> Summary:
    """Convenience entry: read a metrics JSON dir and compute the Summary."""
    s = Summary.from_json_dir(json_dir)
    s._finalize()
    return s
