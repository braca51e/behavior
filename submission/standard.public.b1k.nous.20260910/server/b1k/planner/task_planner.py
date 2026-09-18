"""Eval-time task planner (design.md section 4.2).

Turns a loaded :class:`~b1k.planner.miner.MinedPlan` (static, from
``docs/plans/<task_id>.json``) into an ordered, *active* subgoal queue that the
:mod:`~b1k.planner.progress` state machine drives.  No language model is used at
eval time — everything here is deterministic rule-based progression.

Public surface:
    Subgoal            — one active step (skill + rendered text + budgets)
    TaskPlanner        — initial_subgoals / current / advance / replan / done
    load_mined_plan    — read the per-task JSON (CPU-only)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ServerConfig
from .miner import MinedPlan
from .skill_tax import SKILLS, predicate_families, render_text


@dataclass
class Subgoal:
    skill: str
    args: dict = field(default_factory=dict)
    text: str = ""
    budget_steps: int = 0          # p90 demo duration x budget_mult (in 30 Hz steps)
    mean_steps: int = 0            # mined mean (for RETRY cap = min(budget, 3*mean))
    pred_families: tuple[str, ...] = ()
    retry_cap_steps: int = 0       # min(budget, 3*mean)

    def __post_init__(self):
        if self.skill not in SKILLS:
            # Unknown mined skill: fall back to a generic move so serving never
            # crashes on a malformed plan.
            self.skill = "move_to"
        if not self.text:
            self.text = render_text(self.skill, **self.args)
        if self.pred_families is None:
            self.pred_families = ()
        if not self.pred_families:
            self.pred_families = predicate_families(self.skill)
        if self.retry_cap_steps <= 0:
            self.retry_cap_steps = (
                min(self.budget_steps, int(3 * self.mean_steps))
                if self.mean_steps > 0
                else max(self.budget_steps, 1)
            )

    @property
    def budget_seconds(self) -> float:
        return self.budget_steps / 30.0


def load_mined_plan(task_id: int, cfg: ServerConfig) -> MinedPlan | None:
    """Load ``docs/plans/<task_id>.json`` if present, else None (planner falls
    back to a minimal move_to-only plan built from the task text)."""
    p = Path(cfg.planner.plans_dir) / f"{int(task_id)}.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return MinedPlan.from_dict(json.load(f))
        except Exception:
            return None
    return None


class TaskPlanner:
    """Drives the active subgoal queue for one task/episode."""

    def __init__(self, task_id: int, task_text: str, bddl_goal: str, plan: MinedPlan | None):
        self.task_id = int(task_id)
        self.task_text = task_text or ""
        self.bddl_goal = bddl_goal or ""
        self.plan = plan
        self.queue: list[Subgoal] = []
        self._idx = 0
        self._advance_subgoal_index = 0   # which queue item we are on
        self._init()

    def _init(self):
        if self.plan is not None and self.plan.plan:
            for i, s in enumerate(self.plan.plan):
                self.queue.append(
                    Subgoal(
                        skill=s.skill,
                        args=dict(s.args or {}),
                        text=s.text or render_text(s.skill, object=""),
                        budget_steps=s.p90_steps or int(s.mean_steps * 1.5) or 300,
                        mean_steps=s.mean_steps,
                        pred_families=tuple(s.pred_families or ()) ,
                    )
                )
            if self.queue[0].text == "" or self.queue[0].text == render_text(self.queue[0].skill, object=""):
                # Give the first subgoal a concrete object from the task text.
                self.queue[0].text = render_text(self.queue[0].skill, object=self._first_object())
        else:
            # Minimal fallback: a single move_to toward the task noun.
            self.queue = [
                Subgoal(
                    skill="move_to",
                    args={"object": self._first_object()},
                    text=render_text("move_to", object=self._first_object()),
                    budget_steps=600,
                    mean_steps=300,
                )
            ]
        self._advance_subgoal_index = 0

    def _first_object(self) -> str:
        # Naive: last content word of the task description is usually the target.
        words = [w for w in self.task_text.replace(",", " ").split() if w.isalpha()]
        return words[-1].lower() if words else "target"

    # ---- public API ---------------------------------------------------------
    def initial_subgoals(self) -> list[Subgoal]:
        return list(self.queue)

    def current(self) -> Subgoal:
        if self._advance_subgoal_index < len(self.queue):
            return self.queue[self._advance_subgoal_index]
        return self.queue[-1]

    def done(self) -> bool:
        return self._advance_subgoal_index >= len(self.queue)

    def advance(self) -> None:
        """Move to the next queued subgoal (called by progress on ADVANCE).

        Increments past the end so :meth:`done` can observe completion;
        :meth:`current` clamps to the last item when finished.
        """
        self._advance_subgoal_index += 1

    def replan(self) -> list[Subgoal]:
        """Re-derive the remaining queue: keep the current item, drop satisfied
        predicates' skills from the front if present, and fall back to a final
        move_to to keep the robot productive when the mined plan ran dry."""
        remaining = list(self.queue[self._advance_subgoal_index:])
        if not remaining:
            remaining = [
                Subgoal(
                    skill="move_to",
                    args={"object": self._first_object()},
                    text=render_text("move_to", object=self._first_object()),
                    budget_steps=600,
                    mean_steps=300,
                )
            ]
        self.queue = remaining
        self._advance_subgoal_index = 0
        return self.queue

    def progress_index(self) -> int:
        return self._advance_subgoal_index
