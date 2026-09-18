"""Subgoal progress state machine + watchdogs (design.md section 4.3).

Deterministic, CPU-only.  Consumes the predicate-satisfaction map from the
perception layer and the active :class:`~b1k.planner.task_planner.Subgoal`, and
emits one :class:`Action` per eval step:

    CONTINUE  — keep executing the current subgoal
    ADVANCE   — the subgoal's exit predicates are satisfied; move to the next
    RETRY     — stuck (zero predicate delta over the stuck window); re-execute
                the same subgoal (bounded number of retries)
    REPLAN    — retries exhausted; re-derive the remaining queue
    SEARCH    — navigation subgoal, target unseen for ``search_after_s``;
                emit frontier-search bias (consumed by odometry/controller)
    FINISH    — all goal predicates satisfied (early stop) OR the global time
                budget (1.5x human mean episode length) is exhausted; park

``FINISH`` is the efficiency engine: once every goal predicate is satisfied the
server switches to a safe park pose + no-op actions, so the reported
``normalized_time`` stays well under the 1.5x timeout and ``time_score`` > 0.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import ProgressConfig
from .task_planner import Subgoal

log = logging.getLogger("b1k.progress")


class Act:
    """Action kinds (string constants — trivially JSON/loggable)."""

    CONTINUE = "CONTINUE"
    ADVANCE = "ADVANCE"
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    SEARCH = "SEARCH"
    FINISH = "FINISH"


@dataclass
class Action:
    kind: str
    reason: str = ""
    park: bool = False          # True -> controller serves safe no-op / hold
    requery: bool = False       # True -> controller must re-query the VLA now
    search_bias: bool = False   # True -> odometry/controller biases to frontier
    extra: dict = field(default_factory=dict)


# Navigation-family skills: target "positioned" predicates drive SEARCH/ADVANCE.
_NAV_SKILLS = {"move_to", "turn_to", "search"}

# Detector family -> BDDL predicate names it can satisfy (lowercase, matched as
# a *prefix* of the normalized goal atom).  A BDDL atom normalizes to
# "<predicate><objects...>" (alnum, lowercase), so prefixing on the predicate
# name is precise and avoids substring false-positives (e.g. "in" inside
# "kitchen").
_FAMILY_HINTS: dict[str, tuple[str, ...]] = {
    "lit": ("powered", "lit", "on", "switchon"),
    "unlit": ("powered", "lit", "off", "switchoff"),
    "open": ("open",),
    "closed": ("closed",),
    "contains": ("contains",),
    "empty": ("empty",),
    "object_in_gripper": ("grasped",),
    "attached": ("attachedto",),
    "cut": ("cut",),
    "cleaned": ("cleaned",),
    "sprayed": ("sprayed",),
    "contact": ("contact",),
    "positioned": ("on", "in", "nextto"),
}


class Progress:
    """Tracks subgoal progress and fires watchdog actions.

    Parameters
    ----------
    cfg:
        :class:`ProgressConfig` (hysteresis, budget multipliers, early stop).
    goal_predicates:
        Raw BDDL goal predicate strings for this task, e.g.
        ``["On(Radio, KitchenCounter)", "Powered(Radio)"]``.  These are the
        predicates that ``q_score.final`` counts — satisfying all of them
        yields FINISH.
    episode_mean_steps:
        Human mean episode length in 30 Hz sim steps (from the mined plan /
        task.jsonl).  Drives the global budget.  0 -> no global cap.
    """

    def __init__(
        self,
        cfg: ProgressConfig,
        goal_predicates: list[str],
        episode_mean_steps: float,
        conf_floor: float = 0.6,
        stuck_steps: int = 900,
    ):
        self.cfg = cfg
        self.conf_floor = float(conf_floor)
        self.stuck_steps = int(stuck_steps)
        self.goal_predicates = [g.strip() for g in goal_predicates if g and g.strip()]
        # Normalized id: lowercase, no spaces/parens variance.
        self._goal_ids = [self._norm_id(g) for g in self.goal_predicates]
        self.episode_mean_steps = float(episode_mean_steps)
        self.global_budget_steps = int(self.episode_mean_steps * cfg.global_budget_mult) \
            if self.episode_mean_steps > 0 else 0

        self.satisfied: set[str] = set()
        self.satisfied_conf: dict[str, float] = {}
        self.finished = False
        # Step at which the satisfied-goal set last *grew* (for the FINISH
        # stability window).  Distinct from the per-subgoal delta step: with a
        # single goal predicate the per-subgoal delta stays positive forever
        # after the flip, so FINISH must key off the last satisfaction change.
        self._last_satisfied_change = -30

        # Per-subgoal watchdog state.
        self._subgoal_start_step = 0
        self._satisfied_at_subgoal_start: set[str] = set()
        self._last_delta_step = 0
        self._stuck_count = 0
        self._retry_count = 0
        self._target_seen_recent = False
        self._subgoal = None

    # ---- predicate helpers --------------------------------------------------
    @staticmethod
    def _norm_id(pred: str) -> str:
        return "".join(ch for ch in pred.lower() if ch.isalnum())

    @staticmethod
    def _match_goal(pred_id: str, goal_ids: list[str]) -> int | None:
        """Which goal predicate does ``pred_id`` satisfy?

        Detector keys are family names (e.g. ``"lit"``) or
        ``"family:object"`` (e.g. ``"lit:radio"``).  Goal predicates are BDDL
        atoms (``"Powered(Radio)"``) normalized to ``"<predicate><objects...>"``
        (alnum, lowercase).  A detector key matches a goal atom when the goal's
        predicate name is one of the family's hints (:data:`_FAMILY_HINTS`,
        prefix match) and, if the key names an object, that object appears in
        the goal atom.  Returns the goal index or None.
        """
        if ":" in pred_id:
            family, _, obj = pred_id.partition(":")
        else:
            family, obj = pred_id, ""
        family = family.lower().strip()
        obj = obj.lower().strip()
        hints = _FAMILY_HINTS.get(family, ())
        if not hints:
            return None
        for i, g in enumerate(goal_ids):
            if not any(g == h or g.startswith(h) for h in hints):
                continue
            if obj and obj not in g:
                continue
            return i
        return None

    def _update_satisfied(self, pred_state: dict[str, tuple[bool, float]],
                          step: int) -> None:
        """Fold detector state into the satisfied-goal set (conf floor)."""
        floor = self.conf_floor
        new_sat: set[int] = set()
        for key, (sat, conf) in pred_state.items():
            if not sat or conf < floor:
                continue
            idx = self._match_goal(key, self._goal_ids)
            if idx is not None:
                new_sat.add(idx)
        grown = new_sat - self.satisfied
        # Hysteresis: a predicate stays satisfied once it flips (no de-satisfaction
        # noise); this is intentional — BDDL state transitions in these tasks are
        # monotone in practice and de-flipping causes spurious RETRY loops.
        if grown:
            self.satisfied = self.satisfied | new_sat
            self._last_satisfied_change = step
        for key, (sat, conf) in pred_state.items():
            if sat and conf >= floor:
                self.satisfied_conf[self._norm_id(key)] = max(
                    self.satisfied_conf.get(self._norm_id(key), 0.0), conf
                )

    # ---- lifecycle ----------------------------------------------------------
    def begin_subgoal(self, subgoal: Subgoal, step: int) -> None:
        """Called by the controller when a (new/retried) subgoal becomes active."""
        self._subgoal = subgoal
        self._subgoal_start_step = int(step)
        self._satisfied_at_subgoal_start = set(self.satisfied)
        self._last_delta_step = int(step)
        self._stuck_count = 0
        if subgoal.skill not in _NAV_SKILLS:
            self._retry_count = 0

    @property
    def all_goal_satisfied(self) -> bool:
        return len(self.satisfied) >= len(self._goal_ids) and len(self._goal_ids) > 0

    # ---- main tick ----------------------------------------------------------
    def tick(
        self,
        pred_state: dict[str, tuple[bool, float]],
        subgoal: Subgoal,
        step: int,
        target_seen: bool = False,
    ) -> Action:
        """One eval step.  Returns the watchdog action for the controller.

        Semantics (design.md section 4.3):
          * FINISH  — all goal predicates satisfied (early stop) or the global
                      time budget is exhausted.  ``park=True``.
          * ADVANCE — the current subgoal produced predicate progress (delta>0).
          * RETRY   — stuck (no predicate delta over the window); re-execute the
                      same subgoal, up to 2 retries.
          * REPLAN  — retries exhausted; re-derive the remaining queue.
          * SEARCH  — navigation subgoal whose target has not been seen; bias
                      the controller/odometry toward frontier exploration.
          * CONTINUE — default.
        """
        if self.finished:
            return Action(Act.FINISH, reason="already finished", park=True)

        self._update_satisfied(pred_state, step)
        if self._subgoal is None:
            self.begin_subgoal(subgoal, step)
        self._target_seen_recent = self._target_seen_recent or target_seen

        n_goal = len(self._goal_ids)

        # 1) Early stop: every goal predicate satisfied (stable for >=1 s).
        if n_goal > 0 and self.cfg.early_stop and self.all_goal_satisfied:
            if step - self._last_satisfied_change >= 30:
                self.finished = True
                log.info("FINISH @step %d: all %d goal predicates satisfied", step, n_goal)
                return Action(Act.FINISH, reason="all goal predicates satisfied", park=True)

        # 2) Global time budget exhausted -> park (still reports partial Q).
        if self.global_budget_steps and step >= self.global_budget_steps:
            self.finished = True
            log.info("FINISH @step %d: global budget %d exhausted", step, self.global_budget_steps)
            return Action(Act.FINISH, reason="global time budget exhausted", park=True)

        # 3) Subgoal progress / stuck detection.
        since_start = step - self._subgoal_start_step
        delta = len(self.satisfied - self._satisfied_at_subgoal_start)
        if delta > 0:
            self._last_delta_step = step
            self._stuck_count = 0

        budget = subgoal.budget_steps if subgoal.budget_steps > 0 else 600
        window = max(self.stuck_steps, 60)
        stuck = delta == 0 and (step - self._last_delta_step) >= window
        over_budget = since_start >= budget

        # 4) ADVANCE: the subgoal made real predicate progress.
        if delta > 0 and not stuck:
            self._retry_count = 0
            log.info("ADVANCE @step %d: subgoal %s progressed (delta=%d)", step, subgoal.skill, delta)
            return Action(Act.ADVANCE, reason=f"subgoal {subgoal.skill} advanced", requery=True)

        # 5) SEARCH: navigation subgoal, target unseen, and stuck/over-budget.
        if subgoal.skill in _NAV_SKILLS and not self._target_seen_recent and (stuck or over_budget):
            log.info("SEARCH @step %d: nav subgoal %s, target unseen", step, subgoal.skill)
            return Action(Act.SEARCH, reason="target unseen during nav",
                          requery=True, search_bias=True)

        # 6) RETRY (<=2) then REPLAN.
        if stuck or over_budget:
            if self._retry_count < 2:
                self._retry_count += 1
                log.info("RETRY @step %d: subgoal %s (retry %d/2)", step, subgoal.skill, self._retry_count)
                return Action(Act.RETRY, reason=f"stuck in {subgoal.skill}", requery=True)
            self._retry_count = 0
            log.info("REPLAN @step %d: subgoal %s retries exhausted", step, subgoal.skill)
            return Action(Act.REPLAN, reason=f"retries exhausted in {subgoal.skill}", requery=True)

        return Action(Act.CONTINUE)
