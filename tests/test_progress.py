"""Progress state-machine tests (design.md section 4.3).

Validates the Q-/efficiency engine: predicate gating, ADVANCE on progress,
RETRY/REPLAN on stuck, SEARCH on unseen nav targets, and FINISH early-stop
when all goal predicates are satisfied.  All deterministic, CPU-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b1k.config import ProgressConfig  # noqa: E402
from b1k.planner.progress import Act, Progress  # noqa: E402
from b1k.planner.task_planner import Subgoal  # noqa: E402


def _cfg(**kw):
    base = dict(hysteresis_frames=3, global_budget_mult=1.5, early_stop=True,
                search_after_s=5.0)
    base.update(kw)
    return ProgressConfig(**base)


def _sub(skill="move_to", budget=100, mean=50, fams=()):
    return Subgoal(skill=skill, args={}, text=skill, budget_steps=budget,
                   mean_steps=mean, pred_families=tuple(fams))


def test_continue_when_no_progress_and_within_budget():
    pr = Progress(_cfg(), goal_predicates=["Powered(Radio)"],
                  episode_mean_steps=1000, conf_floor=0.6, stuck_steps=900)
    pr.begin_subgoal(_sub(budget=1000), step=0)
    act = pr.tick({}, _sub(budget=1000), step=10)
    assert act.kind == Act.CONTINUE
    assert not act.park


def test_finish_when_all_goals_satisfied():
    pr = Progress(_cfg(), goal_predicates=["Powered(Radio)"],
                  episode_mean_steps=1000, conf_floor=0.6, stuck_steps=900)
    sg = _sub(skill="turn_on", budget=2000, fams=("lit",))
    pr.begin_subgoal(sg, step=0)
    # Emulate the server loop: on ADVANCE, advance + begin the next subgoal so
    # the per-subgoal delta baseline resets (FINISH then fires after the 30-step
    # stability window once all goals are satisfied).
    act = None
    step = 0
    for _ in range(200):
        act = pr.tick({"lit": (True, 0.95)}, sg, step=step)
        if act.kind == Act.ADVANCE:
            pr.begin_subgoal(sg, step)  # subgoal done; next one starts
        if act.kind == Act.FINISH:
            break
        step += 30
    assert act is not None and act.kind == Act.FINISH
    assert act.park
    assert pr.finished
    assert pr.all_goal_satisfied


def test_finish_respects_early_stop_off():
    pr = Progress(_cfg(early_stop=False), goal_predicates=["Powered(Radio)"],
                  episode_mean_steps=1000, conf_floor=0.6, stuck_steps=900)
    sg = _sub(skill="turn_on", budget=2000, fams=("lit",))
    pr.begin_subgoal(sg, step=0)
    act = pr.tick({"lit": (True, 0.95)}, sg, step=10)
    # early_stop off -> no FINISH from satisfaction; may ADVANCE (progress).
    assert act.kind != Act.FINISH


def test_advance_on_progress():
    pr = Progress(_cfg(), goal_predicates=["Grasped(cup)", "On(cup, sink)"],
                  episode_mean_steps=1000, conf_floor=0.6, stuck_steps=900)
    sg = _sub(skill="pick_up", budget=2000, fams=("object_in_gripper",))
    pr.begin_subgoal(sg, step=0)
    # No progress yet.
    assert pr.tick({}, sg, step=5).kind == Act.CONTINUE
    # Progress: object_in_gripper satisfied.
    act = pr.tick({"object_in_gripper": (True, 0.9)}, sg, step=40)
    assert act.kind == Act.ADVANCE
    assert act.requery


def test_retry_then_replan_on_stuck():
    pr = Progress(_cfg(), goal_predicates=["On(cup, sink)"],
                  episode_mean_steps=1000, conf_floor=0.6, stuck_steps=60)
    sg = _sub(skill="place_in", budget=5000, fams=("contains",))
    pr.begin_subgoal(sg, step=0)
    seen = []
    for step in range(0, 2000):
        act = pr.tick({}, sg, step=step)
        seen.append(act.kind)
        if act.kind in (Act.REPLAN, Act.FINISH):
            break
    assert Act.RETRY in seen
    assert Act.REPLAN in seen


def test_search_on_unseen_nav_target():
    pr = Progress(_cfg(), goal_predicates=["On(cup, sink)"],
                  episode_mean_steps=1000, conf_floor=0.6, stuck_steps=60)
    sg = _sub(skill="move_to", budget=5000, fams=("positioned",))
    pr.begin_subgoal(sg, step=0)
    act = pr.tick({}, sg, step=60, target_seen=False)
    # Nav subgoal, no predicate delta, target unseen -> SEARCH bias.
    assert act.kind == Act.SEARCH
    assert act.search_bias
    assert act.requery


def test_no_search_when_target_seen():
    pr = Progress(_cfg(), goal_predicates=["On(cup, sink)"],
                  episode_mean_steps=1000, conf_floor=0.6, stuck_steps=60)
    sg = _sub(skill="move_to", budget=5000, fams=("positioned",))
    pr.begin_subgoal(sg, step=0)
    act = pr.tick({}, sg, step=60, target_seen=True)
    # Target seen -> not a search; stuck nav falls to RETRY.
    assert act.kind != Act.SEARCH


def test_global_budget_forces_finish():
    pr = Progress(_cfg(), goal_predicates=["On(cup, sink)"],
                  episode_mean_steps=100, conf_floor=0.6, stuck_steps=900)
    sg = _sub(skill="move_to", budget=10**6, fams=("positioned",))
    pr.begin_subgoal(sg, step=0)
    # Global budget = 1.5*100 = 150 steps.
    act = pr.tick({}, sg, step=160)
    assert act.kind == Act.FINISH
    assert pr.finished


def test_finished_stays_finished():
    pr = Progress(_cfg(), goal_predicates=["Powered(Radio)"],
                  episode_mean_steps=100, conf_floor=0.6, stuck_steps=900)
    sg = _sub(skill="turn_on", budget=10**6, fams=("lit",))
    pr.begin_subgoal(sg, step=0)
    pr.tick({"lit": (True, 0.95)}, sg, step=10)
    pr.tick({"lit": (True, 0.95)}, sg, step=50)  # stability window passes
    assert pr.finished
    # Subsequent ticks keep returning FINISH/park.
    for s in (60, 70, 80):
        act = pr.tick({}, sg, step=s)
        assert act.kind == Act.FINISH and act.park


def test_low_confidence_does_not_satisfy():
    pr = Progress(_cfg(), goal_predicates=["Powered(Radio)"],
                  episode_mean_steps=1000, conf_floor=0.6, stuck_steps=900)
    sg = _sub(skill="turn_on", budget=10**6, fams=("lit",))
    pr.begin_subgoal(sg, step=0)
    act = pr.tick({"lit": (True, 0.3)}, sg, step=40)
    # conf 0.3 < floor 0.6 -> not satisfied -> not FINISH.
    assert act.kind != Act.FINISH
    assert not pr.all_goal_satisfied
