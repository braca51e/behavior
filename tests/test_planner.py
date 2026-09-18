"""Planner tests: 31-skill taxonomy, skill-plan miner, eval-time task planner.

Covers the rule-based "System 2" brain (design.md sections 2/4.2) — all
CPU-only.  The miner is exercised on the synthetic annotation fixture
(``tests/fixtures/annotations.jsonl``, contract item (c)) and must produce a
valid MinedPlan per task with duration stats and predicate families.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b1k.planner.miner import MinedPlan, mine_plans, mine_task_plan  # noqa: E402
from b1k.planner.skill_tax import (  # noqa: E402
    SKILL_IDS,
    match_skill,
    predicate_families,
    render_text,
)
from b1k.planner.task_planner import TaskPlanner  # noqa: E402


def test_taxonomy_has_31_skills():
    assert len(SKILL_IDS) == 31
    assert len(set(SKILL_IDS)) == 31


@pytest.mark.parametrize("text,expected", [
    ("Pick up the cup", "pick_up"),
    ("Open the fridge door", "open_door"),
    ("Turn on the radio", "turn_on"),
    ("Move to the kitchen counter", "move_to"),
    ("Pour water into the glass", "pour"),
    ("Wipe the counter", "wipe"),
    ("Slice the onion", "chop"),
    ("Place the book on the shelf", "place_on"),
    ("Close the drawer", "close_drawer"),
    ("Turn off the light", "turn_off"),
    ("Ignite the candle", "ignite"),
    ("Hang the towel on the rack", "hang"),
    ("Press the button", "press"),
    ("Search for the keys", "search"),
    ("Hand over the plate", "hand_over"),
    ("Attach the shelf bracket", "attach"),
    ("Insert the key", "insert"),
    ("Tip over the vase", "tip_over"),
    ("Sweep the floor", "sweep"),
    ("Spray the bottle", "spray"),
    ("Turn to face the door", "turn_to"),
    ("Place the cup in the sink", "place_in"),
])
def test_match_skill(text, expected):
    assert match_skill(text) == expected


def test_match_skill_unknown_returns_none_or_default():
    # Completely unrelated text -> None (miner will default to move_to).
    assert match_skill("") is None


def test_render_text_fills_slots():
    t = render_text("place_on", object="cup", surface="counter")
    assert "cup" in t and "counter" in t


def test_predicate_families_known():
    assert "lit" in predicate_families("turn_on")
    assert "open" in predicate_families("open_door")
    assert "object_in_gripper" in predicate_families("pick_up")


# ---- miner on the synthetic fixture ---------------------------------------
def _load_fixture_annotations():
    lines = []
    with open(REPO / "tests" / "fixtures" / "annotations.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def test_mine_plan_task0():
    anns = _load_fixture_annotations()
    plan = mine_task_plan(anns, task_id=0, task="turning_on_radio")
    assert isinstance(plan, MinedPlan)
    # Task 0 annotations: move_to -> turn_on -> turn_on (compressed: move_to, turn_on).
    skills = [s.skill for s in plan.plan]
    assert "move_to" in skills
    assert "turn_on" in skills
    # turn_on must come after move_to (chronological order preserved).
    assert skills.index("turn_on") > skills.index("move_to")
    # Duration stats are positive and p90 budget >= mean.
    for s in plan.plan:
        assert s.mean_steps > 0
        assert s.p90_steps >= s.mean_steps
    # turn_on predicate family is "lit".
    to = next(s for s in plan.plan if s.skill == "turn_on")
    assert "lit" in to.pred_families
    # Episode mean length is positive (4 episodes ~ (10+3+2)*30 steps each).
    assert plan.episode_mean_steps > 0
    assert plan.num_episodes == 4


def test_mine_plans_multi_task(tmp_path):
    anns = _load_fixture_annotations()
    out = mine_plans(anns, tasks={0: "turning_on_radio", 1: "picking_up_trash",
                                   3: "cleaning_up_plates_and_food"},
                     out_dir=tmp_path)
    assert set(out.keys()) == {0, 1, 3}
    # Task 1: move_to, pick_up, place_in.
    skills1 = [s.skill for s in out[1].plan]
    assert "pick_up" in skills1 and "place_in" in skills1
    # JSON was written for each task.
    for tid in (0, 1, 3):
        p = tmp_path / f"{tid}.json"
        assert p.exists()
        obj = json.loads(p.read_text())
        assert obj["task_id"] == tid
        # Round-trips through MinedPlan.from_dict.
        MinedPlan.from_dict(obj)


def test_mine_plan_fallback_skill():
    # A task with only unmatched text defaults every segment to move_to.
    anns = [{"task_id": 9, "episode_index": 0, "t0": 0, "t1": 5,
             "text": "zzz qqq"}]
    plan = mine_task_plan(anns, task_id=9, task="t")
    assert len(plan.plan) >= 1
    assert plan.plan[0].skill in SKILL_IDS


# ---- eval-time task planner -----------------------------------------------
def test_task_planner_advances():
    plan = MinedPlan(
        task_id=0, task="t",
        plan=[
            _sub("move_to", mean=100, p90=150),
            _sub("pick_up", mean=60, p90=90),
            _sub("place_in", mean=80, p90=120),
        ],
        episode_mean_steps=6000,
    )
    tp = TaskPlanner(task_id=0, task_text="move to cup pick up place in sink",
                     bddl_goal="On(cup, sink)", plan=plan)
    assert len(tp.initial_subgoals()) == 3
    assert tp.current().skill == "move_to"
    tp.advance()
    assert tp.current().skill == "pick_up"
    tp.advance()
    tp.advance()
    assert tp.done()
    # current() stays on the last item when done.
    assert tp.current().skill == "place_in"


def test_task_planner_replan_keeps_remaining():
    plan = MinedPlan(task_id=0, task="t",
                     plan=[_sub("move_to", 100, 150), _sub("pick_up", 60, 90)],
                     episode_mean_steps=3000)
    tp = TaskPlanner(0, "move pick", "", plan)
    tp.advance()  # now on pick_up
    remaining = tp.replan()
    assert [s.skill for s in remaining] == ["pick_up"]
    assert tp.current().skill == "pick_up"


def test_task_planner_fallback_without_plan():
    tp = TaskPlanner(0, "move to the radio", "", plan=None)
    # No mined plan -> a single move_to fallback toward the task noun.
    assert tp.current().skill == "move_to"
    assert "radio" in tp.current().text


def _sub(skill, mean, p90):
    from b1k.planner.miner import MinedSubgoal

    return MinedSubgoal(skill=skill, args={}, text=skill, mean_steps=mean,
                        p90_steps=p90, pred_families=list(predicate_families(skill)),
                        count=1)
