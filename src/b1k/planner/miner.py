"""Training-time skill-plan miner (design.md section 4.2).

Consumes the per-episode *language annotations* shipped with the LeRobot v3
dataset (each annotation segment = (t0, t1, text)) and clusters the free-text
into the 31-skill taxonomy, producing a **MinedPlan** per task: an ordered
skill template with per-subgoal duration statistics (mean / p90, in sim steps)
and the BDDL predicate families each subgoal is expected to flip.

The miner is the only place free text -> canonical skills happens; at eval time
the planner reads the mined JSON and never re-matches language.  The output
layout is::

    docs/plans/<task_id>.json
        {"task_id": 0, "task": "turning_on_radio",
         "plan": [{"skill": "move_to", "args": {...}, "text": "move to radio",
                   "mean_steps": 180, "p90_steps": 270,
                   "pred_families": ["positioned"]}, ...],
         "episode_mean_steps": 21000, "num_episodes": 200}

Input is a JSONL "annotation file" (see ``tests/fixtures/annotations.jsonl``):
one line per episode segment::

    {"task_id": 0, "episode_index": 0, "t0": 0, "t1": 120, "text": "move to radio"}

A real dataset read uses the same line schema (see
``src/training/skill_annotations.py`` for the LeRobot-side reader that emits it).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np

from .skill_tax import SKILLS, match_skill, predicate_families, render_text

FPS = 30.0


@dataclass
class MinedSubgoal:
    skill: str
    args: dict = field(default_factory=dict)
    text: str = ""
    mean_steps: int = 0
    p90_steps: int = 0
    pred_families: list[str] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "args": self.args,
            "text": self.text,
            "mean_steps": int(self.mean_steps),
            "p90_steps": int(self.p90_steps),
            "pred_families": list(self.pred_families),
            "count": int(self.count),
        }


@dataclass
class MinedPlan:
    task_id: int
    task: str
    plan: list[MinedSubgoal]
    episode_mean_steps: float = 0.0
    num_episodes: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "plan": [s.to_dict() for s in self.plan],
            "episode_mean_steps": round(float(self.episode_mean_steps), 1),
            "num_episodes": int(self.num_episodes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MinedPlan":
        return cls(
            task_id=int(d["task_id"]),
            task=str(d.get("task", str(d["task_id"]))),
            plan=[MinedSubgoal(**s) for s in d.get("plan", [])],
            episode_mean_steps=float(d.get("episode_mean_steps", 0.0)),
            num_episodes=int(d.get("num_episodes", 0)),
        )


def _extract_object(text: str) -> str:
    """Best-effort target-object slot: the first content noun after the verb.

    Very small heuristic (strip determiners + verbs); the planner is tolerant
    of empty slots (renders the raw word).  Improving slot quality is a
    training-time refinement and never affects the serving contract.
    """
    stop = {
        "the", "a", "an", "this", "that", "it", "on", "in", "to", "from",
        "near", "next", "under", "over", "and", "with", "of", "for", "at",
        "by", "then", "after", "before", "up", "down", "into", "onto",
    }
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    obj = ""
    for w in words:
        if w in stop or re.match(r"^(move|pick|place|open|close|turn|pour|wipe|spray|search|hand|hang|push|tip|attach|insert|ignite|press|hold|release|slice|chop|sweep|find|look|grasp)$", w):
            continue
        obj = w
        break
    return obj


def mine_task_plan(
    annotations: list[dict],
    task_id: int,
    task: str,
    budget_mult: float = 1.5,
) -> MinedPlan:
    """Mine a per-task skill plan from raw annotation segments.

    Steps:
      1. Map every segment text to a canonical skill (``match_skill``).
      2. Compress consecutive runs of the same skill into one subgoal,
         taking the *first* segment's text and the summed duration.
      3. Order subgoals by first appearance across episodes, then aggregate
         duration statistics (mean / p90) and apply the p90 budget multiplier.
      4. Attach the predicate families the skill may flip.
    """
    by_ep: dict[int, list[tuple[float, str, float, str]]] = defaultdict(list)
    for a in annotations:
        if int(a.get("task_id", task_id)) != int(task_id):
            continue
        text = str(a.get("text", "")).strip()
        t0 = float(a.get("t0", 0.0))
        t1 = float(a.get("t1", 0.0))
        dur = max(t1 - t0, 1.0 / FPS)
        ep = int(a.get("episode_index", -1))
        skill = match_skill(text) or "move_to"
        by_ep[ep].append((t0, text, dur, skill))

    episode_lens: list[float] = []
    skill_order: list[str] = []
    for ep in sorted(by_ep):
        segs = sorted(by_ep[ep], key=lambda s: s[0])  # chronological
        # Compress runs of identical skill: keep first text, sum duration.
        compressed: list[tuple[str, float, str]] = []
        for _t0, text, dur, skill in segs:
            if compressed and compressed[-1][2] == skill:
                pt, pd, _ = compressed[-1]
                compressed[-1] = (pt, pd + dur, skill)
            else:
                compressed.append((text, dur, skill))
        for _text, _dur, skill in compressed:
            if skill not in skill_order:
                skill_order.append(skill)
        episode_lens.append(sum(d for _t, d, _s in compressed))

    per_episode = [sorted(by_ep[ep], key=lambda s: s[0]) for ep in sorted(by_ep)]

    # Aggregate per-skill durations across all episodes (a skill can recur).
    agg: dict[str, list[float]] = defaultdict(list)
    for ep_segs in per_episode:
        runs: list[tuple[str, float]] = []
        for _t0, _text, dur, skill in ep_segs:
            if runs and runs[-1][0] == skill:
                runs[-1] = (skill, runs[-1][1] + dur)
            else:
                runs.append((skill, dur))
        for skill, dur in runs:
            agg[skill].append(dur)

    plan: list[MinedSubgoal] = []
    for skill in skill_order:
        durs = np.asarray(agg[skill], dtype=np.float64)
        mean_steps = float(mean(durs)) * FPS
        p90_steps = float(np.percentile(durs, 90)) * FPS
        plan.append(
            MinedSubgoal(
                skill=skill,
                args={"object": ""},
                text=render_text(skill, object=""),
                mean_steps=int(round(mean_steps)),
                p90_steps=int(round(p90_steps * budget_mult)),
                pred_families=list(predicate_families(skill)),
                count=int(len(durs)),
            )
        )
    ep_mean = float(mean(episode_lens) * FPS) if episode_lens else 0.0
    return MinedPlan(
        task_id=task_id,
        task=task,
        plan=plan,
        episode_mean_steps=ep_mean,
        num_episodes=len(per_episode),
    )


def mine_plans(
    annotation_lines: list[dict] | Path | str,
    tasks: dict[int, str],
    out_dir: str | Path | None = None,
    budget_mult: float = 1.5,
) -> dict[int, MinedPlan]:
    """Mine plans for all (or the given) tasks; optionally write JSON to out_dir.

    ``annotation_lines`` may be a path to a JSONL file (one annotation per
    line) or an already-loaded list of dicts.
    """
    if isinstance(annotation_lines, (Path, str)):
        lines = []
        with open(annotation_lines, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
    else:
        lines = list(annotation_lines)

    out = {}
    for task_id, task in sorted(tasks.items()):
        plan = mine_task_plan(lines, task_id, task, budget_mult=budget_mult)
        out[task_id] = plan
        if out_dir is not None:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{task_id}.json", "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, indent=2)
    return out
