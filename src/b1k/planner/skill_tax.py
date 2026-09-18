"""Canonical 31-skill taxonomy (design.md section 2 / challenge spec section 4.4).

The demo set contains 270,600 skill instances across 31 unique skills,
~27.06 skills per trajectory, ~5.9 min mean length.  This module defines the
skill ids, their language-annotation verb patterns (used by :mod:`miner`), the
default subgoal text template, and the BDDL predicate *families* each skill is
expected to flip (used by :mod:`progress` for gating and early-stop).

The taxonomy is deliberately closed: the miner clusters free-text annotations
into one of these 31 ids, and the planner only ever emits one of these ids.
Keeping the set fixed makes the planner deterministic and CPU-testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    skill_id: str                 # canonical id (see SKILL_IDS)
    pattern: str                  # regex to match the annotation text
    text: str = ""                # language subgoal template ("{object}" slot)
    default_len_s: float = 8.0    # fallback mean subgoal duration if unmined
    pred_families: tuple[str, ...] = ()   # BDDL families it may flip


def _sk(skill_id, pattern, text, default_len_s=8.0, pred_families=()):
    return Skill(skill_id=skill_id, pattern=pattern, text=text,
                 default_len_s=default_len_s, pred_families=tuple(pred_families))


SKILLS: dict[str, Skill] = {
    # --- navigation / positioning ---
    "move_to": _sk("move_to", r"move (?:to|toward|near)", "move to {object}", 6.0, ("positioned",)),
    "turn_to": _sk("turn_to", r"turn (?:to|toward|facing)", "turn to face {object}", 2.0, ()),
    "search": _sk("search", r"search|look for|find", "search for {object}", 12.0, ()),
    # --- pick / place ---
    "pick_up": _sk("pick_up", r"pick up|pick (?:it|this|that) up|grasp", "pick up {object}", 5.0, ("object_in_gripper",)),
    "release": _sk("release", r"release", "release {object}", 2.0, ()),
    "place_in": _sk("place_in", r"place .* in(?!to)", "place {object} in {container}", 6.0, ("contains",)),
    "place_on": _sk("place_on", r"place .* on", "place {object} on {surface}", 6.0, ("positioned",)),
    "place_next_to": _sk("place_next_to", r"place .* next to", "place {object} next to {surface}", 6.0, ("positioned",)),
    "place_under": _sk("place_under", r"place .* under", "place {object} under {surface}", 6.0, ("positioned",)),
    "hand_over": _sk("hand_over", r"hand over", "hand {object} to {hand}", 3.0, ()),
    # --- open / close ---
    "open_door": _sk("open_door", r"open .*(?:door)", "open {object}", 5.0, ("open",)),
    "close_door": _sk("close_door", r"close .*(?:door)", "close {object}", 5.0, ("closed",)),
    "open_drawer": _sk("open_drawer", r"open .*(?:drawer|cabinet)", "open {object}", 4.0, ("open",)),
    "close_drawer": _sk("close_drawer", r"close .*(?:drawer|cabinet)", "close {object}", 4.0, ("closed",)),
    "open_lid": _sk("open_lid", r"open .*(?:lid|fridge|oven|microwave)", "open {object}", 5.0, ("open",)),
    "close_lid": _sk("close_lid", r"close .*(?:lid|fridge|oven|microwave)", "close {object}", 5.0, ("closed",)),
    # --- toggle / appliances ---
    "turn_on": _sk("turn_on", r"turn (?:.* )?on|switch on|turn on", "turn on {object}", 3.0, ("lit",)),
    "turn_off": _sk("turn_off", r"turn (?:.* )?off|switch off|turn off", "turn off {object}", 3.0, ("unlit",)),
    "press": _sk("press", r"press", "press {object}", 2.0, ()),
    "ignite": _sk("ignite", r"ignite|light (?:a )?(?:match|candle|stove|flame)", "ignite {object}", 4.0, ("lit",)),
    # --- manipulation / dexterous ---
    "hold": _sk("hold", r"hold", "hold {object}", 3.0, ()),
    "attach": _sk("attach", r"attach", "attach {object_a} to {object_b}", 5.0, ("attached",)),
    "insert": _sk("insert", r"insert", "insert {object_a} into {object_b}", 4.0, ("contains",)),
    "pour": _sk("pour", r"pour", "pour {object_a} into {object_b}", 6.0, ("contains",)),
    "chop": _sk("chop", r"chop|slice|cut", "chop {object}", 6.0, ("cut",)),
    "tip_over": _sk("tip_over", r"tip over", "tip over {object}", 3.0, ("positioned",)),
    "hang": _sk("hang", r"hang", "hang {object} on {surface}", 5.0, ("positioned",)),
    "push_to": _sk("push_to", r"push .* to", "push {object} to {surface}", 4.0, ("positioned",)),
    "wipe": _sk("wipe", r"wipe", "wipe {surface}", 6.0, ("cleaned",)),
    "sweep": _sk("sweep", r"sweep", "sweep {surface}", 5.0, ("cleaned",)),
    "spray": _sk("spray", r"spray", "spray {object}", 3.0, ("sprayed",)),
}

SKILL_IDS: tuple[str, ...] = tuple(SKILLS.keys())
assert len(SKILL_IDS) == 31, f"expected 31 canonical skills, got {len(SKILL_IDS)}"

# Pre-compiled, in SKILL_IDS order (ties break toward the earlier skill).
_COMPILED: tuple[tuple[str, "re.Pattern[str]", int], ...] = tuple(
    (sid, re.compile(SKILLS[sid].pattern, re.IGNORECASE), i)
    for i, sid in enumerate(SKILL_IDS)
)


def _best_span(s: str) -> str | None:
    """Skill id whose pattern matches ``s`` with the longest span (earliest on tie)."""
    winner_id = None
    winner_len = -1
    for sid, rx, order in _COMPILED:
        m = rx.search(s)
        if not m:
            continue
        span = len(m.group(0))
        if span > winner_len:
            winner_id, winner_len = sid, span
    return winner_id


def match_skill(text: str) -> str | None:
    """Return the canonical skill id for an annotation fragment, or None.

    A fragment may contain several verbs (e.g. "pick up the cup and place it
    on the counter"); the skill with the *longest* matched span wins.  If no
    pattern matches the full fragment, the first two words are retried (handles
    bare-verb annotations that the operator wrote with extra context stripped
    by the miner).
    """
    t = (text or "").strip()
    if not t:
        return None
    winner = _best_span(t)
    if winner is None:
        words = t.split()
        if len(words) > 2:
            winner = _best_span(" ".join(words[:2]))
    return winner


def render_text(skill_id: str, **slots) -> str:
    """Render a skill's language subgoal, filling {slots}; missing -> slot name."""
    sk = SKILLS.get(skill_id)
    if sk is None:
        return skill_id
    return re.sub(r"{([a-z_]+)}", lambda m: str(slots.get(m.group(1), m.group(1))), sk.text)


def predicate_families(skill_id: str) -> tuple[str, ...]:
    return SKILLS.get(skill_id, Skill("", "")).pred_families


# BDDL predicate families used by detectors + progress gating.
PREDICATE_FAMILIES: tuple[str, ...] = (
    "open", "closed", "lit", "unlit", "contains", "empty",
    "object_in_gripper", "contact", "positioned", "attached",
    "cut", "cleaned", "sprayed",
)
