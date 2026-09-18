"""Privileged predicate-label teacher (TRAINING TIME ONLY).

During training the demo dataset carries the *privileged* simulator state
(object poses, gripper contact, BDDL evaluation at each frame), so predicate
labels are **free, dense, and exact**.  This module turns a demo frame into the
teacher labels for each predicate family.  It is quarantined in the
``training`` import tree (``src/b1k/perception/detectors/labels.py``) and is
*never* imported at eval time — at eval the same families are predicted by
``pred_models.py`` from onboard RGB/depth/proprio alone.

The teacher API is deliberately small so the detector training script
(``src/training/detector_train.py``) and the unit tests share it::

    labels = make_teacher_labels(privileged_frame)   # {family: (sat: bool, conf: 1.0)}

``privileged_frame`` is a dict with the demo's privileged fields (all optional;
missing -> the family is labeled ``sat=False, conf=0.0`` and skipped):

    "grasped_objects":  set[str]      # object ids currently in a gripper
    "open_containers":  set[str]      # container ids currently open
    "lit_objects":      set[str]      # appliance ids currently on/lit
    "contains":         dict[str, str]# container_id -> contained_object_id
    "cleaned":          set[str]      # surface ids cleaned
    "attached":         set[tuple]    # (a, b) attachment pairs
"""
from __future__ import annotations

from typing import Any

# Canonical predicate families (must stay in sync with skill_tax.PREDICATE_FAMILIES).
FAMILIES: tuple[str, ...] = (
    "open", "closed", "lit", "unlit", "contains", "empty",
    "object_in_gripper", "contact", "positioned", "attached",
    "cut", "cleaned", "sprayed",
)


def make_teacher_labels(privileged_frame: dict[str, Any]) -> dict[str, tuple[bool, float]]:
    """Emit exact teacher labels (conf=1.0) for every derivable family.

    Returns a ``{family: (sat, conf)}`` map.  A family present in the input
    yields its truth value; families absent from the input are omitted (the
    trainer treats omission as "no label for this frame", not as False).
    """
    labels: dict[str, tuple[bool, float]] = {}

    grasped = set(privileged_frame.get("grasped_objects", ()))
    labels["object_in_gripper"] = (len(grasped) > 0, 1.0)

    open_c = set(privileged_frame.get("open_containers", ()))
    labels["open"] = (len(open_c) > 0, 1.0)
    # "closed" is the negation only when the frame declares a closed set.
    if "closed_containers" in privileged_frame:
        labels["closed"] = (len(privileged_frame["closed_containers"]) > 0, 1.0)

    lit = set(privileged_frame.get("lit_objects", ()))
    labels["lit"] = (len(lit) > 0, 1.0)
    if "unlit_objects" in privileged_frame:
        labels["unlit"] = (len(privileged_frame["unlit_objects"]) > 0, 1.0)

    contains = dict(privileged_frame.get("contains", {}))
    labels["contains"] = (len(contains) > 0, 1.0)

    cleaned = set(privileged_frame.get("cleaned", ()))
    labels["cleaned"] = (len(cleaned) > 0, 1.0)

    attached = set(privileged_frame.get("attached", ()))
    labels["attached"] = (len(attached) > 0, 1.0)

    if "cut_objects" in privileged_frame:
        labels["cut"] = (len(privileged_frame["cut_objects"]) > 0, 1.0)
    if "sprayed_objects" in privileged_frame:
        labels["sprayed"] = (len(privileged_frame["sprayed_objects"]) > 0, 1.0)
    if "contact_pairs" in privileged_frame:
        labels["contact"] = (len(privileged_frame["contact_pairs"]) > 0, 1.0)
    if "positioned_pairs" in privileged_frame:
        labels["positioned"] = (len(privileged_frame["positioned_pairs"]) > 0, 1.0)
    return labels
