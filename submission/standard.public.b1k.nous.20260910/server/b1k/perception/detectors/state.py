"""PredicateTracker: onboard predicate satisfaction with hysteresis (design.md
section 4.3).

Wraps a :class:`~b1k.perception.detectors.pred_models.PredicateModel` and turns
its noisy per-frame outputs into a stable ``{family: (sat, conf)}`` map:

* A predicate is **confirmed satisfied** only after ``K`` consecutive frames
  (default 3) where the model reports sat=True with conf >= ``conf_floor``
  (default 0.6).  This is the "3-frame hysteresis" from the design.
* Once confirmed, satisfaction is *sticky within an episode* (BDDL state
  transitions in these tasks are effectively monotone; de-confirming on a
  single bad frame would cause spurious RETRY loops).  ``reset()`` clears it at
  episode boundaries.
* Families the model never reports stay absent from the map — the progress
  layer treats absence as "unobserved", never as False.

The tracker is intentionally model-agnostic: heuristics (MVP) or trained
classifiers (final) plug in identically.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from ...protocol import Frame
from .pred_models import HeuristicPredicateModel, PredicateModel

log = logging.getLogger("b1k.pred.state")


@dataclass
class _Hysteresis:
    k: int
    conf_floor: float
    # raw sat history per family (most recent K).
    hist: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=3)))
    confirmed: set = field(default_factory=set)

    def observe(self, family: str, sat: bool, conf: float) -> None:
        self.hist[family] = deque((self.hist[family]), maxlen=self.k)
        self.hist[family].append((bool(sat), float(conf)))
        if family in self.confirmed:
            return
        h = self.hist[family]
        if len(h) >= self.k and all(s and c >= self.conf_floor for s, c in h):
            self.confirmed.add(family)
            log.info("predicate %s confirmed satisfied (K=%d, floor=%.2f)", family, self.k, self.conf_floor)

    def state(self, family: str) -> tuple[bool, float]:
        if family in self.confirmed:
            return True, 1.0
        h = self.hist.get(family)
        if h:
            sat, conf = h[-1]
            return sat, conf
        return False, 0.0

    def reset(self) -> None:
        self.hist.clear()
        self.confirmed.clear()


class PredicateTracker:
    """Maintains confirmed satisfaction of predicate families from onboard obs."""

    def __init__(
        self,
        model: PredicateModel | None = None,
        k_frames: int = 3,
        conf_floor: float = 0.6,
    ):
        self.model = model or HeuristicPredicateModel()
        self.k = int(k_frames)
        self.conf_floor = float(conf_floor)
        self._hy = _Hysteresis(k=self.k, conf_floor=self.conf_floor)

    def update(self, frame: Frame, step: int = 0) -> dict[str, tuple[bool, float]]:
        """Run the detector for one frame; return the confirmed predicate map.

        Keys are predicate *families* (e.g. ``"lit"``, ``"object_in_gripper"``)
        or ``"<family>:<object>"`` when a model localizes the object.  The
        progress layer matches these against the task's BDDL goal atoms.
        """
        raw = self.model.predict(frame, step)
        for family, (sat, conf) in raw.items():
            self._hy.observe(family, sat, conf)
        # Report the families we observe: confirmed ones first, then the latest
        # raw reads for any family currently in flight (not yet confirmed).
        out: dict[str, tuple[bool, float]] = {}
        for fam in set(list(self._hy.hist.keys()) + list(self._hy.confirmed)):
            sat, conf = self._hy.state(fam)
            out[fam] = (sat, conf)
        return out

    def confirmed(self) -> list[str]:
        return sorted(self._hy.confirmed)

    @property
    def all_confirmed(self) -> bool:
        """True once every family the model reports has been confirmed."""
        fams = set(self._hy.hist.keys())
        return len(fams) > 0 and fams.issubset(self._hy.confirmed)

    def reset(self) -> None:
        """Episode boundary: clear hysteresis and any model-internal state."""
        self._hy.reset()
        if hasattr(self.model, "reset"):
            self.model.reset()
