"""Action controller: chunk cache, receding horizon, safety clamps (design.md
section 4.5).

Owns the action-chunk lifecycle between the VLA (``policy.SkillConditionedVLA``)
and the server (``server.B1KServer``):

* **Receding horizon** — serve one action per eval step from the cached chunk;
  re-query the VLA every ``chunk_requery`` steps (K=16 by default, matching the
  ``serve_b1k.py`` convention), or immediately on a subgoal ADVANCE/RETRY/
  REPLAN (the progress layer signals ``requery=True``).
* **Chunk-boundary blending** (temporal-ensembling port from GR00T serving) —
  when a fresh chunk arrives, blend the still-remaining tail of the old chunk
  with the new one so the boundary is jerk-free (``policy.chunking.blend_chunks``).
* **Safety** — base velocity clamped to the ``r1pro.yaml`` output limits
  (±0.75 x-y, ±1.0 yaw), and a gripper deadband (commands changing by less than
  ``grip_deadband`` are dropped) to avoid chatter.

Deterministic and CPU-testable (``tests/test_controller.py``).
"""
from __future__ import annotations

import logging

import numpy as np

from .config import ControllerConfig
from .embodiment import ACTION_DIM, BASE_OUT_LIMITS
from .policy.chunking import blend_chunks
from .policy.vla import SkillConditionedVLA
from .planner.task_planner import Subgoal
from .protocol import Frame

log = logging.getLogger("b1k.controller")

_BASE_LO, _BASE_HI = (
    np.asarray(BASE_OUT_LIMITS[0], dtype=np.float32),
    np.asarray(BASE_OUT_LIMITS[1], dtype=np.float32),
)


class ActionController:
    """One controller per episode (server resets it on (re)connect)."""

    def __init__(self, cfg: ControllerConfig, vla: SkillConditionedVLA):
        self.cfg = cfg
        self.vla = vla
        self.chunk: np.ndarray | None = None
        self.offset = 0
        self.last_requery_step = -10**9
        self.task_text = ""
        self._last_grip = np.zeros(2, dtype=np.float32)
        self.parked = False

    # ---- lifecycle ----------------------------------------------------------
    def reset(self, task_text: str = "") -> None:
        """Episode start: drop the chunk cache and gripper deadband state."""
        self.chunk = None
        self.offset = 0
        self.last_requery_step = -10**9
        self.task_text = task_text
        self._last_grip = np.zeros(2, dtype=np.float32)
        self.parked = False
        self.vla.reset()

    # ---- main step ----------------------------------------------------------
    def step(
        self,
        frame: Frame,
        subgoal: Subgoal,
        task_id: int | None = None,
        step: int = 0,
        force_requery: bool = False,
    ) -> np.ndarray:
        """Serve exactly one 23-dim action for this eval step.

        Re-queries the VLA when the chunk is exhausted, every ``chunk_requery``
        steps, or when ``force_requery`` (subgoal transition) is set.
        """
        need = (
            self.chunk is None
            or self.offset >= self.chunk.shape[0]
            or force_requery
            or (step - self.last_requery_step) >= self.cfg.chunk_requery
        )
        if need:
            new = self.vla.act(
                frame,
                subgoal.text,
                self.task_text,
                task_id=task_id,
                step=step,
            )
            old_tail = None
            if self.chunk is not None:
                tail = self.chunk[self.offset:]
                if tail.shape[0] > 0:
                    old_tail = tail
            if old_tail is not None:
                self.chunk = blend_chunks(
                    old_tail,
                    new,
                    step_in_chunk=0,
                    blend_steps=self.cfg.blend_steps,
                )
            else:
                self.chunk = new
            self.offset = 0
            self.last_requery_step = step
            log.debug("VLA re-query @step %d (%d-step chunk)", step, self.chunk.shape[0])

        a = np.asarray(self.chunk[self.offset], dtype=np.float32).copy()
        self.offset += 1
        return self._safety(a)

    def park(self) -> np.ndarray:
        """Serve a safe "park" action (FINISH / stale VLA)."""
        if self.chunk is not None:
            base = np.asarray(self.chunk[-1], dtype=np.float32).copy()
            base[0:3] = 0.0
            self.parked = True
            return self._safety(base)
        base = np.zeros(ACTION_DIM, dtype=np.float32)
        self.parked = True
        return base

    # ---- safety -------------------------------------------------------------
    def _safety(self, a: np.ndarray) -> np.ndarray:
        """Clamp base velocity and deadband gripper commands."""
        a = a.astype(np.float32).copy()
        # Base velocity output limits (r1pro.yaml: x,y ±0.75 m/s, yaw ±1.0 rad/s).
        a[0:3] = np.clip(a[0:3], _BASE_LO, _BASE_HI)
        # Gripper deadband: drop commands that move < grip_deadband from the
        # last commanded grip (MultiFingerGripper smooth mode integrates).
        gl, gr = float(a[14]), float(a[22])
        if abs(gl - self._last_grip[0]) < self.cfg.grip_deadband:
            gl = float(self._last_grip[0])
        if abs(gr - self._last_grip[1]) < self.cfg.grip_deadband:
            gr = float(self._last_grip[1])
        self._last_grip = np.array([gl, gr], dtype=np.float32)
        a[14] = gl
        a[22] = gr
        # NaN/Inf guard (a broken model must never emit garbage to the sim).
        if not np.all(np.isfinite(a)):
            log.warning("non-finite action suppressed @safety; issuing park")
            a[:] = 0.0
        return a
