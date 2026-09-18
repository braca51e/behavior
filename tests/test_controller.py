"""Action-controller tests: safety clamps, gripper deadband, receding horizon,
chunk blending, and park (design.md section 4.5).

Uses a stub VLA (a fixed action chunk) so the tests are deterministic and
CPU-only — they exercise the controller's chunk lifecycle, not the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b1k.config import ControllerConfig  # noqa: E402
from b1k.controller import ActionController  # noqa: E402
from b1k.embodiment import ACTION_DIM  # noqa: E402
from b1k.planner.task_planner import Subgoal  # noqa: E402
from b1k.policy.vla import SkillConditionedVLA  # noqa: E402


def _cfg(**kw):
    base = dict(chunk_requery=16, action_horizon=32, blend_steps=4,
                grip_deadband=0.02, base_clamp=(0.75, 0.75, 1.0),
                stale_vla_s=5.0)
    base.update(kw)
    return ControllerConfig(**base)


def _vla(chunk):
    """A VLA stub that always returns the given (T,23) chunk."""
    v = SkillConditionedVLA(backend="noop")

    class _Stub:
        def act(self, frame, subgoal_text, task_text, task_id=None, step=0):
            return chunk

        def reset(self):
            pass

    return _Stub()


def _sg():
    return Subgoal(skill="move_to", args={}, text="move to x",
                   budget_steps=600, mean_steps=300)


def _frame():
    from b1k.protocol import Frame

    return Frame()


def test_step_returns_23d_finite():
    chunk = np.zeros((32, ACTION_DIM), dtype=np.float32)
    chunk[:, 0] = 0.1
    ctl = ActionController(_cfg(), _vla(chunk))
    ctl.reset("t")
    a = ctl.step(_frame(), _sg(), task_id=0, step=0)
    assert a.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(a))


def test_receding_uses_chunk_in_order():
    # Distinct small constant per row (all within the 0.75 base clamp); zero
    # deadband so gripper dims aren't held.  Verifies receding order.
    chunk = np.zeros((32, ACTION_DIM), dtype=np.float32)
    for i in range(32):
        chunk[i, :] = i * 0.01
    ctl = ActionController(_cfg(grip_deadband=0.0), _vla(chunk))
    ctl.reset("t")
    a0 = ctl.step(_frame(), _sg(), step=0)
    a1 = ctl.step(_frame(), _sg(), step=1)
    assert np.allclose(a0, chunk[0])
    assert np.allclose(a1, chunk[1])


def test_requery_every_k_steps():
    calls = []

    class _CountingVLA:
        def act(self, frame, subgoal_text, task_text, task_id=None, step=0):
            calls.append(step)
            return np.zeros((32, ACTION_DIM), dtype=np.float32)

        def reset(self):
            pass

    ctl = ActionController(_cfg(chunk_requery=16, action_horizon=32), _CountingVLA())
    ctl.reset("t")
    for s in range(40):
        ctl.step(_frame(), _sg(), step=s)
    # Re-query at step 0 (fresh), then every 16 -> 0,16,32.
    assert calls[0] == 0
    assert 16 in calls and 32 in calls


def test_base_velocity_clamped():
    chunk = np.zeros((32, ACTION_DIM), dtype=np.float32)
    chunk[:, 0] = 5.0    # way over the 0.75 x-limit
    chunk[:, 2] = -3.0   # over the -1.0 yaw limit (magnitude)
    ctl = ActionController(_cfg(), _vla(chunk))
    ctl.reset("t")
    a = ctl.step(_frame(), _sg(), step=0)
    assert abs(a[0]) <= 0.75 + 1e-6
    assert a[0] == pytest.approx(0.75, abs=1e-6)
    assert a[2] == pytest.approx(-1.0, abs=1e-6)


def test_gripper_deadband_drops_small_changes():
    ctl = ActionController(_cfg(grip_deadband=0.02), _vla(
        np.zeros((32, ACTION_DIM), dtype=np.float32)))
    ctl.reset("t")
    # First step: grip 0 -> last_grip 0 (no change).
    a0 = ctl.step(_frame(), _sg(), step=0)
    # Feed a chunk with grip = 0.005 (below deadband) -> should be held at 0.
    class _GripVLA:
        def act(self, frame, subgoal_text, task_text, task_id=None, step=0):
            c = np.zeros((32, ACTION_DIM), dtype=np.float32)
            c[:, 14] = 0.005
            return c

        def reset(self):
            pass

    ctl2 = ActionController(_cfg(grip_deadband=0.02), _GripVLA())
    ctl2.reset("t")
    b = ctl2.step(_frame(), _sg(), step=0)
    # 0.005 < deadband from last (0.0) -> held at 0.0.
    assert b[14] == pytest.approx(0.0, abs=1e-6)


def test_nan_action_suppressed_to_park():
    class _NaNVLA:
        def act(self, frame, subgoal_text, task_text, task_id=None, step=0):
            c = np.zeros((32, ACTION_DIM), dtype=np.float32)
            c[:, 0] = np.nan
            return c

        def reset(self):
            pass

    ctl = ActionController(_cfg(), _NaNVLA())
    ctl.reset("t")
    a = ctl.step(_frame(), _sg(), step=0)
    assert np.all(np.isfinite(a))
    assert a[0] == 0.0


def test_park_zeroes_base():
    chunk = np.zeros((32, ACTION_DIM), dtype=np.float32)
    chunk[:, 0] = 0.5
    ctl = ActionController(_cfg(), _vla(chunk))
    ctl.reset("t")
    ctl.step(_frame(), _sg(), step=0)
    p = ctl.park()
    assert p[0] == 0.0
    assert np.all(np.isfinite(p))
    assert p.shape == (ACTION_DIM,)


def test_blend_reduces_boundary_jerk():
    """A fresh chunk differing from the old tail should be blended, not jumped."""
    old_tail = np.zeros((4, ACTION_DIM), dtype=np.float32)
    new = np.ones((4, ACTION_DIM), dtype=np.float32)
    new[:, 0] = 0.0
    from b1k.policy.chunking import blend_chunks

    blended = blend_chunks(old_tail, new, step_in_chunk=0, blend_steps=4)
    # At t=0 alpha=0 -> equals old (0); at t=3 alpha=0.75 -> closer to new.
    assert blended[0, 14] == pytest.approx(0.0, abs=1e-6)
    assert blended[3, 14] == pytest.approx(0.75, abs=1e-3)
    assert blended[3, 14] < 1.0   # not a full jump yet


def test_receding_take_holds_last_when_short():
    from b1k.policy.chunking import receding_take

    chunk = np.arange(4, dtype=np.float32).reshape(4, 1)
    take = receding_take(chunk, offset=2, n=4)
    assert take.shape == (4, 1)
    # Holds the last action for the overrun.
    assert take[2, 0] == take[3, 0] == 3.0
