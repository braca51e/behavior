"""R1Pro embodiment constants for the 2026 BEHAVIOR challenge.

Single source of truth for the 23-dim action layout, 61-dim proprio layout,
camera intrinsics, and controller command limits.  These are copied from the
challenge's canonical ``r1pro.yaml`` and ``omnigibson/eval/utils/eval_utils.py``
(v3.9.2) so that serving-side code never needs to import the simulator.

Do not edit values by hand — regenerate from ``raw/r1pro.yaml`` if the robot
config changes.  All slices are ``np.s_`` objects usable directly on a
(61,) proprio vector or a (23,) action vector.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

ROBOT_NAME = "R1Pro"
ROBOT_ID = "robot_r1"          # default robot namespace in flattened obs
ACTION_DIM = 23
PROPRIO_DIM = 61
HEAD_RES = (720, 720)          # (H, W)
WRIST_RES = (480, 480)         # (H, W)
FPS = 30

# ---------------------------------------------------------------------------
# Action layout (ACTION_QPOS_INDICES["R1Pro"])
# ---------------------------------------------------------------------------
ACTION_SLICES: "OrderedDict[str, slice]" = OrderedDict(
    {
        "base": np.s_[0:3],            # holonomic base velocity cmd (m/s, m/s, rad/s)
        "torso": np.s_[3:7],           # 4-dim trunk position
        "left_arm": np.s_[7:14],       # 7-dim left arm joint position
        "left_gripper": np.s_[14:15],  # 1-dim left gripper
        "right_arm": np.s_[15:22],     # 7-dim right arm joint position
        "right_gripper": np.s_[22:23], # 1-dim right gripper
    }
)

# Base velocity command limits from r1pro.yaml controller_config.base.
BASE_CMD_LIMITS: tuple[np.ndarray, np.ndarray] = (
    np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    np.array([1.0, 1.0, 1.0], dtype=np.float32),
)
BASE_OUT_LIMITS: tuple[np.ndarray, np.ndarray] = (
    np.array([-0.75, -0.75, -1.0], dtype=np.float32),
    np.array([0.75, 0.75, 1.0], dtype=np.float32),
)

# ---------------------------------------------------------------------------
# Proprio layout (PROPRIOCEPTION_INDICES["R1Pro"])
# ---------------------------------------------------------------------------
PROPRIO_SLICES: "OrderedDict[str, slice]" = OrderedDict(
    {
        "base_qvel": np.s_[0:3],
        "arm_left_qpos": np.s_[3:10],
        "arm_left_qvel": np.s_[10:17],
        "eef_left_pos": np.s_[17:20],
        "eef_left_quat": np.s_[20:24],
        "gripper_left_qpos": np.s_[24:26],
        "gripper_left_qvel": np.s_[26:28],
        "arm_right_qpos": np.s_[28:35],
        "arm_right_qvel": np.s_[35:42],
        "eef_right_pos": np.s_[42:45],
        "eef_right_quat": np.s_[45:49],
        "gripper_right_qpos": np.s_[49:51],
        "gripper_right_qvel": np.s_[51:53],
        "trunk_qpos": np.s_[53:57],
        "trunk_qvel": np.s_[57:61],
    }
)

# ---------------------------------------------------------------------------
# Camera intrinsics (CAMERA_INTRINSICS["R1Pro"])
# ---------------------------------------------------------------------------
CAMERA_INTRINSICS: dict[str, np.ndarray] = {
    "head": np.array(
        [[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32
    ),
    "left_wrist": np.array(
        [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    ),
    "right_wrist": np.array(
        [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    ),
}

# Flattened observation role -> camera id (from r1pro.yaml eval.camera_sensor_names).
CAMERA_SENSOR_NAMES: dict[str, str] = {
    "head": f"{ROBOT_ID}:zed_link:Camera:0",
    "left_wrist": f"{ROBOT_ID}:left_realsense_link:Camera:0",
    "right_wrist": f"{ROBOT_ID}:right_realsense_link:Camera:0",
}


def make_no_op_action() -> np.ndarray:
    """A safe "park" action: zero base velocity, hold pose, no grip change."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    # Base: zero (no movement).
    a[ACTION_SLICES["base"]] = 0.0
    # Grippers: leave at last commanded value (0.0 is "hold" for MultiFingerGripper
    # in smooth mode — the controller integrates position deltas).
    return a


def make_hold_action(last_action: np.ndarray) -> np.ndarray:
    """Repeat the last action with zero base velocity (park the robot)."""
    a = np.asarray(last_action, dtype=np.float32).copy()
    if a.size != ACTION_DIM:
        return make_no_op_action()
    a[ACTION_SLICES["base"]] = 0.0
    return a
