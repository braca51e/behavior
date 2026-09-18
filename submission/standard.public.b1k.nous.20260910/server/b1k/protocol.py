"""WebSocket protocol layer: obs parsing and msgpack encode/decode.

Hard contract (mirrors ``omnigibson.eval.utils.flatten_obs_dict`` and the
``WebsocketPolicyServer`` / ``WebsocketPolicy`` evaluator pair, v3.9.2):

- The evaluator sends one **binary** frame per step containing a msgpack-encoded
  ``dict[str, Any]`` of the *flattened* observation.  Keys are the observation
  names joined with the ``::`` separator, e.g.::

      "robot_r1::zed_link::Camera::0::rgb"          # head RGB  (720,720,3) uint8
      "robot_r1::left_realsense_link::Camera::0::rgb"
      "robot_r1::...::depth"                        # linear depth (H,W[,1]) float
      "robot_r1::base_qvel"                         # 3-dim proprio block (or a
      ...                                           # single 61-dim "state" vector)

- The server replies with one **binary** frame containing a msgpack-encoded
  action: ``{"action": [23 floats]}`` (R1Pro ``action_dim`` = 23).  A bare
  23-list is also accepted when decoding, for interop with the baseline
  ``serve_b1k.py`` clients.

Because the exact camera key strings depend on the robot config's sensor names,
:func:`parse_obs` matches keys by *suffix* and *shape* rather than by exact
string, and records the full layout in ``Frame.raw`` so a first real rollout
can verify the contract (design.md week-1 item; ``tests/test_protocol.py``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import msgpack
import numpy as np

from .embodiment import (
    ACTION_DIM,
    PROPRIO_DIM,
    PROPRIO_SLICES,
)

log = logging.getLogger("b1k.protocol")

OBS_SEP = "::"


# ---------------------------------------------------------------------------
# msgpack pack/unpack (native msgpack types: lists of floats / ints)
# ---------------------------------------------------------------------------
def pack(obj: Any) -> bytes:
    """msgpack-encode ``obj`` (dicts of lists/numbers/strings)."""
    return msgpack.packb(obj, use_bin_type=True)


def unpack(data: bytes | bytearray | memoryview) -> Any:
    """msgpack-decode ``data`` into plain Python (dicts, lists)."""
    if isinstance(data, memoryview):
        data = data.tobytes()
    return msgpack.unpackb(bytes(data), raw=False)


@dataclass
class Frame:
    """One observation step, decoded and validated."""

    rgb_head: np.ndarray | None = None        # (720,720,3) uint8
    rgb_left: np.ndarray | None = None        # (480,480,3) uint8
    rgb_right: np.ndarray | None = None       # (480,480,3) uint8
    depth_head: np.ndarray | None = None      # (720,720) float32 (meters)
    depth_left: np.ndarray | None = None      # (480,480) float32
    depth_right: np.ndarray | None = None     # (480,480) float32
    proprio: np.ndarray = field(default_factory=lambda: np.zeros(PROPRIO_DIM, dtype=np.float32))
    raw: dict[str, Any] = field(default_factory=dict)   # full flattened obs (debug/verify)

    @property
    def base_qvel(self) -> np.ndarray:
        """Robot-local base velocity [vx, vy, wz] (proprio slice [0:3])."""
        return self.proprio[PROPRIO_SLICES["base_qvel"]]

    @property
    def gripper_left(self) -> float:
        return float(self.proprio[PROPRIO_SLICES["gripper_left_qpos"]][0])

    @property
    def gripper_right(self) -> float:
        return float(self.proprio[PROPRIO_SLICES["gripper_right_qpos"]][0])

    def __post_init__(self):
        if self.proprio is None:
            self.proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        if self.proprio.shape[0] != PROPRIO_DIM:
            raise ValueError(f"proprio dim {self.proprio.shape[0]} != expected {PROPRIO_DIM}")


# ---------------------------------------------------------------------------
# Key matching (suffix + shape based; tolerant to sensor-name differences)
# ---------------------------------------------------------------------------
def _role_from_key(key: str) -> str | None:
    """Map a flattened obs key to 'head' / 'left_wrist' / 'right_wrist' / None."""
    k = key.lower()
    if "zed" in k:
        return "head"
    if "left" in k:
        return "left_wrist"
    if "right" in k:
        return "right_wrist"
    return None


def _modality_from_key(key: str) -> str | None:
    """Return 'rgb' / 'depth' if the key names a modality; else None.

    If None, :func:`parse_obs` falls back to shape inference (3 channels =>
    rgb, 1 channel => depth) so exact sensor-name strings never matter.
    """
    k = key.lower()
    if "depth" in k:
        return "depth"
    if "rgb" in k or "color" in k:
        return "rgb"
    return None


def parse_obs(payload: bytes | bytearray | memoryview) -> Frame:
    """Decode a raw msgpack payload into a validated :class:`Frame`."""
    obs = unpack(payload)
    if isinstance(obs, (bytes, bytearray, memoryview)):
        # Some clients double-encode; be lenient.
        obs = unpack(obs)
    if not isinstance(obs, dict):
        raise ValueError(f"Expected dict obs, got {type(obs)}")
    # The evaluator may send a wrapper like {"obs": {...}}; unwrap once.
    if isinstance(obs.get("obs"), dict) and len(obs) <= 3:
        obs = obs["obs"]
    return frame_from_dict(obs)


def frame_from_dict(obs: dict) -> Frame:
    """Assemble a :class:`Frame` from a flattened obs dict.

    Values may be numpy arrays (already-decoded) or lists (straight from
    msgpack); ``np.asarray`` handles both.  This is the pure-Python core of
    :func:`parse_obs`, exposed so the server can rebuild a Frame from a
    ``Frame.raw`` dict without a redundant msgpack round-trip.
    """
    if not isinstance(obs, dict):
        raise ValueError(f"Expected dict obs, got {type(obs)}")
    frame = Frame(raw=obs)
    proprio_candidates: list[np.ndarray] = []
    # role -> Frame field suffix ("left_wrist" -> "left", "right_wrist" -> "right").
    _SUFFIX = {"head": "head", "left_wrist": "left", "right_wrist": "right"}

    for key, value in obs.items():
        arr = np.asarray(value)
        role = _role_from_key(key)
        mod = _modality_from_key(key)

        is_image3 = arr.ndim == 3 and arr.shape[2] in (1, 3)
        if role and mod is None and is_image3:
            mod = "rgb" if arr.shape[2] == 3 else "depth"

        if mod == "rgb" and is_image3 and role:
            target = f"rgb_{_SUFFIX[role]}"
            if getattr(frame, target, None) is None:
                a = arr.astype(np.uint8, copy=False)
                if a.shape[2] == 1:
                    a = a.repeat(3, axis=2)
                setattr(frame, target, a)
        elif mod == "depth" and role:
            target = f"depth_{_SUFFIX[role]}"
            if getattr(frame, target, None) is None:
                d = arr.astype(np.float32)
                if d.ndim == 3 and d.shape[2] == 1:
                    d = d[..., 0]
                setattr(frame, target, d)
        elif role is None and arr.ndim == 1 and arr.dtype.kind in "fiu" and arr.size >= 1:
            proprio_candidates.append(arr)

    frame.proprio = _assemble_proprio(proprio_candidates)
    return frame


def _assemble_proprio(candidates: list[np.ndarray]) -> np.ndarray:
    """Build the 61-dim proprio vector from the observed 1-D blocks.

    Preference order:
      1. a single 61-dim vector;
      2. named blocks matched by size to PROPRIO_SLICES (canonical order);
      3. fallback: zero vector with the largest block copied into the head.
    """
    single = [c for c in candidates if c.size == PROPRIO_DIM]
    if single:
        return single[0].astype(np.float32)

    blocks: dict[str, np.ndarray] = {}
    for c in candidates:
        for name, sl in PROPRIO_SLICES.items():
            if (sl.stop - sl.start) == c.size and name not in blocks:
                blocks[name] = c.astype(np.float32)
                break
    if blocks:
        out = np.zeros(PROPRIO_DIM, dtype=np.float32)
        for name, sl in PROPRIO_SLICES.items():
            if name in blocks:
                out[sl] = blocks[name]
        return out

    if candidates:
        biggest = max(candidates, key=lambda x: x.size)
        out = np.zeros(PROPRIO_DIM, dtype=np.float32)
        m = min(biggest.size, PROPRIO_DIM)
        out[:m] = biggest[:m].astype(np.float32)
        log.warning("No 61-dim proprio found; padded largest block (%d) to %d", m, PROPRIO_DIM)
        return out
    return np.zeros(PROPRIO_DIM, dtype=np.float32)


def encode_action(action: np.ndarray) -> bytes:
    """Encode a 23-dim action as the msgpack response payload."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size != ACTION_DIM:
        raise ValueError(f"action size {a.size} != expected {ACTION_DIM}")
    return pack({"action": a.tolist()})


def decode_action(payload: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a server response payload into a 23-dim action array."""
    obj = unpack(payload)
    if isinstance(obj, dict) and "action" in obj:
        obj = obj["action"]
    a = np.asarray(obj, dtype=np.float32).reshape(-1)
    if a.size != ACTION_DIM:
        raise ValueError(f"decoded action size {a.size} != {ACTION_DIM}")
    return a
