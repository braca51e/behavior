"""Protocol tests: obs parsing, msgpack round-trip, action encode/decode.

Validates the WebSocket serving contract (design.md section 4.1) against:
  * a small synthetic flattened-obs dict (fast, per-test), and
  * the **recorded** 15 MB wire fixture (``tests/fixtures/obs_payload.bin``) —
    the exact msgpack payload shape a baseline rollout would produce — to
    confirm the full-res camera + 61-dim proprio layout parses to a Frame.

The recorded fixture also pins the action round-trip (contract item (a)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b1k.embodiment import ACTION_DIM, PROPRIO_DIM  # noqa: E402
from b1k.protocol import (  # noqa: E402
    Frame,
    decode_action,
    encode_action,
    frame_from_dict,
    parse_obs,
    pack,
    unpack,
)


def test_small_frame_parses(small_frame):
    f = small_frame
    assert f.rgb_head is not None and f.rgb_head.dtype == np.uint8
    assert f.rgb_left is not None and f.rgb_right is not None
    assert f.depth_head is not None and f.depth_head.dtype == np.float32
    assert f.proprio.shape == (PROPRIO_DIM,)
    # base_qvel slice [0:3] is exposed.
    assert f.base_qvel.shape == (3,)


def test_action_roundtrip_small(small_frame):
    a = np.linspace(-1, 1, ACTION_DIM).astype(np.float32)
    payload = encode_action(a)
    back = decode_action(payload)
    assert back.shape == (ACTION_DIM,)
    assert np.allclose(back, a, atol=1e-6)


def test_encode_action_rejects_wrong_dim():
    with pytest.raises(ValueError):
        encode_action(np.zeros(10))


def test_recorded_fixture_parses(fixtures_dir):
    """The recorded baseline-rollout obs payload must parse to a full Frame."""
    raw = (fixtures_dir / "obs_payload.bin").read_bytes()
    frame = parse_obs(raw)
    assert frame.rgb_head is not None and frame.rgb_head.shape == (720, 720, 3)
    assert frame.rgb_left is not None and frame.rgb_left.shape == (480, 480, 3)
    assert frame.rgb_right is not None and frame.rgb_right.shape == (480, 480, 3)
    assert frame.depth_head is not None and frame.depth_head.shape == (720, 720)
    assert frame.depth_left is not None and frame.depth_left.shape == (480, 480)
    assert frame.proprio.shape == (61,)
    # proprio base_qvel is non-zero in the fixture.
    assert np.linalg.norm(frame.base_qvel) > 0


def test_recorded_action_decodes(fixtures_dir):
    """The recorded 23-float action must decode to exactly 23 floats."""
    raw = (fixtures_dir / "action_23d.bin").read_bytes()
    a = decode_action(raw)
    assert a.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(a))


def test_msgpack_pack_unpack_dict(fixtures_dir):
    obs = unpack((fixtures_dir / "obs_payload.bin").read_bytes())
    assert isinstance(obs, dict)
    # Round-trip a small dict.
    d = {"a": [1.0, 2.0], "b": "x", "c": 3}
    assert unpack(pack(d)) == d


def test_frame_from_dict_accepts_arrays_and_lists():
    obs = {
        "robot_r1::zed_link::Camera::0::rgb":
            np.zeros((32, 32, 3), dtype=np.uint8).tolist(),
        "robot_r1::proprio": np.zeros(61, dtype=np.float32).tolist(),
    }
    f = frame_from_dict(obs)
    assert f.rgb_head is not None
    assert f.proprio.shape == (61,)
