"""pytest configuration + shared helpers.

Adds ``src/`` to ``sys.path`` so ``import b1k`` / ``import evalharness`` /
``import training`` resolve, and provides a small synthetic :class:`Frame`
builder for fast unit tests (the 15 MB recorded fixture is used by exactly one
wire-format test).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
FIX = REPO / "tests" / "fixtures"

# Make the package importable without installation.
for p in (str(SRC), str(SRC / "b1k")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIX


@pytest.fixture
def small_frame() -> "np.ndarray":
    """A small (non-fixture) synthetic flattened obs dict: 64x64 cameras.

    Cheap to msgpack/parse per test.  Uses the real camera *role* suffixes so
    ``protocol.parse_obs`` exercises the same key-matching path.
    """
    from b1k.protocol import frame_from_dict

    rng = np.random.default_rng(0)
    obs = {
        "robot_r1::zed_link::Camera::0::rgb":
            rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).tolist(),
        "robot_r1::left_realsense_link::Camera::0::rgb":
            rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).tolist(),
        "robot_r1::right_realsense_link::Camera::0::rgb":
            rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).tolist(),
        "robot_r1::zed_link::Camera::0::depth":
            rng.uniform(0.5, 4.0, (64, 64)).astype(np.float32).tolist(),
        "robot_r1::proprio": _proprio(rng).tolist(),
    }
    return frame_from_dict(obs)


def _proprio(rng) -> np.ndarray:
    p = np.zeros(61, dtype=np.float32)
    p[0:3] = [0.1, 0.0, 0.0]
    p[24:26] = [0.05, 0.05]
    p[49:51] = [0.05, 0.05]
    p[53:57] = [1.025, -1.45, -0.47, 0.0]
    return p


@pytest.fixture
def closed_gripper_frame(small_frame):
    """small_frame with both grippers reported closed (qpos ~0)."""
    from b1k.protocol import frame_from_dict

    obs = dict(small_frame.raw)
    p = np.asarray(obs["robot_r1::proprio"], dtype=np.float32).copy()
    p[24:26] = 0.0
    p[49:51] = 0.0
    obs["robot_r1::proprio"] = p.tolist()
    return frame_from_dict(obs)
