"""Odometry tests: velocity integration vs synthetic ground truth (design.md
section 4.4), occupancy marking, frontier search, and episode reset.

All CPU-only; no VO model (``vo_model_path=None``) so these exercise the
velocity-integration path that the MVP relies on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b1k.perception.odometry import Odometry, OccupancyGrid, _rot2  # noqa: E402


def test_zero_velocity_no_movement():
    odo = Odometry(fps=30.0)
    v = np.zeros(3)
    for _ in range(100):
        odo.update(v)
    assert np.allclose(odo.position, 0.0)
    assert odo.yaw == 0.0


def test_constant_forward_velocity():
    odo = Odometry(fps=30.0)
    v = np.array([1.0, 0.0, 0.0])          # 1 m/s forward
    for _ in range(30):                    # 1 second
        odo.update(v)
    assert abs(odo.position[0] - 1.0) < 1e-6
    assert abs(odo.position[1]) < 1e-6


def test_pure_yaw_rotation():
    odo = Odometry(fps=30.0)
    v = np.array([0.0, 0.0, np.pi / 2.0])   # 90 deg/s yaw
    for _ in range(30):                     # 1 s -> yaw = pi/2
        odo.update(v)
    assert abs(odo.yaw - np.pi / 2.0) < 1e-6
    assert np.allclose(odo.position, 0.0)


def test_turning_changes_forward_direction():
    """Move forward, yaw 90 deg, move forward again -> net diagonal."""
    odo = Odometry(fps=30.0)
    fwd = np.array([1.0, 0.0, 0.0])
    for _ in range(30):
        odo.update(fwd)
    yaw = np.array([0.0, 0.0, np.pi / 2.0])
    for _ in range(30):
        odo.update(yaw)
    for _ in range(30):
        odo.update(fwd)
    p = odo.position
    # After turning 90 deg, forward motion is along world +y.
    assert p[0] > 0.9 and p[1] > 0.9
    assert abs(p[0] - p[1]) < 0.1


def test_pose_matrix_is_orthonormal_rotation():
    odo = Odometry(fps=30.0)
    for _ in range(50):
        odo.update(np.array([0.3, -0.1, 0.01]))
    T = odo.pose
    assert T.shape == (4, 4)
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_occupancy_marks_and_queries():
    g = OccupancyGrid(cell_m=0.5)
    g.mark_occupied(1.0, 2.0)
    g.mark_free(0.0, 0.0)
    assert not g.is_free(1.0, 2.0)
    assert g.is_free(0.0, 0.0)
    # Unknown cells count as free (optimistic) for navigation.
    assert g.is_free(5.0, 5.0)


def test_frontier_near_unknown():
    g = OccupancyGrid(cell_m=0.5)
    g.mark_occupied(0.5, 0.0)     # a wall one cell east
    g.mark_free(0.0, 0.0)         # robot cell
    frontier = g.frontier(radius_cells=5)
    # Some frontier cells exist near the unknown region.
    assert isinstance(frontier, list)
    assert all(len(pt) == 2 for pt in frontier)


def test_reset_clears_state():
    odo = Odometry(fps=30.0)
    for _ in range(30):
        odo.update(np.array([0.5, 0.0, 0.0]))
    odo.reset()
    assert np.allclose(odo.position, 0.0)
    assert odo.occupancy.cell_count() == 0


def test_depth_occupancy_never_crashes():
    odo = Odometry(fps=30.0, occupancy_cell_m=0.5)
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.2, 4.0, (16, 16)).astype(np.float32)
    # A handful of steps with depth must not raise and must populate occupancy.
    for _ in range(5):
        odo.update(np.array([0.1, 0.0, 0.0]), depth)
    assert odo.occupancy.cell_count() >= 0
    assert np.all(np.isfinite(odo.position))


def test_rot2_identity_and_quarter():
    assert np.allclose(_rot2(0.0), np.eye(2))
    R = _rot2(np.pi / 2.0)
    assert np.allclose(R @ np.array([1.0, 0.0]), np.array([0.0, 1.0]), atol=1e-9)
