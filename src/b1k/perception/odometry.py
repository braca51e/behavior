"""Odometry & room-scale occupancy (design.md section 4.4).

Fuses two signals available onboard (no global pose):

1. **Velocity integration** of ``obs.state[0:3]`` — the R1Pro robot-local base
   velocity ``[vx, vy, wz]`` (July-2026 convention, matches the action frame).
   Integrated per 30 Hz step to a relative pose since episode start.
2. **Head-depth visual odometry** (pretrained DPV2, optional).  When
   ``vo_model_path`` is configured it runs sub-sampled (every ``vo_subsample``)
   and *corrects* integration drift with an exponential-smoothing blend of the
   per-step displacement (gain = ``vo_blend_gain``).  When absent (the default
   for CPU/MVP) integration alone is used — still valid, just drift-prone.

A coarse **occupancy grid** (``occupancy_cell_m`` = 0.5 m) is built from head
depth (near pixels -> occupied) for room-scale frontier search.  All state is
*relative* to the episode start frame (the only thing observable at eval).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..embodiment import HEAD_RES

log = logging.getLogger("b1k.odometry")


@dataclass
class OccupancyGrid:
    """Coarse relative occupancy: 1 = occupied, 0 = free, -1 = unknown.

    Centered on the episode start; grows as the robot moves (cells are
    allocated lazily so memory stays bounded by explored area).
    """

    cell_m: float = 0.5
    _cells: dict = field(default_factory=dict)   # (ix, iy) -> {0,1,-1}

    def mark_occupied(self, x_m: float, y_m: float) -> None:
        ix, iy = int(round(x_m / self.cell_m)), int(round(y_m / self.cell_m))
        if (ix, iy) not in self._cells:
            self._cells[(ix, iy)] = 0
        self._cells[(ix, iy)] = 1

    def mark_occupied_many(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Vectorized batch of :meth:`mark_occupied` (numpy arrays of x/y in m).

        Deduplicates to unique cells (numpy), so the cost is ~O(unique cells),
        independent of the number of depth points.
        """
        ix = np.round(np.asarray(xs, dtype=np.float64) / self.cell_m).astype(np.int64)
        iy = np.round(np.asarray(ys, dtype=np.float64) / self.cell_m).astype(np.int64)
        flat = ix.astype(np.int64) * (1 << 32) + iy.astype(np.int64)
        for cell in np.unique(flat):
            self._cells[(int(cell >> 32), int(cell & 0xFFFFFFFF))] = 1

    def mark_free(self, x_m: float, y_m: float) -> None:
        ix, iy = int(round(x_m / self.cell_m)), int(round(y_m / self.cell_m))
        self._cells.setdefault((ix, iy), 0)

    def is_free(self, x_m: float, y_m: float) -> bool:
        return self._cells.get((int(round(x_m / self.cell_m)), int(round(y_m / self.cell_m))), -1) in (0, -1)

    def frontier(self, radius_cells: int = 40, max_points: int = 64) -> list[tuple[float, float]]:
        """Free cells adjacent to at least one unknown cell, near the robot.

        Used by the SEARCH bias to pick the next direction to explore.  Returns
        world (x, y) meters sorted by Manhattan distance to the origin (episode
        start), capped at ``max_points``.
        """
        candidates: list[tuple[float, float, int]] = []
        for (ix, iy), v in list(self._cells.items()):
            if v != 0:
                continue
            touches_unknown = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if self._cells.get((ix + dx, iy + dy), -1) == -1:
                        touches_unknown = True
                        break
                if touches_unknown:
                    break
            if not touches_unknown:
                continue
            dist = abs(ix) + abs(iy)
            if dist <= radius_cells:
                candidates.append((ix * self.cell_m, iy * self.cell_m, dist))
        candidates.sort(key=lambda p: p[2])
        return [(x, y) for x, y, _ in candidates[:max_points]]

    def cell_count(self) -> int:
        return len(self._cells)


def _rot2(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class Odometry:
    """Relative pose + occupancy from onboard signals only."""

    def __init__(
        self,
        fps: float = 30.0,
        vo_subsample: int = 3,
        vo_blend_gain: float = 0.1,
        occupancy_cell_m: float = 0.5,
        occupancy_radius_cells: int = 40,
        vo_model_path=None,
    ):
        self.fps = float(fps)
        self.vo_subsample = max(1, int(vo_subsample))
        self.vo_blend_gain = float(vo_blend_gain)
        self.occupancy = OccupancyGrid(cell_m=occupancy_cell_m)
        self.occupancy_radius = int(occupancy_radius_cells)
        self._pose = np.zeros(3, dtype=np.float64)   # [x, y, yaw] relative to t0
        self._step = 0
        self._vo_model = None
        if vo_model_path is not None:
            self._vo_model = self._load_vo_model(vo_model_path)

    def _load_vo_model(self, path):
        """Lazy-load a pretrained depth VO (DPV2).  CPU-safe: returns None on
        any import/model failure so integration-only mode still works."""
        try:
            from ..perception._vo import DepthVO  # optional heavy dep

            return DepthVO(path)
        except Exception as e:  # noqa: BLE001 - VO is an optional enhancement
            log.warning("VO model at %s unavailable (%s); using velocity integration only", path, e)
            return None

    # ---- core update --------------------------------------------------------
    def update(self, base_qvel: np.ndarray, depth_head: np.ndarray | None = None) -> None:
        """Integrate one step of robot-local base velocity [vx, vy, wz] (m/s).

        ``depth_head`` is used (a) to mark occupancy around the robot and (b) as
        the input to the optional VO drift correction.
        """
        v = np.asarray(base_qvel, dtype=np.float64).reshape(-1)
        if v.size < 3:
            return
        vx, vy, wz = v[0], v[1], v[2]
        dt = 1.0 / self.fps
        yaw = self._pose[2]
        R = _rot2(yaw)
        disp = R @ np.array([vx, vy], dtype=np.float64) * dt

        if self._vo_model is not None and depth_head is not None:
            if self._step % self.vo_subsample == 0:
                try:
                    vo_disp = self._vo_model.displacement(depth_head)
                    if vo_disp is not None:
                        g = self.vo_blend_gain
                        disp = (1.0 - g) * disp + g * np.asarray(vo_disp, dtype=np.float64)[:2]
                except Exception:  # noqa: BLE001 - VO must never break stepping
                    pass

        self._pose[0] += disp[0]
        self._pose[1] += disp[1]
        self._pose[2] = (self._pose[2] + wz * dt) % (2.0 * np.pi)

        if depth_head is not None:
            self._update_occupancy(depth_head)
        self._step += 1

    def _update_occupancy(self, depth_head: np.ndarray) -> None:
        """Project near head-depth pixels to the world grid (robot frame -> rel).

        Uses the head intrinsics from ``embodiment``; only pixels with depth in
        (0.1, 3.0) m are marked occupied (walls/furniture).  Vectorized (numpy)
        so it costs ~sub-ms, and sub-sampled 16x for speed — the occupancy grid
        is a coarse search aid, not a map, so this is deliberate.
        """
        try:
            from ..embodiment import CAMERA_INTRINSICS

            d = np.asarray(depth_head, dtype=np.float32)
            if d.size == 0:
                return
            H, W = d.shape[:2]
            intr = CAMERA_INTRINSICS["head"]
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            sub = 16
            ys, xs = np.mgrid[0:H:sub, 0:W:sub]
            dd = d[ys, xs]
            mask = (dd > 0.1) & (dd < 3.0)
            if not mask.any():
                return
            xs_m = (xs[mask].astype(np.float64) - cx) * dd[mask] / fx
            ys_m = (ys[mask].astype(np.float64) - cy) * dd[mask] / fy
            pts = np.stack([xs_m, -ys_m], axis=1)
            R = _rot2(self._pose[2])
            pts_xy = (R @ pts.T).T
            world_x = self._pose[0] + pts_xy[:, 0]
            world_y = self._pose[1] + pts_xy[:, 1]
            self.occupancy.mark_occupied_many(world_x, world_y)
            # Mark the robot's own cell free.
            self.occupancy.mark_free(float(self._pose[0]), float(self._pose[1]))
        except Exception:  # noqa: BLE001 - occupancy must never break stepping
            pass

    # ---- accessors ----------------------------------------------------------
    @property
    def pose(self) -> np.ndarray:
        """4x4 relative pose (robot frame at episode start)."""
        c, s = np.cos(self._pose[2]), np.sin(self._pose[2])
        T = np.eye(4, dtype=np.float64)
        T[0, 0], T[0, 1], T[0, 3] = c, -s, self._pose[0]
        T[1, 0], T[1, 1], T[1, 3] = s, c, self._pose[1]
        T[2, 3] = 0.0
        return T

    @property
    def position(self) -> np.ndarray:
        return np.array([self._pose[0], self._pose[1], 0.0], dtype=np.float64)

    @property
    def yaw(self) -> float:
        return float(self._pose[2])

    def reset(self) -> None:
        """Episode boundary: drop all relative state."""
        self._pose = np.zeros(3, dtype=np.float64)
        self.occupancy = OccupancyGrid(cell_m=self.occupancy.cell_m)
        self._step = 0
