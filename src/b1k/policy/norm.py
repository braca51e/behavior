"""Normalization-statistics load/apply (design.md section 4.5).

The OpenPI fork ships per-dataset ``meta/stats.json`` (min/max/mean/std per
action dimension).  Because the Aug-2026 velocity fix changed
``observation.state`` slices, the design mandates **recomputing** norm stats
post-fix before training (see ``src/training/stats.py``).  At serve time this
module applies the recorded mean/std to map a model-space chunk into the
robot's action space (``action_normalize: false`` on R1Pro means the VLA
outputs are already in action units — but if a checkpoint was trained with
normalized actions, this is the inverse transform).

Public:
    NormStats.load(path)        -> NormStats (mean/std per dim)
    NormStats.apply(chunk)      -> denormalized chunk
    NormStats.normalize(chunk)  -> normalized chunk
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..embodiment import ACTION_DIM


@dataclass
class NormStats:
    mean: "np.ndarray | None" = None
    std: "np.ndarray | None" = None

    def __post_init__(self):
        if self.mean is None or self.std is None:
            self.mean = np.zeros(ACTION_DIM)
            self.std = np.ones(ACTION_DIM)

    @classmethod
    def load(cls, path: str | Path | None) -> "NormStats":
        if not path:
            return cls.identity()
        p = Path(path)
        if not p.exists():
            return cls.identity()
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Accept {"action": {"mean": [...], "std": [...]}} or top-level mean/std.
        node = raw.get("action", raw) if isinstance(raw, dict) else {}
        mean = np.asarray(node.get("mean", np.zeros(ACTION_DIM)), dtype=np.float64).reshape(-1)
        std = np.asarray(node.get("std", np.ones(ACTION_DIM)), dtype=np.float64).reshape(-1)
        return cls(mean=mean, std=std)

    @classmethod
    def identity(cls) -> "NormStats":
        return cls(mean=np.zeros(ACTION_DIM), std=np.ones(ACTION_DIM))

    def _fit_dims(self) -> int:
        n = int(self.mean.size)
        return n if n > 0 else ACTION_DIM

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        """Denormalize: out = mean + in * std (in model space)."""
        a = np.asarray(chunk, dtype=np.float64)
        d = self._fit_dims()
        m = self.mean[:d].reshape(1, d) if a.ndim == 2 else self.mean[:d]
        s = self.std[:d].reshape(1, d) if a.ndim == 2 else self.std[:d]
        return (m + a * s).astype(np.float32)

    def normalize(self, chunk: np.ndarray) -> np.ndarray:
        """Normalize: out = (in - mean) / std."""
        a = np.asarray(chunk, dtype=np.float64)
        d = self._fit_dims()
        m = self.mean[:d].reshape(1, d) if a.ndim == 2 else self.mean[:d]
        s = self.std[:d].reshape(1, d) if a.ndim == 2 else self.std[:d]
        s = np.where(s < 1e-8, 1.0, s)
        return ((a - m) / s).astype(np.float32)
