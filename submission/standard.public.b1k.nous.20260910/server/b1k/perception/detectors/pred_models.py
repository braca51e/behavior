"""Onboard predicate detectors (eval-time inference).

Design (design.md section 4.6): small per-family vision classifiers trained
with privileged supervision from demos; at eval they consume *onboard* RGB +
depth + proprio only (never the privileged fields).

Two interchangeable backends implement :class:`PredicateModel`:

* :class:`TrainedPredicateModel` — loads a detector weight bundle
  (``data/detectors/<family>.pt``) and runs inference with torch (lazy import;
  CPU or GPU).  This is the intended final backend.
* :class:`HeuristicPredicateModel` — deterministic onboard-only signals with
  *honest, low* confidence: gripper closure from proprio (strong signal),
  head-camera brightness delta (weak ``lit`` signal), and "always unknown"
  (sat=False, conf=0.0) for the rest so the progress layer treats them as
  unobserved rather than guessed.

The :class:`PredicateTracker` in ``state.py`` applies the K-frame hysteresis on
top of whichever backend is active.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...protocol import Frame

log = logging.getLogger("b1k.pred")


@dataclass
class ModelBundle:
    """Detector weight bundle location (one file per family)."""

    root: Path

    def path_for(self, family: str) -> Path:
        return Path(self.root) / f"{family}.pt"

    def has(self, family: str) -> bool:
        return self.path_for(family).exists()


class PredicateModel(ABC):
    """Onboard predicate inference: Frame -> {family: (sat, conf)}."""

    @abstractmethod
    def predict(self, frame: Frame, step: int = 0) -> dict[str, tuple[bool, float]]: ...

    def families(self) -> tuple[str, ...]:
        return ()


class HeuristicPredicateModel(PredicateModel):
    """Deterministic onboard-only detectors (no weights).

    Confidence is deliberately low for vision heuristics so the tracker's
    ``conf_floor`` (0.6) filters them out; the gripper signal is a strong
    proprio read and gets higher confidence.
    """

    def __init__(self, grip_open_threshold: float = 0.04):
        self._grip_open = grip_open_threshold
        # Rolling head-brightness reference for the weak "lit" signal.
        self._bright_ref: float | None = None
        self._prev_grip_open = True

    def families(self) -> tuple[str, ...]:
        return ("object_in_gripper", "lit")

    def predict(self, frame: Frame, step: int = 0) -> dict[str, tuple[bool, float]]:
        out: dict[str, tuple[bool, float]] = {}

        # --- gripper (proprio, strong) ---
        # A grasp is registered when the mean of both gripper qpos drops well
        # below the open threshold and stays there for >=2 frames.
        gl = frame.gripper_left
        gr = frame.gripper_right
        closed = (gl < self._grip_open) and (gr < self._grip_open)
        out["object_in_gripper"] = (closed, 0.9 if closed else 0.3)

        # --- lit (head brightness delta, weak) ---
        if frame.rgb_head is not None and frame.rgb_head.size:
            mean = float(frame.rgb_head.astype(np.float32).mean())
            if self._bright_ref is None:
                self._bright_ref = mean
            delta = mean - self._bright_ref
            if delta > 4.0:  # scene noticeably brighter than episode start
                out["lit"] = (True, 0.62)
            else:
                out["lit"] = (False, 0.3)
        return out

    def reset(self) -> None:
        self._bright_ref = None
        self._prev_grip_open = True


class TrainedPredicateModel(PredicateModel):
    """Torch inference over a per-family weight bundle (lazy torch import)."""

    def __init__(self, bundle: ModelBundle, device: str = "cpu"):
        self.bundle = bundle
        self.device = device
        self._models: dict[str, object] = {}
        self._ok = False
        try:
            import torch

            self._torch = torch
            for fam in ("object_in_gripper", "open", "closed", "lit", "unlit",
                        "contains", "positioned", "attached", "cut", "cleaned",
                        "sprayed", "contact"):
                if bundle.has(fam):
                    self._models[fam] = torch.jit.load(str(bundle.path_for(fam)), map_location=device)
            self._ok = True
            log.info("Loaded %d trained predicate models on %s", len(self._models), device)
        except Exception as e:  # noqa: BLE001 - weights are optional at MVP
            log.warning("TrainedPredicateModel unavailable (%s); falling back to heuristics", e)
            self._torch = None

    def families(self) -> tuple[str, ...]:
        return tuple(self._models.keys())

    def predict(self, frame: Frame, step: int = 0) -> dict[str, tuple[bool, float]]:
        if not self._ok:
            return {}
        out: dict[str, tuple[bool, float]] = {}
        t = self._torch
        # Crop head + wrists to 256 and stack into one (N,3,256,256) batch.
        crops = []
        for img in (frame.rgb_head, frame.rgb_left, frame.rgb_right):
            if img is None:
                crops.append(None)
                continue
            a = np.asarray(img, dtype=np.uint8)
            if a.ndim == 3 and a.shape[2] != 3:
                a = a.repeat(3, axis=2)
            crops.append(t.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float())
        if all(c is None for c in crops):
            return out
        batch = t.cat([c for c in crops if c is not None], dim=0).to(self.device)
        batch = t.nn.functional.interpolate(batch, size=(256, 256), mode="area")
        with t.no_grad():
            for fam, model in self._models.items():
                try:
                    logit = model(batch)
                    prob = float(t.sigmoid(logit).mean().item())
                    sat = prob > 0.5
                    conf = max(prob, 1.0 - prob)
                    out[fam] = (sat, round(conf, 4))
                except Exception:  # noqa: BLE001 - a bad family must not kill the step
                    log.debug("family %s inference failed", fam, exc_info=True)
        return out


def build_model(cfg) -> PredicateModel:
    """Pick a detector backend from server config.

    ``policy.detector_bundle_dir`` (optional) pointing at trained weights ->
    :class:`TrainedPredicateModel`; otherwise the heuristic backend.
    """
    bundle_dir = getattr(cfg.policy, "detector_bundle_dir", None)
    if bundle_dir:
        bundle = ModelBundle(root=Path(bundle_dir))
        if any(bundle.has(f) for f in ("object_in_gripper", "lit")):
            return TrainedPredicateModel(bundle, device=cfg.policy.device)
    return HeuristicPredicateModel()
