"""Skill-conditioned VLA inference wrapper (design.md section 4.5).

One shared policy for all 100 tasks: pi0.5 (OpenPI 'behavior' fork, ``pi05_b1k``
config) fine-tuned once on all 20k demos with language conditioning
``task_text + " · subgoal: " + subgoal.text``.  Output: a 32-step action chunk
(23-dim), served receding-horizon by the controller.

To keep the serving contract verifiable **without** the multi-GB checkpoint or
a GPU, three backends share the same ``act()`` surface:

* ``"vla"``   — real pi0.5 inference (lazy torch + checkpoint load).  The final
  backend; selected via ``policy.backend: vla`` + ``checkpoint_dir``.
* ``"echo"``  — MVP: replays the nearest *recorded demo action chunk* for the
  task (``data/demos/<task_id>/actions.npy``).  CPU-only, no weights.  Lets the
  full server + harness + scoring pipeline run end-to-end in a fixture, exactly
  like the design's "baseline pi0.5 checkpoint serving" milestone, without
  shipping 3.27 TB of data or a GPU.
* ``"noop"``  — safe park action (zero base, hold pose).  Used for the
  episode-end-semantics verification fixture (design week-1 risk #5).

Only the ``"vla"`` backend touches torch; the others are pure numpy so the
package imports and serves on a CPU-only box.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..embodiment import ACTION_DIM, make_no_op_action
from ..protocol import Frame
from .norm import NormStats

log = logging.getLogger("b1k.vla")


class SkillConditionedVLA:
    """act(frame, subgoal_text, task_text) -> (horizon, 23) action chunk."""

    def __init__(
        self,
        backend: str = "echo",
        checkpoint_dir: str | Path | None = None,
        norm_stats_path: str | Path | None = None,
        device: str = "cpu",
        action_horizon: int = 32,
        demos_root: str | Path | None = None,
        seed: int = 0,
    ):
        self.backend = backend
        self.device = device
        self.action_horizon = int(action_horizon)
        self.demos_root = Path(demos_root) if demos_root else None
        self.seed = int(seed)
        self.norm = NormStats.load(norm_stats_path)
        self._demo_cache: dict[int, np.ndarray] = {}
        self._torch_model = None
        if backend == "vla":
            self._init_vla(checkpoint_dir)
        elif backend not in ("echo", "noop"):
            raise ValueError(f"unknown policy backend: {backend!r}")

    # ---- backend init -------------------------------------------------------
    def _init_vla(self, checkpoint_dir: str | Path | None) -> None:
        """Load the pi0.5 checkpoint (OpenPI 'behavior' fork).  Lazy torch."""
        if checkpoint_dir is None:
            raise ValueError('backend="vla" requires policy.checkpoint_dir')
        try:
            import torch
        except ImportError as e:  # pragma: no cover - torch is a hard dep for vla
            raise RuntimeError('backend="vla" needs torch installed') from e
        self._torch = torch
        # OpenPI 'behavior' fork exposes pi05_b1k inference via its serve path;
        # we load the exported torchscript / safetensors bundle.  The exact
        # loader mirrors wensi-ai/openpi serve_b1k.py (design week-1 verifies).
        cp = Path(checkpoint_dir)
        if not cp.exists():
            raise FileNotFoundError(f"VLA checkpoint dir missing: {cp}")
        self._ckpt_dir = cp
        self._torch.manual_seed(self.seed)
        log.info("VLA backend initialised (checkpoint=%s, device=%s)", cp, self.device)

    def _load_demo_actions(self, task_id: int) -> np.ndarray | None:
        """Recorded demo action sequence for a task (echo backend), if present."""
        if task_id in self._demo_cache:
            return self._demo_cache[task_id]
        arr = None
        if self.demos_root is not None:
            p = self.demos_root / f"{int(task_id)}" / "actions.npy"
            if p.exists():
                arr = np.load(p).astype(np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                arr = arr[:, :ACTION_DIM]
                self._demo_cache[task_id] = arr
        return arr

    # ---- main API -----------------------------------------------------------
    def act(self, frame: Frame, subgoal_text: str, task_text: str,
            task_id: int | None = None, step: int = 0) -> np.ndarray:
        """Return an (action_horizon, 23) chunk in robot action units."""
        if self.backend == "noop":
            base = make_no_op_action()
            return np.repeat(base[None, :], self.action_horizon, axis=0)

        if self.backend == "echo":
            return self._act_echo(frame, task_id, step)

        # backend == "vla"
        return self._act_vla(frame, subgoal_text, task_text)

    # ---- backends -----------------------------------------------------------
    def _act_echo(self, frame: Frame, task_id: int | None, step: int = 0) -> np.ndarray:
        """Replay a recorded demo chunk, offset by the episode step (deterministic
        w.r.t. task_id + step).  Falls back to a no-op chunk when no demo is
        cached for the task (still a valid 23-dim action, so serving never
        crashes on an unseen task)."""
        H = self.action_horizon
        demo = self._load_demo_actions(task_id if task_id is not None else -1)
        out = np.zeros((H, ACTION_DIM), dtype=np.float32)
        if demo is not None and demo.shape[0] > 0:
            n = demo.shape[0]
            # Recede through the demo, wrapping (demos are ~1000+ steps; a
            # fixture demo may be shorter, so wrap).
            for h in range(H):
                idx = (step + h) % n
                out[h] = demo[idx]
        else:
            out[:] = make_no_op_action()
        # Echo demos are recorded in robot action units (R1Pro has
        # action_normalize: false), so no norm transform is applied.
        return out

    def _act_vla(self, frame: Frame, subgoal_text: str, task_text: str) -> np.ndarray:
        """Real pi0.5 skill-conditioned inference (torch).  Lazy, GPU path."""
        t = self._torch
        # Build the language conditioning string exactly as the OpenPI fork's
        # serve_b1k.py does: task text + subgoal.
        lang = f"{task_text} · subgoal: {subgoal_text}"
        # Inputs: head RGB 720, L/R wrist RGB 480, L/R depth 480, proprio 61.
        imgs = []
        for im in (frame.rgb_head, frame.rgb_left, frame.rgb_right):
            if im is None:
                imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                imgs.append(im)
        # NOTE: The exact pi05_b1k preprocessor (resize/tokenize) is invoked via
        # the OpenPI fork; the call below is the integration seam.  Week-1 of the
        # plan wires this to the fork's `policy.act()` with the recorded obs.
        try:
            from openpi.policies import policy as openpi_policy  # type: ignore
        except Exception as e:  # pragma: no cover - depends on fork install
            raise RuntimeError(
                "openpi 'behavior' fork not importable; install it and set "
                f"policy.checkpoint_dir to run the real VLA ({e})"
            ) from e
        obs = openpi_policy.preprocess_r1pro_obs(frame, lang)  # fork-specific
        chunk = openpi_policy.infer(self._ckpt_dir, obs, horizon=self.action_horizon)
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        chunk = chunk[:, :ACTION_DIM]
        if chunk.shape[0] < self.action_horizon:
            pad = np.repeat(chunk[-1:], self.action_horizon - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return self.norm.apply(chunk)

    # ---- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        """Episode boundary (called by the server on WS (re)connect).

        The demo-action cache is kept across episodes (it is per-task and
        small); only the real VLA's KV/recency state would be reset here.
        """
        if self.backend == "vla" and self._torch_model is not None:
            pass  # fork-specific KV reset seam
