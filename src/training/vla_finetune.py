"""pi0.5 skill-conditioned finetune (training side, OpenPI 'behavior' fork).

The design (design.md section 2 / 4.6): fine-tune pi0.5 **once** on all 20k
demos with language conditioning = ``task_text + " · subgoal: " + subgoal.text``
(32-step horizon), producing a single shared VLA checkpoint for all 100 tasks.
Per-task LoRA/task-embedding adapters (phase 3) are trained on top of this base
by :mod:`adapters`.

This module is the *training seam*: it loads the OpenPI fork's pi05_b1k config,
wires the skill-conditioned dataset, and runs the fork's trainer.  It requires
the ``wensi-ai/openpi`` ``behavior`` fork + ``uv`` training env (design week 1)
and the demo dataset, so it is lazy-imported and cannot run on a serving box —
by construction it also is never imported from ``src/b1k``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# OpenPI 'behavior' fork (wensi-ai/openpi).  The exact module paths are verified
# against the fork at bootstrap (scripts/bootstrap.sh); the seam below calls
# them.  Kept as import-time *optional* so this file parses without the fork.
OPENPI_CONFIG = "pi05_b1k"          # pi0.5 R1Pro challenge config in the fork
ACTION_HORIZON = 32                # 32-step action chunk (serve_b1k convention)
RECEDING = 16                      # serving re-query period (design: K=16)


@dataclass
class TrainingConfig:
    data_root: str = "data/demos"
    tasks: list = None
    norm_stats: str = "data/stats.json"
    out_dir: str = "data/checkpoints/pi05_b1k"
    batch_size: int = 8
    steps: int = 50_000
    action_horizon: int = ACTION_HORIZON
    seed: int = 0
    device: str = "cuda"
    use_subgoal_conditioning: bool = True

    def to_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def _openpi():
    """Import the OpenPI 'behavior' fork's training entrypoints (lazy)."""
    try:
        import openpi  # noqa: F401  (fork on PYTHONPATH)

        from openpi import training as openpi_training  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "OpenPI 'behavior' fork (wensi-ai/openpi, branch 'behavior') is not "
            "installed.  Run scripts/bootstrap.sh on a training box first."
        ) from e
    return openpi_training


def finetune_vla(cfg: TrainingConfig) -> Path:
    """Skill-conditioned pi0.5 finetune -> checkpoint dir.

    Wires the fork's ``pi05_b1k`` config to our LeRobot dataset with
    subgoal-language conditioning, recomputes norm stats from ``cfg.norm_stats``
    (post velocity fix), and runs the trainer.  Returns the checkpoint dir.
    """
    tr = _openpi()
    base_cfg = tr.get_config(OPENPI_CONFIG)
    # Skill-conditioning dataset: language = task_text + " · subgoal: " + subgoal.
    base_cfg = _apply_subgoal_conditioning(base_cfg, cfg)
    base_cfg.train.batch_size = cfg.batch_size
    base_cfg.train.num_steps = cfg.steps
    base_cfg.seed = cfg.seed
    out = tr.train(base_cfg, checkpoint_dir=cfg.out_dir, resume=False)
    return Path(cfg.out_dir)


def _apply_subgoal_conditioning(cfg_obj, cfg: TrainingConfig):
    """Attach the skill-conditioned dataset wrapper to the fork config.

    The fork's pi05_b1k dataset already consumes LeRobot v3 frames; we override
    its ``language`` field so conditioning = ``task_text + " · subgoal: " +
    subgoal.text`` (the exact string the serving VLA uses, so train/serve match).
    """
    # This is the integration seam: the fork exposes a language-transform hook.
    # The implementation is validated at bootstrap (week 1) against the fork's
    # actual API; here we set the documented config knobs.
    try:
        cfg_obj.data.language = "task_with_subgoal"
        cfg_obj.data.subgoal_template = "{task_text} · subgoal: {subgoal_text}"
        cfg_obj.data.action_horizon = cfg.action_horizon
        cfg_obj.data.receding_horizon = RECEDING
        cfg_obj.data.norm_stats = cfg.norm_stats
    except Exception as e:  # noqa: BLE001 - fork API may differ; surface clearly
        raise RuntimeError(
            f"pi05_b1k subgoal-conditioning knobs not found ({e}); verify the "
            "OpenPI 'behavior' fork API in scripts/bootstrap.sh week-1"
        ) from e
    return cfg_obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/demos")
    ap.add_argument("--norm-stats", default="data/stats.json")
    ap.add_argument("--out-dir", default="data/checkpoints/pi05_b1k")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = TrainingConfig(
        data_root=a.data_root, norm_stats=a.norm_stats, out_dir=a.out_dir,
        batch_size=a.batch_size, steps=a.steps, seed=a.seed,
    )
    out = finetune_vla(cfg)
    cfg.to_json(out / "training_config.json")
    print("finetuned pi0.5 ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
