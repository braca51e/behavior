"""Per-task LoRA / task-embedding adapters (training side, phase 3).

The design (design.md section 4.5): after the shared pi0.5 finetune, identify
the bottom-quartile tasks from self-eval and train a *tiny* adapter per task
(LoRA r=8 on attention, or a learned task-embedding head) so one base model +
<=20 adapters still fit the 24 GB serving budget.  At serve time the adapter is
selected by ``task_id`` (the VLA wrapper loads it on a task boundary).

This is the training seam for that: it wraps the OpenPI fork's LoRA trainer on
the shared checkpoint, scoped to one task's 200 demos.  Lazy-imports the fork
like :mod:`vla_finetune`; never imported from ``src/b1k``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .vla_finetune import OPENPI_CONFIG, _openpi


@dataclass
class AdapterConfig:
    task_id: int
    base_checkpoint: str = "data/checkpoints/pi05_b1k"
    out_dir: str = ""
    lora_rank: int = 8
    steps: int = 2_000
    batch_size: int = 8
    seed: int = 0
    device: str = "cuda"

    def resolved_out(self) -> str:
        return self.out_dir or f"data/checkpoints/pi05_b1k/adapter_task_{self.task_id:03d}"


def train_adapter(task_id: int, base: str, out: str | None = None, **kw) -> Path:
    """Train one per-task adapter on top of the shared base checkpoint.

    ``base`` is the pi0.5 shared finetune dir; ``out`` (default
    ``base/adapter_task_<id>``) holds the LoRA weights + a task-embedding head.
    Returns the adapter dir.
    """
    cfg = AdapterConfig(task_id=int(task_id), base_checkpoint=base, out_dir=out or "", **kw)
    tr = _openpi()
    base_cfg = tr.get_config(OPENPI_CONFIG)
    base_cfg.model.checkpoint_dir = cfg.base_checkpoint
    # LoRA on attention (r=8) + a task-embedding head; scoped to this task's
    # 200 demos.  The fork exposes a lora-train hook; the exact knobs are
    # verified at bootstrap week 1.
    try:
        base_cfg.train.lora = True
        base_cfg.train.lora_rank = cfg.lora_rank
        base_cfg.train.task_embedding = True
        base_cfg.data.task_id = cfg.task_id
        base_cfg.train.num_steps = cfg.steps
        base_cfg.train.batch_size = cfg.batch_size
        base_cfg.seed = cfg.seed
    except Exception as e:  # noqa: BLE001 - surface fork API mismatch clearly
        raise RuntimeError(
            f"pi05_b1k LoRA/training knobs not found ({e}); verify the OpenPI "
            "'behavior' fork API in scripts/bootstrap.sh week-1"
        ) from e
    out_dir = cfg.resolved_out()
    tr.train(base_cfg, checkpoint_dir=out_dir, resume=False)
    p = Path(out_dir)
    (p / "adapter_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--base", default="data/checkpoints/pi05_b1k")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = train_adapter(a.task, a.base, a.out, lora_rank=a.lora_rank,
                        steps=a.steps, batch_size=a.batch_size, seed=a.seed)
    print("trained adapter ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
