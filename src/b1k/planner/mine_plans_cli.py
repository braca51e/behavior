"""CLI for mining per-task skill plans (docs/plans/<task_id>.json).

``miner.mine_plans`` has no CLI and the serving pipeline reads
``docs/plans/<task_id>.json`` at eval time (falling back to a
move_to-only plan when the file is absent), so this wrapper closes the gap
between ``src.training.skill_annotations`` (which writes the flat
annotation JSONL) and the plans the planner actually loads.

Usage::

    PYTHONPATH=$PWD/src python3 -m b1k.planner.mine_plans_cli \
        --annotations data/annotations.jsonl --tasks 0 1 3

Task display names come from ``docs/task-descriptions-100.jsonl``
(``task_index`` -> ``task_name``), mirroring ``self_eval._task_names``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .miner import mine_plans

log = logging.getLogger("miner.cli")

REPO_ROOT = Path(__file__).resolve().parents[3]


def _task_names(tasks: list[int]) -> dict[int, str]:
    p = REPO_ROOT / "docs" / "task-descriptions-100.jsonl"
    names: dict[int, str] = {}
    if not p.exists():
        return {t: f"task_{t}" for t in tasks}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        tid = int(o.get("task_id", o.get("task_index", -1)))
        if tid in names:
            continue
        names[tid] = o.get("task_name") or o.get("task") or ""
    return {t: names.get(t, f"task_{t}") for t in tasks}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Mine per-task skill plans.")
    ap.add_argument("--annotations", default="data/annotations.jsonl",
                    help="JSONL from src.training.skill_annotations")
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(100)),
                    help="space-separated task ids")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "plans"))
    a = ap.parse_args(argv)

    ann_path = Path(a.annotations)
    if not ann_path.exists():
        print(f"FAIL: annotations file not found: {ann_path} "
              "(run src.training.skill_annotations first)", file=sys.stderr)
        return 2
    names = _task_names(a.tasks)
    mine_plans(ann_path, tasks=names, out_dir=Path(a.out_dir))
    written = sorted(Path(a.out_dir).glob("*.json"))
    print(f"plans -> {', '.join(w.name for w in written) if written else '(none)'} "
          f"in {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
