"""Recompute LeRobot ``meta/stats.json`` (training side).

The Aug-2026 velocity fix changed several ``observation.state`` slices
(``arm_left_qvel`` [10:17], ``gripper_left_qvel`` [26:28], ``arm_right_qvel``
[35:42], ``gripper_right_qvel`` [51:53], ``trunk_qvel`` [57:61]); the design
mandates recomputing norm stats *after* re-syncing so the VLA is trained/served
against the corrected distribution.

This module computes per-dimension mean/std for ``action`` and
``observation.state`` over a task's chunk and writes the LeRobot stats schema
(``{"action": {"mean","std","min","max"}, "observation.state": {...}}``).
Heavy deps (pandas) are lazy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import lerobot_io

_CHUNK_RE = re.compile(r"^chunk-(\d{3})$")


def _stats(arr: np.ndarray) -> dict:
    a = arr.astype(np.float64)
    return {
        "mean": a.mean(axis=0).tolist(),
        "std": np.where(a.std(axis=0) < 1e-8, 1.0, a.std(axis=0)).tolist(),
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
    }


def discover_task_ids(data_root: str | Path) -> list[int]:
    """Return sorted task ids that have a ``data/chunk-NNN`` directory on disk."""
    data = Path(data_root) / "data"
    if not data.is_dir():
        return []
    ids: list[int] = []
    for p in data.iterdir():
        m = _CHUNK_RE.match(p.name)
        if m and p.is_dir():
            ids.append(int(m.group(1)))
    return sorted(ids)


def compute_task_stats(data_root: str | Path, task_id: int,
                       fields=("action", "observation.state")) -> dict:
    """Compute norm stats for one task's chunk over the given fields."""
    pd = lerobot_io._pandas()
    data_dir = lerobot_io.chunk_dir(data_root, task_id)
    if not data_dir.exists():
        raise FileNotFoundError(f"No LeRobot chunk dir for task {task_id}: {data_dir}")
    df = pd.read_parquet(data_dir, columns=list(fields))
    out = {}
    for f in fields:
        out[f] = _stats(np.stack(df[f].to_numpy()))
    return out


def recompute_norm_stats(data_root: str | Path, tasks: list[int],
                         out: str | Path | None = None) -> dict:
    """Aggregate per-task stats into a dataset-level ``stats.json``."""
    if not tasks:
        raise FileNotFoundError(
            f"No tasks to score under {data_root} "
            f"(expected data/chunk-NNN/…). Download demos first or pass --tasks."
        )
    merged = {}
    used: list[int] = []
    for tid in tasks:
        data_dir = lerobot_io.chunk_dir(data_root, tid)
        if not data_dir.exists():
            print(f"skip task {tid}: missing {data_dir}")
            continue
        s = compute_task_stats(data_root, tid)
        used.append(tid)
        for k, v in s.items():
            # LeRobot stats.json holds dataset-wide per-dim stats; we average
            # the per-task summaries (a training refinement — exact values are
            # recomputed over the full batch in vla_finetune).
            if k not in merged:
                merged[k] = {key: np.array(val) for key, val in v.items()}
            else:
                for key, val in v.items():
                    merged[k][key] = merged[k][key] + np.array(val)
    if not used:
        raise FileNotFoundError(
            f"None of the requested tasks have chunks under {data_root}/data/"
        )
    final = {}
    for k, v in merged.items():
        n = len(used)
        final[k] = {key: (val / n).tolist() for key, val in v.items()}
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
    print(f"stats from tasks {used}")
    return final


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/demos")
    ap.add_argument(
        "--tasks", type=int, nargs="*", default=None,
        help="task ids (default: auto-detect data/chunk-NNN under --data-root)",
    )
    ap.add_argument("--out", default="data/stats.json")
    a = ap.parse_args()
    tasks = a.tasks if a.tasks is not None else discover_task_ids(a.data_root)
    recompute_norm_stats(a.data_root, tasks, a.out)
    print("wrote", a.out)
