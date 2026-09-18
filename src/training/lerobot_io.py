"""LeRobot v3.0 frame/episode/video reader (training side).

Wraps the challenge dataset layout (``meta/``, ``data/``, ``videos/``,
``annotations/``) so training scripts can iterate episodes/frames without
caring about chunk paths.  Heavy deps (pandas, pyarrow) are imported lazily so
``import src.training`` stays cheap and the import boundary (no privileged
leak) is testable.

Public::

    episodes(data_root, task_id)            -> list[(episode_index, length)]
    frames(data_root, task_id, episode)     -> iterator of frame dicts
    annotation_segments(data_root, task_id) -> list[{"t0","t1","text"}]
"""
from __future__ import annotations

import json
from pathlib import Path


def _pandas():
    import pandas as pd  # lazy

    return pd


def chunk_dir(data_root: str | Path, task_id: int) -> Path:
    return Path(data_root) / "data" / f"chunk-{int(task_id):03d}"


def episodes(data_root: str | Path, task_id: int) -> list[tuple[int, int]]:
    """Return (episode_index, length_frames) pairs for a task's chunk."""
    root = Path(data_root)
    meta = root / "meta" / "episodes" / f"chunk-{int(task_id):03d}"
    if not meta.exists():
        return []
    out: list[tuple[int, int]] = []
    for pf in sorted(meta.glob("file-*.parquet")):
        df = _pandas().read_parquet(pf, columns=["episode_index", "length"])
        for row in df.itertuples(index=False):
            out.append((int(row.episode_index), int(row.length)))
    return out


def frames(data_root: str | Path, task_id: int, episode_index: int):
    """Yield per-frame dicts: ``{"timestamp","observation.state","action",...}``."""
    root = Path(data_root)
    df = _pandas().read_parquet(
        chunk_dir(data_root, task_id),
        columns=["episode_index", "timestamp", "observation.state", "action",
                 "frame_index", "next.reward", "next.terminated", "next.truncated"],
    )
    sub = df[df["episode_index"] == episode_index]
    for row in sub.itertuples(index=False):
        import numpy as np

        yield {
            "frame_index": int(row.frame_index),
            "timestamp": float(row.timestamp),
            "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
            "action": np.asarray(row.action, dtype=np.float32),
            "next.reward": float(row["next.reward"]),
            "next.terminated": bool(row["next.terminated"]),
            "next.truncated": bool(row["next.truncated"]),
        }


def annotation_segments(data_root: str | Path, task_id: int) -> list[dict]:
    """Read per-episode language annotation segments for a task.

    The dataset ships annotations as JSONL under ``annotations/``; the exact
    path varies by release, so this scans the task chunk dir for ``*.jsonl``
    and yields ``{"episode_index","t0","t1","text"}`` dicts (the schema the
    b1k miner consumes).
    """
    root = Path(data_root)
    ann = root / "annotations" / f"chunk-{int(task_id):03d}"
    segs: list[dict] = []
    if not ann.exists():
        return segs
    for jf in sorted(ann.glob("*.jsonl")):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                segs.append(
                    {
                        "episode_index": int(o.get("episode_index", -1)),
                        "t0": float(o.get("t0", 0.0)),
                        "t1": float(o.get("t1", 0.0)),
                        "text": str(o.get("text", "")),
                    }
                )
    return segs
