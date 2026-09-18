"""Per-episode language -> skill-segment parse (training side).

Turns the dataset's per-episode language annotations into the flat
``{task_id, episode_index, t0, t1, text}`` lines that
:func:`b1k.planner.miner.mine_plans` consumes.  This is the training-time
bridge between the LeRobot reader and the b1k miner; it also validates that
every segment maps to a canonical skill (unmatched fragments are logged so the
taxonomy can be extended before freezing).

Usage::

    python -m src.training.skill_annotations --data-root data/demos --tasks 0 \
        --out data/annotations.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from . import lerobot_io

log = logging.getLogger("training.skill_annotations")


def parse_skill_segments(data_root: str | Path, tasks: list[int]) -> list[dict]:
    """Return flat annotation lines for the given tasks."""
    lines: list[dict] = []
    unmatched: Counter = Counter()
    for tid in tasks:
        try:
            segs = lerobot_io.annotation_segments(data_root, tid)
        except Exception as e:  # noqa: BLE001 - a bad chunk must not kill the run
            log.warning("task %d annotations unreadable: %s", tid, e)
            continue
        # Validate against the canonical taxonomy.
        from b1k.planner.skill_tax import match_skill

        for s in segs:
            if match_skill(s["text"]) is None:
                unmatched[s["text"][:40]] += 1
            lines.append(
                {
                    "task_id": tid,
                    "episode_index": s["episode_index"],
                    "t0": s["t0"],
                    "t1": s["t1"],
                    "text": s["text"],
                }
            )
    if unmatched:
        log.warning("%d unmatched annotation fragments (top: %s)",
                    sum(unmatched.values()), unmatched.most_common(5))
    return lines


def write_lines(lines: list[dict], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for o in lines:
            f.write(json.dumps(o) + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/demos")
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(100)))
    ap.add_argument("--out", default="data/annotations.jsonl")
    a = ap.parse_args(argv)
    lines = parse_skill_segments(a.data_root, a.tasks)
    write_lines(lines, a.out)
    print(f"wrote {len(lines)} annotation lines -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
