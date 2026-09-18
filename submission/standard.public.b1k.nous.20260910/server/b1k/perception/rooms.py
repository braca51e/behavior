"""Room prior from task metadata (design.md section 4.4).

``B100_task_misc.csv`` ships a per-task "Rooms to include" column (v3.9.2
evaluator loads exactly those rooms, so it is legal task metadata at eval).
This module parses that CSV into a ``RoomPrior`` giving, per task, the ordered
room list the evaluation scene contains.  The planner/progress layer uses it to
bias cross-room search ("the target is not in front of me; which room next?").

Format (see ``raw/2025_misc.csv`` — the 2026 file ``B100_task_misc.csv`` has the
same columns)::

    Task ID,Task,Rooms to inlcude,Task Ready for Slurm,Task Ready for local,Task Ready to Test
    0,turning_on_radio,"corridor_0
    garden_0
    kitchen_0",...

Note the column header typo ``inlcude`` is preserved verbatim from the
challenge CSV (do not "fix" it — it is how the real file spells it).
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("b1k.rooms")

# The real 2026 metadata column header (sic).
ROOMS_COLUMN = "Rooms to inlcude"
ROOMS_COLUMN_FIXED = "Rooms to include"


@dataclass
class RoomPrior:
    """Ordered list of loaded rooms per task id (0..99)."""

    by_task_id: dict[int, list[str]]
    by_task_name: dict[str, list[str]]

    @classmethod
    def from_csv(cls, path: str | Path) -> "RoomPrior":
        by_id: dict[int, list[str]] = {}
        by_name: dict[str, list[str]] = {}
        path = Path(path)
        if not path.exists():
            log.warning("Room prior CSV %s not found (RoomPrior will be empty)", path)
            return cls(by_id, by_name)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Task") or "").strip()
                tid_raw = (row.get("Task ID") or "").strip()
                rooms_cell = row.get(ROOMS_COLUMN) or row.get(ROOMS_COLUMN_FIXED) or ""
                rooms = [r.strip() for r in rooms_cell.splitlines() if r.strip()]
                if tid_raw:
                    try:
                        by_id[int(tid_raw)] = rooms
                    except ValueError:
                        pass
                if name:
                    by_name[name] = rooms
        return cls(by_id, by_name)

    def rooms_for(self, task_id: int | None = None, task_name: str | None = None) -> list[str]:
        if task_name is not None and task_name in self.by_task_name:
            return list(self.by_task_name[task_name])
        if task_id is not None and task_id in self.by_task_id:
            return list(self.by_task_id[task_id])
        return []

    def __len__(self) -> int:
        return len(self.by_task_id)
