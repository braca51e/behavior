"""Per-chunk Hugging Face download for the 2026 challenge demos.

Canonical pattern (challenge spec section 2.1): the LeRobot v3.0 dataset
``behavior-1k/2026-challenge-demos`` is ~3.27 TB across 100 task chunks
(``chunk-000`` = task 0 ... ``chunk-099`` = task 99).  Download only the chunks
you need; this module builds the ``hf download`` command per task and can
run them (heavy network) when invoked.

Usage::

    python -m src.training.download --task 0 --root data/demos
    python -m src.training.download --tasks 0 1 2 --root data/demos --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

DATASET = "behavior-1k/2026-challenge-demos"
REPO_TYPE = "dataset"


def _hf_bin() -> str:
    """Prefer the current ``hf`` CLI; fall back to legacy ``huggingface-cli``."""
    return shutil.which("hf") or shutil.which("huggingface-cli") or "hf"


def chunk_name(task_id: int) -> str:
    return f"chunk-{int(task_id):03d}"


def download_args(task_id: int, data_root: str | Path) -> list[str]:
    """Return the ``hf download`` argv for one task chunk."""
    c = chunk_name(task_id)
    return [
        _hf_bin(), "download", DATASET,
        "--repo-type", REPO_TYPE,
        "--local-dir", str(data_root),
        "--include", f"data/{c}/**",
        "--include", f"meta/episodes/{c}/**",
        "--include", f"videos/*/{c}/**",
        "--include", "meta/info.json",
        "--include", "meta/stats.json",
        "--include", "meta/tasks.parquet",
        "--include", "meta/tasks.jsonl",
    ]


def download_chunks(task_ids: list[int], data_root: str | Path,
                    run: bool = True) -> list[list[str]]:
    """Build (and optionally run) per-task download commands."""
    cmds = [download_args(t, data_root) for t in task_ids]
    for cmd in cmds:
        print("+", " ".join(cmd))
        if run:
            subprocess.run(cmd, check=True)
    return cmds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--tasks", type=int, nargs="*", default=None,
                    help="space-separated list of task ids")
    ap.add_argument("--root", default="data/demos")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without running")
    args = ap.parse_args(argv)
    ids: list[int] = []
    if args.task is not None:
        ids.append(args.task)
    if args.tasks:
        ids.extend(args.tasks)
    if not ids:
        ids = list(range(100))
    download_chunks(sorted(set(ids)), args.root, run=not args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
