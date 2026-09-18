"""Per-chunk Hugging Face download for the 2026 challenge demos.

Canonical pattern (challenge spec section 2.1): the LeRobot v3.0 dataset
``behavior-1k/2026-challenge-demos`` is ~3.27 TB across 100 task chunks
(``chunk-000`` = task 0 ... ``chunk-099`` = task 99).  Download only the chunks
you need.

Uses ``huggingface_hub`` Python API (no ``hf`` CLI required). The CLI is
optional if present.

Usage::

    python -m src.training.download --task 0 --root data/demos
    python -m src.training.download --tasks 0 1 2 --root data/demos --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "behavior-1k/2026-challenge-demos"
REPO_TYPE = "dataset"


def chunk_name(task_id: int) -> str:
    return f"chunk-{int(task_id):03d}"


def allow_patterns(task_id: int) -> list[str]:
    c = chunk_name(task_id)
    return [
        f"data/{c}/**",
        f"meta/episodes/{c}/**",
        f"videos/*/{c}/**",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/tasks.jsonl",
    ]


def _hf_bin() -> str | None:
    return shutil.which("hf") or shutil.which("huggingface-cli")


def download_args(task_id: int, data_root: str | Path) -> list[str]:
    """Return the ``hf download`` argv for one task chunk (for --dry-run / docs)."""
    bin_ = _hf_bin() or "hf"
    c = chunk_name(task_id)
    return [
        bin_, "download", DATASET,
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


def _download_via_hub(task_id: int, data_root: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "FAIL: need package huggingface_hub (or install the hf CLI).\n"
            "  pip install -U 'huggingface_hub[cli]'\n"
            "  # or:  docker run --rm -e HF_TOKEN -v \"$PWD/data/demos:/data/demos\" "
            "b1k-groot shell -c 'hf download …'"
        ) from e

    patterns = allow_patterns(task_id)
    print(f"+ snapshot_download {DATASET} chunk-{task_id:03d} -> {data_root}")
    for p in patterns:
        print(f"    include {p}")
    snapshot_download(
        repo_id=DATASET,
        repo_type=REPO_TYPE,
        local_dir=str(data_root),
        allow_patterns=patterns,
        token=True,  # use HF_TOKEN / cached login if present
    )


def _download_via_cli(task_id: int, data_root: Path) -> None:
    cmd = download_args(task_id, data_root)
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_chunks(task_ids: list[int], data_root: str | Path,
                    run: bool = True) -> list[list[str]]:
    """Build (and optionally run) per-task download commands / API calls."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    cmds = [download_args(t, root) for t in task_ids]
    for tid, cmd in zip(task_ids, cmds):
        if not run:
            print("+", " ".join(cmd))
            continue
        # Prefer Python API so hosts without `hf` on PATH still work.
        if _hf_bin() is not None:
            _download_via_cli(tid, root)
        else:
            _download_via_hub(tid, root)
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
    info = Path(args.root) / "meta" / "info.json"
    if not args.dry_run and not info.is_file():
        print(f"WARN: expected {info} after download", file=sys.stderr)
        return 1
    print(f"demos -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
