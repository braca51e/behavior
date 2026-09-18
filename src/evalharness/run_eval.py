"""Wrap the v3.9.2 evaluator (design.md section 4.7).

One call runs ``omnigibson.eval.eval`` for a single task across a set of
instance indices, pointing at our policy server on ``--port``.  This is the
exact command shape from the challenge spec (section 7)::

    python -m omnigibson.eval.eval \
      --task-name turning_on_radio \
      --host 127.0.0.1 --port 8000 \
      --instance-indices 0 1 2 3 4 5 6 7 8 9 \
      --num-rollouts 1 \
      --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
      --output-dir outputs/b1k_eval --write-video

The metrics JSONs land in ``<output_dir>/json/``; videos in
``<output_dir>/videos/``.  We return the json dir so :mod:`aggregate` can score
it.  The evaluator is a *subprocess* (it owns the simulator); this module only
builds and runs the command and checks that the expected JSONs appeared.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

DEFAULT_INSTANCE_INDICES = tuple(range(10))   # report on the first 10 public


def eval_cmd(
    task_name: str,
    host: str,
    port: int,
    instance_indices: Iterable[int] = DEFAULT_INSTANCE_INDICES,
    num_rollouts: int = 1,
    output_dir: str | Path = "outputs/b1k_eval",
    max_steps: int | None = None,
    write_video: bool = True,
    headless: bool = True,
    robot_config: str | None = None,
    env: str = "behavior",
) -> list[str]:
    """Build the ``python -m omnigibson.eval.eval`` argv."""
    cmd = [
        "python", "-m", "omnigibson.eval.eval",
        "--task-name", task_name,
        "--host", host, "--port", str(port),
        "--instance-indices", *[str(i) for i in instance_indices],
        "--num-rollouts", str(num_rollouts),
        "--env-wrapper", "omnigibson.eval.wrappers.RGBDFullResWrapper",
        "--output-dir", str(output_dir),
    ]
    if write_video:
        cmd.append("--write-video")
    if headless:
        cmd.append("--headless")
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if robot_config:
        cmd += ["--robot-config", robot_config]
    return cmd


def run_task(
    task_name: str,
    port: int,
    instance_indices: Iterable[int] = DEFAULT_INSTANCE_INDICES,
    output_dir: str | Path = "outputs/b1k_eval",
    host: str = "127.0.0.1",
    max_steps: int | None = None,
    check: bool = True,
) -> Path:
    """Run the evaluator for one task; return the metrics JSON dir.

    ``check=True`` raises if the evaluator exits non-zero or the expected
    per-rollout JSONs are missing — the self-eval gate relies on this.
    """
    cmd = eval_cmd(task_name, host, port, instance_indices, output_dir=output_dir,
                   max_steps=max_steps)
    print("+", shlex.join(cmd))
    proc = subprocess.run(cmd, check=False)
    json_dir = Path(output_dir) / "json"
    if check:
        if proc.returncode != 0:
            raise RuntimeError(
                f"evaluator exited {proc.returncode} for {task_name}: {cmd}"
            )
        expected = len(list(instance_indices))
        if json_dir.is_dir():
            got = len(list(json_dir.glob(f"{task_name}_*.json")))
            if got < expected:
                raise RuntimeError(
                    f"expected {expected} JSONs for {task_name}, found {got} in {json_dir}"
                )
    return json_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--instances", type=int, nargs="*", default=list(DEFAULT_INSTANCE_INDICES))
    ap.add_argument("--output-dir", default="outputs/b1k_eval")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        print(shlex.join(eval_cmd(task_name=a.task, host=a.host, port=a.port,
                                  instance_indices=a.instances, output_dir=a.output_dir,
                                  max_steps=a.max_steps)))
        return 0
    json_dir = run_task(a.task, a.port, a.instances, a.output_dir, a.host, a.max_steps)
    print("metrics ->", json_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
