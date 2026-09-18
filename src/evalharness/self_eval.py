"""Reproducible self-eval entrypoint (design.md: ``scripts/self_eval.sh``).

Produces ``outputs/<date>/report.md`` (plus ``summary.json``).  Two modes:

* ``--mode fixture`` (default; runs on CPU, no simulator) — verifies the
  *serving pipeline* end-to-end in-process (spins a ``B1KServer`` per fixture
  task, feeds the recorded obs frames for a bounded number of steps, asserts
  every action is a finite 23-dim vector and the loop never crashes / parks
  correctly), then exercises the *scoring math* on the five fixture metrics
  JSONs with the exact ``score_utils`` formula.  The report is explicitly
  labeled "fixture verification" and does **not** claim a leaderboard Q —
  real Q comes from the simulator.

* ``--mode sim`` (requires ``omnigibson`` + a GPU box with BEHAVIOR v3.9.2) —
  fans out the real evaluator (``run_eval`` + ``parallel``) over the selected
  tasks/instances against our live server, collects the per-rollout metrics
  JSONs the simulator writes, and aggregates them into the report.  This is the
  path that produces a real ``Q``.

``scripts/self_eval.sh`` runs the local test suite first (it must pass), then
this module.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from b1k.config import load_config  # noqa: E402
from b1k.protocol import decode_action  # noqa: E402
from b1k.server import B1KServer  # noqa: E402
from evalharness.aggregate import aggregate  # noqa: E402
from evalharness.report import render_report  # noqa: E402

log = logging.getLogger("self_eval")


# ---------------------------------------------------------------------------
# Fixture mode
# ---------------------------------------------------------------------------
FIXTURE_TASKS = [0, 1, 3]          # tasks with BDDL goals in data/bddl_goals.json
STEPS_PER_TASK = 30                # bounded rollout length for the pipeline check


def _serve_fixture_task(cfg, task_id: int, steps: int = STEPS_PER_TASK) -> dict:
    """Drive one fixture task through the live serving pipeline in-process."""
    from b1k.protocol import frame_from_dict

    server = B1KServer(cfg, task=task_id)
    obs_bytes = (REPO / "tests" / "fixtures" / "obs_payload.bin").read_bytes()
    from b1k.protocol import parse_obs

    frame = parse_obs(obs_bytes)
    result = {"task_id": task_id, "task": server.task.task_name,
              "steps": 0, "actions": [], "finite": True, "parks": 0, "error": None}
    for s in range(steps):
        try:
            a = server.handle_frame(frame)
            a = np.asarray(a, dtype=np.float32)
            result["steps"] += 1
            if a.shape != (23,) or not np.all(np.isfinite(a)):
                result["finite"] = False
            result["actions"].append(a[0:3].tolist())
            if server._finished:
                result["parks"] += 1
        except Exception as e:  # noqa: BLE001 - record and stop
            result["error"] = f"{type(e).__name__}: {e}"
            break
    result["finished"] = bool(server._finished)
    result["confirmed_preds"] = server.tracker.confirmed()
    return result


def run_fixture_mode(out_dir: Path) -> int:
    """Verify the serving pipeline + scoring math; write the report."""
    cfg = load_config(REPO / "configs" / "server.yaml")
    print("[fixture] verifying serving pipeline per task ...")
    per_task = [ _serve_fixture_task(cfg, t) for t in FIXTURE_TASKS ]
    for r in per_task:
        status = "OK" if (r["finite"] and r["error"] is None and r["steps"] > 0) else "FAIL"
        print(f"  task {r['task_id']:>2} ({r['task']:<28}): {r['steps']}/{STEPS_PER_TASK} "
              f"steps finite={r['finite']} finished={r['finished']} -> {status}")

    serving_ok = all(r["finite"] and r["error"] is None and r["steps"] > 0 for r in per_task)

    # Scoring math on the five fixture metrics (real values).
    from evalharness.aggregate import aggregate as _agg

    print("[fixture] scoring the 5 fixture metrics JSONs (exact score_utils formula) ...")
    summ = _agg(REPO / "tests" / "fixtures")
    o = summ.overall
    print(f"  Q={o['q_score']:.4f}  task_sr={o['task_sr']:.4f}  "
          f"time_score={o['time_score']:.4f}  rollouts={o['num_rollouts']}")

    report = render_report(summ, out_dir / "report.md", team="b1k")
    # Append the fixture-verification block (so the report shows both halves).
    extra = _fixture_block(per_task, serving_ok)
    with open(report, "a", encoding="utf-8") as f:
        f.write("\n" + extra + "\n")
    print(f"[fixture] report -> {report}")
    return 0 if serving_ok else 1


def _fixture_block(per_task, serving_ok) -> str:
    lines = ["## Fixture verification (serving pipeline — not a leaderboard Q)",
             "",
             f"- serving pipeline: **{'OK' if serving_ok else 'FAIL'}**",
             "",
             "| task | id | steps | finite 23-dim | finished(park) | confirmed predicates | error |",
             "|---|---|---|---|---|---|---|"]
    for r in per_task:
        lines.append(
            f"| {r['task']} | {r['task_id']} | {r['steps']} | "
            f"{'yes' if r['finite'] else 'NO'} | {'yes' if r['finished'] else 'no'} | "
            f"{', '.join(r['confirmed_preds']) or '—'} | {r['error'] or '—'} |"
        )
    lines += ["",
              "> This section is a CPU-only verification that the WebSocket serving",
              "> contract, action safety, and progress/park logic run end-to-end on the",
              "> recorded obs fixture.  It is **not** a sim score.  Real Q requires",
              "> ``--mode sim`` on an OmniGibson v3.9.2 + GPU box.",
              ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sim mode
# ---------------------------------------------------------------------------
def run_sim_mode(out_dir: Path, tasks: list[int], instances: list[int],
                 base_port: int, host: str) -> int:
    """Run the real evaluator fan-out against our live server (needs OmniGibson)."""
    from evalharness import parallel, run_eval

    cfg = load_config(REPO / "configs" / "server.yaml")
    # Check the simulator is importable.
    try:
        import omnigibson  # noqa: F401
    except ImportError as e:
        print("FAIL: omnigibson is not installed — sim mode requires the v3.9.2 "
              "evaluator (run scripts/bootstrap.sh on a GPU box).", file=sys.stderr)
        print(f"  ({e})", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "b1k_eval"
    n_ports = min(len(tasks), 50)
    names = _task_names(tasks)
    assign = parallel.assign_tasks(n_ports, tasks, names, base=base_port)
    parallel.spawn_fleet(assign, config=str(REPO / "configs" / "server.yaml"))
    for a in assign:
        if not parallel.wait_healthz(a.port, timeout_s=180.0, host=host):
            print(f"FAIL: server on port {a.port} did not become healthy", file=sys.stderr)
            return 2
    json_dirs = []
    port_by_task = {a.task_id: a.port for a in assign}
    for t in tasks:
        name = names[t]
        port = port_by_task.get(t, base_port)
        try:
            jd = run_eval.run_task(name, port, instances,
                                    output_dir=eval_dir, host=host)
            json_dirs.append(jd)
            print(f"  task {name} (port {port}) -> {jd}")
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: task {name} failed: {e}", file=sys.stderr)
    # Aggregate every produced JSON dir.
    combined = out_dir / "json"
    combined.mkdir(parents=True, exist_ok=True)
    _copy_jsons(json_dirs, combined)
    summ = aggregate(combined)
    report = render_report(summ, out_dir / "report.md", team="b1k")
    print(f"[sim] report -> {report}  Q={summ.overall['q_score']:.4f} "
          f"rollouts={summ.overall['num_rollouts']}")
    return 0


def _copy_jsons(src_dirs, dst: Path) -> None:
    import shutil

    for d in src_dirs:
        d = Path(d)
        if d.is_dir():
            for f in d.glob("*.json"):
                shutil.copy2(f, dst / f.name)


def _task_names(tasks: list[int]) -> list[str]:
    names = []
    p = REPO / "docs" / "task-descriptions-100.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                names.append(o.get("task_name") or o.get("task") or "")
    # tasks may be a subset; map index -> name
    by_idx = {}
    for line in (p.read_text().splitlines() if p.exists() else []):
        if line.strip():
            o = json.loads(line)
            by_idx[int(o.get("task_id", o.get("task_index", -1)))] = \
                o.get("task_name") or o.get("task") or ""
    out = []
    for t in tasks:
        out.append(by_idx.get(t, f"task_{t}"))
    return out


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixture", "sim"], default="fixture")
    ap.add_argument("--tasks", type=int, nargs="*", default=None,
                    help="sim mode: task ids to run (default: fixture tasks)")
    ap.add_argument("--instances", type=int, nargs="*",
                    default=list(range(10)))
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default=None,
                    help="output dir (default: outputs/<date>)")
    a = ap.parse_args(argv)

    date = _dt.datetime.now().strftime("%Y%m%d")
    out_dir = Path(a.out) if a.out else REPO / "outputs" / date
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    print(f"self_eval mode={a.mode} out={out_dir}")
    if a.mode == "sim":
        tasks = a.tasks or FIXTURE_TASKS
        return run_sim_mode(out_dir, tasks, a.instances, a.base_port, a.host)
    return run_fixture_mode(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
