"""Multi-port server fan-out (design.md section 4.7 / 50-port IP fallback).

For parallel self-eval (and the IP-based serving option, which requires >=50
ports) we run N independent ``B1KServer`` workers, one per port, each pinned to
a task so the 40-90 sim episodes of a task all hit a warm, task-specific server
(controller chunk + VLA cache are per-task).  A simple round-robin job queue
maps ``task_id -> port`` and the evaluator fan-out (``run_eval.run_task``) uses
that port.

Usage::

    # spawn 50 servers (ports 8000..8049) across the 100 tasks
    python -m src.evalharness.parallel --ports 50 --config configs/server.yaml --task-map

The worker is a *subprocess* (``python -m b1k.server --port P --task T``) so a
crash in one rollout cannot take down the fleet.  ``parallel.py`` itself never
imports the simulator — it only orchestrates our server processes.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PortAssignment:
    port: int
    task_name: str
    task_id: int


def base_port(port: int = 8000) -> int:
    return int(port)


def assign_tasks(num_ports: int, task_ids: list[int], task_names: list[str],
                 base: int = 8000) -> list[PortAssignment]:
    """Round-robin map tasks onto ports (a port may serve several tasks over
    its lifetime; at any instant each port holds one task's server)."""
    out: list[PortAssignment] = []
    for i in range(num_ports):
        if i < len(task_ids):
            out.append(PortAssignment(port=base + i, task_name=task_names[i],
                                      task_id=task_ids[i]))
    return out


def server_cmd(port: int, task: str, config: str = "configs/server.yaml",
               host: str = "127.0.0.1") -> list[str]:
    return ["python", "-m", "b1k.server", "--config", config,
            "--port", str(port), "--host", host, "--task", task]


def wait_healthz(port: int, timeout_s: float = 60.0, host: str = "127.0.0.1") -> bool:
    """Poll ``GET /healthz`` until 200 or timeout (mirrors the evaluator's
    readiness check before it opens the WebSocket)."""
    import urllib.request

    url = f"http://{host}:{port}/healthz"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def spawn_fleet(assignments: list[PortAssignment], config: str = "configs/server.yaml",
                start: bool = True) -> list[PortAssignment]:
    """Launch the per-port server subprocesses (returns assignments)."""
    for a in assignments:
        if start:
            cmd = server_cmd(a.port, a.task_name, config)
            print("+", shlex.join(cmd))
            subprocess.Popen(cmd)
    return assignments


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", type=int, default=50)
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--config", default="configs/server.yaml")
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--no-start", action="store_true",
                    help="print the commands instead of spawning")
    a = ap.parse_args(argv)

    task_ids = a.tasks if a.tasks is not None else list(range(100))
    # Task names from the descriptions jsonl.
    from b1k.config import REPO_ROOT

    names = []
    p = REPO_ROOT / "docs" / "task-descriptions-100.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                names.append(o.get("task_name") or o.get("task") or "")
    names = names + ["task_%d" % i for i in range(len(names), len(task_ids))]

    assign = assign_tasks(a.ports, task_ids, names, base=a.base_port)
    if a.no_start:
        for x in assign:
            print(shlex.join(server_cmd(x.port, x.task_name, a.config)))
        return 0
    spawn_fleet(assign, config=a.config)
    # Wait for healthz on the first few ports as a readiness signal.
    for x in assign[: min(4, len(assign))]:
        ok = wait_healthz(x.port, timeout_s=120.0)
        print(f"healthz {x.port} ({x.task_name}): {'OK' if ok else 'TIMEOUT'}")
    print(f"fleet of {len(assign)} servers up on ports "
          f"{assign[0].port}..{assign[-1].port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
