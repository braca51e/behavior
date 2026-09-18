"""End-to-end WebSocket serving test (the acceptance criterion).

Spawns the real entrypoint ``python -m b1k.server --config configs/server.yaml
--port <p> --task 0``, then:
  1. polls ``GET /healthz`` until 200;
  2. opens the WebSocket;
  3. sends the recorded 15 MB flattened-obs payload;
  4. receives a msgpack action and checks it is a finite 23-float;
  5. sends a few more steps (the progress machine + receding horizon advance);
  6. sends a JSON ``reset`` control message and confirms the ack;
  7. tears the server down.

Exit 0 on success.  This is the closest local stand-in for the organizers'
``omnigibson.eval.eval`` driving our server over the same WebSocket contract.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b1k.protocol import decode_action  # noqa: E402

PORT = 8137
HOST = "127.0.0.1"


def wait_healthz(timeout_s: float = 90.0) -> bool:
    url = f"http://{HOST}:{PORT}/healthz"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    import websockets.sync.client as ws

    global PORT
    PORT = free_port()

    proc = subprocess.Popen(
        [sys.executable, "-m", "b1k.server",
         "--config", str(REPO / "configs" / "server.yaml"),
         "--port", str(PORT), "--host", HOST, "--task", "0",
         "--log-level", "INFO"],
        cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": __import__("os").environ["PATH"]},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_healthz(timeout_s=120.0):
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            print("FAIL: /healthz never returned 200\n", out[-4000:])
            return 1
        print("healthz OK")

        uri = f"ws://{HOST}:{PORT}"
        with ws.connect(uri, max_size=64 * 1024 * 1024) as conn:
            print("WS connected:", uri)
            # Official WebsocketClientPolicy reads metadata first.
            from b1k.protocol import unpack
            meta = unpack(conn.recv())
            print("server metadata:", meta)

            obs = (REPO / "tests" / "fixtures" / "obs_payload.bin").read_bytes()
            # Step 1.
            t0 = time.perf_counter()
            conn.send(obs)
            resp = conn.recv()
            dt = time.perf_counter() - t0
            a = decode_action(resp)
            print(f"step 1 action in {dt:.2f}s: shape={a.shape} finite={bool(np.all(np.isfinite(a)))} "
                  f"base={a[0:3].tolist()}")
            assert a.shape == (23,), a.shape
            assert np.all(np.isfinite(a)), "non-finite action"

            # A few more steps (receding horizon advances; progress ticks).
            for i in range(4):
                conn.send(obs)
                a = decode_action(conn.recv())
                assert a.shape == (23,) and np.all(np.isfinite(a))
            print("4 more steps OK")

            # Control message: reset (ack expected).
            conn.send(json.dumps({"type": "reset"}).encode())
            ack = json.loads(conn.recv())
            print("reset ack:", ack)
            assert ack.get("type") == "reset_ok"

            # After reset the server still answers with a 23-dim action.
            conn.send(obs)
            a = decode_action(conn.recv())
            assert a.shape == (23,) and np.all(np.isfinite(a))
            print("post-reset step OK")

        print("WS END-TO-END TEST PASSED")
        return 0
    finally:
        proc.terminate()
        try:
            out = proc.stdout.read().decode(errors="replace")
        except Exception:
            out = ""
        proc.wait(timeout=10)
        # Show the server log tail for the obs-layout verification + FINISH lines.
        tail = "\n".join(out.splitlines()[-25:])
        print("\n----- server log tail -----\n" + tail)


if __name__ == "__main__":
    raise SystemExit(main())
