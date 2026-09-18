"""Verify the documented entrypoint binds and answers /healthz (no PYTHONPATH,
relies on the editable install)."""
import os
import socket
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


port = free_port()
env = dict(os.environ)
env.pop("PYTHONPATH", None)
proc = subprocess.Popen(
    [sys.executable, "-m", "b1k.server", "--config", "configs/server.yaml",
     "--port", str(port), "--host", "127.0.0.1", "--task", "0"],
    cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
ok = False
try:
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            print("server exited early:", proc.stdout.read().decode()[-1500:])
            break
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3)
            if r.status == 200:
                print(f"healthz OK on {port} (documented entrypoint works, no PYTHONPATH)")
                ok = True
                break
        except Exception:
            time.sleep(0.5)
finally:
    proc.terminate()
    try:
        out = proc.stdout.read().decode(errors="replace")
    except Exception:
        out = ""
    proc.wait(timeout=10)
    tail = "\n".join(out.splitlines()[-8:])
    print("----- log tail -----\n" + tail)
sys.exit(0 if ok else 1)
