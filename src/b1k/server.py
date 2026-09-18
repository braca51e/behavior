"""WebSocket policy server (design.md section 4.1, 5).

Entrypoint::

    python -m b1k.server --config configs/server.yaml --port 8000

Serving contract (mirrors ``omnigibson.eval.policies.WebsocketPolicy``):
  * ``GET /healthz`` -> 200 once the model is loaded and ready.
  * On a WS connection: reset episode state, then for each binary frame
    decode the flattened obs (msgpack), run the pipeline, and reply with one
    msgpack-encoded 23-dim action.

The heavy lifting is in :meth:`B1KServer.handle_obs`, which is pure-Python and
fully unit-testable (feed a decoded obs dict, get an action back) — the WS
transport is a thin async wrapper around it.

Per-step pipeline (design.md section 5)::

    obs dict -> Frame
      1. PredicateTracker.update(frame)          # onboard predicate map
      2. Progress.tick(pred_state, subgoal, t)   # CONTINUE/ADVANCE/RETRY/...
         CONTINUE ->
      3.   Odometry.update(base_qvel, depth_head)
      4.   controller.step() -> one cached-chunk action  (VLA re-query every K)
         ADVANCE/RETRY/REPLAN -> re-query VLA on the (new) subgoal
         SEARCH -> re-query + frontier bias
         FINISH  -> park pose, no-op actions
      5. msgpack-encode(action) -> ws
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ServerConfig, load_config
from .controller import ActionController
from .embodiment import ACTION_DIM, make_no_op_action
from .planner.miner import MinedPlan
from .planner.progress import Act, Action as PAction, Progress
from .planner.task_planner import TaskPlanner, load_mined_plan
from .policy.vla import SkillConditionedVLA
from .perception.detectors.state import PredicateTracker
from .perception.odometry import Odometry
from .perception.rooms import RoomPrior
from .protocol import Frame, encode_action, frame_from_dict, parse_obs

log = logging.getLogger("b1k.server")


# ---------------------------------------------------------------------------
# Task identity + BDDL goals (legal at eval: BDDL is identical train/eval)
# ---------------------------------------------------------------------------
@dataclass
class TaskMeta:
    task_id: int
    task_name: str
    task_text: str
    goal_predicates: list[str]
    episode_mean_steps: float
    rooms: list[str]


class _TaskIndex:
    """name<->id + description + goal predicates + human episode length."""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}
        self.texts: dict[int, str] = {}
        self.goals: dict[int, list[str]] = {}
        self.ep_mean: dict[int, float] = {}
        self._load_descriptions()
        self._load_goals()

    def _load_descriptions(self) -> None:
        p = Path(self.cfg.task_descriptions_jsonl)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    tid = int(o.get("task_id", o.get("task_index", -1)))
                    name = o.get("task_name") or o.get("task") or str(tid)
                    text = o.get("description") or o.get("task_text") or ""
                    if tid < 0:
                        continue
                    self.name_to_id[name] = tid
                    self.id_to_name[tid] = name
                    self.texts[tid] = text

    def _load_goals(self) -> None:
        # data/bddl_goals.json : { "<task_id>": {"predicates": [...], "episode_mean_steps": N} }
        # data/human_stats.jsonl: per-task human stats (episode length in steps),
        #   from the challenge task.jsonl (score_utils.HUMAN_STATS_KEYS "length").
        from .config import REPO_ROOT

        goals_path = REPO_ROOT / "data" / "bddl_goals.json"
        if goals_path.exists():
            with open(goals_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                tid = int(k)
                self.goals[tid] = [str(x) for x in v.get("predicates", [])]
                if v.get("episode_mean_steps"):
                    self.ep_mean[tid] = float(v["episode_mean_steps"])

        human_path = REPO_ROOT / "data" / "human_stats.jsonl"
        if human_path.exists():
            with open(human_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    tid = int(o.get("task_index", -1))
                    if tid >= 0 and o.get("length"):
                        # "length" is the mean episode length in steps (fps-scaled).
                        self.ep_mean.setdefault(tid, float(o["length"]))

    def resolve(self, task: str | int | None) -> TaskMeta | None:
        if task is None:
            return None
        if isinstance(task, int):
            tid = int(task)
            name = self.id_to_name.get(tid, str(tid))
        else:
            t = str(task).strip()
            tid = self.name_to_id.get(t, None)
            name = t
            if tid is None:
                # maybe a numeric string id
                if t.isdigit():
                    tid = int(t)
                    name = self.id_to_name.get(tid, t)
                else:
                    tid = -1
        if tid < 0:
            return None
        return TaskMeta(
            task_id=tid,
            task_name=name,
            task_text=self.texts.get(tid, ""),
            goal_predicates=self.goals.get(tid, []),
            episode_mean_steps=self.ep_mean.get(tid, 0.0),
            rooms=[],
        )


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------
class B1KServer:
    """One stateful policy server (a WS connection == one episode/rollout)."""

    def __init__(self, cfg: ServerConfig, task: str | int | None = None):
        self.cfg = cfg
        self.index = _TaskIndex(cfg)
        self.rooms = RoomPrior.from_csv(cfg.task_rooms_csv)

        pcfg = cfg.policy
        self.vla = SkillConditionedVLA(
            backend=pcfg.backend,
            checkpoint_dir=pcfg.checkpoint_dir,
            norm_stats_path=pcfg.norm_stats_path,
            device=pcfg.device,
            action_horizon=cfg.controller.action_horizon,
            demos_root=pcfg.demos_root,
            seed=pcfg.seed,
        )
        self.controller = ActionController(cfg.controller, self.vla)
        self.tracker = PredicateTracker(k_frames=cfg.progress.hysteresis_frames)
        # build_model reads cfg.policy.detector_bundle_dir if set.
        from .perception.detectors.pred_models import build_model

        self.tracker.model = build_model(cfg)
        self.odo = Odometry(
            fps=cfg.odometry.fps,
            vo_subsample=cfg.odometry.vo_subsample,
            vo_blend_gain=cfg.odometry.vo_blend_gain,
            occupancy_cell_m=cfg.odometry.occupancy_cell_m,
            occupancy_radius_cells=cfg.odometry.occupancy_radius_cells,
            vo_model_path=cfg.odometry.vo_model_path,
        )
        self.task: TaskMeta | None = None
        self.planner: TaskPlanner | None = None
        self.progress: Progress | None = None
        if task is not None:
            self.set_task(task)
        self._steps = 0
        self._finished = False
        self._verified_layout = False

    # ---- task selection -----------------------------------------------------
    def set_task(self, task: str | int) -> TaskMeta:
        meta = self.index.resolve(task)
        if meta is None:
            raise ValueError(f"Unknown task {task!r}; index has {len(self.index.id_to_name)} tasks")
        meta.rooms = self.rooms.rooms_for(task_id=meta.task_id, task_name=meta.task_name)
        self.task = meta
        self.planner = TaskPlanner(
            task_id=meta.task_id,
            task_text=meta.task_text,
            bddl_goal=" & ".join(meta.goal_predicates),
            plan=load_mined_plan(meta.task_id, self.cfg),
        )
        self.progress = Progress(
            cfg=self.cfg.progress,
            goal_predicates=meta.goal_predicates,
            episode_mean_steps=meta.episode_mean_steps,
            conf_floor=self.cfg.planner.conf_floor,
            stuck_steps=self.cfg.planner.stuck_steps,
        )
        self.controller.reset(task_text=meta.task_text)
        self.tracker.reset()
        self.odo.reset()
        self._steps = 0
        self._finished = False
        log.info("Task set: %s (id=%d, goals=%d, ep_mean=%.0f steps)",
                 meta.task_name, meta.task_id, len(meta.goal_predicates),
                 meta.episode_mean_steps)
        return meta

    # ---- episode reset ------------------------------------------------------
    def reset_episode(self, task: str | int | None = None) -> None:
        """WS (re)connect: new rollout.  Reset all per-episode state.

        If ``task`` is given (the evaluator may reconnect for a different task),
        switch to it; otherwise re-initialise the *current* task as a fresh
        rollout (cleared planner queue, progress, controller chunk, trackers).
        """
        if task is not None:
            self.set_task(task)
        elif self.task is not None:
            self.set_task(self.task.task_name)
        else:
            self._steps = 0
            self._finished = False

    # ---- the per-step pipeline --------------------------------------------
    def _pred_state(self, frame: Frame) -> dict[str, tuple[bool, float]]:
        return self.tracker.update(frame, step=self._steps)

    def _target_seen(self) -> bool:
        """Proxy: is the current subgoal's target in view / effected?

        MVP: True if any predicate family the active subgoal can flip is
        confirmed by the tracker (i.e. we are making physical progress on it).
        """
        sub = self.planner.current() if self.planner else None
        if sub is None or not sub.pred_families:
            return False
        conf = set(self.tracker.confirmed())
        return any(f in conf for f in sub.pred_families)

    def handle_frame(self, frame: Frame) -> np.ndarray:
        """Run the per-step pipeline on an already-parsed :class:`Frame`.

        Returns exactly one 23-dim action (float32)."""
        # Week-1 verification: log the obs key layout once (design risk item).
        if self.cfg.verify_obs_layout and not self._verified_layout:
            self._verified_layout = True
            self._log_obs_layout(frame)

        if self.task is None:
            # No task known yet: serve a safe no-op (the harness/evaluator will
            # either pass --task or a reset message; we must still answer).
            self._steps += 1
            return make_no_op_action()

        pred_state = self._pred_state(frame)
        self.odo.update(frame.base_qvel, frame.depth_head)

        sub = self.planner.current()
        act = self.progress.tick(pred_state, sub, step=self._steps,
                                 target_seen=self._target_seen())
        self._steps += 1

        if act.kind == Act.ADVANCE:
            self.planner.advance()
            sub = self.planner.current()
            self.progress.begin_subgoal(sub, self._steps)
            return self.controller.step(frame, sub, task_id=self.task.task_id,
                                        step=self._steps, force_requery=True)

        if act.kind in (Act.RETRY, Act.REPLAN):
            if act.kind == Act.REPLAN:
                self.planner.replan()
                sub = self.planner.current()
                self.progress.begin_subgoal(sub, self._steps)
            # RETRY: re-execute the same subgoal (re-query VLA for a fresh chunk).
            return self.controller.step(frame, sub, task_id=self.task.task_id,
                                        step=self._steps, force_requery=True)

        if act.kind == Act.SEARCH:
            # Frontier-search bias: re-query the VLA on the nav subgoal; the
            # controller serves the chunk.  (A real nav bias would modulate the
            # base-velocity rows here — see odometry.frontier().)
            return self.controller.step(frame, sub, task_id=self.task.task_id,
                                        step=self._steps, force_requery=True)

        if act.kind == Act.FINISH or self.progress.finished:
            self._finished = True
            log.info("Server FINISH @step %d: %s", self._steps, act.reason)
            return self.controller.park()

        # CONTINUE
        return self.controller.step(frame, sub, task_id=self.task.task_id,
                                    step=self._steps)

    def handle_obs(self, obs: dict[str, Any]) -> np.ndarray:
        """One eval step from a *decoded* flattened obs dict.  Thin wrapper:
        assemble the Frame then run :meth:`handle_frame`."""
        return self.handle_frame(frame_from_dict(obs))

    def _log_obs_layout(self, frame: Frame) -> None:
        try:
            lines = []
            for k, v in frame.raw.items():
                a = np.asarray(v)
                lines.append(f"    {k!r}: shape={a.shape} dtype={a.dtype}")
            log.info("Obs layout (%d keys, first rollout of this connection):\n%s",
                     len(frame.raw), "\n".join(lines[:40]))
        except Exception:  # noqa: BLE001
            pass

    # ---- raw-bytes convenience (used by the WS layer & fixtures) ----------
    def handle_payload(self, payload: bytes) -> bytes:
        """Decode a msgpack obs payload and return a msgpack action payload."""
        frame = parse_obs(payload)
        action = self.handle_frame(frame)
        return encode_action(action)

    # ---- HTTP health --------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.task is not None or self.cfg.policy.backend in ("echo", "noop")


# ---------------------------------------------------------------------------
# WebSocket transport
# ---------------------------------------------------------------------------
async def _ws_handler(server: B1KServer, conn) -> None:
    """websockets.asyncio server handler matching ``WebsocketPolicyServer``.

    Contract (omnigibson.eval.utils.network_utils v3.9.2):
      1. On connect, server immediately sends a msgpack metadata dict.
      2. Evaluator may send msgpack ``{"reset": True}`` (no reply expected).
      3. Each obs is a msgpack flattened dict; reply is msgpack ``{"action": ...}``.

    Local harnesses may also send JSON ``{"type": "reset"|"set_task", ...}``
    control frames (with ack), which the official evaluator never uses.
    """
    from .protocol import pack, unpack, encode_action

    log.info("WS connected from %s; resetting episode", conn.remote_address)
    server.reset_episode()
    # Official WebsocketClientPolicy does unpackb(conn.recv()) right after
    # connect — must send metadata before waiting for any obs.
    meta = {
        "robot": server.cfg.robot,
        "backend": server.cfg.policy.backend,
        "task": None if server.task is None else server.task.task_name,
    }
    await conn.send(pack(meta))
    try:
        async for raw in conn:
            ctrl = _maybe_control(raw)
            if ctrl is not None:
                await _handle_control(server, conn, ctrl)
                continue
            # Official client reset: msgpack {"reset": True} (no reply).
            try:
                obj = unpack(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else None
            except Exception:
                obj = None
            if isinstance(obj, dict) and obj.get("reset") is True and "action" not in obj:
                server.reset_episode(obj.get("task"))
                log.info("WS reset (msgpack) from evaluator")
                continue
            if isinstance(obj, dict) and "type" not in obj and any(
                isinstance(k, str) and ("::" in k or k.endswith(("rgb", "depth", "qpos", "qvel", "state")))
                for k in obj
            ):
                action = await asyncio.to_thread(server.handle_obs, obj)
                await conn.send(encode_action(action))
                continue
            action_bytes = await asyncio.to_thread(server.handle_payload, raw)
            await conn.send(action_bytes)
    except Exception as e:  # noqa: BLE001 - a bad client must not kill the server
        log.warning("WS handler error: %s", e, exc_info=True)
    finally:
        log.info("WS disconnected")


def _maybe_control(raw) -> dict | None:
    """Return the decoded control dict if ``raw`` is a JSON control frame,
    else None (meaning: treat ``raw`` as a binary obs payload)."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if not text.startswith("{"):
            return None
        try:
            msg = json.loads(text)
        except Exception:
            return None
        return msg if isinstance(msg, dict) and "type" in msg else None
    if isinstance(raw, str):
        try:
            msg = json.loads(raw)
        except Exception:
            return None
        return msg if isinstance(msg, dict) and "type" in msg else None
    return None


async def _handle_control(server: B1KServer, conn, msg: dict) -> None:
    mtype = msg.get("type")
    if mtype == "reset":
        server.reset_episode(msg.get("task"))
        await conn.send(json.dumps({"type": "reset_ok"}).encode())
    elif mtype == "set_task":
        meta = server.set_task(msg.get("task"))
        await conn.send(json.dumps({"type": "task_ok", "task": meta.task_name,
                                    "id": meta.task_id}).encode())
    else:
        await conn.send(json.dumps({"type": "ignored", "got": mtype}).encode())


def _make_process_request():
    """Answer ``GET /healthz`` with HTTP 200; let the WS upgrade through.

    In websockets 15 (asyncio) the ``process_request(conn, request)`` hook runs
    for every inbound HTTP request.  Returning a :class:`Response` answers it
    as plain HTTP (the evaluator's pre-WS health probe); returning ``None``
    hands the request to the WebSocket upgrade path.  Only ``/healthz`` is
    answered as HTTP — every other path (including ``/``) must reach the WS
    upgrade or the evaluator's ``WebsocketPolicy`` client would get a 200
    instead of a socket.
    """
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    async def process_request(conn, request):
        if request.path in ("/healthz", "/healthz/"):
            body = b"ok"
            headers = Headers()
            headers["Content-Type"] = "text/plain"
            headers["Content-Length"] = str(len(body))
            return Response(200, "OK", headers, body)
        return None

    return process_request


async def _run_async(cfg: ServerConfig, task: str | int | None) -> None:
    import websockets.asyncio.server as wsserver

    server = B1KServer(cfg, task=task)
    host, port = cfg.host, cfg.port
    stop = asyncio.Event()
    process_request = _make_process_request()

    async with wsserver.serve(
        lambda conn: _ws_handler(server, conn),
        host,
        port,
        process_request=process_request,
        max_size=cfg.max_size or None,
        # Match official WebsocketClientPolicy (ping_interval=60, ping_timeout=300)
        # so long Isaac Sim steps don't drop the socket.
        ping_interval=60,
        ping_timeout=300,
        compression=None,
    ) as wss:
        log.info("b1k server up on ws://%s:%d (task=%s, backend=%s)",
                 host, port, task if task is not None else "<none until reset>",
                 cfg.policy.backend)
        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass
        finally:
            wss.close()
            await wss.wait_closed()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m b1k.server")
    ap.add_argument("--config", default="configs/server.yaml", type=str)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", type=str, default=None)
    ap.add_argument("--task", type=str, default=None,
                    help="task name or id to serve (matches serve_b1k.py --task)")
    ap.add_argument("--log-level", default="INFO", type=str)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        from .config import REPO_ROOT

        cfg_path = REPO_ROOT / cfg_path
    cfg = load_config(cfg_path)
    if args.port is not None:
        cfg.port = int(args.port)
    if args.host is not None:
        cfg.host = args.host

    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_async(cfg, args.task))
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down")
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
