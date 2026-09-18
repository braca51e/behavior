"""Generate the b1k test fixtures (deterministic, CPU-only).

Produces everything the unit tests and the self-eval need *without* the 3.27 TB
demo dataset or a GPU:

* ``tests/fixtures/obs_payload.bin``  — one msgpack-encoded *flattened* obs dict
  (head RGB 720x720x3 uint8, both wrist RGB 480x480x3, both+head depth float32,
  61-dim proprio) shaped exactly like the evaluator's ``flatten_obs_dict`` +
  ``WebsocketPolicyServer`` wire format (contract item (a)).
* ``tests/fixtures/action_23d.bin``   — the recorded 23-float action paired
  with the obs above (one baseline-rollout step).
* ``tests/fixtures/metrics_*.json``   — 5 canned per-rollout metrics JSONs
  covering q_score.final = 0 / 0.5 / 1.0 with and without early stop, plus a
  time_score case (contract item (b)).
* ``tests/fixtures/annotations.jsonl``— synthetic per-episode language
  annotations exercising the 31-skill miner on 3 tasks (contract item (c)).
* ``data/demos/<task_id>/actions.npy``— short recorded demo action sequences
  (N x 23) for the "echo" VLA backend (task 0 and 1).
* ``data/bddl_goals.json``            — goal BDDL predicates + human episode
  mean length for a few tasks (legal at eval: BDDL identical train/eval).

Run::

    python3 tests/fixtures/make_fixtures.py

Regenerating is idempotent and seeded, so the fixtures are reproducible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from b1k.protocol import encode_action, pack  # noqa: E402

SEED = 12345
FIX = REPO / "tests" / "fixtures"


def _flattened_obs(rng: np.random.Generator) -> dict:
    """Build a flattened obs dict in the evaluator's wire format."""
    head = rng.integers(0, 256, size=(720, 720, 3), dtype=np.uint8)
    wl = rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    wr = rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    d_head = rng.uniform(0.5, 4.0, size=(720, 720, 1)).astype(np.float32)
    d_l = rng.uniform(0.1, 1.5, size=(480, 480, 1)).astype(np.float32)
    d_r = rng.uniform(0.1, 1.5, size=(480, 480, 1)).astype(np.float32)
    # 61-dim proprio: base_qvel small, joints plausible, grippers open (0.05).
    proprio = np.zeros(61, dtype=np.float32)
    proprio[0:3] = [0.05, -0.01, 0.02]          # base_qvel (robot-local)
    proprio[3:10] = rng.uniform(-1.0, 1.0, 7)   # arm_left_qpos
    proprio[10:17] = rng.uniform(-0.5, 0.5, 7)  # arm_left_qvel
    proprio[17:20] = [0.3, 0.4, 1.1]            # eef_left_pos
    proprio[20:24] = [1.0, 0, 0, 0]             # eef_left_quat
    proprio[24:26] = [0.05, 0.05]               # gripper_left_qpos (open)
    proprio[26:28] = 0.0
    proprio[28:35] = rng.uniform(-1.0, 1.0, 7)
    proprio[35:42] = rng.uniform(-0.5, 0.5, 7)
    proprio[42:45] = [-0.3, 0.4, 1.1]
    proprio[45:49] = [1.0, 0, 0, 0]
    proprio[49:51] = [0.05, 0.05]
    proprio[51:53] = 0.0
    proprio[53:57] = [1.025, -1.45, -0.47, 0.0]  # trunk_qpos (JoyLo default)
    proprio[57:61] = 0.0
    return {
        "robot_r1::zed_link::Camera::0::rgb": head.tolist(),
        "robot_r1::left_realsense_link::Camera::0::rgb": wl.tolist(),
        "robot_r1::right_realsense_link::Camera::0::rgb": wr.tolist(),
        "robot_r1::zed_link::Camera::0::depth": d_head.tolist(),
        "robot_r1::left_realsense_link::Camera::0::depth": d_l.tolist(),
        "robot_r1::right_realsense_link::Camera::0::depth": d_r.tolist(),
        "robot_r1::proprio": proprio.tolist(),
    }


def _recorded_action(rng: np.random.Generator) -> np.ndarray:
    a = np.zeros(23, dtype=np.float32)
    a[0:3] = [0.1, 0.0, 0.05]
    a[3:7] = [1.02, -1.44, -0.46, 0.0]
    a[7:14] = rng.uniform(-0.6, 0.6, 7)
    a[14] = 0.05
    a[15:22] = rng.uniform(-0.6, 0.6, 7)
    a[22] = 0.05
    return a


def _metric_json(task: str, instance: int, rollout: int, q: float, nt: float,
                 success: bool, steps: int, base_d: float, left_d: float,
                 right_d: float) -> dict:
    return {
        "task": task,
        "instance_id": int(instance),
        "rollout_id": int(rollout),
        "steps": steps,
        "success": success,
        "agent_distance": {"base": round(base_d, 3), "left": round(left_d, 3),
                           "right": round(right_d, 3)},
        "normalized_agent_distance": {"base": round(base_d, 3),
                                      "left": round(left_d, 3),
                                      "right": round(right_d, 3)},
        "q_score": {"final": q},
        "time": {"simulator_steps": steps,
                 "simulator_time": round(steps / 30.0, 3),
                 "normalized_time": round(nt, 4)},
    }


def make_annotations() -> list[dict]:
    """Synthetic skill annotations for 3 tasks (miner fixture)."""
    out: list[dict] = []
    # Task 0: turning_on_radio — walk to the radio, turn it on, verify lit.
    for ep in range(4):
        t = 0.0
        for text, dur in [
            ("Move to the radio", 10 + ep),
            ("Turn on the radio switch", 3),
            ("Turn on the radio", 2),
        ]:
            out.append({"task_id": 0, "episode_index": ep, "t0": t,
                        "t1": t + dur, "text": text})
            t += dur
    # Task 1: picking_up_trash — walk to bin, pick up trash, place in bin.
    for ep in range(4):
        t = 0.0
        for text, dur in [
            ("Move to the trash can", 12),
            ("Pick up the trash bag", 4),
            ("Place the trash bag in the trash can", 5),
        ]:
            out.append({"task_id": 1, "episode_index": ep, "t0": t,
                        "t1": t + dur, "text": text})
            t += dur
    # Task 3: cleaning_up_plates_and_food — pick up plates, place in sink.
    for ep in range(4):
        t = 0.0
        for text, dur in [
            ("Move to the kitchen counter", 9),
            ("Pick up the plate", 4),
            ("Place the plate on the sink", 5),
            ("Wipe the counter", 6),
        ]:
            out.append({"task_id": 3, "episode_index": ep, "t0": t,
                        "t1": t + dur, "text": text})
            t += dur
    return out


def make_demo_actions(task_id: int, n: int = 200, seed: int = 0) -> np.ndarray:
    """A short recorded demo action sequence (N x 23) for the echo backend."""
    rng = np.random.default_rng(seed + task_id)
    a = np.zeros((n, 23), dtype=np.float32)
    # A plausible "walk forward then reach" profile.
    t = np.arange(n, dtype=np.float32)
    a[:, 0] = 0.2 * np.exp(-t / 60.0)                       # base vx decay
    a[:, 2] = 0.05 * np.sin(t / 20.0) * np.exp(-t / 100.0)  # base yaw wiggle
    a[:, 3:7] = [1.02, -1.45, -0.47, 0.0]
    a[:, 7:14] = 0.3 * rng.normal(0, 1, (n, 7)).cumsum(axis=0) / 20
    a[:, 14] = np.where(t < n // 2, 0.05, 0.0)             # grip closes mid
    a[:, 15:22] = -0.3 * rng.normal(0, 1, (n, 7)).cumsum(axis=0) / 20
    a[:, 22] = np.where(t < n // 2, 0.05, 0.0)
    return a


def main() -> int:
    rng = np.random.default_rng(SEED)
    FIX.mkdir(parents=True, exist_ok=True)

    # (a) flattened obs + recorded action
    obs = _flattened_obs(rng)
    payload = pack(obs)
    (FIX / "obs_payload.bin").write_bytes(payload)
    action = _recorded_action(rng)
    (FIX / "action_23d.bin").write_bytes(encode_action(action))

    # (b) 5 canned metrics JSONs (q 0 / 0.5 / 1.0, +/- early stop, time_score)
    T = "turning_on_radio"
    metrics = {
        f"{T}_301_0.json": _metric_json(T, 301, 0, 0.0, 1.6, False, 11000, 4.2, 6.1, 5.8),
        f"{T}_302_0.json": _metric_json(T, 302, 0, 0.5, 1.2, False, 8300, 3.1, 4.0, 3.9),
        f"{T}_303_0.json": _metric_json(T, 303, 0, 1.0, 0.8, True, 5500, 2.0, 3.2, 3.0),
        f"{T}_304_0.json": _metric_json(T, 304, 0, 1.0, 0.6, True, 4100, 1.5, 2.1, 2.0),
        f"{T}_305_0.json": _metric_json(T, 305, 0, 0.5, 1.5, False, 10200, 3.8, 5.0, 4.9),
    }
    for name, obj in metrics.items():
        with open(FIX / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    # (c) synthetic skill annotations
    anns = make_annotations()
    with open(FIX / "annotations.jsonl", "w", encoding="utf-8") as f:
        for o in anns:
            f.write(json.dumps(o) + "\n")

    # echo-backend demo action sequences (task 0, 1)
    for tid in (0, 1):
        d = REPO / "data" / "demos" / str(tid)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "actions.npy", make_demo_actions(tid))

    # BDDL goals + human episode mean steps for the fixture tasks (legal at eval)
    goals = {
        "0": {"predicates": ["Powered(Radio)", "On(Radio)"], "episode_mean_steps": 9000},
        "1": {"predicates": ["In(TrashBag, TrashCan)", "On(TrashBag, Floor)"],
              "episode_mean_steps": 12000},
        "3": {"predicates": ["In(Plate, Sink)", "Cleaned(Counter)"],
              "episode_mean_steps": 14000},
    }
    data_dir = REPO / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "bddl_goals.json", "w", encoding="utf-8") as f:
        json.dump(goals, f, indent=2)

    # Human stats (episode length in steps) for the global budget / time_score.
    with open(data_dir / "human_stats.jsonl", "w", encoding="utf-8") as f:
        for tid, g in goals.items():
            f.write(json.dumps({"task_index": int(tid),
                                "length": g["episode_mean_steps"]}) + "\n")

    # Mined per-task skill plans (docs/plans/<task_id>.json) from the fixture
    # annotations — demonstrates the miner -> planner flow and gives the
    # serving pipeline real (if synthetic) plans instead of the move_to fallback.
    from b1k.planner.miner import mine_plans  # noqa: E402

    plans_dir = REPO / "docs" / "plans"
    task_names = {}
    desc = REPO / "docs" / "task-descriptions-100.jsonl"
    if desc.exists():
        for line in desc.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                task_names[int(o.get("task_id", o.get("task_index", -1)))] = \
                    o.get("task_name") or o.get("task") or ""
    mine_plans(anns, tasks={0: task_names.get(0, "turning_on_radio"),
                            1: task_names.get(1, "picking_up_trash"),
                            3: task_names.get(3, "cleaning_up_plates_and_food")},
               out_dir=plans_dir)
    print("  mined plans ->", plans_dir, "for tasks 0,1,3")

    print("fixtures written to", FIX)
    print("  obs_payload.bin:", (FIX / "obs_payload.bin").stat().st_size, "bytes")
    print("  action_23d.bin:", (FIX / "action_23d.bin").stat().st_size, "bytes")
    print("  metrics:", list(metrics))
    print("  annotations:", len(anns), "segments")
    print("  demo actions: tasks 0,1 (200x23)")
    print("  bddl goals: tasks", list(goals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
