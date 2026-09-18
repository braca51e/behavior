"""Runtime configuration (dataclass) for the b1k server.

Loaded from ``configs/server.yaml`` (see :func:`load_config`).  All paths are
resolved relative to the repository root unless absolute.  Values here drive
the serving pipeline end-to-end; nothing privileged (no omnigibson, no training
import) is referenced at serve time.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PlannerConfig:
    # Skill sequence planning (rule-based; no LLM at eval).
    plans_dir: Path = REPO_ROOT / "docs" / "plans"
    # p90 demo-duration multiplier applied to each subgoal budget (design: 1.5).
    budget_mult: float = 1.5
    # Stuck detection: RETRY after this many steps with zero predicate delta.
    stuck_steps: int = 900                 # 30 s @ 30 Hz
    # RETRY cap: a subgoal may not exceed min(budget, 3 x demo-mean length).
    retry_cap_mult: float = 3.0
    # Confidence floor for detector predicates.
    conf_floor: float = 0.6


@dataclass
class ProgressConfig:
    # K consecutive consistent frames required before a predicate flip sticks.
    hysteresis_frames: int = 3
    # Global episode budget: 1.5x human mean episode length (design default).
    global_budget_mult: float = 1.5
    # FINISH: command park pose + no-op actions once all predicates satisfied.
    early_stop: bool = True
    # SEARCH: emit frontier-search bias after target unseen for this long.
    search_after_s: float = 5.0


@dataclass
class OdometryConfig:
    # Velocity integration of obs.state[0:3] (30 Hz).
    fps: float = 30.0
    # Head-depth visual odometry: sub-sample factor (1 = every frame; design: 3).
    vo_subsample: int = 3
    # Exponential-smoothing blend gain for VO drift correction (0..1).
    vo_blend_gain: float = 0.1
    # Occupancy grid cell size (meters) for room-scale search.
    occupancy_cell_m: float = 0.5
    # Max grid half-extent (cells) around the robot.
    occupancy_radius_cells: int = 40
    # VO model path (pretrained DPV2); None -> depth-free integration only.
    vo_model_path: Path | None = None


@dataclass
class ControllerConfig:
    # Receding horizon: re-query the VLA every K steps.
    chunk_requery: int = 16
    # VLA output chunk length (23-dim actions per query).
    action_horizon: int = 32
    # Chunk-boundary blend (temporal-ensembling port): a=(1-a)old+a*new, ramp
    # 0->1 over this many steps from the chunk start.
    blend_steps: int = 4
    # Gripper deadband: ignore grip commands changing by less than this.
    grip_deadband: float = 0.02
    # Base velocity clamp (output limits from r1pro.yaml).
    base_clamp: tuple[float, float, float] = (0.75, 0.75, 1.0)
    # Safety: zero base + hold arms after this many seconds without VLA update.
    stale_vla_s: float = 5.0


@dataclass
class PolicyConfig:
    # VLA backend: "vla" loads the fine-tuned pi0.5 checkpoint (needs torch +
    # checkpoint); "echo" replays the nearest recorded demo action (MVP, no
    # model weights); "noop" returns a park action.  echo/noop run CPU-only and
    # are the defaults so the serving contract is verifiable without training.
    backend: str = "echo"
    # Checkpoint dir for the "vla" backend (OpenPI 'behavior' fork, pi05_b1k).
    checkpoint_dir: Path | None = None
    # Norm stats json (recomputed post Aug-2026 velocity fix).
    norm_stats_path: Path | None = None
    # Device for torch inference.  Default "cpu" (safe at import time); use
    # "cuda" on a GPU box for backend=vla.  Resolved by ``resolve_device`` if
    # set to "auto".
    device: str = "cpu"
    # Per-task LoRA / task-embedding adapter dir (phase 3; may be empty).
    adapters_dir: Path | None = None
    # Recorded demo-action fixtures root (for the "echo" backend).
    demos_root: Path = REPO_ROOT / "data" / "demos"
    # Detector weight bundle dir (trained predicate models); None -> heuristics.
    detector_bundle_dir: Path | None = None
    seed: int = 0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: str) -> str:
    """Map 'auto' -> 'cuda' if available else 'cpu'; pass through 'cuda'/'cpu'."""
    if device and str(device).lower() == "auto":
        return "cuda" if _cuda_available() else "cpu"
    return device or "cpu"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    robot: str = "R1Pro"
    # Max obs payload (bytes): 720^2*3 + 2*480^2*3 + depths + proprio ~= 5 MB.
    max_size: int = 64 * 1024 * 1024
    # Task metadata (room prior + NL descriptions).
    task_rooms_csv: Path = REPO_ROOT / "data" / "B100_task_misc.csv"
    task_descriptions_jsonl: Path = REPO_ROOT / "docs" / "task-descriptions-100.jsonl"
    # Log dir (per-episode obs-verify dumps land here).
    log_dir: Path = REPO_ROOT / "outputs" / "logs"
    # Reset the server state on every new WebSocket connection (multi-instance
    # serving: each evaluator rollout reconnects).
    reset_on_connect: bool = True
    # Echo the parsed obs layout once per connection at INFO (week-1 verify).
    verify_obs_layout: bool = True

    planner: PlannerConfig = field(default_factory=PlannerConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def _as_path(v):
    """Coerce a config value to an absolute Path if it is a non-empty string."""
    if isinstance(v, str) and v.strip():
        p = Path(v).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)
    return None


def load_config(path: str | Path) -> ServerConfig:
    """Load a ServerConfig from a YAML file.  Missing sections use defaults.

    Relative path values are resolved against the repository root, so the
    config is portable as long as it ships inside the repo (as ``configs/
    server.yaml`` does).
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    def section(cls, d):
        """Instantiate a dataclass section, ignoring unknown keys."""
        import dataclasses

        d = d or {}
        keys = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in d.items() if k in keys}
        for k in list(clean):
            if k in (
                "vo_model_path",
                "checkpoint_dir",
                "norm_stats_path",
                "adapters_dir",
                "demos_root",
                "plans_dir",
                "detector_bundle_dir",
            ):
                p = _as_path(clean[k])
                if p is not None:
                    clean[k] = p
        if "device" in clean and isinstance(clean["device"], str):
            clean["device"] = resolve_device(clean["device"])
        return cls(**clean)

    cfg = ServerConfig(
        host=raw.get("host", "127.0.0.1"),
        port=int(raw.get("port", 8000)),
        robot=raw.get("robot", "R1Pro"),
        max_size=int(raw.get("max_size", ServerConfig.max_size)),
        task_rooms_csv=_as_path(raw.get("task_rooms_csv")) or ServerConfig.task_rooms_csv,
        task_descriptions_jsonl=_as_path(raw.get("task_descriptions_jsonl"))
        or ServerConfig.task_descriptions_jsonl,
        log_dir=_as_path(raw.get("log_dir")) or ServerConfig.log_dir,
        reset_on_connect=bool(raw.get("reset_on_connect", True)),
        verify_obs_layout=bool(raw.get("verify_obs_layout", True)),
        planner=section(PlannerConfig, raw.get("planner")),
        progress=section(ProgressConfig, raw.get("progress")),
        odometry=section(OdometryConfig, raw.get("odometry")),
        controller=section(ControllerConfig, raw.get("controller")),
        policy=section(PolicyConfig, raw.get("policy")),
    )
    return cfg


def dump_config(cfg: ServerConfig) -> str:
    """Serialize for debugging / reproducibility (no secrets in server config)."""
    return yaml.safe_dump(asdict(cfg), sort_keys=False)
