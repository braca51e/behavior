"""Submission-validation core for the 2026 BEHAVIOR challenge (pure Python, CPU).

This module reproduces the organizers' scoring *exactly*, from the shipped
``raw/score_utils.py`` / ``raw/eval_utils.py`` (v3.9.2):

* ``time_score = EVAL_TIMEOUT_MULTIPLIER/(EVAL_TIMEOUT_MULTIPLIER-1)
   - 1/((EVAL_TIMEOUT_MULTIPLIER-1)*normalized_time)``  (= 3 - 2/nt).
* Per-task scores divide by ``NUM_{PUBLIC,HIDDEN}_TEST_INSTANCES`` (20) and
  ``task_sr`` counts rollouts with ``q == 1`` — so **missing instances and
  missing tasks count as 0** in the official math.
* Overall Q averages the per-task Q over **all 100 tasks**.
* The submission folder name must be
  ``<track>.<testset>.<team>.<affiliation>.<date>`` and every metrics file must
  be named ``<task_name>_<instance_id>_<rollout_id>.json`` with an *exact*
  task name, an instance id in ``TEST_INSTANCE_IDS`` (301..340; public =
  301..320, hidden = 321..340) and ``rollout_id == 0``.

The validators in :func:`validate_submission` / :func:`get_scores` are the
single source of truth for the local harness, the pytest gate, and the
``scripts/validate_submission.sh`` one-liner.  They never touch the simulator,
so they run on any CPU box — which is the point of a *local* validation
harness: we verify the solution is functional and submission-ready without
relying on external challenge testing.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---- official constants (raw/eval_utils.py, v3.9.2) -------------------------
EVAL_TIMEOUT_MULTIPLIER = 1.5
NUM_TEST_INSTANCES = 40
NUM_PUBLIC_TEST_INSTANCES = 20
NUM_HIDDEN_TEST_INSTANCES = NUM_TEST_INSTANCES - NUM_PUBLIC_TEST_INSTANCES
TEST_INSTANCE_IDS = list(range(301, 341))          # 301..340 inclusive
PUBLIC_INSTANCE_IDS = TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES]
HIDDEN_INSTANCE_IDS = TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:]
TRACKS = ("standard", "privileged")
TESTSETS = ("public", "hidden")
REPO = Path(__file__).resolve().parents[1]

_FILE_RE = re.compile(r"^(.+)_(\d+)_(\d+)\.json$")


def time_score_from_normalized(normalized_time: float) -> float:
    """Official per-rollout time score: ``3 - 2/normalized_time``."""
    nt = float(normalized_time)
    return (
        EVAL_TIMEOUT_MULTIPLIER / (EVAL_TIMEOUT_MULTIPLIER - 1.0)
        - 1.0 / ((EVAL_TIMEOUT_MULTIPLIER - 1.0) * nt)
    )


_TASK_NAMES_CACHE: list[str] | None = None


def load_task_names(path: Path | str | None = None) -> list[str]:
    """The 100 official task names, index-aligned to ``docs/task-descriptions-100.jsonl``.

    Cached: it is consulted for every metrics file during validation, so we
    parse the jsonl at most once per process (the default path only).
    """
    global _TASK_NAMES_CACHE
    if path is None and _TASK_NAMES_CACHE is not None:
        return _TASK_NAMES_CACHE
    p = Path(path) if path else REPO / "docs" / "task-descriptions-100.jsonl"
    names: list[str] = []
    for line in p.read_text().splitlines():
        if line.strip():
            names.append(str(json.loads(line)["task_name"]))
    if path is None:
        _TASK_NAMES_CACHE = names
    return names


def parse_submission_name(folder: str) -> dict | None:
    """Parse ``<track>.<testset>.<team>.<affiliation>.<date>``; None if malformed."""
    parts = str(folder).split(".")
    if len(parts) != 5:
        return None
    track, testset, team, affiliation, date = parts
    return {
        "track": track,
        "testset": testset,
        "team": team,
        "affiliation": affiliation,
        "date": date,
        "ok": (
            track in TRACKS
            and testset in TESTSETS
            and bool(team)
            and bool(affiliation)
            and bool(date)
        ),
    }


def file_pattern_valid(filename: str, testset: str | None = None) -> bool:
    """True if ``filename`` is a legal ``<task>_<instance>_<rollout>.json``.

    The task name is checked against the *full* 100-task list (the official
    scorer's ``possible_filenames`` is built that way), the instance id must be
    in ``TEST_INSTANCE_IDS``, and — when a testset is given — must fall in that
    testset's 20-instance window.  ``rollout_id`` must be 0 (1 rollout/instance
    for scoring).
    """
    m = _FILE_RE.match(filename)
    if not m:
        return False
    task, inst, roll = m.group(1), int(m.group(2)), int(m.group(3))
    if task not in load_task_names():
        return False
    if inst not in TEST_INSTANCE_IDS:
        return False
    if roll != 0:
        return False
    if testset is not None:
        window = PUBLIC_INSTANCE_IDS if testset == "public" else HIDDEN_INSTANCE_IDS
        if inst not in window:
            return False
    return True


@dataclass
class Rollout:
    task: str
    instance_id: int
    rollout_id: int
    q: float
    time_score: float
    base_dist: float
    left_dist: float
    right_dist: float
    steps: int = 0
    success: bool = False


@dataclass
class Validation:
    """Outcome of validating one submission folder."""

    folder: str = ""
    track: str = ""
    testset: str = ""
    team: str = ""
    affiliation: str = ""
    date: str = ""
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rollouts: int = 0
    valid_rollouts: list[Rollout] = field(default_factory=list)
    per_task_q: dict[str, float] = field(default_factory=dict)
    per_task_sr: dict[str, float] = field(default_factory=dict)
    overall_q: float = 0.0
    overall_task_sr: float = 0.0
    overall_time_score: float = 0.0


def get_scores(
    rollouts: list[Rollout],
    n_tasks: int,
    n_instances_per_task: int,
    task_names: list[str],
) -> dict:
    """Reproduce the official ``compute_final_q_score`` aggregation.

    Faithful to ``raw/score_utils.py`` (v3.9.2): *every* per-task average —
    Q, task_sr, time_score and the three distance metrics — divides by the
    full instance count (``n_instances_per_task`` = 20), so missing instances
    and missing tasks count as 0.  ``task_sr`` counts rollouts with
    ``q == 1``.  The overall averages span all ``n_tasks`` (100) tasks.
    """
    n_inst = max(1, int(n_instances_per_task))
    n_task = max(1, int(n_tasks))
    per_task_q: dict[str, float] = {}
    per_task_sr: dict[str, float] = {}
    per_task_time: dict[str, float] = {}
    per_task_base: dict[str, float] = {}
    per_task_left: dict[str, float] = {}
    per_task_right: dict[str, float] = {}
    for t in task_names:
        rs = [r for r in rollouts if r.task == t]
        per_task_q[t] = sum(r.q for r in rs) / n_inst
        per_task_sr[t] = sum(1 for r in rs if r.q == 1) / n_inst
        per_task_time[t] = sum(r.time_score for r in rs) / n_inst
        per_task_base[t] = sum(r.base_dist for r in rs) / n_inst
        per_task_left[t] = sum(r.left_dist for r in rs) / n_inst
        per_task_right[t] = sum(r.right_dist for r in rs) / n_inst
    return {
        "n_rollouts": len(rollouts),
        "n_tasks": n_task,
        "n_instances_per_task": n_inst,
        "overall_q": sum(per_task_q.values()) / n_task,
        "overall_task_sr": sum(per_task_sr.values()) / n_task,
        "overall_time_score": sum(per_task_time.values()) / n_task,
        "overall_base_distance": sum(per_task_base.values()) / n_task,
        "overall_left_distance": sum(per_task_left.values()) / n_task,
        "overall_right_distance": sum(per_task_right.values()) / n_task,
        "per_task_q": per_task_q,
        "per_task_sr": per_task_sr,
        "per_task_time_score": per_task_time,
    }


def _schema_check(obj, path: str) -> tuple[list[str], dict | None]:
    """Validate one metrics JSON against the spec §7.1 schema.

    Returns ``(errors, rollout)``; ``rollout`` is None when errors are
    structural (missing keys) so it cannot be scored.
    """
    errs: list[str] = []
    if not isinstance(obj, dict):
        return [f"{path}: not a JSON object"], None
    for key in ("task", "instance_id", "rollout_id", "q_score", "time",
                "normalized_agent_distance"):
        if key not in obj:
            errs.append(f"{path}: missing key {key!r}")
    if errs:
        return errs, None

    q = obj.get("q_score", {})
    qv = q.get("final") if isinstance(q, dict) else None
    if not isinstance(qv, (int, float)) or isinstance(qv, bool):
        errs.append(f"{path}: q_score.final not numeric")
    else:
        qf = float(qv)
        if math.isnan(qf) or not (0.0 <= qf <= 1.0):
            errs.append(f"{path}: q_score.final out of [0,1] ({qv})")

    t = obj.get("time", {})
    nt = t.get("normalized_time") if isinstance(t, dict) else None
    if not isinstance(nt, (int, float)) or isinstance(nt, bool) or math.isnan(nt):
        errs.append(f"{path}: time.normalized_time missing/non-finite")

    n = obj.get("normalized_agent_distance", {})
    dist: dict[str, float] = {}
    for side in ("base", "left", "right"):
        v = n.get(side) if isinstance(n, dict) else None
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or v < 0:
            errs.append(f"{path}: normalized_agent_distance.{side} invalid ({v!r})")
        else:
            dist[side] = float(v)

    task = str(obj.get("task"))
    inst = int(obj.get("instance_id"))
    roll = int(obj.get("rollout_id"))
    rollout = None
    if not errs and nt is not None:
        rollout = Rollout(
            task=task,
            instance_id=inst,
            rollout_id=roll,
            q=float(qf),
            time_score=time_score_from_normalized(nt),
            base_dist=dist.get("base", 0.0),
            left_dist=dist.get("left", 0.0),
            right_dist=dist.get("right", 0.0),
            steps=int(obj.get("steps", 0) or 0),
            success=bool(obj.get("success", False)),
        )
    return errs, rollout


def validate_submission(submission_dir: Path | str) -> Validation:
    """Validate a ``<track>.<testset>.<team>.<affiliation>.<date>/`` folder.

    Checks (format): folder name, file naming, schema, instance bounds,
    duplicates, one-rollout convention.  Checks (scoring): computes the exact
    official Q / task_sr / time_score over the *valid* rollouts.  ``passed``
    is True only when there are no errors (warnings, e.g. incomplete task
    coverage, do not fail the gate).
    """
    sub = Path(submission_dir)
    folder = sub.name
    v = Validation(folder=folder)
    parsed = parse_submission_name(folder)
    if parsed is None:
        v.errors.append(f"folder {folder!r} not <track>.<testset>.<team>.<affiliation>.<date>")
        return v
    v.track, v.testset, v.team, v.affiliation, v.date = (
        parsed["track"], parsed["testset"], parsed["team"], parsed["affiliation"], parsed["date"])
    if not parsed["ok"]:
        v.errors.append(f"folder invalid: track must be {TRACKS}, testset {TESTSETS}, non-empty team/affiliation/date")
        return v

    json_dir = sub / "json"
    if not json_dir.is_dir():
        v.errors.append(f"missing {json_dir!s}/ (the evaluator writes metrics here)")
        return v

    task_names = load_task_names()
    n_inst = NUM_PUBLIC_TEST_INSTANCES if v.testset == "public" else NUM_HIDDEN_TEST_INSTANCES
    window = PUBLIC_INSTANCE_IDS if v.testset == "public" else HIDDEN_INSTANCE_IDS

    rollouts: list[Rollout] = []
    files = sorted(p for p in json_dir.iterdir() if p.suffix == ".json")
    if not files:
        v.errors.append(f"{json_dir!s}/ contains no *.json metrics")
    for p in files:
        if not file_pattern_valid(p.name, v.testset):
            v.errors.append(f"{p.name}: illegal filename (task name / instance {window} / rollout 0)")
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            v.errors.append(f"{p.name}: unreadable JSON ({e})")
            continue
        errs, r = _schema_check(obj, p.name)
        v.errors.extend(errs)
        if r is None:
            continue
        m = _FILE_RE.match(p.name)
        f_task, f_inst, f_roll = m.group(1), int(m.group(2)), int(m.group(3))
        if r.task != f_task:
            v.errors.append(f"{p.name}: task {r.task!r} != filename task {f_task!r}")
            continue
        if r.instance_id != f_inst:
            v.errors.append(f"{p.name}: instance_id {r.instance_id} != filename {f_inst}")
            continue
        if r.rollout_id != f_roll:
            v.errors.append(f"{p.name}: rollout_id {r.rollout_id} != filename {f_roll}")
            continue
        if f_roll != 0:
            v.warnings.append(f"{p.name}: rollout_id != 0 (scoring uses 1 rollout/instance)")
        rollouts.append(r)

    v.valid_rollouts = rollouts
    v.n_rollouts = len(rollouts)

    # Completeness warnings (do not fail the gate).
    present = {r.task for r in rollouts}
    for t in present:
        have = {r.instance_id for r in rollouts if r.task == t}
        missing = [i for i in window if i not in have]
        if missing:
            v.warnings.append(f"completeness: {t} has {len(missing)}/20 scored instances missing ({missing[0]}..{missing[-1]})")

    scores = get_scores(rollouts, len(task_names), n_inst, task_names)
    v.per_task_q = scores["per_task_q"]
    v.per_task_sr = scores["per_task_sr"]
    v.overall_q = scores["overall_q"]
    v.overall_task_sr = scores["overall_task_sr"]
    v.overall_time_score = scores["overall_time_score"]
    v.passed = len(v.errors) == 0
    return v


def score_to_json(v: Validation) -> dict:
    """Machine-readable score block for reports / the submission manifest."""
    return {
        "overall_q": v.overall_q,
        "overall_task_sr": v.overall_task_sr,
        "overall_time_score": v.overall_time_score,
        "n_rollouts": v.n_rollouts,
        "n_tasks": len(load_task_names()),
        "per_task_q": {k: val for k, val in v.per_task_q.items() if val != 0.0},
    }


def validation_to_dict(v: Validation) -> dict:
    d = score_to_json(v)
    d.update({
        "folder": v.folder,
        "track": v.track,
        "testset": v.testset,
        "team": v.team,
        "affiliation": v.affiliation,
        "date": v.date,
        "passed": v.passed,
        "n_errors": len(v.errors),
        "n_warnings": len(v.warnings),
        "errors": list(v.errors),
        "warnings": list(v.warnings),
    })
    return d
