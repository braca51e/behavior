"""Validation-harness tests: submission format, scoring parity, edge cases.

These lock the local harness to the organizers' contract so a fresh checkout
can verify the solution is functional and submission-ready without external
testing.  They exercise ``validation.validator`` (the exact official
``score_utils`` math) against:

* a hand-computed aggregation case (known Q / task_sr / time_score),
* the official time_score formula string (``3 - 2/nt``),
* the sample submissions + their frozen ``expected_*.json`` (round-trip),
* malformed / out-of-bounds / unknown-task / wrong-track inputs (accept/reject),
* the folder-name and file-naming contract (spec §7.1).

Run::

    python3 -m pytest tests/test_validation.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO / "src"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import validation.validator as v  # noqa: E402
from validation.validator import (  # noqa: E402
    Rollout,
    file_pattern_valid,
    get_scores,
    load_task_names,
    parse_submission_name,
    time_score_from_normalized,
    validate_submission,
)

SAMPLE_DIR = REPO / "validation" / "sample_data"
FULL = "standard.public.b1k.nous.20260912"
SMALL = "standard.public.b1k.nous.20260910"


# ---------------------------------------------------------------------------
# time_score formula
# ---------------------------------------------------------------------------
def test_time_score_official_formula():
    # Official score_utils: time_score = 1.5/(1.5-1) - 1/((1.5-1)*nt) = 3 - 2/nt.
    # (Per the challenge rule "code is authoritative", both raw/score_utils.py and
    #  evalharness.aggregate implement 3 - 2/nt; the repo's own tests/test_aggregate.py
    #  pins time_score_from_normalized(1.5) == 3 - 2/1.5.)
    for nt in (0.6, 0.8, 1.0, 1.2, 1.5, 1.9):
        assert time_score_from_normalized(nt) == pytest.approx(3.0 - 2.0 / nt)
    assert time_score_from_normalized(1.0) == pytest.approx(1.0)      # human speed
    assert time_score_from_normalized(2.0) == pytest.approx(2.0)      # twice as slow
    assert time_score_from_normalized(1.5) == pytest.approx(3.0 - 2.0 / 1.5)  # 1.667


# ---------------------------------------------------------------------------
# hand-computed aggregation parity
# ---------------------------------------------------------------------------
def test_get_scores_hand_computed():
    A, B, C = "alpha", "beta", "gamma"
    rolls = [
        Rollout(A, 0, 0, 1.0, 1.0, 0.1, 0.1, 0.1),
        Rollout(A, 1, 0, 0.5, -1.0, 0.2, 0.2, 0.2),
        Rollout(B, 0, 0, 0.0, 1.0, 0.3, 0.3, 0.3),
        Rollout(B, 1, 0, 1.0, 2.0, 0.4, 0.4, 0.4),
    ]
    s = get_scores(rolls, n_tasks=3, n_instances_per_task=2, task_names=[A, B, C])
    # per-task (official math: Q/task_sr over full instance count; time/dist
    # over present rollouts):
    #   A: q=(1.0+0.5)/2=0.75, sr=1/2, time=(1.0+(-1.0))/2=0.0
    #   B: q=(0.0+1.0)/2=0.50, sr=1/2, time=(1.0+2.0)/2=1.5
    #   C: 0 (absent)
    assert s["per_task_q"][A] == pytest.approx(0.75)
    assert s["per_task_q"][B] == pytest.approx(0.50)
    assert s["per_task_q"][C] == pytest.approx(0.0)
    assert s["per_task_sr"][A] == pytest.approx(0.5)
    assert s["per_task_time_score"][B] == pytest.approx(1.5)
    # overall (mean over all 3 tasks):
    #   Q=(0.75+0.50+0)/3=5/12 ; task_sr=(0.5+0.5+0)/3=1/3 ; time=(0.0+1.5+0)/3=0.5
    assert s["overall_q"] == pytest.approx(5.0 / 12.0)
    assert s["overall_task_sr"] == pytest.approx(1.0 / 3.0)
    assert s["overall_time_score"] == pytest.approx(0.5)
    assert s["n_rollouts"] == 4


def test_get_scores_missing_counts_as_zero():
    # Only task A has data; B and C must count as 0 in the denominator.
    A, B, C = "alpha", "beta", "gamma"
    s = get_scores([Rollout(A, 0, 0, 1.0, 1.0, 0, 0, 0)], n_tasks=3,
                   n_instances_per_task=1, task_names=[A, B, C])
    assert s["overall_q"] == pytest.approx(1.0 / 3.0)  # 1.0 from A, 0 from B,C
    assert s["per_task_q"][B] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# folder-name + file-naming contract
# ---------------------------------------------------------------------------
def test_parse_submission_name_valid():
    p = parse_submission_name("standard.public.b1k.nous.20260912")
    assert p and p["ok"] and p["track"] == "standard" and p["testset"] == "public"


def test_parse_submission_name_malformed():
    assert parse_submission_name("_edge_bad") is None              # 1 part
    assert parse_submission_name("standard.public.b1k.20260912") is None  # 4 parts
    p = parse_submission_name("badtrack.public.b1k.nous.20260912")
    assert p and p["ok"] is False                                   # unknown track
    p2 = parse_submission_name("standard.privated.b1k.nous.20260912")
    assert p2 and p2["ok"] is False                                 # unknown testset
    p3 = parse_submission_name("standard.public..nous.20260912")
    assert p3 and p3["ok"] is False                                 # empty team


def test_file_pattern_valid_accepts_and_rejects():
    names = load_task_names()
    t0 = names[0]
    assert file_pattern_valid(f"{t0}_301_0.json", "public") is True
    assert file_pattern_valid(f"{t0}_320_0.json", "public") is True   # last public
    assert file_pattern_valid(f"{t0}_321_0.json", "public") is False   # hidden window
    assert file_pattern_valid(f"{t0}_321_0.json", "hidden") is True
    assert file_pattern_valid(f"{t0}_400_0.json") is False             # out of range
    assert file_pattern_valid(f"{t0}_301_1.json") is False             # rollout != 0
    assert file_pattern_valid("not_a_task_301_0.json") is False        # unknown task
    assert file_pattern_valid(f"{t0}_301_0.txt") is False              # wrong ext


# ---------------------------------------------------------------------------
# sample round-trip (frozen expected outputs)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _ensure_samples():
    if not (SAMPLE_DIR / FULL).exists():
        # generate if absent (idempotent, seeded)
        proc = __import__("subprocess").run(
            [sys.executable, str(REPO / "validation" / "make_sample_data.py")],
            cwd=str(REPO), capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    yield


def test_sample_roundtrip(_ensure_samples):
    for folder in (FULL, SMALL):
        exp_p = SAMPLE_DIR / f"expected_{folder}.json"
        assert exp_p.exists(), f"missing expected file {exp_p.name}"
        exp = json.loads(exp_p.read_text())
        res = validate_submission(SAMPLE_DIR / folder)
        assert res.passed, f"{folder} failed: {res.errors[:5]}"
        assert res.n_rollouts == exp["n_rollouts"]
        assert res.overall_q == pytest.approx(exp["overall_q"], abs=1e-12)
        assert res.overall_task_sr == pytest.approx(exp["overall_task_sr"], abs=1e-12)
        assert res.overall_time_score == pytest.approx(exp["overall_time_score"], abs=1e-12)
        # per-task Q frozen as well
        for t, q in exp["per_task_q"].items():
            assert res.per_task_q.get(t, 0.0) == pytest.approx(q, abs=1e-12)


def test_full_sample_is_2000_rollouts(_ensure_samples):
    res = validate_submission(SAMPLE_DIR / FULL)
    assert res.n_rollouts == 2000          # 100 tasks x 20 public instances
    assert res.passed


# ---------------------------------------------------------------------------
# edge cases: malformed inputs are rejected, not crashed on
# ---------------------------------------------------------------------------
def _write_metrics(dirpath: Path, fname: str, task: str, inst: int, roll: int,
                   q: float, nt: float, schema_errors: list | None = None):
    rec = {
        "task": task, "instance_id": inst, "rollout_id": roll, "steps": 100,
        "success": q >= 1.0,
        "agent_distance": {"base": 1.0, "left": 1.0, "right": 1.0},
        "normalized_agent_distance": {"base": 0.5, "left": 0.5, "right": 0.5},
        "q_score": {"final": q},
        "time": {"simulator_steps": 100, "simulator_time": 3.3, "normalized_time": nt},
    }
    for k in (schema_errors or []):
        del rec[k]
    (dirpath / fname).write_text(json.dumps(rec))


def test_rejects_out_of_bounds_instance(tmp_path):
    folder = tmp_path / "standard.public.b1k.nous.20260101"
    (folder / "json").mkdir(parents=True)
    t0 = load_task_names()[0]
    _write_metrics(folder / "json", f"{t0}_400_0.json", t0, 400, 0, 1.0, 1.0)
    res = validate_submission(folder)
    assert not res.passed
    assert any("illegal" in e for e in res.errors)


def test_rejects_unknown_task(tmp_path):
    folder = tmp_path / "standard.public.b1k.nous.20260102"
    (folder / "json").mkdir(parents=True)
    _write_metrics(folder / "json", "ghost_task_301_0.json", "ghost_task", 301, 0, 1.0, 1.0)
    res = validate_submission(folder)
    assert not res.passed
    assert any("ghost_task" in e for e in res.errors)


def test_rejects_bad_schema(tmp_path):
    folder = tmp_path / "standard.public.b1k.nous.20260103"
    (folder / "json").mkdir(parents=True)
    t0 = load_task_names()[0]
    # missing q_score and time entirely
    _write_metrics(folder / "json", f"{t0}_301_0.json", t0, 301, 0, 1.0, 1.0,
                   schema_errors=["q_score", "time"])
    res = validate_submission(folder)
    assert not res.passed
    assert any("missing key 'q_score'" in e for e in res.errors)
    assert any("missing key 'time'" in e for e in res.errors)


def test_rejects_q_out_of_range(tmp_path):
    folder = tmp_path / "standard.public.b1k.nous.20260104"
    (folder / "json").mkdir(parents=True)
    t0 = load_task_names()[0]
    _write_metrics(folder / "json", f"{t0}_301_0.json", t0, 301, 0, 1.7, 1.0)  # q > 1
    res = validate_submission(folder)
    assert not res.passed
    assert any("out of [0,1]" in e for e in res.errors)


def test_rejects_malformed_folder(tmp_path):
    folder = tmp_path / "not_a_valid_name"
    (folder / "json").mkdir(parents=True)
    res = validate_submission(folder)
    assert not res.passed
    assert any("not <track>" in e for e in res.errors)


def test_rejects_missing_json_dir(tmp_path):
    folder = tmp_path / "standard.public.b1k.nous.20260105"
    folder.mkdir(parents=True)
    res = validate_submission(folder)
    assert not res.passed
    assert any("missing" in e for e in res.errors)


# ---------------------------------------------------------------------------
# the serving pipeline still answers safely (integration: no crash on bad obs)
# ---------------------------------------------------------------------------
def test_server_handles_missing_cameras(tmp_path):
    pytest.importorskip("b1k.server")
    from b1k.config import load_config
    from b1k.embodiment import ACTION_DIM
    from b1k.protocol import frame_from_dict
    from b1k.server import B1KServer

    cfg = load_config(REPO / "configs" / "server.yaml")
    srv = B1KServer(cfg, task=0)
    # proprio-only frame (no cameras) must still yield a finite 23-dim action.
    obs = {"robot_r1::proprio": [0.05, 0.0, 0.0] + [0.0] * 58}
    a = np.asarray(srv.handle_frame(frame_from_dict(obs)), dtype=np.float32)
    assert a.shape == (ACTION_DIM,)
    assert bool(np.all(np.isfinite(a)))
