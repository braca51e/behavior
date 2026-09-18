"""Aggregate tests: Q / time_score recomputed exactly per the challenge's
score_utils formula, validated against the 5 fixture metrics JSONs
(contract item (b)).

The fixture dir (``tests/fixtures``) holds 5 per-rollout JSONs for one task
covering q_score.final = 0 / 0.5 / 1.0 with and without early stop, plus a
time_score case.  ``aggregate`` must reproduce the leaderboard math.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evalharness.aggregate import (  # noqa: E402
    Summary,
    aggregate,
    time_score_from_normalized,
)

FIX = REPO / "tests" / "fixtures"


def test_time_score_formula():
    # time_score = 3 - 2/normalized_time (EVAL_TIMEOUT_MULTIPLIER = 1.5),
    # exactly as score_utils.compute_final_q_score computes it from the
    # metrics JSON `time.normalized_time`.  (The code, not the spec prose, is
    # authoritative per the challenge-spec rule.)
    assert time_score_from_normalized(1.6) == pytest.approx(3 - 2 / 1.6)   # 1.75
    assert time_score_from_normalized(1.0) == pytest.approx(1.0)
    assert time_score_from_normalized(1.5) == pytest.approx(3 - 2 / 1.5)   # 1.6667
    assert time_score_from_normalized(0.5) == pytest.approx(-1.0)         # 3 - 4
    assert time_score_from_normalized(2.0) == pytest.approx(2.0)          # 3 - 1
    assert time_score_from_normalized(0.75) == pytest.approx(0.33333, abs=1e-4)
    assert time_score_from_normalized(0.0) == 0.0                         # defensive


def test_aggregate_fixture_q():
    s = aggregate(FIX)
    assert isinstance(s, Summary)
    # The 5 fixtures are all the same task; Q = mean of their q finals.
    q_vals = [0.0, 0.5, 1.0, 1.0, 0.5]
    expected_q = sum(q_vals) / len(q_vals)
    assert s.overall["q_score"] == pytest.approx(expected_q)
    assert s.overall["num_rollouts"] == 5
    assert s.overall["num_tasks"] == 1


def test_aggregate_task_sr():
    s = aggregate(FIX)
    # task_sr = fraction of rollouts with q == 1.0 -> 2 of 5.
    assert s.overall["task_sr"] == pytest.approx(0.4)


def test_aggregate_time_score():
    s = aggregate(FIX)
    # Per-rollout normalized_time: 1.6, 1.2, 0.8, 0.6, 1.5.
    expected = np.mean([
        time_score_from_normalized(1.6),
        time_score_from_normalized(1.2),
        time_score_from_normalized(0.8),
        time_score_from_normalized(0.6),
        time_score_from_normalized(1.5),
    ])
    assert s.overall["time_score"] == pytest.approx(expected)


def test_summary_json_roundtrip(tmp_path):
    s = aggregate(FIX)
    from evalharness.report import summary_to_json

    out = tmp_path / "summary.json"
    summary_to_json(s, out)
    obj = json.loads(out.read_text())
    assert obj["overall"]["q_score"] == pytest.approx(s.overall["q_score"])
    assert "per_task" in obj


def test_report_renders(tmp_path):
    s = aggregate(FIX)
    from evalharness.report import render_report

    p = render_report(s, tmp_path / "report.md", team="b1k")
    text = p.read_text()
    assert "Self-eval report" in text
    assert "Q (mean task BDDL predicate fraction)" in text
    # per-task table present
    assert "turning_on_radio" in text
    assert (tmp_path / "summary.json").exists()
